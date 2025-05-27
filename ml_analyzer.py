import logging
import random
import json
import re
from typing import Dict, List, Tuple, Optional, Callable
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from budget_guard import BudgetGuard

logger = logging.getLogger(__name__)

class MLAnalyzer:
    def __init__(self, config: Dict):
        """Initialize ML analyzer with configuration."""
        self.config = config
        # Initialize budget guard
        budget_config = config.get('budget', {})
        max_budget = budget_config.get('max_budget', 45.0)
        buffer = budget_config.get('buffer', 5.0)
        self.budget_guard = BudgetGuard(max_budget=max_budget, buffer=buffer)
        
        # Model configuration
        self.local_model_name = config.get('local_model', 'google/gemma-2b')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Analysis context
        self.log_context = None
        self.adaptive_prompt = None
        
        # Initialize model components
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the local model if available."""
        try:
            logger.info(f"Loading local model: {self.local_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self.model = None
            self.tokenizer = None
    
    def _sample_log_lines(self, log_lines: List[str], sample_size: int = 10) -> List[str]:
        """Sample random lines from the log file for context discovery."""
        if len(log_lines) <= sample_size:
            return log_lines
        
        # Sample lines from different parts of the file for better representation
        file_length = len(log_lines)
        indices = []
        
        # Get samples from beginning, middle, and end
        for section in range(3):
            start_idx = (file_length // 3) * section
            end_idx = (file_length // 3) * (section + 1)
            section_samples = min(sample_size // 3 + 1, end_idx - start_idx)
            section_indices = random.sample(range(start_idx, end_idx), section_samples)
            indices.extend(section_indices)
        
        # Fill remaining slots with completely random samples
        remaining = sample_size - len(indices)
        if remaining > 0:
            available_indices = [i for i in range(file_length) if i not in indices]
            if available_indices:
                additional_indices = random.sample(
                    available_indices, 
                    min(remaining, len(available_indices))
                )
                indices.extend(additional_indices)
        
        return [log_lines[i] for i in sorted(indices[:sample_size])]
    
    def _discover_log_context(self, sample_lines: List[str]) -> Dict:
        """
        Analyze sample lines to discover log structure and patterns.
        Uses three-tier approach: Cloud → Gemma-2B → Keywords
        """
        if not sample_lines:
            return {}
        
        logger.info("Starting context discovery with cloud-first approach...")
        
        # Tier 1: Try cloud analysis first (most accurate)
        try:
            if self.budget_guard and self.budget_guard.can_make_request():
                logger.info("Attempting cloud-based context discovery...")
                context = self._discover_context_cloud(sample_lines)
                if context and context.get('LOG_TYPE') and context.get('LOG_TYPE') != 'Unknown':
                    logger.info(f"✅ Cloud context discovery successful: {context.get('LOG_TYPE')}")
                    return context
                else:
                    logger.info("⚠️ Cloud context discovery returned unclear results")
            else:
                logger.info("💰 Budget limit reached, skipping cloud context discovery")
        except Exception as e:
            logger.warning(f"❌ Cloud context discovery failed: {e}")
        
        # Tier 2: Try local Gemma-2B model
        try:
            if self.model:
                logger.info("Attempting Gemma-2B context discovery...")
                context = self._discover_context_gemma(sample_lines)
                if context and context.get('LOG_TYPE') and context.get('LOG_TYPE') != 'Unknown':
                    logger.info(f"✅ Gemma-2B context discovery successful: {context.get('LOG_TYPE')}")
                    return context
                else:
                    logger.info("⚠️ Gemma-2B context discovery returned unclear results")
            else:
                logger.info("🤖 Gemma-2B model not available")
        except Exception as e:
            logger.warning(f"❌ Gemma-2B context discovery failed: {e}")
        
        # Tier 3: Fallback to keyword-based analysis
        logger.info("Using keyword-based fallback context discovery...")
        context = self._basic_context_analysis(sample_lines)
        logger.info(f"📝 Keyword-based detection result: {context.get('LOG_TYPE')}")
        return context
    
    def _discover_context_cloud(self, sample_lines: List[str]) -> Dict:
        """Use cloud API (OpenAI) for context discovery."""
        import openai
        import os
        
        # Check if API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise Exception("OpenAI API key not found")
        
        # Create context discovery prompt
        sample_text = "\n".join(sample_lines)
        
        prompt = f"""Analyze these log file samples and identify the log type and characteristics:

LOG SAMPLES:
{sample_text}

Please analyze and provide a JSON response with:
1. "LOG_TYPE": What type of logs are these? (options: "Linux System", "Web Server", "Database", "Application", "Network", "Security", or "Unknown")
2. "LOG_FORMAT": Brief description of the structure/format
3. "SEVERITY_INDICATORS": List of severity levels found (ERROR, WARN, INFO, etc.)
4. "KEY_CHARACTERISTICS": List of key identifying features
5. "CONFIDENCE": Your confidence level (0.0-1.0)

Focus on identifying Linux/Unix system logs (ftpd, sshd, kernel, systemd, cron, syslog, etc.)

Respond ONLY with valid JSON:"""

        try:
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            # Track budget usage
            if self.budget_guard:
                # Estimate tokens (rough calculation)
                input_tokens = len(prompt.split()) * 1.3  # Rough token estimate
                output_tokens = len(response.choices[0].message.content.split()) * 1.3
                
                # Update budget usage for input and output tokens
                self.budget_guard.update_usage(int(input_tokens), is_input=True)
                self.budget_guard.update_usage(int(output_tokens), is_input=False)
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                context_data = json.loads(content)
                logger.info("Cloud context discovery successful")
                return context_data
            except json.JSONDecodeError:
                # Try to extract key info if JSON parsing fails
                logger.warning("Cloud response not valid JSON, attempting extraction")
                return self._extract_context_from_text(content, sample_lines)
                
        except Exception as e:
            logger.error(f"Cloud context discovery failed: {e}")
            raise
    
    def _discover_context_gemma(self, sample_lines: List[str]) -> Dict:
        """Use local Gemma-2B model for context discovery."""
        if not self.model:
            return {}
        
        try:
            # Create context discovery prompt
            sample_text = "\n".join(sample_lines)
            
            context_prompt = f"""Analyze these log file samples and provide a structured analysis:

LOG SAMPLES:
{sample_text}

Please analyze and provide:
1. LOG_TYPE: What type of logs are these? (web server, application, system, database, etc.)
2. LOG_FORMAT: What's the structure/format? (timestamp format, fields, delimiters)
3. KEY_FIELDS: What are the main fields present? (timestamp, level, source, message, etc.)
4. SEVERITY_INDICATORS: How are error levels indicated? (ERROR, WARN, INFO, etc.)
5. COMMON_PATTERNS: What normal patterns do you see?
6. ANOMALY_INDICATORS: What would indicate anomalies in this type of log?
7. CONTEXT_KEYWORDS: Important keywords or identifiers specific to this system

Respond in JSON format only:"""

            # Generate context analysis
            inputs = self.tokenizer(context_prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            context_response = response.split("Respond in JSON format only:")[-1].strip()
            
            # Try to parse JSON response
            try:
                context_data = json.loads(context_response)
                logger.info("Gemma-2B context discovery successful")
                return context_data
            except json.JSONDecodeError:
                # Fallback: extract key information using regex
                logger.warning("Gemma-2B response not valid JSON, using fallback extraction")
                return self._extract_context_fallback(context_response, sample_lines)
                
        except Exception as e:
            logger.error(f"Gemma-2B context discovery failed: {e}")
            return {}
    
    def _extract_context_from_text(self, text: str, sample_lines: List[str]) -> Dict:
        """Extract context information from free-form text response."""
        context = {
            'LOG_TYPE': 'Unknown',
            'LOG_FORMAT': 'Mixed',
            'SEVERITY_INDICATORS': [],
            'KEY_CHARACTERISTICS': [],
            'CONFIDENCE': 0.5
        }
        
        # Look for log type mentions
        text_upper = text.upper()
        if any(term in text_upper for term in ['LINUX', 'UNIX', 'SYSTEM', 'SYSLOG']):
            context['LOG_TYPE'] = 'Linux System'
        elif any(term in text_upper for term in ['WEB', 'HTTP', 'APACHE', 'NGINX']):
            context['LOG_TYPE'] = 'Web Server'
        elif any(term in text_upper for term in ['DATABASE', 'SQL', 'MYSQL', 'POSTGRES']):
            context['LOG_TYPE'] = 'Database'
        
        # Extract severity indicators
        for level in ['ERROR', 'WARN', 'INFO', 'DEBUG', 'FATAL', 'TRACE']:
            if level in text_upper:
                context['SEVERITY_INDICATORS'].append(level)
        
        return context
    
    def _extract_context_fallback(self, response: str, sample_lines: List[str]) -> Dict:
        """Fallback method to extract context information."""
        context = {}
        
        # Extract information using regex patterns
        log_type_match = re.search(r'LOG_TYPE[:\s]+([^\n]+)', response, re.IGNORECASE)
        if log_type_match:
            context['LOG_TYPE'] = log_type_match.group(1).strip()
        
        format_match = re.search(r'LOG_FORMAT[:\s]+([^\n]+)', response, re.IGNORECASE)
        if format_match:
            context['LOG_FORMAT'] = format_match.group(1).strip()
        
        # Basic analysis of sample lines
        severity_levels = set()
        for line in sample_lines:
            for level in ['ERROR', 'WARN', 'INFO', 'DEBUG', 'FATAL', 'TRACE']:
                if level in line.upper():
                    severity_levels.add(level)
        
        context['SEVERITY_INDICATORS'] = list(severity_levels)
        context['SAMPLE_COUNT'] = len(sample_lines)
        
        return context
    
    def _basic_context_analysis(self, sample_lines: List[str]) -> Dict:
        """Basic rule-based context analysis as ultimate fallback."""
        context = {
            'LOG_TYPE': 'Unknown',
            'LOG_FORMAT': 'Mixed',
            'SEVERITY_INDICATORS': [],
            'COMMON_PATTERNS': [],
            'SAMPLE_COUNT': len(sample_lines)
        }
        
        # Detect common log types
        combined_text = ' '.join(sample_lines).upper()
        
        # Linux/Unix system logs (check first as they're specific)
        linux_indicators = ['UNIX', 'LINUX', 'FTPD', 'SSHD', 'KERNEL', 'SYSTEMD', 'CRON', 'SYSLOG', 'DAEMON', 'AUTH', 'MAIL', 'SECURITY']
        if any(indicator in combined_text for indicator in linux_indicators):
            context['LOG_TYPE'] = 'Linux System'
            # Log which indicators were found for debugging
            found_indicators = [ind for ind in linux_indicators if ind in combined_text]
            logger.info(f"Linux system log detected based on indicators: {found_indicators}")
        elif any(indicator in combined_text for indicator in ['HTTP', 'GET', 'POST', 'RESPONSE']):
            context['LOG_TYPE'] = 'Web Server'
        elif any(indicator in combined_text for indicator in ['SQL', 'DATABASE', 'QUERY']):
            context['LOG_TYPE'] = 'Database'
        elif any(indicator in combined_text for indicator in ['KERNEL', 'SYSTEM', 'BOOT']):
            context['LOG_TYPE'] = 'System'
        elif any(indicator in combined_text for indicator in ['APPLICATION', 'APP', 'SERVICE']):
            context['LOG_TYPE'] = 'Application'
        
        # Find severity indicators
        for line in sample_lines:
            for level in ['ERROR', 'WARN', 'INFO', 'DEBUG', 'FATAL', 'TRACE']:
                if level in line.upper() and level not in context['SEVERITY_INDICATORS']:
                    context['SEVERITY_INDICATORS'].append(level)
        
        return context
    
    def _create_adaptive_prompt(self, context: Dict) -> str:
        """Create a tailored prompt based on discovered context."""
        if not context:
            return self._get_default_prompt()
        
        log_type = context.get('LOG_TYPE', 'Unknown')
        severity_indicators = context.get('SEVERITY_INDICATORS', [])
        log_format = context.get('LOG_FORMAT', 'Unknown')
        anomaly_indicators = context.get('ANOMALY_INDICATORS', [])
        
        # Build adaptive prompt
        prompt = f"""You are analyzing {log_type} logs with the following characteristics:
- Format: {log_format}
- Known severity levels: {', '.join(severity_indicators) if severity_indicators else 'Standard levels'}

"""
        
        if anomaly_indicators:
            prompt += f"Known anomaly patterns for this log type: {', '.join(anomaly_indicators)}\n\n"
        
        prompt += f"""For each log line, determine if it indicates an anomaly. Consider:
1. Severity level (focus on: {', '.join(severity_indicators) if severity_indicators else 'ERROR, WARN, FATAL'})
2. {log_type}-specific error patterns
3. Unusual patterns or values
4. Performance issues or timeouts

Respond with: NORMAL or ANOMALY followed by confidence (0.0-1.0) and brief reason.
Format: [NORMAL|ANOMALY] confidence=X.X reason="brief explanation"

Log line to analyze:"""
        
        return prompt
    
    def _get_default_prompt(self) -> str:
        """Default prompt when context discovery fails."""
        return """Analyze this log line for anomalies. Look for errors, warnings, failures, timeouts, or unusual patterns.

Respond with: NORMAL or ANOMALY followed by confidence (0.0-1.0) and brief reason.
Format: [NORMAL|ANOMALY] confidence=X.X reason="brief explanation"

Log line to analyze:"""
    
    def analyze_logs(self, log_lines: List[str], shutdown_check: Optional[Callable[[], bool]] = None) -> Dict:
        """
        Analyze logs using two-phase approach:
        1. Context discovery phase
        2. Adaptive line-by-line analysis
        """
        results = {
            'total_lines': len(log_lines),
            'anomalies': [],
            'context': {},
            'analysis_method': 'adaptive_context_aware',
            'stats': {
                'normal_count': 0,
                'anomaly_count': 0,
                'context_discovery_success': False
            }
        }
        
        if not log_lines:
            return results
        
        logger.info("Starting adaptive context-aware log analysis...")
        
        # Phase 1: Context Discovery
        logger.info("Phase 1: Discovering log context...")
        sample_lines = self._sample_log_lines(log_lines, 10)
        self.log_context = self._discover_log_context(sample_lines)
        
        if self.log_context:
            results['context'] = self.log_context
            results['stats']['context_discovery_success'] = True
            logger.info(f"Context discovered: {self.log_context.get('LOG_TYPE', 'Unknown')} logs")
        
        # Phase 2: Adaptive Analysis
        logger.info("Phase 2: Adaptive line-by-line analysis...")
        self.adaptive_prompt = self._create_adaptive_prompt(self.log_context)
        
        # Analyze each log line with adaptive context
        for i, line in enumerate(log_lines):
            # Check for shutdown request
            if shutdown_check and shutdown_check():
                logger.info(f"Shutdown requested, stopping analysis at line {i + 1}/{len(log_lines)}")
                break
                
            if not line.strip():
                continue
                
            try:
                analysis_result = self._analyze_single_line_adaptive(line.strip())
                
                if analysis_result['is_anomaly']:
                    results['anomalies'].append({
                        'line_number': i + 1,
                        'content': line.strip(),
                        'confidence': analysis_result['confidence'],
                        'reason': analysis_result['reason'],
                        'method': 'gemma_adaptive'
                    })
                    results['stats']['anomaly_count'] += 1
                else:
                    results['stats']['normal_count'] += 1
                    
                # Log progress and check shutdown every 10 lines for responsiveness
                if (i + 1) % 10 == 0:
                    if shutdown_check and shutdown_check():
                        logger.info(f"Shutdown requested, stopping analysis at line {i + 1}/{len(log_lines)}")
                        break
                        
                if (i + 1) % 100 == 0:
                    logger.info(f"Analyzed {i + 1}/{len(log_lines)} lines")
                    
            except Exception as e:
                logger.error(f"Error analyzing line {i + 1}: {e}")
                continue
        
        logger.info(f"Analysis complete: {results['stats']['anomaly_count']} anomalies found in {results['total_lines']} lines")
        return results
    
    def _analyze_single_line_adaptive(self, line: str) -> Dict:
        """Analyze a single log line using adaptive context-aware prompt."""
        if not self.model:
            return self._heuristic_analysis(line)
        
        try:
            # Use adaptive prompt with discovered context
            full_prompt = f"{self.adaptive_prompt}\n{line}"
            
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            analysis_part = response.split("Log line to analyze:")[-1].strip()
            
            return self._parse_analysis_response(analysis_part)
            
        except Exception as e:
            logger.error(f"Adaptive analysis failed: {e}")
            return self._heuristic_analysis(line)
    
    def _parse_analysis_response(self, response: str) -> Dict:
        """Parse the model's analysis response."""
        # Default result
        result = {
            'is_anomaly': False,
            'confidence': 0.5,
            'reason': 'Analysis unclear'
        }
        
        try:
            # Look for ANOMALY or NORMAL in response
            response_upper = response.upper()
            
            if 'ANOMALY' in response_upper:
                result['is_anomaly'] = True
            elif 'NORMAL' in response_upper:
                result['is_anomaly'] = False
            else:
                # Fallback to heuristics if unclear
                return self._heuristic_analysis(response)
            
            # Extract confidence
            conf_match = re.search(r'confidence[=\s]+([0-9.]+)', response, re.IGNORECASE)
            if conf_match:
                confidence = float(conf_match.group(1))
                result['confidence'] = max(0.0, min(1.0, confidence))
            
            # Extract reason
            reason_match = re.search(r'reason[=\s]*["\']([^"\']+)["\']', response, re.IGNORECASE)
            if reason_match:
                result['reason'] = reason_match.group(1)
            elif 'ANOMALY' in response_upper:
                result['reason'] = 'Potential anomaly detected'
            else:
                result['reason'] = 'Normal log entry'
                
        except Exception as e:
            logger.warning(f"Error parsing response: {e}")
            
        return result
    
    def _heuristic_analysis(self, line: str) -> Dict:
        """Fallback heuristic analysis when model fails."""
        line_upper = line.upper()
        
        # Critical indicators
        critical_keywords = ['ERROR', 'FATAL', 'CRITICAL', 'EXCEPTION', 'FAILED', 'TIMEOUT']
        warning_keywords = ['WARN', 'WARNING', 'DEPRECATED', 'RETRY']
        
        if any(keyword in line_upper for keyword in critical_keywords):
            return {
                'is_anomaly': True,
                'confidence': 0.8,
                'reason': 'Critical error indicator detected'
            }
        elif any(keyword in line_upper for keyword in warning_keywords):
            return {
                'is_anomaly': True,
                'confidence': 0.6,
                'reason': 'Warning level indicator detected'
            }
        else:
            return {
                'is_anomaly': False,
                'confidence': 0.7,
                'reason': 'No anomaly indicators found'
            }
    
    def get_context_info(self) -> Dict:
        """Get discovered log context information."""
        return {
            'context': self.log_context or {},
            'adaptive_prompt_active': self.adaptive_prompt is not None,
            'model_available': self.model is not None
        }