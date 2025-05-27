from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import logging
from datetime import datetime
import os
import re
import hashlib

class LogParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # We'll use a simpler approach to avoid Drain3 initialization issues
        try:
            # Try to initialize Drain3, but have a fallback if it fails
            config = TemplateMinerConfig()
            self.template_miner = TemplateMiner()
            self.use_drain3 = True
            self.logger.info("Using Drain3 for template mining")
        except Exception as e:
            # Fall back to a simple regex-based parser
            self.logger.warning(f"Failed to initialize Drain3: {str(e)}. Using fallback parser.")
            self.use_drain3 = False
            self.templates = {}  # Store templates as {template_hash: template}
            self.template_counter = 0
    
    def parse(self, log_line):
        """
        Parse a log line and return template_id, variables, and timestamp.
        
        Args:
            log_line (str): Raw log line to parse
            
        Returns:
            tuple: (template_id, variables, timestamp)
        """
        try:
            # Extract timestamp (assuming it's at the start of the line)
            timestamp = None
            try:
                # Try to parse timestamp from the beginning of the line
                timestamp_str = log_line.split()[0]
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            except (ValueError, IndexError):
                self.logger.warning(f"Could not parse timestamp from line: {log_line[:100]}...")
                timestamp = datetime.now()

            # Process the log line to extract template and variables
            if self.use_drain3:
                return self._parse_with_drain3(log_line, timestamp)
            else:
                return self._parse_with_fallback(log_line, timestamp)

        except Exception as e:
            self.logger.error(f"Error parsing log line: {str(e)}")
            raise

    def _parse_with_drain3(self, log_line, timestamp):
        """Use Drain3 to parse the log line."""
        result = self.template_miner.add_log_message(log_line)
        
        if result["change_type"] != "none":
            self.logger.debug(f"New template found: {result['template_mined']}")
        
        # Extract variables from the template
        variables = self._extract_variables(log_line, result["template_mined"])
        
        return result["cluster_id"], variables, timestamp

    def _parse_with_fallback(self, log_line, timestamp):
        """Simple fallback parser that uses regex patterns."""
        # Create a template by replacing numbers and identifiers with placeholders
        template = re.sub(r'\b\d+\b', '<*>', log_line)  # Replace numbers
        template = re.sub(r'\b[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\b', '<*>', template)  # Replace identifiers
        
        # Generate a hash for the template
        template_hash = hashlib.md5(template.encode()).hexdigest()
        
        # Store the template if it's new
        if template_hash not in self.templates:
            self.template_counter += 1
            self.templates[template_hash] = {
                'id': self.template_counter,
                'template': template,
                'count': 0
            }
        
        # Update template count
        self.templates[template_hash]['count'] += 1
        template_id = self.templates[template_hash]['id']
        
        # Extract variables by comparing log line with template
        variables = self._extract_variables(log_line, template)
        
        return template_id, variables, timestamp

    def _extract_variables(self, log_line, template):
        """
        Extract variables from a log line based on its template.
        
        Args:
            log_line (str): Raw log line
            template (str): Template with <*> placeholders
            
        Returns:
            dict: Dictionary of variable names and values
        """
        variables = {}
        template_parts = template.split()
        log_parts = log_line.split()
        
        if len(template_parts) != len(log_parts):
            return variables
            
        for i, (t_part, l_part) in enumerate(zip(template_parts, log_parts)):
            if t_part == "<*>":
                variables[f"var_{i}"] = l_part
                
        return variables 