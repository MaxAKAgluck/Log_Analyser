import json
import logging
from datetime import datetime
from colorama import init, Fore, Style
from collections import defaultdict
from typing import Dict

class LogReporter:
    def __init__(self):
        init()  # Initialize colorama
        self.logger = logging.getLogger(__name__)
        self.anomalies = defaultdict(list)
        self.severity_colors = {
            'high': Fore.RED,
            'medium': Fore.YELLOW,
            'low': Fore.GREEN
        }

    def add_anomaly(self, log_line, template_id, probability):
        """
        Add an anomaly to the report.
        
        Args:
            log_line (str): The anomalous log line
            template_id: ID of the template
            probability (float): Probability of the anomaly
        """
        # Determine severity based on probability
        severity = self._get_severity(probability)
        
        # Create anomaly entry
        anomaly = {
            'timestamp': datetime.now().isoformat(),
            'log_line': log_line,
            'template_id': template_id,
            'probability': probability,
            'severity': severity
        }
        
        # Add to anomalies collection
        self.anomalies[severity].append(anomaly)
        self.logger.debug(f"Added {severity} severity anomaly")

    def print_report(self):
        """Print the analysis report with colorized output."""
        print("\n=== Log Analysis Report ===\n")
        
        # Print by severity
        for severity in ['high', 'medium', 'low']:
            color = self.severity_colors[severity]
            anomalies = self.anomalies[severity]
            
            if anomalies:
                print(f"\n{color}{severity.upper()} Severity Anomalies ({len(anomalies)}):{Style.RESET_ALL}")
                
                for anomaly in anomalies:
                    print(f"\n{color}Timestamp: {anomaly['timestamp']}")
                    print(f"Probability: {anomaly['probability']:.2f}")
                    print(f"Template ID: {anomaly['template_id']}")
                    print(f"Log Line: {anomaly['log_line']}{Style.RESET_ALL}")
        
        # Print summary
        total = sum(len(anomalies) for anomalies in self.anomalies.values())
        print(f"\n{Fore.CYAN}Total Anomalies Found: {total}{Style.RESET_ALL}")

    def save_json(self, output_path):
        """
        Save the analysis results to a JSON file.
        
        Args:
            output_path (str): Path to save the JSON file
        """
        try:
            # Convert defaultdict to regular dict for JSON serialization
            report = {
                'timestamp': datetime.now().isoformat(),
                'anomalies': dict(self.anomalies),
                'summary': {
                    severity: len(anomalies)
                    for severity, anomalies in self.anomalies.items()
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            self.logger.info(f"Report saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {str(e)}")
            raise

    def _get_severity(self, probability):
        """
        Determine severity based on probability.
        
        Args:
            probability (float): Probability of the anomaly
            
        Returns:
            str: Severity level ('high', 'medium', or 'low')
        """
        if probability >= 0.75:
            return 'high'
        elif probability >= 0.6:
            return 'medium'
        else:
            return 'low'

    def print_analysis_summary(self, analysis_results: Dict):
        """Print comprehensive analysis summary."""
        print(f"\n{Fore.BOLD}{'='*60}")
        print(f"📊 LOG ANALYSIS SUMMARY")
        print(f"{'='*60}{Style.RESET_ALL}")
        
        # Context Discovery Results
        if 'context' in analysis_results and analysis_results['context']:
            context = analysis_results['context']
            print(f"\n{Fore.CYAN}🔍 DISCOVERED LOG CONTEXT:{Style.RESET_ALL}")
            print(f"   Log Type: {context.get('LOG_TYPE', 'Unknown')}")
            print(f"   Format: {context.get('LOG_FORMAT', 'Unknown')}")
            if context.get('SEVERITY_INDICATORS'):
                print(f"   Severity Levels: {', '.join(context['SEVERITY_INDICATORS'])}")
            if context.get('KEY_FIELDS'):
                print(f"   Key Fields: {', '.join(context['KEY_FIELDS'])}")
            print(f"   Context Discovery: {'✅ Success' if analysis_results.get('stats', {}).get('context_discovery_success') else '❌ Failed'}")
        
        # Analysis Method
        method = analysis_results.get('analysis_method', 'unknown')
        print(f"\n{Fore.BLUE}🔧 Analysis Method: {method.replace('_', ' ').title()}{Style.RESET_ALL}")
        
        # Basic stats
        total_lines = analysis_results.get('total_lines', 0)
        anomaly_count = len(analysis_results.get('anomalies', []))
        
        print(f"\n{Fore.GREEN}📈 STATISTICS:{Style.RESET_ALL}")
        print(f"   Total lines analyzed: {total_lines:,}")
        print(f"   Normal entries: {analysis_results.get('stats', {}).get('normal_count', 0):,}")
        print(f"   Anomalies detected: {anomaly_count:,}")
        
        if total_lines > 0:
            anomaly_rate = (anomaly_count / total_lines) * 100
            print(f"   Anomaly rate: {anomaly_rate:.2f}%")
        
        # ... rest of existing method remains the same ... 