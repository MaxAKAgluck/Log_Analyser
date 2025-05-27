import numpy as np
from sklearn.ensemble import IsolationForest
from collections import defaultdict
import logging
from datetime import datetime, timedelta
import re

class LogStats:
    def __init__(self, window_size=1000, contamination=0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.template_counts = defaultdict(int)
        self.template_windows = defaultdict(list)
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.logger = logging.getLogger(__name__)
        self.window_start = datetime.now()
        self.current_window = []
        
        # Keywords for simple anomaly detection
        self.high_severity_words = [
            "error", "exception", "fail", "critical", "crash", "timeout", "unavailable",
            "killed", "fatal", "panic", "died", "corruption", "denied", "broken", "violation"
        ]
        self.medium_severity_words = [
            "warn", "warning", "retry", "slow", "high", "delay", "degraded", 
            "limited", "unusual", "dropped", "missing", "reset", "overflow"
        ]
        
        # For training the Isolation Forest
        self.min_samples_to_train = 10  # Need at least this many samples to train
        self.model_trained = False

    def update(self, template_id, variables, timestamp):
        """
        Update statistics for a template and check for anomalies.
        
        Args:
            template_id: ID of the template
            variables: Dictionary of variables extracted from the log line
            timestamp: Timestamp of the log entry
        """
        # Update template counts
        self.template_counts[template_id] += 1
        
        # Add to current window
        self.current_window.append({
            'template_id': template_id,
            'timestamp': timestamp,
            'variables': variables
        })
        
        # Check if we need to process the current window
        if len(self.current_window) >= self.window_size:
            self._process_window()
            self.current_window = []
            self.window_start = timestamp

    def is_anomaly(self, template_id, log_line=""):
        """
        Check if a template is anomalous based on its frequency, patterns or content.
        
        Args:
            template_id: ID of the template to check
            log_line: The original log line text (for keyword checking)
            
        Returns:
            bool: True if the template is anomalous, False otherwise
        """
        self.logger.info(f"Checking anomaly for template_id: {template_id}")
        
        # Always return True for first pass to ensure logs are analyzed
        # This ensures even on first run when model isn't trained, we'll run ML analysis
        if not self.model_trained:
            self.logger.info("Model not yet trained, sending for ML analysis")
            return True
            
        # 1. Simple keyword-based heuristic detection
        if log_line:
            lowercase_line = log_line.lower()
            
            # Check for high severity keywords
            for word in self.high_severity_words:
                if word in lowercase_line:
                    self.logger.info(f"High severity word '{word}' found in log")
                    return True
                    
            # Check for medium severity keywords
            for word in self.medium_severity_words:
                if word in lowercase_line:
                    self.logger.info(f"Medium severity word '{word}' found in log")
                    return True
        
        # 2. Frequency-based detection
        # Get the template's frequency in the current window
        window_freq = sum(1 for entry in self.current_window 
                         if entry['template_id'] == template_id)
                         
        # Rare templates are suspicious
        if self.template_counts[template_id] <= 3:
            self.logger.info(f"Rare template (count {self.template_counts[template_id]}) detected")
            return True
            
        # Sudden spike in frequency is suspicious
        if window_freq > 5 * (self.template_counts[template_id] / max(1, len(self.current_window))):
            self.logger.info(f"Frequency spike detected for template")
            return True
        
        # 3. Statistical anomaly detection using Isolation Forest
        if self.model_trained and len(self.current_window) >= 5:
            # Calculate features for anomaly detection
            features = self._extract_features(template_id, window_freq)
            
            # Reshape for sklearn
            features = np.array(features).reshape(1, -1)
            
            # Predict if anomalous
            prediction = self.isolation_forest.predict(features)
            
            if prediction[0] == -1:  # -1 indicates anomaly in Isolation Forest
                self.logger.info(f"Isolation Forest detected anomaly")
                return True
                
        self.logger.info(f"No anomaly detected for template_id: {template_id}")
        return False

    def _process_window(self):
        """Process the current window of log entries."""
        if not self.current_window:
            return

        # Calculate features for each template in the window
        templates_features = []
        for entry in self.current_window:
            template_id = entry['template_id']
            window_freq = sum(1 for e in self.current_window 
                            if e['template_id'] == template_id)
            features = self._extract_features(template_id, window_freq)
            templates_features.append(features)
        
        # Update Isolation Forest with new data
        if templates_features and len(templates_features) >= self.min_samples_to_train:
            self.logger.info(f"Training Isolation Forest with {len(templates_features)} samples")
            X = np.array(templates_features)
            self.isolation_forest.fit(X)
            self.model_trained = True
            self.logger.info("Isolation Forest model trained")

    def _extract_features(self, template_id, window_freq):
        """
        Extract features for anomaly detection.
        
        Args:
            template_id: ID of the template
            window_freq: Frequency of the template in the current window
            
        Returns:
            list: Feature vector for anomaly detection
        """
        # Calculate various features
        total_count = self.template_counts[template_id]
        avg_freq = total_count / max(1, len(self.current_window))
        
        # Time-based features
        time_since_last = 0
        if self.template_windows[template_id]:
            last_occurrence = self.template_windows[template_id][-1]
            time_since_last = (datetime.now() - last_occurrence).total_seconds()
        
        # Combine features
        features = [
            window_freq,  # Current window frequency
            total_count,  # Total count
            avg_freq,     # Average frequency
            time_since_last,  # Time since last occurrence
            len(self.template_windows[template_id])  # Number of windows seen
        ]
        
        return features 