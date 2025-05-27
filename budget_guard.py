import logging
from datetime import datetime, timedelta

class BudgetGuard:
    def __init__(self, max_budget=45.0, buffer=5.0):
        """
        Initialize the budget guard.
        
        Args:
            max_budget (float): Maximum budget in USD
            buffer (float): Buffer amount to maintain in USD
        """
        self.max_budget = max_budget
        self.buffer = buffer
        self.current_usage = 0.0
        self.last_reset = datetime.now()
        self.logger = logging.getLogger(__name__)
        
        # GPT-3.5-turbo pricing (as of 2024)
        self.input_price_per_1k = 0.0015  # $0.0015 per 1K input tokens
        self.output_price_per_1k = 0.002  # $0.002 per 1K output tokens

    def can_make_request(self):
        """
        Check if a new API request can be made within budget constraints.
        
        Returns:
            bool: True if a request can be made, False otherwise
        """
        # Check if we need to reset the usage counter (e.g., monthly)
        self._check_reset()
        
        # Calculate remaining budget
        remaining = self.max_budget - self.current_usage
        
        # Ensure we maintain the buffer
        if remaining <= self.buffer:
            self.logger.warning(f"Budget limit reached. Remaining: ${remaining:.2f}")
            return False
            
        return True

    def update_usage(self, total_tokens, is_input=True):
        """
        Update the current usage based on token count.
        
        Args:
            total_tokens (int): Number of tokens used
            is_input (bool): Whether the tokens are input or output tokens
        """
        # Calculate cost based on token type
        price_per_1k = self.input_price_per_1k if is_input else self.output_price_per_1k
        cost = (total_tokens / 1000) * price_per_1k
        
        # Update current usage
        self.current_usage += cost
        
        # Log the update
        self.logger.debug(f"Updated usage: ${self.current_usage:.2f} (added ${cost:.2f})")

    def get_remaining_budget(self):
        """
        Get the remaining budget.
        
        Returns:
            float: Remaining budget in USD
        """
        self._check_reset()
        return self.max_budget - self.current_usage

    def _check_reset(self):
        """Check if the usage counter needs to be reset."""
        # Example: Reset monthly
        if datetime.now() - self.last_reset > timedelta(days=30):
            self.current_usage = 0.0
            self.last_reset = datetime.now()
            self.logger.info("Usage counter reset") 