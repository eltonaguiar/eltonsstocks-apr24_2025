"""
Rate limiter for API calls to prevent exceeding rate limits.
"""
import time
from datetime import datetime, timedelta
import logging
import random
from functools import wraps

class RateLimiter:
    """
    Rate limiter for API calls with exponential backoff and request tracking.
    """
    def __init__(self, rate_limits):
        """
        Initialize the rate limiter.
        
        Args:
            rate_limits (dict): Dictionary mapping API names to their rate limits (calls per minute)
        """
        self.rate_limits = rate_limits
        self.api_calls = {api: [] for api in rate_limits}
        self.api_failures = {api: {'count': 0, 'until': None, 'dead': False} for api in rate_limits}
        self.logger = logging.getLogger('rate_limiter')
    
    def can_make_request(self, api):
        """
        Check if a request can be made to the specified API.
        
        Args:
            api (str): The API name
            
        Returns:
            bool: True if the request can be made, False otherwise
        """
        now = time.time()
        
        # Remove timestamps older than 60 seconds
        self.api_calls[api] = [t for t in self.api_calls[api] if now - t < 60]
        
        # Check if API is in cooldown period
        info = self.api_failures[api]
        if info['dead'] or (info['until'] and datetime.now() < info['until']):
            return False
        
        # Check if request would exceed rate limit
        if len(self.api_calls[api]) < self.rate_limits[api]:
            self.api_calls[api].append(now)
            return True
        
        return False
    
    def record_failure(self, api, status_code=None):
        """
        Record an API failure and implement backoff strategy.
        
        Args:
            api (str): The API name
            status_code (int, optional): HTTP status code of the failure
        """
        info = self.api_failures[api]
        
        # Handle rate limit errors specially
        if status_code == 429:
            info['count'] += 1
            backoff_minutes = min(2 ** (info['count'] - 1), 30)  # Exponential backoff up to 30 minutes
            info['until'] = datetime.now() + timedelta(minutes=backoff_minutes)
            self.logger.warning(
                f"Rate limit hit for {api}. Backing off for {backoff_minutes} minutes. "
                f"Failure count: {info['count']}"
            )
            
            # Mark API as dead if too many failures
            if info['count'] >= 5:
                info['dead'] = True
                self.logger.error(f"API {api} marked as dead after {info['count']} failures")
        else:
            # For non-rate-limit errors, use a smaller backoff
            info['count'] += 1
            if info['count'] > 3:
                backoff_seconds = min(5 * info['count'], 60)
                info['until'] = datetime.now() + timedelta(seconds=backoff_seconds)
                self.logger.warning(f"API {api} errors mounting. Backing off for {backoff_seconds}s")
    
    def reset_api(self, api):
        """
        Reset the failure counter for an API.
        
        Args:
            api (str): The API name
        """
        if api in self.api_failures:
            self.api_failures[api] = {'count': 0, 'until': None, 'dead': False}
            self.logger.info(f"Reset failure counter for API {api}")
    
    def get_next_available_time(self, api):
        """
        Get the next time when the API will be available.
        
        Args:
            api (str): The API name
            
        Returns:
            datetime or None: The next available time, or None if available now
        """
        info = self.api_failures[api]
        if info['until'] and datetime.now() < info['until']:
            return info['until']
        
        now = time.time()
        self.api_calls[api] = [t for t in self.api_calls[api] if now - t < 60]
        
        if len(self.api_calls[api]) >= self.rate_limits[api]:
            # Get oldest timestamp and calculate when it will expire
            oldest = min(self.api_calls[api])
            seconds_until_available = 60 - (now - oldest)
            return datetime.now() + timedelta(seconds=seconds_until_available)
            
        return None
    
    def with_rate_limit(self, api):
        """
        Decorator to apply rate limiting to a function.
        
        Args:
            api (str): The API name
            
        Returns:
            function: Decorated function with rate limiting
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.can_make_request(api):
                    next_time = self.get_next_available_time(api)
                    if next_time:
                        wait_time = (next_time - datetime.now()).total_seconds()
                        if wait_time > 0:
                            self.logger.info(f"Rate limit for {api} - waiting {wait_time:.2f}s")
                            time.sleep(wait_time + random.uniform(0.1, 1.0))  # Add jitter
                    
                    # If API is dead, return error indicator
                    if self.api_failures[api]['dead']:
                        return None
                
                try:
                    result = func(*args, **kwargs)
                    # Successful call, reset failure counter
                    if self.api_failures[api]['count'] > 0:
                        self.reset_api(api)
                    return result
                except Exception as e:
                    status_code = getattr(e, 'response', {}).get('status_code', None)
                    self.record_failure(api, status_code)
                    self.logger.error(f"API {api} call failed: {str(e)}")
                    raise
            
            return wrapper
        return decorator