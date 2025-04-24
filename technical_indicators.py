"""
Technical indicators for stock analysis.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('indicators')

class DataValidationError(Exception):
    """Exception raised for data validation errors in indicator calculations."""
    pass

def validate_data(data, min_length=None, name="data"):
    """
    Validate data before calculation to avoid silent failures.
    
    Args:
        data: numpy array or list - The data to validate
        min_length: int - Minimum required length
        name: str - Name of the data for error messages
        
    Returns:
        numpy array - The validated data
        
    Raises:
        DataValidationError: If data doesn't meet requirements
    """
    if data is None:
        raise DataValidationError(f"{name} cannot be None")
        
    if not isinstance(data, (list, np.ndarray, pd.Series)):
        raise DataValidationError(f"{name} must be list, numpy array or pandas Series")
    
    if len(data) == 0:
        raise DataValidationError(f"{name} cannot be empty")
        
    if min_length and len(data) < min_length:
        raise DataValidationError(f"{name} length must be at least {min_length}, got {len(data)}")
        
    # Convert to numpy array for consistent calculations
    return np.array(data)

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: array-like - Price series
        period: int - RSI period
        
    Returns:
        float - The RSI value (0-100)
        
    Raises:
        DataValidationError: If input data is invalid
    """
    try:
        prices = validate_data(prices, min_length=period+1, name="prices")
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100
        
        rs = up / down
        return 100 - (100 / (1 + rs))
    except DataValidationError as e:
        logger.warning(f"RSI calculation error: {str(e)}")
        return 50  # Neutral RSI value
    except Exception as e:
        logger.error(f"RSI calculation unexpected error: {str(e)}")
        return 50

def calculate_sharpe_ratio(returns, risk_free_rate=0.01, annualization=252):
    """
    Calculate Sharpe ratio - reward to variability ratio.
    
    Args:
        returns: array-like - Daily returns (not prices)
        risk_free_rate: float - Annual risk-free rate (0.01 = 1%)
        annualization: int - Number of periods in a year
        
    Returns:
        float - Sharpe ratio
        
    Raises:
        DataValidationError: If input data is invalid
    """
    try:
        returns = validate_data(returns, min_length=5, name="returns")
        
        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/annualization) - 1
        
        excess_returns = returns - daily_rf
        if len(excess_returns) == 0:
            return 0
            
        std = np.std(excess_returns, ddof=1)
        if std == 0:
            return 0
            
        sharpe = (np.mean(excess_returns) / std) * np.sqrt(annualization)
        return sharpe
    except DataValidationError as e:
        logger.warning(f"Sharpe ratio calculation error: {str(e)}")
        return 0
    except Exception as e:
        logger.error(f"Sharpe ratio unexpected error: {str(e)}")
        return 0

def detect_volatility_squeeze(prices, lookback=20, stdev_multiplier=2):
    """
    Detect volatility squeeze (low volatility period often preceding a breakout).
    
    Args:
        prices: array-like - Price series
        lookback: int - Lookback period
        stdev_multiplier: float - Standard deviation multiplier for Bollinger Bands
        
    Returns:
        bool - True if squeeze is detected
        
    Raises:
        DataValidationError: If input data is invalid
    """
    try:
        prices = validate_data(prices, min_length=lookback*2, name="prices")
        
        # Convert to pandas Series for rolling calculations
        s = pd.Series(prices)
        bandwidth = s.rolling(lookback).std() * stdev_multiplier
        
        # Average bandwidth over the whole period
        avg_bandwidth = bandwidth.mean()
        
        # Current bandwidth
        current_bandwidth = bandwidth.iloc[-1]
        
        # Squeeze is when bandwidth is significantly below the average
        return current_bandwidth < avg_bandwidth * 0.6
    except DataValidationError as e:
        logger.warning(f"Squeeze detection error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Squeeze detection unexpected error: {str(e)}")
        return False

def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown percentage.
    
    Args:
        prices: array-like - Price series
        
    Returns:
        float - Maximum drawdown as a percentage (0-100)
        
    Raises:
        DataValidationError: If input data is invalid
    """
    try:
        prices = validate_data(prices, min_length=2, name="prices")
        
        # Calculate the maximum drawdown
        peak = prices[0]
        max_dd = 0
        
        for price in prices:
            if price > peak:
                peak = price
            else:
                dd = (peak - price) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                    
        return max_dd * 100  # Convert to percentage
    except DataValidationError as e:
        logger.warning(f"Max drawdown calculation error: {str(e)}")
        return 0
    except Exception as e:
        logger.error(f"Max drawdown unexpected error: {str(e)}")
        return 0

def calculate_moving_average_ratio(prices, short_period=50, long_period=200):
    """
    Calculate ratio between short and long moving averages.
    
    Args:
        prices: array-like - Price series
        short_period: int - Short moving average period
        long_period: int - Long moving average period
        
    Returns:
        float - Ratio between short and long MAs
        
    Raises:
        DataValidationError: If input data is invalid
    """
    try:
        prices = validate_data(prices, min_length=long_period, name="prices")
        
        # Convert to pandas Series for rolling calculations
        s = pd.Series(prices)
        
        # Calculate moving averages
        ma_short = s.rolling(short_period).mean().iloc[-1]
        ma_long = s.rolling(long_period).mean().iloc[-1]
        
        # Calculate ratio
        if ma_long == 0:
            return 1.0
            
        return ma_short / ma_long
    except DataValidationError as e:
        logger.warning(f"MA ratio calculation error: {str(e)}")
        return 1.0  # Neutral value
    except Exception as e:
        logger.error(f"MA ratio unexpected error: {str(e)}")
        return 1.0

def calculate_momentum(prices, period=20):
    """
    Calculate momentum as percentage change over period.
    
    Args:
        prices: array-like - Price series
        period: int - Lookback period
        
    Returns:
        float - Momentum as percentage (-100 to +inf)
        
    Raises:
        DataValidationError: If input data is invalid
    """
    try:
        prices = validate_data(prices, min_length=period+1, name="prices")
        
        start_price = prices[-period-1]
        end_price = prices[-1]
        
        if start_price == 0:
            return 0
            
        momentum = (end_price / start_price - 1) * 100
        return momentum
    except DataValidationError as e:
        logger.warning(f"Momentum calculation error: {str(e)}")
        return 0
    except Exception as e:
        logger.error(f"Momentum unexpected error: {str(e)}")
        return 0

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: array-like - Price series
        fast_period: int - Fast EMA period
        slow_period: int - Slow EMA period
        signal_period: int - Signal line period
        
    Returns:
        tuple - (MACD line, Signal line)
        
    Raises:
        DataValidationError: If input data is invalid
    """
    try:
        prices = validate_data(prices, min_length=slow_period+signal_period, name="prices")
        
        # Convert to pandas Series for EMA calculations
        close = pd.Series(prices)
        
        # Calculate EMAs
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        return macd_line.iloc[-1], signal_line.iloc[-1]
    except DataValidationError as e:
        logger.warning(f"MACD calculation error: {str(e)}")
        return 0, 0
    except Exception as e:
        logger.error(f"MACD unexpected error: {str(e)}")
        return 0, 0

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: array-like - Price series
        window: int - Window for moving average
        num_std: int/float - Number of standard deviations for bands
        
    Returns:
        tuple - (middle band, upper band, lower band)
        
    Raises:
        DataValidationError: If input data is invalid
    """
    try:
        prices = validate_data(prices, min_length=window, name="prices")
        
        # Convert to pandas Series
        s = pd.Series(prices)
        
        # Calculate middle band (SMA)
        middle_band = s.rolling(window=window).mean()
        
        # Calculate standard deviation
        std = s.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        # Return most recent values
        return (
            middle_band.iloc[-1],
            upper_band.iloc[-1],
            lower_band.iloc[-1]
        )
    except DataValidationError as e:
        logger.warning(f"Bollinger Bands calculation error: {str(e)}")
        return prices[-1] if len(prices) > 0 else 0, 0, 0
    except Exception as e:
        logger.error(f"Bollinger Bands unexpected error: {str(e)}")
        return prices[-1] if len(prices) > 0 else 0, 0, 0