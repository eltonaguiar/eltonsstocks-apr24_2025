import yfinance as yf
import pandas as pd
import numpy as np
import ta
import requests
import io
import time
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

"""
=== STOCK SPIKE PREDICTOR ===

This tool analyzes stocks using multiple technical indicators to find potential candidates 
for significant price increases ("spikes").

== FOR NON-TECHNICAL USERS ==

TECHNICAL INDICATORS EXPLAINED:

1. RSI (Relative Strength Index):
   - A measure of how "oversold" or "overbought" a stock is
   - RSI below 30 suggests a stock is oversold and may be due for a rebound
   - Think of it like a spring that's been compressed too much and is ready to bounce back

2. Bollinger Bands:
   - Imagine a "price channel" with an upper and lower boundary around the stock price
   - The middle line is a 20-day average price
   - The upper and lower bands show where the price is considered abnormally high or low
   - When price hits the lower band, it often bounces back up
   - When price hits the upper band, it often drops back down

3. Bollinger Band Squeeze:
   - When the upper and lower bands get very close together (narrow)
   - This indicates a period of low volatility (price isn't moving much)
   - Historically, periods of low volatility are followed by high volatility
   - It's like a spring being compressed - the tighter it gets, the bigger the eventual move
   - A squeeze suggests a big price move is coming soon (though direction is uncertain)

4. Moving Averages:
   - Smooth out price action to show the overall trend
   - EMA (Exponential Moving Average) emphasizes more recent price action
   - When shorter EMAs cross above longer EMAs, it often signals an uptrend beginning

5. Support and Resistance:
   - Support: Price level where a stock tends to stop falling and bounce back up
   - Resistance: Price level where a stock tends to stop rising and fall back down
   - Think of support as a floor and resistance as a ceiling
   - When price breaks through resistance, that level often becomes the new support
"""

# API Keys for multiple data sources
API_KEYS = {
    # Alpha Vantage API Keys - Temporarily marked as inactive for today
    'ALPHA_VANTAGE_KEY': '1618K3H5MFCONH92',  # Marked as inactive for today
    'ALPHA_VANTAGE_KEY_FAILOVER': 'OJUUINBH3E50UGWO',  # Marked as inactive for today
    
    # RapidAPI Key (can be used for various market data APIs)
    'RAPIDAPI_KEY': 'b1ee5c7d46msh15d35e04e051ad2p1b5e14jsncfbcbb51b443',
    
    # Finnhub API Key
    'FINNHUB_KEY': 'cvstlkhr01qhup0t0j7gcvstlkhr01qhup0t0j80',
    
    # Polygon.io API Key
    'POLYGON_KEY': '2xiTaFKgrJA8eGyxd_tF5GTu3OTXMUWC',
    
    # Marketstack API Key
    'MARKETSTACK_KEY': 'f4a2dd73fbd3cbb068a7dc56b6192cfc',
    
    # Tiingo API Key
    'TIINGO_KEY': '2247aa4e93338de698597f58f44136f08e17694d',
    
    # Alpaca Markets API Keys
    'ALPACA_API_KEY': 'PKNMXSCXURUCGRHDY6C3',
    'ALPACA_SECRET_KEY': '2l6gkJA7j4biMNK4tU70c055mBb5qkGeD6q7IVFz',
    'ALPACA_PAPER': True,
    
    # Nasdaq Data Link API Key
    'NASDAQ_DATALINK_KEY': 'GCRzsLSfx9DmxyCbM6u3',
    
    # Google Custom Search
    'GOOGLE_CUSTOM_SEARCH_KEY': 'AIzaSyB3jhUkndfV6_c99tCh_h0byKpTjTh3ETU',
    'GOOGLE_CSE_ID': 'd0432542ea931417b'
}

# API Rate Limiting Settings
API_LIMITS = {
    'ALPHA_VANTAGE': {'calls_per_minute': 5, 'calls_per_day': 500, 'last_call_time': 0, 'calls_today': 0, 'active': False},  # Set to inactive
    'ALPHA_VANTAGE_FAILOVER': {'calls_per_minute': 5, 'calls_per_day': 500, 'last_call_time': 0, 'calls_today': 0, 'active': False},  # Set to inactive
    'FINNHUB': {'calls_per_minute': 60, 'calls_per_day': 500, 'last_call_time': 0, 'calls_today': 0, 'active': True},
    'POLYGON': {'calls_per_minute': 5, 'calls_per_day': 200, 'last_call_time': 0, 'calls_today': 0, 'active': True},
    'MARKETSTACK': {'calls_per_minute': 5, 'calls_per_day': 1000, 'last_call_time': 0, 'calls_today': 0, 'active': True},
    'TIINGO': {'calls_per_minute': 5, 'calls_per_day': 500, 'last_call_time': 0, 'calls_today': 0, 'active': True},
    'ALPACA': {'calls_per_minute': 200, 'calls_per_day': 5000, 'last_call_time': 0, 'calls_today': 0, 'active': True},
    'YAHOO_FINANCE': {'calls_per_minute': 2, 'calls_per_day': 500, 'last_call_time': 0, 'calls_today': 0, 'active': True}
}

# Define stock universe (limited to 3 stocks for testing)
tickers = [
    'AAPL',  # Apple - Large cap tech, highly liquid
    'PLTR',  # Palantir - Mid cap tech with volatility
    'WMT'    # Walmart - Large cap retail/consumer staple
]

# Parameters for screening - loosened criteria
LOOKBACK_DAYS = 90  # Increased from 60 to 90 for better support/resistance calculation
MIN_MARKET_CAP = 10e6  # 10M minimum market cap
MAX_MARKET_CAP = 20e9  # 20B maximum market cap
MIN_SHORT_INTEREST = 0.05  # 5% minimum short interest
PRICE_THRESHOLD = 20.0  # Maximum price threshold
CONSOLIDATION_RANGE = 0.30  # Maximum price range for consolidation (30% of mean price)
VOL_DROP_THRESHOLD = 0.9  # Volume below 90% of average in recent days
RSI_OVERSOLD_THRESHOLD = 30  # RSI below this is considered oversold (changed from 40 to 30 per requirement)
RSI_LOOKBACK = 14  # Standard lookback period for RSI calculation
RSI_ONLY_MODE = True  # Special mode that prioritizes RSI regardless of other criteria

# Additional parameters for improved technical analysis
BOLL_SQUEEZE_LOOKBACK = 60  # Lookback period for Bollinger Band squeeze detection
BOLL_SQUEEZE_THRESHOLD = 1.1  # Maximum width multiplier for squeeze detection
EMA_FAST = 8  # Fast EMA period
EMA_SLOW = 21  # Slow EMA period
MACD_SIGNAL = 9  # MACD signal line period
SMA_SHORT = 5  # Short SMA period (for crossover)
SMA_MEDIUM = 10  # Medium SMA period (for crossover)
MOMENTUM_DAYS = 5  # Days to look back for momentum calculation
RETURN_DAYS = 7  # Days to look back for return calculation
MIN_7D_RETURN = 0.0  # Minimum 7-day return (looking for stocks starting to move)

# Fundamental filters - Updated per requirements
MIN_EPS = 0  # Must be positive (changed from -1 to 0 per requirement)
MAX_DEBT_TO_EQUITY = 200  # Updated from 300 to 200 per requirement

# Spike definition for backtesting
FUTURE_DAYS = 20  # Number of days forward to check for spikes
MIN_SPIKE_PCT = 0.20  # Minimum 20% increase to qualify as spike
BACKTEST_PERIODS = 6  # Number of historical periods to test
BACKTEST_PERIOD_LENGTH = 30  # Length of each backtest period in days

# API Selection configuration
# Order of APIs to try when fetching data
API_PRIORITY = [
    'YAHOO_FINANCE',      # Try Yahoo Finance first (no key needed)
    'TIINGO',             # Then try Tiingo
    'POLYGON',            # Then try Polygon
    'FINNHUB',            # Then try Finnhub
    'MARKETSTACK',        # Then try Marketstack
    'ALPACA'              # Finally try Alpaca
    # Alpha Vantage APIs removed from priority list temporarily
]

def check_api_rate_limit(api_name):
    """
    Check if we can make a call to the specified API without hitting rate limits
    Returns True if it's safe to make a call, False otherwise
    Also updates the last call time and call count if it's safe to proceed
    """
    if api_name not in API_LIMITS:
        return True  # No limit tracking for this API
        
    # Check if the API is marked as active
    if not API_LIMITS[api_name].get('active', True):
        print(f"{api_name} is currently marked as inactive")
        return False
        
    now = time.time()
    limit_info = API_LIMITS[api_name]
    
    # Check if we've hit the daily limit
    if limit_info['calls_today'] >= limit_info['calls_per_day']:
        print(f"{api_name} daily limit reached")
        return False
        
    # Check if we need to wait for rate limiting
    time_since_last_call = now - limit_info['last_call_time']
    seconds_per_call = 60 / limit_info['calls_per_minute']
    
    if time_since_last_call < seconds_per_call:
        # Need to wait a bit to respect rate limits
        sleep_time = seconds_per_call - time_since_last_call
        print(f"Rate limiting {api_name}, waiting {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)
    
    # Update API call tracking
    API_LIMITS[api_name]['last_call_time'] = time.time()
    API_LIMITS[api_name]['calls_today'] += 1
    
    return True

def test_api_connectivity():
    """
    Test all APIs to check which ones are working
    Updates the API_LIMITS dictionary to mark non-working APIs as inactive
    """
    print("=" * 70)
    print("TESTING API CONNECTIVITY")
    print("=" * 70)
    
    test_ticker = 'AAPL'  # Common stock that should be available on all platforms
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)  # Just get 10 days of data for testing
    
    working_apis = []
    failed_apis = []
    
    # Test each API separately
    print("Testing Yahoo Finance...")
    if API_LIMITS['YAHOO_FINANCE'].get('active', True):
        data = get_stock_data_from_yahoo(test_ticker, start_date, end_date)
        if data is not None and not data.empty:
            working_apis.append('YAHOO_FINANCE')
            print("✅ Yahoo Finance API is working")
        else:
            failed_apis.append('YAHOO_FINANCE')
            API_LIMITS['YAHOO_FINANCE']['active'] = False
            print("❌ Yahoo Finance API is not working")
    
    # Skip Alpha Vantage testing since it's marked as inactive for today
    print("Skipping Alpha Vantage APIs (marked inactive for today)...")
    
    print("Testing Tiingo...")
    if API_LIMITS['TIINGO'].get('active', True):
        data = get_stock_data_from_tiingo(test_ticker, start_date, end_date)
        if data is not None and not data.empty:
            working_apis.append('TIINGO')
            print("✅ Tiingo API is working")
        else:
            failed_apis.append('TIINGO')
            API_LIMITS['TIINGO']['active'] = False
            print("❌ Tiingo API is not working")
    
    print("Testing Polygon.io...")
    if API_LIMITS['POLYGON'].get('active', True):
        data = get_stock_data_from_polygon(test_ticker, start_date, end_date)
        if data is not None and not data.empty:
            working_apis.append('POLYGON')
            print("✅ Polygon.io API is working")
        else:
            failed_apis.append('POLYGON')
            API_LIMITS['POLYGON']['active'] = False
            print("❌ Polygon.io API is not working")
    
    print("Testing Finnhub...")
    if API_LIMITS['FINNHUB'].get('active', True):
        data = get_stock_data_from_finnhub(test_ticker, start_date, end_date)
        if data is not None and not data.empty:
            working_apis.append('FINNHUB')
            print("✅ Finnhub API is working")
        else:
            failed_apis.append('FINNHUB')
            API_LIMITS['FINNHUB']['active'] = False
            print("❌ Finnhub API is not working")
    
    print("Testing Marketstack...")
    if API_LIMITS['MARKETSTACK'].get('active', True):
        data = get_stock_data_from_marketstack(test_ticker, start_date, end_date)
        if data is not None and not data.empty:
            working_apis.append('MARKETSTACK')
            print("✅ Marketstack API is working")
        else:
            failed_apis.append('MARKETSTACK')
            API_LIMITS['MARKETSTACK']['active'] = False
            print("❌ Marketstack API is not working")
    
    print("Testing Alpaca...")
    if API_LIMITS['ALPACA'].get('active', True):
        data = get_stock_data_from_alpaca(test_ticker, start_date, end_date)
        if data is not None and not data.empty:
            working_apis.append('ALPACA')
            print("✅ Alpaca API is working")
        else:
            failed_apis.append('ALPACA')
            API_LIMITS['ALPACA']['active'] = False
            print("❌ Alpaca API is not working")
    
    # Update API priority list to only include working APIs
    global API_PRIORITY
    API_PRIORITY = [api for api in API_PRIORITY if api in working_apis]
    
    print("\nAPI TESTING SUMMARY:")
    print(f"Working APIs: {', '.join(working_apis)}")
    print(f"Failed APIs: {', '.join(failed_apis)}")
    print(f"Updated API priority: {', '.join(API_PRIORITY)}")
    print("=" * 70)
    
    return len(working_apis) > 0

def run_scan():
    """
    Main function to run the stock scanner
    """
    print("=" * 70)
    print("STOCK SPIKE PREDICTOR")
    print("A multi-source API stock analysis system")
    print("=" * 70)
    
    # First, test API connectivity to determine which APIs are working
    if not test_api_connectivity():
        print("ERROR: No working APIs found. Cannot proceed with analysis.")
        return
    
    print(f"Scanning {len(tickers)} tickers...")
    print(f"Using multiple data sources: {', '.join(API_PRIORITY)}")
    print("-" * 70)
    
    results = []
    errors = 0
    
    for idx, ticker in enumerate(tickers):
        print(f"Processing {ticker} ({idx+1}/{len(tickers)})...")
        
        try:
            result = analyze_stock(ticker)
            if result:
                results.append(result)
                # Show summary of result
                print(f"  ✅ {ticker}: RSI={result['RSI']:.1f}, Price=${result['Price']:.2f}, EPS=${result['EPS'] if result['EPS'] is not None else 'N/A'}")
                print(f"     D/E={result['DebtToEquity'] if result['DebtToEquity'] is not None else 'N/A'}, Score={result['TotalScore']:.1f}, Pattern={result['Classification']}")
            else:
                errors += 1
                print(f"  ❌ {ticker}: Analysis failed")
        except Exception as e:
            print(f"  ❌ {ticker}: Error - {str(e)[:100]}")
            errors += 1
    
    print("\n" + "-" * 70)
    print(f"Scan complete: {len(results)} stocks analyzed successfully, {errors} errors")
    
    # Apply filters and save results
    if results:
        # Filter oversold stocks (RSI < 30)
        oversold = filter_oversold_stocks(results, RSI_OVERSOLD_THRESHOLD)
        save_results_to_csv(oversold, "oversold_stocks.csv")
        
        # Filter squeeze candidates
        squeeze = filter_squeeze_candidates(results)
        save_results_to_csv(squeeze, "squeeze_stocks.csv")
        
        # Filter top overall candidates
        top = filter_top_candidates(results)
        save_results_to_csv(top, "top_candidates.csv")
        
        # Display top candidates
        print("\nTOP CANDIDATES:")
        print_results_table(top)
        
        print("\nOVERSOLD STOCKS:")
        print_results_table(oversold)
        
        print("\nBOLLINGER BAND SQUEEZE CANDIDATES:")
        print_results_table(squeeze)
    else:
        print("No valid results to process")

if __name__ == "__main__":
    run_scan()