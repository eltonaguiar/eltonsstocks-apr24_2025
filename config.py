"""Configuration settings for the stock update application.
Store all configurable values here to separate them from the main code."""

import os
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).parent
SERVICE_ACCOUNT_FILE = BASE_DIR / "credentials.json"

# Google Sheets configuration
SPREADSHEET_ID = "1bWANjyQeU6srKRZO0fWNFTItd_gR3kbdjsycToaJXKo"
MAIN_SHEET = "Sheet1"
TEMP_SHEET = "_Temp"

# API keys (in production, these would be environment variables)
API_KEYS = {
    'finnhub': os.environ.get('FINNHUB_API_KEY', 'cvstlkhr01qhup0t0j7gcvstlkhr01qhup0t0j80'),
    'alpha_vantage': os.environ.get('ALPHA_VANTAGE_API_KEY', '1618K3H5MFCONH92'),
    'financial_modeling_prep': os.environ.get('FMP_API_KEY', 'iF4K10WedJZINDhUWGXlGAiA57rn4sRD'),
    'twelvedata': os.environ.get('TWELVEDATA_API_KEY', '43e686519f7b4155a4a90eaae82fb63a')
}

# Transaction costs and slippage configuration
TRANSACTION_COST = float(os.environ.get('TRANSACTION_COST', '0.001'))  # Default 0.1% per trade
SLIPPAGE = float(os.environ.get('SLIPPAGE', '0.001'))  # Default 0.1% slippage

# API rate limits (requests per minute)
RATE_LIMITS = {
    'finnhub': 30,
    'alpha_vantage': 5,
    'financial_modeling_prep': 50,  # FMP has higher limits than Alpha Vantage
    'yahoo': 100,  # Yahoo Finance doesn't have official rate limits, but we'll be conservative
    'twelvedata': 8  # Twelve Data Basic Plan: 8 credits per minute, 800 per day
}

# Daily API credit limits
DAILY_LIMITS = {
    'twelvedata': 800  # 800 credits per day for Twelve Data Basic Plan
}

# API endpoints
API_ENDPOINTS = {
    'financial_modeling_prep': {
        'base_url': 'https://financialmodelingprep.com/api/v3',
        'profile': '/profile/{symbol}',
        'quote': '/quote/{symbol}',
        'income': '/income-statement/{symbol}',
        'balance': '/balance-sheet-statement/{symbol}',
        'cash': '/cash-flow-statement/{symbol}'
    },
    'twelvedata': {
        'base_url': 'https://api.twelvedata.com',
        'price': '/price',
        'quote': '/quote',
        'time_series': '/time_series',
        'profile': '/profile',
        'fundamentals': '/fundamentals',
        'statistics': '/statistics',
        'earnings': '/earnings',
        'technicals': '/technicals'
    }
}

# Google Sheets API limits
SHEETS_BATCH_SIZE = 20  # Recommended by Google for batch updates

# Column mapping - use index values instead of hardcoded letters
# This makes column rearrangement easier
COLUMNS = {
    'symbol': 0,       # A
    'source': 1,       # B
    'date': 2,         # C
    'price_1w': 3,     # D
    'price_1d': 4,     # E
    'price_now': 5,    # F
    'last_updated': 6, # G
    'support': 7,      # H
    'resistance': 8,   # I
    'support_resistance': 9, # J
    'price_date': 10,  # K
    'price_diff': 11,  # L
    'verdict': 12,     # M
    'today': 13,       # N
    'rsi': 14,         # O
    'sharpe': 15,      # P
    'squeeze': 16,     # Q
    'drawdown': 17,    # R
    'beta': 18,        # S
    'composite_score': 19, # T
    'ma_ratio': 20,    # U
    'momentum': 21,    # V
    'pe': 22,          # W
    'de': 23,          # X
    'explanation': 24  # Y
}

# Convert numeric indices to column letters for Google Sheets
def idx_to_col(idx: int) -> str:
    """Convert 0-based index to Excel-style column letter (A, B, ..., Z, AA, AB, ...)"""
    result = ""
    while True:
        idx, remainder = divmod(idx, 26)
        result = chr(65 + remainder) + result
        if idx == 0:
            break
        idx -= 1
    return result

# Create column letter constants
for name, idx in COLUMNS.items():
    col_name = f"C_{name.upper()}"
    globals()[col_name] = idx_to_col(idx)

# Temp sheet header
HEADER = [
    'Symbol', 'Source', 'Date', 'Price in 1 week', 'Price in 1 day', 'Price now', 'Last updated',
    'Support (20-day low)', 'Resistance (20-day high)', 'Support & Resistance',
    'Price in date pulled', 'Price diff', 'Verdict', 'Today\'s date',
    'RSI (0-100)', 'Sharpe Ratio (annualized)', 'Bollinger Squeeze?', 'Max Drawdown %',
    'Beta', 'Composite Score (0-1)', 'MA50/200 Ratio', '20d Momentum %',
    'P/E Ratio', 'Debt/Equity', 'Explanation'
]

# Scoring weights - having these in config makes them easy to adjust
SIMPLE_SCORE_WEIGHTS = {
    'rsi': 0.15,
    'sharpe': 0.25,
    'drawdown': 0.10,
    'ma_ratio': 0.20,
    'momentum': 0.15,
    'pe': 0.10,
    'de': 0.05
}

ADVANCED_SCORE_WEIGHTS = {
    'rsi': 0.10,
    'macd': 0.08,
    'sharpe': 0.12,
    'squeeze': 0.05,
    'drawdown': 0.08,
    'ma_ratio': 0.10,
    'momentum': 0.10,
    'valuation': 0.12,  # P/E relative to sector
    'growth': 0.10,  # Revenue & earnings growth
    'sentiment': 0.05,  # News sentiment
    'bbands': 0.10  # Position relative to Bollinger Bands
}

# Cell formatting options
CELL_FORMATS = {
    'support_resistance': {
        "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 1}  # Light blue
    },
    'rsi': {
        "backgroundColor": {"red": 1, "green": 0.95, "blue": 0.8}  # Light orange
    },
    'score_verdict': {
        "backgroundColor": {"red": 0.8, "green": 1, "blue": 0.8}  # Light green
    }
}

# Logging configuration
LOG_FILE = BASE_DIR / "stock_update.log"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'DEBUG'  # Could be DEBUG, INFO, WARNING, ERROR, CRITICAL

# Backtesting configuration
BACKTEST_START_DATE = '2020-01-01'
BACKTEST_END_DATE = '2023-12-31'
INITIAL_CAPITAL = 100000  # Initial capital for backtesting
