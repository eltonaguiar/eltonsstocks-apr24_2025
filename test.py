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

# Import the DataFetcher class from data_fetchers module
from data_fetchers import DataFetcher
from config import API_KEYS, API_ENDPOINTS

# Create a DataFetcher instance with API keys and rate limits
data_fetcher = DataFetcher(API_KEYS, API_LIMITS)

# Define wrapper functions to match the expected function names in the test script
def get_stock_data_from_yahoo(ticker, start_date, end_date):
    """Wrapper for DataFetcher to get data from Yahoo Finance"""
    try:
        # Convert to string format if datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        # Use DataFetcher's fetch_price_history method
        period = f"{(datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days}d"
        return data_fetcher.fetch_price_history(ticker, period=period)
    except Exception as e:
        print(f"Error fetching Yahoo data: {str(e)}")
        return None

def get_stock_data_from_tiingo(ticker, start_date, end_date):
    """Wrapper for DataFetcher to get data from Tiingo"""
    try:
        # Use a DataFetcher method that would connect to Tiingo
        # This is a placeholder - we'll need to implement this in DataFetcher
        # For now, return None to indicate failure
        print("Tiingo API not fully implemented in DataFetcher yet")
        return None
    except Exception as e:
        print(f"Error fetching Tiingo data: {str(e)}")
        return None

def get_stock_data_from_polygon(ticker, start_date, end_date):
    """Wrapper for DataFetcher to get data from Polygon"""
    try:
        # Use a DataFetcher method that would connect to Polygon
        # This is a placeholder - we'll need to implement this in DataFetcher
        # For now, return None to indicate failure
        print("Polygon API not fully implemented in DataFetcher yet")
        return None
    except Exception as e:
        print(f"Error fetching Polygon data: {str(e)}")
        return None

def get_stock_data_from_finnhub(ticker, start_date, end_date):
    """Wrapper for DataFetcher to get data from Finnhub"""
    try:
        # Use a DataFetcher method that would connect to Finnhub
        # This is a placeholder - we'll need to implement this in DataFetcher
        # For now, return None to indicate failure
        print("Finnhub API not fully implemented in DataFetcher yet")
        return None
    except Exception as e:
        print(f"Error fetching Finnhub data: {str(e)}")
        return None

def get_stock_data_from_marketstack(ticker, start_date, end_date):
    """Wrapper for DataFetcher to get data from Marketstack"""
    try:
        # Use a DataFetcher method that would connect to Marketstack
        # This is a placeholder - we'll need to implement this in DataFetcher
        # For now, return None to indicate failure
        print("Marketstack API not fully implemented in DataFetcher yet")
        return None
    except Exception as e:
        print(f"Error fetching Marketstack data: {str(e)}")
        return None

def get_stock_data_from_alpaca(ticker, start_date, end_date):
    """Wrapper for DataFetcher to get data from Alpaca"""
    try:
        # Use a DataFetcher method that would connect to Alpaca
        # This is a placeholder - we'll need to implement this in DataFetcher
        # For now, return None to indicate failure
        print("Alpaca API not fully implemented in DataFetcher yet")
        return None
    except Exception as e:
        print(f"Error fetching Alpaca data: {str(e)}")
        return None