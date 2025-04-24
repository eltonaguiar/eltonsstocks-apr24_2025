"""
Main module for the Stock Spike Replicator.

This script orchestrates the entire process of fetching, analyzing, and recommending stocks.
It processes both user-provided symbols and additional cheap stocks, combines the results,
and updates the Google Sheets with the findings, including a summary of the algorithm's advantages.

To run the algorithm:
1. Ensure all required dependencies are installed (see requirements.txt)
2. Set up your Google Sheets API credentials (SERVICE_ACCOUNT_FILE in config.py)
3. Run this script using: python main.py [--num_stocks N] [--verbose]
4. Check the console output for the stock recommendations
5. View the full results in the connected Google Sheet

Performance optimizations:
- Improved async processing with proper error handling
- Added timeouts to prevent hanging on API calls or processing
- Implemented caching for expensive operations
- Batch processing to avoid overwhelming resources
- Simplified ML model for faster processing
"""

import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import time
import argparse
import sys
from datetime import datetime
from data_fetchers import DataFetcher
from ml_backtesting import EnhancedBacktester
from scoring import run_scoring
from model_retrainer import schedule_retraining
from sheets_handler import SheetsHandler
from config import SPREADSHEET_ID, MAIN_SHEET, SERVICE_ACCOUNT_FILE
from tqdm import tqdm

logger = logging.getLogger(__name__)

async def fetch_and_process_stock(symbol: str, data_fetcher: DataFetcher, verbose: bool, **kwargs) -> Dict[str, Any]:
    """
    Fetch historical data for a given symbol and process it using EnhancedBacktester and scoring.
    Enhanced to properly populate price, support, resistance, and other required fields.

    Args:
        symbol (str): Stock symbol to process.
        data_fetcher (DataFetcher): DataFetcher instance to use for fetching data.
        verbose (bool): Whether to print verbose output.

    Returns:
        Dict[str, Any]: Processed results for the symbol.
    """
    try:
        # Use a cache key based on symbol and current date to enable caching
        cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d')}"
        
        # Check if we have cached results (simple in-memory cache)
        # Skip cache if force_refresh is True
        force_refresh = kwargs.get('force_refresh', True)  # Default to True for now until fundamentals are stable
        
        if not force_refresh and hasattr(fetch_and_process_stock, 'cache') and cache_key in fetch_and_process_stock.cache:
            if verbose:
                print(f"🔄 CACHED: Using cached results for {symbol} (from today)")
            logger.info(f"Using cached results for {symbol}")
            return fetch_and_process_stock.cache[cache_key]
            
        if verbose:
            print(f"Fetching historical data for {symbol}")
        logger.info(f"Fetching historical data for {symbol}")
        
        # Fetch fundamental data in parallel with historical data
        fundamental_task = asyncio.create_task(data_fetcher.fetch_fundamentals(symbol))
        
        # Add timeout to the API call
        try:
            historical_data = await asyncio.wait_for(
                data_fetcher.fetch_historical_data(symbol),
                timeout=30  # 30-second timeout for API call
            )
        except asyncio.TimeoutError:
            if verbose:
                print(f"Timeout fetching data for {symbol}")
            logger.error(f"Timeout fetching data for {symbol}")
            return None
        
        if not historical_data or not historical_data.get('historical_data'):
            if verbose:
                print(f"No historical data available for {symbol}")
            logger.error(f"No historical data available for {symbol}")
            return None

        # Convert historical_data to DataFrame
        df = pd.DataFrame(historical_data['historical_data'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.astype(float)
        
        # Add symbol column to DataFrame for tracking
        df['symbol'] = symbol

        # Get fundamental data
        try:
            fundamental_data = await fundamental_task
            if verbose:
                print(f"Fetched fundamental data for {symbol}")
            logger.info(f"Fetched fundamental data for {symbol}")
        except Exception as e:
            if verbose:
                print(f"Error fetching fundamental data for {symbol}: {str(e)}")
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            fundamental_data = {
                'pe_ratio': None,
                'debt_equity': None,
                'sector': None,
                'industry': None
            }

        if verbose:
            print(f"Preparing data for {symbol}")
        logger.info(f"Preparing data for {symbol}")
        
        # Use a simplified backtester for faster processing
        backtester = EnhancedBacktester(
            use_simplified_model=True,  # Use a simpler model for faster processing
            n_estimators=50,            # Reduce number of estimators
            max_depth=3                 # Limit tree depth
        )
        
        prepared_data = backtester.prepare_data(df, symbol)

        if verbose:
            print(f"Training model for {symbol}")
        logger.info(f"Training model for {symbol}")
        
        # Train model with timeout
        try:
            await asyncio.wait_for(
                asyncio.to_thread(backtester.train_model, prepared_data),
                timeout=60  # 60-second timeout for model training
            )
        except asyncio.TimeoutError:
            if verbose:
                print(f"Model training timed out for {symbol}, using simplified approach")
            logger.warning(f"Model training timed out for {symbol}, using simplified approach")
            # Use a fallback approach if training times out
            backtester.use_fallback_model()
        
        if verbose:
            print(f"Running backtest for {symbol}")
        logger.info(f"Running backtest for {symbol}")
        
        # Run backtest with timeout
        try:
            ml_results = await asyncio.wait_for(
                asyncio.to_thread(backtester.backtest, df),
                timeout=30  # 30-second timeout for backtesting
            )
        except asyncio.TimeoutError:
            if verbose:
                print(f"Backtesting timed out for {symbol}, using simplified results")
            logger.warning(f"Backtesting timed out for {symbol}, using simplified results")
            # Generate simplified results if backtesting times out
            ml_results = backtester.generate_simplified_results(df)
        
        if verbose:
            print(f"Scoring results for {symbol}")
        logger.info(f"Scoring results for {symbol}")
        
        # Calculate support and resistance levels
        support = None
        resistance = None
        
        try:
            # Calculate 20-day low as support
            lows = df['low'].iloc[:20]
            support = lows.min()
            
            # Calculate 20-day high as resistance
            highs = df['high'].iloc[:20]
            resistance = highs.max()
        except (KeyError, IndexError):
            # Fallback if low/high not available
            if 'close' in df.columns:
                close_price = df['close'].iloc[-1]
                support = close_price * 0.95  # 5% below current price
                resistance = close_price * 1.05  # 5% above current price
        
        # Get current price (latest close)
        current_price = df['close'].iloc[-1] if 'close' in df.columns else None
        
        # Calculate MA50/200 ratio
        ma_50 = df['ma_50'].iloc[-1] if 'ma_50' in df.columns else None
        ma_200 = df['ma_200'].iloc[-1] if 'ma_200' in df.columns else None
        ma50_200_ratio = None
        
        if ma_50 is not None and ma_200 is not None and ma_200 > 0:
            ma50_200_ratio = ma_50 / ma_200
        
        # Calculate 20-day momentum
        momentum_20d = None
        if len(df) >= 20 and 'close' in df.columns:
            current = df['close'].iloc[-1]
            past = df['close'].iloc[-20] if len(df) >= 20 else df['close'].iloc[0]
            if past > 0:
                momentum_20d = ((current / past) - 1) * 100  # as percentage
        
        # Check for Bollinger squeeze
        bollinger_squeeze = False
        if 'bollinger_upper' in df.columns and 'bollinger_lower' in df.columns and 'close' in df.columns:
            upper = df['bollinger_upper'].iloc[-1]
            lower = df['bollinger_lower'].iloc[-1]
            close = df['close'].iloc[-1]
            if close > 0:
                band_width = (upper - lower) / close
                bollinger_squeeze = band_width < 0.05  # 5% threshold
        
        # Prepare metrics for scoring
        metrics = {
            'ml_prediction': ml_results.get('ml_prediction', 0),
            'returns': ml_results.get('returns', []),
            'positions': ml_results.get('positions', []),
            'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
            'macd': df['macd'].iloc[-1] if 'macd' in df.columns else 0,
            'macd_signal': df['signal'].iloc[-1] if 'signal' in df.columns else 0,
            'price': current_price,
            'bb_lower': df['bollinger_lower'].iloc[-1] if 'bollinger_lower' in df.columns else (current_price * 0.95 if current_price else None),
            'bb_upper': df['bollinger_upper'].iloc[-1] if 'bollinger_upper' in df.columns else (current_price * 1.05 if current_price else None),
            'ma_50': ma_50,
            'ma_200': ma_200,
            'volatility': df['volatility'].iloc[-1] if 'volatility' in df.columns else df['close'].std() if 'close' in df.columns else None,
        }
        
        # Run scoring with timeout
        try:
            scoring_results = await asyncio.wait_for(
                asyncio.to_thread(run_scoring, metrics, {symbol: df['close'].values}),
                timeout=15  # 15-second timeout for scoring
            )
        except asyncio.TimeoutError:
            if verbose:
                print(f"Scoring timed out for {symbol}, using default values")
            logger.warning(f"Scoring timed out for {symbol}, using default values")
            # Use default values if scoring times out
            scoring_results = {
                'score': 0.5,
                'verdict': 'Hold',
                'transaction_cost_impact': 0.01,
                'slippage_impact': 0.005,
                'cost_adjusted_sharpe_ratio': 1.0
            }
        
        # Generate explanation based on metrics
        explanation = generate_explanation(
            symbol,
            scoring_results.get('verdict', 'Hold'),
            scoring_results.get('score', 0.5),
            metrics.get('rsi', 50),
            ml_results.get('sharpe_ratio', 1.0),
            ml_results.get('max_drawdown', 0.1),
            metrics.get('ma_50', None),
            metrics.get('ma_200', None),
            ma50_200_ratio,
            momentum_20d,
            bollinger_squeeze,
            fundamental_data.get('pe_ratio'),
            fundamental_data.get('sector')
        )

        result = {
            'symbol': symbol,
            'score': scoring_results.get('score', 0.5),
            'verdict': scoring_results.get('verdict', 'Hold'),
            'total_return': ml_results.get('total_return', 0),
            'sharpe_ratio': ml_results.get('sharpe_ratio', 1.0),
            'sortino_ratio': ml_results.get('sortino_ratio', 1.0),
            'calmar_ratio': ml_results.get('calmar_ratio', 1.0),
            'max_drawdown': ml_results.get('max_drawdown', 0.1),
            'win_rate': ml_results.get('win_rate', 0.5),
            'avg_win': ml_results.get('avg_win', 0.02),
            'avg_loss': ml_results.get('avg_loss', -0.01),
            'transaction_cost_impact': scoring_results.get('transaction_cost_impact', 0.01),
            'slippage_impact': scoring_results.get('slippage_impact', 0.005),
            'cost_adjusted_sharpe_ratio': scoring_results.get('cost_adjusted_sharpe_ratio', 1.0),
            'current_price': current_price,
            'support': support,
            'resistance': resistance,
            'rsi': metrics.get('rsi', 50),
            'bollinger_squeeze': bollinger_squeeze,
            'ma50_200_ratio': ma50_200_ratio,
            'momentum_20d': momentum_20d,
            'pe_ratio': fundamental_data.get('pe_ratio'),
            'debt_equity': fundamental_data.get('debt_equity'),
            'sector': fundamental_data.get('sector'),
            'industry': fundamental_data.get('industry'),
            'explanation': explanation,
            'historical_data': historical_data.get('historical_data', [])
        }
        
        # Cache the result
        if not hasattr(fetch_and_process_stock, 'cache'):
            fetch_and_process_stock.cache = {}
        fetch_and_process_stock.cache[cache_key] = result
        
        if verbose:
            print(f"Successfully processed {symbol}")
        logger.info(f"Successfully processed {symbol}")
        return result
    except Exception as e:
        if verbose:
            print(f"Error processing {symbol}: {str(e)}")
        logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
        return None

def generate_explanation(symbol, verdict, score, rsi, sharpe_ratio, max_drawdown, ma_50, ma_200, ma_ratio, momentum, bollinger_squeeze, pe_ratio=None, sector=None):
    """
    Generate a detailed explanation for the stock's rating.
    
    Args:
        symbol (str): Stock symbol
        verdict (str): Trading verdict (Strong Buy, Buy, Hold, Sell, Strong Sell)
        score (float): Composite score
        rsi (float): Relative Strength Index
        sharpe_ratio (float): Sharpe ratio
        max_drawdown (float): Maximum drawdown
        ma_50 (float): 50-day moving average
        ma_200 (float): 200-day moving average
        ma_ratio (float): MA50/200 ratio
        momentum (float): 20-day momentum percentage
        bollinger_squeeze (bool): Whether there's a Bollinger squeeze
        pe_ratio (float, optional): Price to Earnings ratio
        sector (str, optional): Stock sector
        
    Returns:
        str: Detailed explanation
    """
    explanation = f"The stock's composite score is {score:.2f}, resulting in a {verdict} verdict. "
    
    if verdict in ['Strong Buy', 'Buy']:
        explanation += f"{symbol} is considered a buy because: "
    elif verdict == 'Hold':
        explanation += f"{symbol} is not currently recommended as a buy or sell because: "
    else:
        explanation += f"{symbol} is not recommended as a buy because: "
    
    factors = []
    
    # RSI analysis
    if rsi is not None:
        if rsi < 30:
            factors.append(f"The RSI ({rsi:.2f}) indicates the stock may be oversold")
        elif rsi > 70:
            factors.append(f"The RSI ({rsi:.2f}) indicates the stock may be overbought")
        else:
            factors.append(f"The RSI ({rsi:.2f}) is in a neutral range")
    
    # Sharpe ratio analysis
    if sharpe_ratio is not None:
        if sharpe_ratio > 1:
            factors.append(f"The Sharpe Ratio ({sharpe_ratio:.2f}) suggests good risk-adjusted returns")
        elif sharpe_ratio < 0:
            factors.append(f"The Sharpe Ratio ({sharpe_ratio:.2f}) suggests poor risk-adjusted returns")
        else:
            factors.append(f"The Sharpe Ratio ({sharpe_ratio:.2f}) suggests moderate risk-adjusted returns")
    
    # Max drawdown analysis
    if max_drawdown is not None:
        if max_drawdown > 0.2:
            factors.append(f"The Max Drawdown ({max_drawdown:.2%}) is relatively high, indicating increased risk")
        else:
            factors.append(f"The Max Drawdown ({max_drawdown:.2%}) is at an acceptable level")
    
    # Moving average analysis
    if ma_ratio is not None:
        if ma_ratio > 1.05:
            factors.append(f"The MA50/200 Ratio ({ma_ratio:.2f}) indicates a strong bullish trend")
        elif ma_ratio > 1:
            factors.append(f"The MA50/200 Ratio ({ma_ratio:.2f}) indicates a bullish trend")
        elif ma_ratio < 0.95:
            factors.append(f"The MA50/200 Ratio ({ma_ratio:.2f}) indicates a strong bearish trend")
        elif ma_ratio < 1:
            factors.append(f"The MA50/200 Ratio ({ma_ratio:.2f}) indicates a bearish trend")
        else:
            factors.append(f"The MA50/200 Ratio ({ma_ratio:.2f}) indicates a neutral trend")
    
    # Momentum analysis
    if momentum is not None:
        if momentum > 10:
            factors.append(f"The 20d Momentum ({momentum:.2f}%) shows very strong positive momentum")
        elif momentum > 5:
            factors.append(f"The 20d Momentum ({momentum:.2f}%) shows strong positive momentum")
        elif momentum < -10:
            factors.append(f"The 20d Momentum ({momentum:.2f}%) shows very strong negative momentum")
        elif momentum < -5:
            factors.append(f"The 20d Momentum ({momentum:.2f}%) shows strong negative momentum")
        else:
            factors.append(f"The 20d Momentum ({momentum:.2f}%) shows relatively neutral momentum")
    
    # Bollinger squeeze analysis
    if bollinger_squeeze:
        factors.append("A Bollinger Squeeze is detected, suggesting a potential breakout may occur soon")
    
    # P/E ratio analysis
    if pe_ratio is not None:
        if pe_ratio > 25:
            factors.append(f"The P/E Ratio ({pe_ratio:.2f}) suggests the stock may be overvalued")
        elif pe_ratio < 15:
            factors.append(f"The P/E Ratio ({pe_ratio:.2f}) suggests the stock may be undervalued")
        else:
            factors.append(f"The P/E Ratio ({pe_ratio:.2f}) is in a reasonable range")
    
    # Add sector information if available
    if sector:
        factors.append(f"The stock belongs to the {sector} sector")
    
    explanation += ". ".join(factors) + "."
    
    explanation += "\n\nPlease note that this analysis is based on historical data and current market conditions. Always conduct your own research and consider your personal financial situation before making investment decisions."
    
    return explanation

# Helper functions
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(prices, window=20, num_std=2):
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    return upper_band, lower_band

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_obv(close, volume):
    return (np.sign(close.diff()) * volume).cumsum()

def calculate_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad)

def calculate_stochastic_oscillator(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

async def fetch_and_process_stocks(symbols: List[str], verbose: bool, force_refresh: bool = True) -> List[Dict[str, Any]]:
    """
    Fetch and process multiple stock symbols concurrently.

    Args:
        symbols (List[str]): List of stock symbols to process.
        verbose (bool): Whether to print verbose output.
        force_refresh (bool, optional): Whether to force refresh data even if cached. Defaults to True.

    Returns:
        List[Dict[str, Any]]: List of processed results for each symbol.
    """
    data_fetcher = DataFetcher()
    tasks = []
    for symbol in symbols:
        task = fetch_and_process_stock(symbol, data_fetcher, verbose, force_refresh=force_refresh)
        tasks.append(task)
    
    # Add explicit logging for debugging
    logger.info(f"Processing {len(tasks)} tasks for symbols: {symbols}")
    
    results = await asyncio.gather(*tasks)
    
    # Add validation for results
    valid_results = [result for result in results if result is not None]
    logger.info(f"Obtained {len(valid_results)} valid results out of {len(symbols)} symbols")
    
    if len(valid_results) < len(symbols):
        failed_symbols = [symbol for symbol, result in zip(symbols, results) if result is None]
        logger.warning(f"Failed to process these symbols: {failed_symbols}")
    
    return valid_results

async def main(num_stocks: int = None, verbose: bool = False, force_refresh: bool = True):
    start_time = time.time()
    print("Starting Stock Spike Replicator")
    logger.info("Starting Stock Spike Replicator")
    
    # Set up logging based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize model retraining scheduler
    retraining_task = None
    try:
        # Set a timeout for the entire process
        async with asyncio.timeout(1800):  # 30-minute timeout for the entire process
            # Initialize model retraining scheduler with lower priority
            retraining_task = asyncio.create_task(schedule_retraining())
            print("Model retraining task initialized")
            
            # Initialize Google Sheets handler
            if verbose:
                print("Initializing Google Sheets handler")
            logger.info("Initializing Google Sheets handler")
            sheets_handler = SheetsHandler(SPREADSHEET_ID, SERVICE_ACCOUNT_FILE)
            
            # Add timeout for Google Sheets initialization
            try:
                sheets_init_success = await asyncio.wait_for(
                    asyncio.to_thread(sheets_handler.initialize),
                    timeout=30  # 30-second timeout for initialization
                )
                if not sheets_init_success:
                    print("Failed to initialize Google Sheets connection")
                    logger.error("Failed to initialize Google Sheets connection")
                    return
            except asyncio.TimeoutError:
                print("Google Sheets initialization timed out")
                logger.error("Google Sheets initialization timed out")
                return
            
            if verbose:
                print("Fetching user-provided symbols from Google Sheet")
            logger.info("Fetching user-provided symbols from Google Sheet")
            
            # Get worksheet with timeout
            try:
                main_worksheet = await asyncio.wait_for(
                    asyncio.to_thread(sheets_handler.get_or_create_worksheet, MAIN_SHEET),
                    timeout=20  # 20-second timeout
                )
            except asyncio.TimeoutError:
                print("Timed out getting worksheet")
                logger.error("Timed out getting worksheet")
                return
            
            # Get symbols with timeout
            try:
                user_symbols_raw = await asyncio.wait_for(
                    asyncio.to_thread(sheets_handler.get_range, main_worksheet, 'A2:A'),
                    timeout=20  # 20-second timeout
                )
                user_symbols = [symbol[0] for symbol in user_symbols_raw if symbol]  # Flatten and remove empty cells
            except asyncio.TimeoutError:
                print("Timed out fetching symbols from sheet")
                logger.error("Timed out fetching symbols from sheet")
                return
            
            # Process user-provided symbols
            if user_symbols:
                symbols_to_process = user_symbols if num_stocks is None else user_symbols[:num_stocks]
            else:
                print("No symbols found in the sheet. Please add symbols to process.")
                logger.warning("No symbols found in the sheet.")
                return
            
            print(f"Processing {len(symbols_to_process)} stocks: {', '.join(symbols_to_process)}")
            logger.info(f"Processing {len(symbols_to_process)} stocks: {', '.join(symbols_to_process)}")
            
            # Estimate processing time (more optimistic with our improvements)
            estimated_time_per_stock = 30  # seconds (reduced from 60)
            total_estimated_time = len(symbols_to_process) * estimated_time_per_stock
            print(f"Estimated processing time: {total_estimated_time // 60} minutes {total_estimated_time % 60} seconds")
            
            # Process stocks with progress reporting
            print("Starting stock processing...")
            processing_start = time.time()
            
            results = await fetch_and_process_stocks(symbols_to_process, verbose, force_refresh)
            
            processing_time = time.time() - processing_start
            print(f"Stock processing completed in {processing_time:.2f} seconds")
            
            if not results:
                print("No valid results were obtained. Please check the logs for errors.")
                logger.error("No valid results were obtained")
                return
            
            # Sort results by score in descending order
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

            if verbose:
                print("Updating Google Sheet with results")
            logger.info("Updating Google Sheet with results")
            
            # Update sheet with timeout
            try:
                update_success = await asyncio.wait_for(
                    asyncio.to_thread(sheets_handler.update_sheet_with_results, main_worksheet, sorted_results),
                    timeout=60  # 60-second timeout for sheet update
                )
                if not update_success:
                    logger.error("Failed to update Google Sheet with results")
                    print("Failed to update Google Sheet with results")
            except asyncio.TimeoutError:
                print("Timed out updating Google Sheet")
                logger.error("Timed out updating Google Sheet")

            if verbose:
                print("Updating summary tab")
            logger.info("Updating summary tab")
            
            # Update summary with timeout
            try:
                summary_worksheet = await asyncio.wait_for(
                    asyncio.to_thread(sheets_handler.get_or_create_worksheet, 'Summary'),
                    timeout=20  # 20-second timeout
                )
                summary_success = await asyncio.wait_for(
                    asyncio.to_thread(sheets_handler.update_summary_tab, summary_worksheet),
                    timeout=30  # 30-second timeout
                )
                if not summary_success:
                    logger.error("Failed to update summary tab")
                    print("Failed to update summary tab")
            except asyncio.TimeoutError:
                print("Timed out updating summary tab")
                logger.error("Timed out updating summary tab")

            # Print recommendations and usage instructions
            print("\nStock Recommendations:")
            for result in sorted_results:
                print(f"Symbol: {result['symbol']}")
                print(f"Score: {result['score']:.2f}")
                print(f"Verdict: {result['verdict']}")
                print(f"Total Return: {result['total_return']:.2%}")
                print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                print(f"Sortino Ratio: {result['sortino_ratio']:.2f}")
                print(f"Calmar Ratio: {result['calmar_ratio']:.2f}")
                print(f"Max Drawdown: {result['max_drawdown']:.2%}")
                print(f"Win Rate: {result['win_rate']:.2%}")
                print(f"Average Win: {result['avg_win']:.2%}")
                print(f"Average Loss: {result['avg_loss']:.2%}")
                print("---")

            end_time = time.time()
            total_time = end_time - start_time
            print(f"\nStock Spike Replicator completed successfully in {total_time:.2f} seconds")
            logger.info(f"Stock Spike Replicator completed successfully in {total_time:.2f} seconds")

            print("\nImportant Information:")
            print(f"- Main results are in the '{MAIN_SHEET}' sheet of your Google Sheets document")
            print(f"- Backup sheets are created with the prefix 'Backup_' when there are 30+ data points")
            print("- The scheduler runs independently within the application and does not require Windows Task Scheduler")
            print("- Daily updates are performed automatically when the application is running")
            print("\nPlease check the 'Summary' sheet in your Google Sheets document for more details on the algorithm's advantages and usage instructions.")
    except asyncio.TimeoutError:
        print("Operation timed out after 30 minutes")
        logger.error("Operation timed out after 30 minutes")
    except asyncio.CancelledError:
        print("Operation was cancelled")
        logger.info("Operation was cancelled")
    except KeyboardInterrupt:
        print("Operation was interrupted by user")
        logger.info("Operation was interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
    finally:
        # Cancel the retraining task if it exists
        if retraining_task:
            retraining_task.cancel()
            try:
                await retraining_task
            except asyncio.CancelledError:
                print("Retraining task cancelled")
                logger.info("Retraining task cancelled")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Spike Replicator")
    parser.add_argument("--num_stocks", type=int, default=None, help="Number of stocks to process (default: all symbols)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh data even if cached")
    parser.add_argument("--use-cache", action="store_true", help="Use cached data if available (overrides --force-refresh)")
    args = parser.parse_args()
    
    # Determine whether to force refresh
    # Default to True for now until fundamentals are stable
    force_refresh = True
    if args.use_cache:
        force_refresh = False
    elif args.force_refresh:
        force_refresh = True
    
    try:
        asyncio.run(main(args.num_stocks, args.verbose, force_refresh))
    except KeyboardInterrupt:
        print("Script execution was interrupted by user")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
    sys.exit(0)

print("""
Instructions for using the Stock Spike Replicator:

Option 1: Google Sheets Integration
1. Ensure all required dependencies are installed (see requirements.txt)
2. Set up your Google Sheets API credentials (SERVICE_ACCOUNT_FILE in config.py)
3. Add the stock symbols you want to analyze in the 'A' column of the main sheet in your Google Sheets document
4. Run this script using: python main.py [--num_stocks N] [--verbose] [--force-refresh] [--use-cache]
   Where:
   - N is the number of stocks you want to process (default is all symbols)
   - --verbose enables detailed output
   - --force-refresh forces refreshing data even if cached (default behavior currently)
   - --use-cache uses cached data if available (overrides --force-refresh)
5. Check the console output for the top 5 stock recommendations
6. View the full results in the connected Google Sheet

Option 2: Web Interface (Recommended)
1. Run 'start_servers.bat' in the project directory
2. Open http://localhost:3000 in your web browser
3. Create an account or log in to access all features
4. Use the interactive dashboard for real-time analysis and portfolio management

The web interface offers additional benefits such as real-time updates, interactive backtesting, and user-friendly watchlist management. See the Summary tab in your Google Sheet for more details on web interface advantages.
""")