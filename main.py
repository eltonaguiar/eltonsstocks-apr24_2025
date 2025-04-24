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
"""

import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
import time
import argparse
import sys
from data_fetchers import DataFetcher
from ml_backtesting import EnhancedBacktester
from scoring import run_scoring
from model_retrainer import schedule_retraining
from sheets_handler import SheetsHandler
from config import SPREADSHEET_ID, MAIN_SHEET, SERVICE_ACCOUNT_FILE
from tqdm import tqdm

logger = logging.getLogger(__name__)

async def fetch_and_process_stock(symbol: str, data_fetcher: DataFetcher, verbose: bool) -> Dict[str, Any]:
    """
    Fetch historical data for a given symbol and process it using EnhancedBacktester and scoring.

    Args:
        symbol (str): Stock symbol to process.
        data_fetcher (DataFetcher): DataFetcher instance to use for fetching data.
        verbose (bool): Whether to print verbose output.

    Returns:
        Dict[str, Any]: Processed results for the symbol.
    """
    try:
        if verbose:
            print(f"Fetching historical data for {symbol}")
        logger.info(f"Fetching historical data for {symbol}")
        historical_data = await data_fetcher.fetch_historical_data(symbol)
        
        if not historical_data['historical_data']:
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

        if verbose:
            print(f"Preparing data for {symbol}")
        logger.info(f"Preparing data for {symbol}")
        backtester = EnhancedBacktester()
        prepared_data = backtester.prepare_data(df, symbol)

        if verbose:
            print(f"Training model for {symbol}")
        logger.info(f"Training model for {symbol}")
        backtester.train_model(prepared_data)
        
        if verbose:
            print(f"Running backtest for {symbol}")
        logger.info(f"Running backtest for {symbol}")
        ml_results = backtester.backtest(df)
        
        if verbose:
            print(f"Scoring results for {symbol}")
        logger.info(f"Scoring results for {symbol}")
        
        # Prepare metrics for scoring
        metrics = {
            'ml_prediction': ml_results['ml_prediction'],
            'returns': ml_results['returns'],
            'positions': ml_results['positions'],
            'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else None,
            'macd': df['macd'].iloc[-1] if 'macd' in df.columns else None,
            'macd_signal': df['signal'].iloc[-1] if 'signal' in df.columns else None,
            'price': df['close'].iloc[-1],
            'bb_lower': df['bollinger_lower'].iloc[-1] if 'bollinger_lower' in df.columns else None,
            'bb_upper': df['bollinger_upper'].iloc[-1] if 'bollinger_upper' in df.columns else None,
            'ma_50': df['ma_50'].iloc[-1] if 'ma_50' in df.columns else None,
            'ma_200': df['ma_200'].iloc[-1] if 'ma_200' in df.columns else None,
            'volatility': df['volatility'].iloc[-1] if 'volatility' in df.columns else None,
        }
        
        scoring_results = run_scoring(metrics, {symbol: df['close'].values})

        result = {
            'symbol': symbol,
            'score': scoring_results['score'],
            'verdict': scoring_results['verdict'],
            'total_return': ml_results['total_return'],
            'sharpe_ratio': ml_results['sharpe_ratio'],
            'sortino_ratio': ml_results['sortino_ratio'],
            'calmar_ratio': ml_results['calmar_ratio'],
            'max_drawdown': ml_results['max_drawdown'],
            'win_rate': ml_results['win_rate'],
            'avg_win': ml_results['avg_win'],
            'avg_loss': ml_results['avg_loss'],
            'transaction_cost_impact': scoring_results['transaction_cost_impact'],
            'slippage_impact': scoring_results['slippage_impact'],
            'cost_adjusted_sharpe_ratio': scoring_results['cost_adjusted_sharpe_ratio'],
        }
        if verbose:
            print(f"Successfully processed {symbol}")
        logger.info(f"Successfully processed {symbol}")
        return result
    except Exception as e:
        if verbose:
            print(f"Error processing {symbol}: {str(e)}")
        logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
        return None

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

async def fetch_and_process_stocks(symbols: List[str], verbose: bool) -> List[Dict[str, Any]]:
    """
    Fetch and process multiple stock symbols concurrently.

    Args:
        symbols (List[str]): List of stock symbols to process.
        verbose (bool): Whether to print verbose output.

    Returns:
        List[Dict[str, Any]]: List of processed results for each symbol.
    """
    data_fetcher = DataFetcher()
    tasks = []
    for symbol in symbols:
        task = fetch_and_process_stock(symbol, data_fetcher, verbose)
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

async def main(num_stocks: int = 31, verbose: bool = False):
    start_time = time.time()
    print("Starting Stock Spike Replicator")
    logger.info("Starting Stock Spike Replicator")
    
    # Set up logging based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize model retraining scheduler
    retraining_task = None
    try:
        retraining_task = asyncio.create_task(schedule_retraining())
        print("Model retraining task initialized")
        
        # Initialize Google Sheets handler
        if verbose:
            print("Initializing Google Sheets handler")
        logger.info("Initializing Google Sheets handler")
        sheets_handler = SheetsHandler(SPREADSHEET_ID, SERVICE_ACCOUNT_FILE)
        if not sheets_handler.initialize():
            print("Failed to initialize Google Sheets connection")
            logger.error("Failed to initialize Google Sheets connection")
            return
        
        if verbose:
            print("Fetching user-provided symbols from Google Sheet")
        logger.info("Fetching user-provided symbols from Google Sheet")
        main_worksheet = sheets_handler.get_or_create_worksheet(MAIN_SHEET)
        user_symbols = sheets_handler.get_range(main_worksheet, 'A2:A')
        user_symbols = [symbol[0] for symbol in user_symbols if symbol]  # Flatten and remove empty cells
        
        # Process user-provided symbols
        if user_symbols:
            symbols_to_process = user_symbols[:num_stocks]
        else:
            print("No symbols found in the sheet. Please add symbols to process.")
            logger.warning("No symbols found in the sheet.")
            return
        
        print(f"Processing {len(symbols_to_process)} stocks: {', '.join(symbols_to_process)}")
        logger.info(f"Processing {len(symbols_to_process)} stocks: {', '.join(symbols_to_process)}")
        
        # Estimate processing time
        estimated_time_per_stock = 60  # seconds
        total_estimated_time = len(symbols_to_process) * estimated_time_per_stock
        print(f"Estimated processing time: {total_estimated_time // 60} minutes {total_estimated_time % 60} seconds")
        
        results = await fetch_and_process_stocks(symbols_to_process, verbose)
        
        # Sort results by score in descending order
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

        if verbose:
            print("Updating Google Sheet with results")
        logger.info("Updating Google Sheet with results")
        update_success = sheets_handler.update_sheet_with_results(main_worksheet, sorted_results)
        if not update_success:
            logger.error("Failed to update Google Sheet with results")
            print("Failed to update Google Sheet with results")

        if verbose:
            print("Updating summary tab")
        logger.info("Updating summary tab")
        summary_worksheet = sheets_handler.get_or_create_worksheet('Summary')
        summary_success = sheets_handler.update_summary_tab(summary_worksheet)
        if not summary_success:
            logger.error("Failed to update summary tab")
            print("Failed to update summary tab")

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
    parser.add_argument("--num_stocks", type=int, default=1, help="Number of stocks to process (default: 1)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.num_stocks, args.verbose))
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
4. Run this script using: python main.py [--num_stocks N]
   Where N is the number of stocks you want to process (default is 5)
5. Check the console output for the top 5 stock recommendations
6. View the full results in the connected Google Sheet

Option 2: Web Interface (Recommended)
1. Run 'start_servers.bat' in the project directory
2. Open http://localhost:3000 in your web browser
3. Create an account or log in to access all features
4. Use the interactive dashboard for real-time analysis and portfolio management

The web interface offers additional benefits such as real-time updates, interactive backtesting, and user-friendly watchlist management. See the Summary tab in your Google Sheet for more details on web interface advantages.
""")