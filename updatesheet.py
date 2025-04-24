"""
Stock update application for maintaining a Google Sheets dashboard with stock metrics.
This script fetches financial data, calculates technical indicators, and updates a Google Sheet.

Improvements:
- Modular architecture with separate components
- Asynchronous API calls for improved performance
- Rate limit handling with automatic fallback between providers
- Error handling with detailed logging
- Configurable metrics and calculations
- Caching mechanism for frequently accessed data
- Visualization of top stock pick
"""
import asyncio
import logging
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
from io import BytesIO

from config import (
    API_KEYS, RATE_LIMITS, SPREADSHEET_ID, MAIN_SHEET, TEMP_SHEET,
    SERVICE_ACCOUNT_FILE, HEADER, SIMPLE_SCORE_WEIGHTS, ADVANCED_SCORE_WEIGHTS,
    CELL_FORMATS, LOG_FORMAT, LOG_LEVEL, LOG_FILE, API_ENDPOINTS,
    idx_to_col, COLUMNS
)
from technical_indicators import (
    calculate_sharpe_ratio, detect_volatility_squeeze,
    calculate_max_drawdown, calculate_moving_average_ratio, calculate_momentum,
    calculate_macd, calculate_bollinger_bands, calculate_rsi
)
from scoring import ScoreCalculator
from data_fetchers import DataFetcher
from sheets_handler import SheetsHandler
from ml_backtesting import run_ml_backtesting
from visualizations import plot_top_stock_pick

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)

# Create console handler for output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger('main')


class StockUpdateApp:
    """Main application class for updating the stock data sheet."""
    
    def __init__(self):
        """Initialize the application components."""
        self.data_fetcher = DataFetcher(API_KEYS, RATE_LIMITS)
        self.sheets_handler = SheetsHandler(SPREADSHEET_ID, SERVICE_ACCOUNT_FILE, batch_size=20)
        self.score_calculator = ScoreCalculator({
            'simple': SIMPLE_SCORE_WEIGHTS,
            'advanced': ADVANCED_SCORE_WEIGHTS
        })
        self.sector_cache = {}
        self.symbol_data_cache = {}
        self.historical_data = {}

    @lru_cache(maxsize=1000)
    def get_cached_sector_data(self, symbol: str) -> Tuple[str, Dict]:
        """Get cached sector data for a symbol."""
        sector = self.sector_cache.get(symbol, {}).get('sector', 'Unknown')
        sector_metrics = self.sector_cache.get(symbol, {}).get('metrics', {})
        return sector, sector_metrics

    @lru_cache(maxsize=1000)
    def get_cached_symbol_data(self, symbol: str, date_str: str) -> Dict:
        """Get cached symbol data."""
        return self.symbol_data_cache.get((symbol, date_str), {})

    def cache_sector_data(self, sectors: Dict, sector_metrics: Dict):
        """Cache sector data for all symbols."""
        for symbol, sector in sectors.items():
            self.sector_cache[symbol] = {
                'sector': sector,
                'metrics': sector_metrics.get(sector, {})
            }

    def cache_symbol_data(self, symbol: str, date_str: str, data: Dict):
        """Cache symbol data."""
        self.symbol_data_cache[(symbol, date_str)] = data
        
    async def run(self):
        """Run the main update process."""
        start_time = datetime.now()
        logger.info(f"Starting stock update at {start_time:%H:%M:%S}")
        print(f"🚀 Starting stock update at {start_time:%H:%M:%S}")
        
        try:
            # Initialize sheets
            main_sheet, temp_sheet = self.initialize_sheets()
            
            # Get all rows from the main sheet
            all_rows = self.sheets_handler.get_all_values(main_sheet)
            if not all_rows or len(all_rows) <= 1:  # Account for header row
                logger.error("No data found in the main sheet")
                print("❌ No data found in the main sheet")
                return False
            
            # Extract symbols and dates (skip header row)
            symbol_data = []
            for i, row in enumerate(all_rows[1:], 1):
                if not row or not row[0].strip():
                    continue
                
                symbol = row[0].strip()
                date_str = row[2].strip() if len(row) > 2 else ""
                symbol_data.append({
                    'row': i+1,  # Add 1 to account for header row & 1-based indexing
                    'symbol': symbol,
                    'date': date_str
                })
            
            total_symbols = len(symbol_data)
            logger.info(f"Found {total_symbols} symbols to update")
            print(f"Found {total_symbols} symbols to update")
            
            # Fetch sector data to provide context for scoring
            if total_symbols > 5:
                print("🔍 Fetching sector data...")
                sectors = await self.get_sector_data([d['symbol'] for d in symbol_data])
            else:
                sectors = ({}, {})
            
            # Clear old data and prepare sheets
            await self.prepare_sheets(main_sheet, temp_sheet, total_symbols)
            
            # Process symbols in batches to avoid overloading APIs
            batch_size = 5
            results_temp = []
            results_main = []
            
            # Process all symbols in batches
            for i in range(0, total_symbols, batch_size):
                batch = symbol_data[i:i + batch_size]
                
                # Process the batch of symbols
                batch_results = await self.process_symbols_batch(batch, sectors)
                
                # Add batch results to our aggregated results
                for temp_buf, main_buf in batch_results:
                    results_temp.extend(temp_buf)
                    results_main.extend(main_buf)
                
                # Update the sheets with the batch results
                if results_temp:
                    self.sheets_handler.batch_update(temp_sheet, results_temp)
                    print(f"✅ Updated temp sheet for batch {i//batch_size + 1}/{(total_symbols-1)//batch_size + 1}")
                    results_temp = []
                
                if results_main:
                    self.sheets_handler.batch_update(main_sheet, results_main)
                    print(f"✅ Updated main sheet for batch {i//batch_size + 1}/{(total_symbols-1)//batch_size + 1}")
                    results_main = []
                
                if i + batch_size < total_symbols:
                    print("Cooldown period between batches. Please wait...")
                    time.sleep(60)  # 60-second cooldown
            
            # Apply cell formatting
            self.apply_formatting(main_sheet, total_symbols+1)  # +1 for header row
            
            # Run ML and backtesting
            self.run_ml_backtesting()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Finished update at {end_time:%H:%M:%S} (duration: {duration:.2f} seconds)")
            print(f"🎉 Finished update at {end_time:%H:%M:%S} (duration: {duration:.2f} seconds)")
            
            # Log cache statistics
            logger.info(f"Sector cache hits: {self.get_cached_sector_data.cache_info().hits}")
            logger.info(f"Symbol data cache hits: {self.get_cached_symbol_data.cache_info().hits}")
            
            return True
            
        except Exception as e:
            logger.critical(f"Critical error in update process: {str(e)}", exc_info=True)
            print(f"❌ Critical error: {str(e)}")
            return False

    def clear_caches(self):
        """Clear all caches."""
        self.get_cached_sector_data.cache_clear()
        self.get_cached_symbol_data.cache_clear()
        self.sector_cache.clear()
        self.symbol_data_cache.clear()
        logger.info("All caches cleared")

    def run_ml_backtesting(self):
        """Run machine learning and backtesting on historical data."""
        try:
            print("🧠 Running machine learning and backtesting...")
            cumulative_returns = run_ml_backtesting(self.historical_data)
            
            # Create a new sheet for ML and backtesting results
            ml_sheet = self.sheets_handler.get_or_create_worksheet("ML_Backtesting")
            ml_sheet.clear()
            
            # Prepare data for the sheet
            data = [["Date", "Cumulative Returns"]]
            for date, returns in cumulative_returns.items():
                data.append([date, returns])
            
            # Update the sheet
            self.sheets_handler.batch_update(ml_sheet, [{'range': 'A1', 'values': data}])
            
            print("✅ Machine learning and backtesting completed")
        except Exception as e:
            logger.error(f"Error in ML and backtesting: {str(e)}")
            print("❌ Error occurred during machine learning and backtesting")
    
    def initialize_sheets(self) -> Tuple[Any, Any]:
        """Initialize Google Sheets connection and get required worksheets."""
        if not self.sheets_handler.initialize():
            raise RuntimeError("Failed to initialize Google Sheets connection")
        
        try:
            main_sheet = self.sheets_handler.get_or_create_worksheet(MAIN_SHEET)
            temp_sheet = self.sheets_handler.get_or_create_worksheet(TEMP_SHEET)
            return main_sheet, temp_sheet
        except Exception as e:
            logger.error(f"Failed to get worksheets: {str(e)}")
            raise
    
    async def prepare_sheets(self, main_sheet, temp_sheet, total_rows):
        """Prepare the sheets by clearing old data and setting up headers."""
        try:
            # Map column names to letters using idx_to_col and COLUMNS
            col_price_now = idx_to_col(COLUMNS['price_now'])
            col_today = idx_to_col(COLUMNS['today'])
            col_support = idx_to_col(COLUMNS['support'])
            col_resist = idx_to_col(COLUMNS['resistance'])
            col_composite_score = idx_to_col(COLUMNS['composite_score'])
            col_verdict = idx_to_col(COLUMNS['verdict'])
            col_rsi = idx_to_col(COLUMNS['rsi'])
            col_sharpe = idx_to_col(COLUMNS['sharpe'])
            col_squeeze = idx_to_col(COLUMNS['squeeze'])
            col_drawdown = idx_to_col(COLUMNS['drawdown'])
            col_beta = idx_to_col(COLUMNS['beta'])
            col_ma_ratio = idx_to_col(COLUMNS['ma_ratio'])
            col_momentum = idx_to_col(COLUMNS['momentum'])
            col_pe = idx_to_col(COLUMNS['pe'])
            col_de = idx_to_col(COLUMNS['de'])
            col_explanation = idx_to_col(COLUMNS['explanation'])
            
            # Clear old data in main sheet
            columns_to_clear = [
                col_price_now, col_today, col_support, col_resist, col_composite_score, col_verdict,
                col_rsi, col_sharpe, col_squeeze, col_drawdown, col_beta,
                col_ma_ratio, col_momentum, col_pe, col_de, col_explanation
            ]
            ranges_to_clear = [f"{col}2:{col}{total_rows+1}" for col in columns_to_clear]
            self.sheets_handler.batch_clear(main_sheet, ranges_to_clear)
            
            # Reset temp sheet and set header
            temp_sheet.clear()
            self.sheets_handler.batch_update(temp_sheet, [
                {'range': 'A1', 'values': [HEADER]}
            ])
            
            # Check if column L has formula for price diff calculation
            try:
                row2_formulas = main_sheet.get('L2:L2', value_render_option='FORMULA')
                has_formula = False
                formula_text = ""
                
                if row2_formulas and row2_formulas[0] and row2_formulas[0][0]:
                    cell_value = row2_formulas[0][0]
                    if cell_value.startswith('='):
                        has_formula = True
                        formula_text = cell_value
                        logger.info(f"Found existing formula in L2: {formula_text}")
            except Exception as e:
                logger.error(f"Error checking for formulas: {str(e)}")
                has_formula = False
                
            # Apply formula to all rows if it exists, or create a default one
            formula_cells = []
            for i in range(2, total_rows + 2):
                formula = formula_text if has_formula else f'=F{i}-K{i}'
                formula_cells.append({
                    'range': f'L{i}',
                    'values': [[formula]]
                })
            
            # Apply formulas to all rows at once
            if formula_cells:
                self.sheets_handler.batch_update(main_sheet, formula_cells)
            
            # Double-check formula application
            check_row = main_sheet.get('L2:L2', value_render_option='FORMULA')
            if check_row and check_row[0] and check_row[0][0].startswith('='):
                logger.info("Formula successfully applied to column L")
            else:
                logger.warning("Formula may not have been applied correctly to column L")
                
        except Exception as e:
            logger.error(f"Error preparing sheets: {str(e)}")
            raise
    
    async def get_sector_data(self, symbols: List[str]) -> Tuple[Dict, Dict]:
        """Fetch sector data for all symbols for industry benchmarking."""
        try:
            sectors, sector_metrics = await asyncio.to_thread(
                self.data_fetcher.fetch_sector_data, symbols
            )
            logger.info(f"Collected sector data for {len(sectors)} symbols across {len(sector_metrics)} sectors")
            self.cache_sector_data(sectors, sector_metrics)
            return sectors, sector_metrics
        except Exception as e:
            logger.error(f"Error fetching sector data: {str(e)}")
            return {}, {}
    
    async def process_symbols_batch(self, symbol_batch: List[Dict], sectors: Tuple[Dict, Dict]) -> List[Tuple[List, List]]:
        """Process a batch of symbols in parallel."""
        sectors_map, sector_metrics = sectors
        
        # Create tasks for each symbol in the batch
        tasks = []
        for data in symbol_batch:
            task = self.process_symbol(data, sectors_map, sector_metrics)
            tasks.append(task)
        
        # Run all tasks concurrently and gather results
        batch_results = await asyncio.gather(*tasks)
        return batch_results
    
    async def process_symbol(self, data: Dict, sectors_map: Dict, sector_metrics: Dict) -> Tuple[List, List]:
        """Process a single symbol and generate updates for both sheets."""
        symbol = data.get('symbol')
        row_idx = data.get('row')
        date_str = data.get('date')

        if not symbol or not isinstance(symbol, str):
            logger.error(f"Invalid symbol: {symbol}")
            return self._generate_empty_updates("INVALID", row_idx or 0)

        if not row_idx or not isinstance(row_idx, int):
            logger.error(f"Invalid row index for symbol {symbol}: {row_idx}")
            return self._generate_empty_updates(symbol, 0)

        if date_str:
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                logger.error(f"Invalid date format for symbol {symbol}: {date_str}")
                date_str = ""

        try:
            logger.info(f"Processing symbol {symbol} for date {date_str}")
            
            # Check cache for symbol data
            all_data = self.get_cached_symbol_data(symbol, date_str)
            
            if not all_data:
                # Fetch all data for this symbol asynchronously if not in cache
                all_data = await self.data_fetcher.fetch_all_data_for_symbol(symbol, date_str)
                if all_data:
                    self.cache_symbol_data(symbol, date_str, all_data)
            
            if not all_data:
                logger.warning(f"No data found for symbol {symbol}")
                return self._generate_empty_updates(symbol, row_idx)
            
            logger.debug(f"Data fetched for {symbol}: {all_data.keys()}")
            
            # Store historical data for ML and backtesting
            if 'price_history' in all_data:
                self.historical_data[symbol] = all_data['price_history']
            
            # Process the fetched data
            processed_data = self._process_symbol_data(symbol, all_data, sectors_map, sector_metrics)
            
            # Generate updates for temp and main sheets
            temp_updates = self._generate_temp_updates(symbol, row_idx, processed_data)
            main_updates = self._generate_main_updates(row_idx, processed_data)
            
            return temp_updates, main_updates
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {str(e)}")
            return self._generate_empty_updates(symbol, row_idx)

    def _process_symbol_data(self, symbol: str, all_data: Dict, sectors_map: Dict, sector_metrics: Dict) -> Dict:
        """Process the fetched data for a symbol."""
        processed_data = {}
        
        # Extract and process data
        processed_data['source'] = all_data.get('source', 'N/A')
        processed_data['date'] = all_data.get('date', 'N/A')
        processed_data['price_1w'] = all_data.get('price_1w', 'N/A')
        processed_data['price_1d'] = all_data.get('price_1d', 'N/A')
        processed_data['price'] = all_data.get('price', 0)
        processed_data['last_updated'] = all_data.get('last_updated', 'N/A')
        processed_data['price_on_date'] = all_data.get('price_on_date', 'N/A')
        
        # Process support and resistance
        processed_data['support'] = self._calculate_support(all_data)
        processed_data['resistance'] = self._calculate_resistance(all_data)
        
        # Process other metrics
        processed_data['rsi'] = self._calculate_rsi(all_data)
        processed_data['sharpe'] = all_data.get('sharpe', 0) or 0
        processed_data['squeeze'] = all_data.get('squeeze', False)
        processed_data['drawdown'] = all_data.get('drawdown', 0) or 0
        processed_data['beta'] = all_data.get('beta') or 1.0
        processed_data['ma_ratio'] = all_data.get('ma_ratio') or 1.0
        processed_data['momentum'] = self._calculate_momentum(all_data)
        processed_data['pe_ratio'] = all_data.get('pe_ratio') or 15
        processed_data['de_ratio'] = all_data.get('de_ratio') or 0.8
        
        # Get sector information
        processed_data['sector'] = sectors_map.get(symbol, "Unknown")
        processed_data['sector_pe'] = sector_metrics.get(processed_data['sector'], {}).get('avg_pe', 15)
        
        # Calculate score and verdict
        processed_data['score'] = self._calculate_score(processed_data)
        processed_data['verdict'] = self._generate_verdict(processed_data['score'])
        
        # Generate explanation
        processed_data['explanation'] = self._generate_explanation(processed_data)
        
        return processed_data

    def _calculate_rsi(self, all_data: Dict) -> float:
        if 'price_history' in all_data and all_data['price_history'] is not None:
            prices = all_data['price_history']['Close'].values
            if len(prices) > 14:
                return calculate_rsi(prices, 14)  # Using 14 as the default period for RSI
        return 50  # Neutral RSI as fallback

    def _generate_explanation(self, processed_data: Dict) -> str:
        """Generate a detailed explanation for the stock's rating."""
        verdict = processed_data['verdict']
        score = processed_data['score']
        
        explanation = f"The stock's composite score is {score:.2f}, resulting in a {verdict} verdict. "
        
        if verdict in ['Strong Buy', 'Buy']:
            explanation += "This stock is considered a buy because: "
        elif verdict == 'Hold':
            explanation += "This stock is not currently recommended as a buy or sell because: "
        else:
            explanation += "This stock is not recommended as a buy because: "
        
        factors = []
        if processed_data['rsi'] < 30:
            factors.append(f"The RSI ({processed_data['rsi']:.2f}) indicates the stock may be oversold")
        elif processed_data['rsi'] > 70:
            factors.append(f"The RSI ({processed_data['rsi']:.2f}) indicates the stock may be overbought")
        
        if processed_data['sharpe'] > 1:
            factors.append(f"The Sharpe Ratio ({processed_data['sharpe']:.2f}) suggests good risk-adjusted returns")
        elif processed_data['sharpe'] < 0:
            factors.append(f"The Sharpe Ratio ({processed_data['sharpe']:.2f}) suggests poor risk-adjusted returns")
        
        if processed_data['drawdown'] > 20:
            factors.append(f"The Max Drawdown ({processed_data['drawdown']:.2f}%) is relatively high, indicating increased risk")
        
        if processed_data['beta'] > 1.5:
            factors.append(f"The Beta ({processed_data['beta']:.2f}) suggests the stock is more volatile than the market")
        elif processed_data['beta'] < 0.5:
            factors.append(f"The Beta ({processed_data['beta']:.2f}) suggests the stock is less volatile than the market")
        
        if processed_data['ma_ratio'] > 1:
            factors.append(f"The MA50/200 Ratio ({processed_data['ma_ratio']:.2f}) indicates a bullish trend")
        elif processed_data['ma_ratio'] < 1:
            factors.append(f"The MA50/200 Ratio ({processed_data['ma_ratio']:.2f}) indicates a bearish trend")
        
        if processed_data['momentum'] > 5:
            factors.append(f"The 20d Momentum ({processed_data['momentum']:.2f}%) shows strong positive momentum")
        elif processed_data['momentum'] < -5:
            factors.append(f"The 20d Momentum ({processed_data['momentum']:.2f}%) shows strong negative momentum")
        
        if processed_data['pe_ratio'] > 25:
            factors.append(f"The P/E Ratio ({processed_data['pe_ratio']:.2f}) suggests the stock may be overvalued")
        elif processed_data['pe_ratio'] < 15:
            factors.append(f"The P/E Ratio ({processed_data['pe_ratio']:.2f}) suggests the stock may be undervalued")
        
        if processed_data['de_ratio'] > 2:
            factors.append(f"The D/E Ratio ({processed_data['de_ratio']:.2f}) indicates high leverage")
        elif processed_data['de_ratio'] < 0.5:
            factors.append(f"The D/E Ratio ({processed_data['de_ratio']:.2f}) indicates low leverage")
        
        explanation += " ".join(factors)
        
        explanation += "\n\nPlease note that this analysis is based on historical data and current market conditions. Always conduct your own research and consider your personal financial situation before making investment decisions."
        
        return explanation

    def _calculate_support(self, all_data: Dict) -> float:
        price = all_data.get('price', 0)
        if 'price_history' in all_data and all_data['price_history'] is not None:
            prices = all_data['price_history']['Close'].values
            if len(prices) > 20:
                return min(prices[-20:]) * 0.98  # 2% below 20-day min as support estimate
        return price * 0.95  # 5% below current price as fallback

    def _calculate_resistance(self, all_data: Dict) -> float:
        price = all_data.get('price', 0)
        if 'price_history' in all_data and all_data['price_history'] is not None:
            prices = all_data['price_history']['Close'].values
            if len(prices) > 20:
                return max(prices[-20:]) * 1.02  # 2% above 20-day max as resistance estimate
        return price * 1.05  # 5% above current price as fallback

    def _calculate_momentum(self, all_data: Dict) -> float:
        if 'price_history' in all_data and all_data['price_history'] is not None:
            prices = all_data['price_history']['Close'].values
            if len(prices) > 21:
                return calculate_momentum(prices, 20)
        return 0

    def _calculate_score(self, processed_data: Dict) -> float:
        return self.score_calculator.calculate_advanced_score(processed_data)

    def _generate_verdict(self, score: float) -> str:
        return self.score_calculator.get_verdict(score)

    def _generate_temp_updates(self, symbol: str, row_idx: int, processed_data: Dict) -> List[Dict]:
        today = datetime.now().strftime('%Y-%m-%d')
        squeeze_str = 'Yes' if processed_data['squeeze'] else 'No'
        
        return [{
            'range': f'A{row_idx}:Y{row_idx}',
            'values': [[
                symbol,
                processed_data.get('source', ''),
                processed_data.get('date', ''),
                processed_data.get('price_1w', ''),
                processed_data.get('price_1d', ''),
                processed_data['price'],
                processed_data.get('last_updated', ''),
                round(processed_data['support'], 2),
                round(processed_data['resistance'], 2),
                f"{round(processed_data['support'], 2)} - {round(processed_data['resistance'], 2)}",
                processed_data['price_on_date'],
                '',  # Price diff (calculated by formula)
                processed_data['verdict'],
                today,
                round(processed_data['rsi'], 2),
                round(processed_data['sharpe'], 3),
                squeeze_str,
                round(processed_data['drawdown'], 2),
                round(processed_data['beta'], 3),
                round(processed_data['score'], 3),
                round(processed_data['ma_ratio'], 3),
                round(processed_data['momentum'], 3),
                round(processed_data['pe_ratio'], 2),
                round(processed_data['de_ratio'], 2),
                processed_data.get('explanation', '')
            ]]
        }]

    def _generate_main_updates(self, row_idx: int, processed_data: Dict) -> List[Dict]:
        today = datetime.now().strftime('%Y-%m-%d')
        squeeze_str = 'Yes' if processed_data['squeeze'] else 'No'
        
        return [
            {'range': f'{idx_to_col(COLUMNS["price_now"])}{row_idx}', 'values': [[processed_data['price']]]},
            {'range': f'{idx_to_col(COLUMNS["last_updated"])}{row_idx}', 'values': [[processed_data.get('last_updated', '')]]},
            {'range': f'{idx_to_col(COLUMNS["support"])}{row_idx}', 'values': [[round(processed_data['support'], 2)]]},
            {'range': f'{idx_to_col(COLUMNS["resistance"])}{row_idx}', 'values': [[round(processed_data['resistance'], 2)]]},
            {'range': f'{idx_to_col(COLUMNS["support_resistance"])}{row_idx}', 'values': [[f"{round(processed_data['support'], 2)} - {round(processed_data['resistance'], 2)}"]]},
            {'range': f'{idx_to_col(COLUMNS["price_date"])}{row_idx}', 'values': [[processed_data['price_on_date']]]},
            {'range': f'{idx_to_col(COLUMNS["verdict"])}{row_idx}', 'values': [[processed_data['verdict']]]},
            {'range': f'{idx_to_col(COLUMNS["today"])}{row_idx}', 'values': [[today]]},
            {'range': f'{idx_to_col(COLUMNS["rsi"])}{row_idx}', 'values': [[round(processed_data['rsi'], 2)]]},
            {'range': f'{idx_to_col(COLUMNS["sharpe"])}{row_idx}', 'values': [[round(processed_data['sharpe'], 3)]]},
            {'range': f'{idx_to_col(COLUMNS["squeeze"])}{row_idx}', 'values': [[squeeze_str]]},
            {'range': f'{idx_to_col(COLUMNS["drawdown"])}{row_idx}', 'values': [[round(processed_data['drawdown'], 2)]]},
            {'range': f'{idx_to_col(COLUMNS["beta"])}{row_idx}', 'values': [[round(processed_data['beta'], 3)]]},
            {'range': f'{idx_to_col(COLUMNS["composite_score"])}{row_idx}', 'values': [[round(processed_data['score'], 3)]]},
            {'range': f'{idx_to_col(COLUMNS["ma_ratio"])}{row_idx}', 'values': [[round(processed_data['ma_ratio'], 3)]]},
            {'range': f'{idx_to_col(COLUMNS["momentum"])}{row_idx}', 'values': [[round(processed_data['momentum'], 3)]]},
            {'range': f'{idx_to_col(COLUMNS["pe"])}{row_idx}', 'values': [[round(processed_data['pe_ratio'], 2)]]},
            {'range': f'{idx_to_col(COLUMNS["de"])}{row_idx}', 'values': [[round(processed_data['de_ratio'], 2)]]},
            {'range': f'{idx_to_col(COLUMNS["explanation"])}{row_idx}', 'values': [[processed_data.get('explanation', '')]]}
        ]
                
    async def process_symbol(self, data: Dict, sectors_map: Dict, sector_metrics: Dict) -> Tuple[List, List]:
        """Process a single symbol and generate updates for both sheets."""
        symbol = data.get('symbol')
        row_idx = data.get('row')
        date_str = data.get('date')

        if not symbol or not isinstance(symbol, str):
            logger.error(f"Invalid symbol: {symbol}")
            return self._generate_empty_updates("INVALID", row_idx or 0)

        if not row_idx or not isinstance(row_idx, int):
            logger.error(f"Invalid row index for symbol {symbol}: {row_idx}")
            return self._generate_empty_updates(symbol, 0)

        if date_str:
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                logger.error(f"Invalid date format for symbol {symbol}: {date_str}")
                date_str = ""

        try:
            logger.info(f"Processing symbol {symbol} for date {date_str}")
            
            # Fetch all data for this symbol asynchronously
            all_data = await self.data_fetcher.fetch_all_data_for_symbol(symbol, date_str)
            
            if not all_data:
                logger.warning(f"No data found for symbol {symbol}")
                return self._generate_empty_updates(symbol, row_idx)
            
            logger.debug(f"Data fetched for {symbol}: {all_data.keys()}")
            
            # Process the fetched data
            processed_data = self._process_symbol_data(symbol, all_data, sectors_map, sector_metrics)
            
            # Generate updates for temp and main sheets
            temp_updates = self._generate_temp_updates(symbol, row_idx, processed_data)
            main_updates = self._generate_main_updates(row_idx, processed_data)
            
            return temp_updates, main_updates
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {str(e)}")
            return self._generate_empty_updates(symbol, row_idx)
    
    def _generate_empty_updates(self, symbol: str, row_idx: int) -> Tuple[List, List]:
        """Generate empty but non-blank updates for a symbol when data fetching fails."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Map column names to letters using idx_to_col and COLUMNS
        col_price_now = idx_to_col(COLUMNS['price_now'])
        col_today = idx_to_col(COLUMNS['today'])
        col_support = idx_to_col(COLUMNS['support'])
        col_resist = idx_to_col(COLUMNS['resistance'])
        col_support_resistance = idx_to_col(COLUMNS['support_resistance'])
        col_price_date = idx_to_col(COLUMNS['price_date'])
        col_composite_score = idx_to_col(COLUMNS['composite_score'])
        col_rsi = idx_to_col(COLUMNS['rsi'])
        col_sharpe = idx_to_col(COLUMNS['sharpe'])
        col_squeeze = idx_to_col(COLUMNS['squeeze'])
        col_drawdown = idx_to_col(COLUMNS['drawdown'])
        col_beta = idx_to_col(COLUMNS['beta'])
        col_ma_ratio = idx_to_col(COLUMNS['ma_ratio'])
        col_momentum = idx_to_col(COLUMNS['momentum'])
        col_pe = idx_to_col(COLUMNS['pe'])
        col_de = idx_to_col(COLUMNS['de'])
        col_verdict = idx_to_col(COLUMNS['verdict'])
        col_explanation = idx_to_col(COLUMNS['explanation'])
        
        # For temp sheet - provide fallback values for all columns
        temp_updates = [{
            'range': f'A{row_idx}:Y{row_idx}',
            'values': [[
                symbol,           # Symbol
                'N/A',            # Source
                '',               # Date
                '',               # Price in 1 week
                '',               # Price in 1 day
                'N/A',            # Price now
                '',               # Last updated
                0,                # Support
                0,                # Resistance
                '0 - 0',          # Support & Resistance
                'N/A',            # Price in date pulled
                '',               # Price diff
                'Hold',           # Verdict
                today,            # Today's date
                50,               # RSI - neutral
                0,                # Sharpe
                'No',             # Squeeze
                0,                # Drawdown
                1.0,              # Beta - market average
                0.5,              # Composite Score - neutral
                1.0,              # MA ratio - neutral
                0,                # Momentum
                15,               # P/E - average
                0.8,              # D/E - moderate
                'Insufficient data to generate explanation'  # Explanation
            ]]
        }]
        
        # For main sheet - provide fallback values for all columns
        main_updates = [
            {'range': f'{col_price_now}{row_idx}', 'values': [['N/A']]},
            {'range': f'{col_today}{row_idx}', 'values': [[today]]},
            {'range': f'{col_support}{row_idx}', 'values': [[0]]},
            {'range': f'{col_resist}{row_idx}', 'values': [[0]]},
            {'range': f'{col_support_resistance}{row_idx}', 'values': [['0 - 0']]},
            {'range': f'{col_price_date}{row_idx}', 'values': [['N/A']]},
            {'range': f'{col_composite_score}{row_idx}', 'values': [[0.5]]},
            {'range': f'{col_rsi}{row_idx}', 'values': [[50]]},
            {'range': f'{col_sharpe}{row_idx}', 'values': [[0]]},
            {'range': f'{col_squeeze}{row_idx}', 'values': [['No']]},
            {'range': f'{col_drawdown}{row_idx}', 'values': [[0]]},
            {'range': f'{col_beta}{row_idx}', 'values': [[1.0]]},
            {'range': f'{col_ma_ratio}{row_idx}', 'values': [[1.0]]},
            {'range': f'{col_momentum}{row_idx}', 'values': [[0]]},
            {'range': f'{col_pe}{row_idx}', 'values': [[15]]},
            {'range': f'{col_de}{row_idx}', 'values': [[0.8]]},
            {'range': f'{col_verdict}{row_idx}', 'values': [['Hold']]},
            {'range': f'{col_explanation}{row_idx}', 'values': [['Insufficient data to generate explanation']]}
        ]
        
        return temp_updates, main_updates
    
    def apply_formatting(self, worksheet, total_rows):
        """Apply cell formatting to highlight important columns."""
        try:
            # Map column names to letters using idx_to_col and COLUMNS
            col_support = idx_to_col(COLUMNS['support'])
            col_resist = idx_to_col(COLUMNS['resistance'])
            col_support_resistance = idx_to_col(COLUMNS['support_resistance'])
            col_composite_score = idx_to_col(COLUMNS['composite_score'])
            col_verdict = idx_to_col(COLUMNS['verdict'])
            col_rsi = idx_to_col(COLUMNS['rsi'])
            
            # Highlight Support and Resistance columns
            self.sheets_handler.format_cells(
                worksheet,
                f"{col_support}2:{col_support}{total_rows}",
                CELL_FORMATS['support_resistance']
            )
            self.sheets_handler.format_cells(
                worksheet,
                f"{col_resist}2:{col_resist}{total_rows}",
                CELL_FORMATS['support_resistance']
            )
            self.sheets_handler.format_cells(
                worksheet,
                f"{col_support_resistance}2:{col_support_resistance}{total_rows}",
                CELL_FORMATS['support_resistance']
            )
            
            # Highlight Composite Score and Verdict columns
            self.sheets_handler.format_cells(
                worksheet,
                f"{col_composite_score}2:{col_composite_score}{total_rows}",
                CELL_FORMATS['score_verdict']
            )
            self.sheets_handler.format_cells(
                worksheet,
                f"{col_verdict}2:{col_verdict}{total_rows}",
                CELL_FORMATS['score_verdict']
            )
            
            # Highlight RSI column
            self.sheets_handler.format_cells(
                worksheet,
                f"{col_rsi}2:{col_rsi}{total_rows}",
                CELL_FORMATS['rsi']
            )
            
            logger.info("Applied cell formatting")
        except Exception as e:
            logger.error(f"Error applying cell formatting: {str(e)}")

    def create_summary_tab(self):
        """Create and populate the summary tab."""
        try:
            summary_sheet = self.sheets_handler.get_or_create_worksheet("Summary")
            summary_sheet.clear()

            total_stocks = len(self.symbol_data_cache)
            strong_buys = []
            buys = []
            holds = []
            sells = []
            strong_sells = []

            for symbol, data in self.symbol_data_cache.items():
                verdict = data.get('verdict', '')
                if verdict == 'Strong Buy':
                    strong_buys.append(symbol)
                elif verdict == 'Buy':
                    buys.append(symbol)
                elif verdict == 'Hold':
                    holds.append(symbol)
                elif verdict == 'Sell':
                    sells.append(symbol)
                elif verdict == 'Strong Sell':
                    strong_sells.append(symbol)

            summary_content = [
                ["Stock Spike Replicator - Summary"],
                [""],
                [f"Total stocks analyzed: {total_stocks}"],
                [f"Strong Buys: {len(strong_buys)}"],
                [f"Buys: {len(buys)}"],
                [f"Holds: {len(holds)}"],
                [f"Sells: {len(sells)}"],
                [f"Strong Sells: {len(strong_sells)}"],
                [""],
                ["Market Overview:"],
                [self._generate_market_overview(total_stocks, strong_buys, buys, holds, sells, strong_sells)],
                [""],
                ["Top Stock Picks:"],
                ["Strong Buys:"] + (strong_buys[:5] if strong_buys else ["None"]),
                ["Buys:"] + (buys[:5] if buys else ["None"]),
                [""],
                ["Example Stock Analysis:"],
            ]

            # Add example stock analysis
            example_stocks = strong_buys[:2] + buys[:2] + holds[:1]
            for symbol in example_stocks[:5]:  # Analyze up to 5 stocks
                analysis = self._generate_stock_analysis(symbol)
                summary_content.extend(analysis)

            # Add the rest of the summary content
            summary_content.extend([
                [""],
                ["Variables Used and Their Meanings:"],
                ["RSI (Relative Strength Index)", "Measures the magnitude of recent price changes to evaluate overbought or oversold conditions."],
                ["Sharpe Ratio", "Measures the risk-adjusted performance of the stock."],
                ["Max Drawdown", "The largest peak-to-trough decline in the stock's value."],
                ["Beta", "Measures the stock's volatility in relation to the overall market."],
                ["MA50/200 Ratio", "Ratio of 50-day moving average to 200-day moving average, indicates trend direction."],
                ["20d Momentum", "Measures the rate of change in price over the last 20 days."],
                ["P/E Ratio", "Price-to-Earnings ratio, a valuation metric."],
                ["D/E Ratio", "Debt-to-Equity ratio, measures a company's financial leverage."],
                [""],
                ["Scoring Weights:"],
                ["Simple Score Weights:", f"{SIMPLE_SCORE_WEIGHTS}"],
                ["Advanced Score Weights:", f"{ADVANCED_SCORE_WEIGHTS}"],
                [""],
                ["Verdict Thresholds:"],
                ["Strong Buy", "Composite Score > 0.8 or ML prediction > 5%"],
                ["Buy", "Composite Score > 0.6 or ML prediction > 2%"],
                ["Hold", "0.4 <= Composite Score <= 0.6 or -2% <= ML prediction <= 2%"],
                ["Sell", "Composite Score < 0.4 or ML prediction < -2%"],
                ["Strong Sell", "Composite Score < 0.2 or ML prediction < -5%"],
                [""],
                ["To adjust these values, modify the SIMPLE_SCORE_WEIGHTS and ADVANCED_SCORE_WEIGHTS dictionaries in the config.py file."],
                ["The verdict thresholds can be adjusted in the ScoreCalculator class in the scoring.py file."],
                [""],
                ["Note: This tool provides analysis based on historical data and current market conditions. Always conduct your own research and consider your personal financial situation before making investment decisions."],
            ])

            self.sheets_handler.batch_update(summary_sheet, [{'range': 'A1', 'values': summary_content}])
            logger.info("Created and populated summary tab")
        except Exception as e:
            logger.error(f"Error creating summary tab: {str(e)}")

    def _generate_market_overview(self, total_stocks, strong_buys, buys, holds, sells, strong_sells):
        """Generate a market overview based on the analyzed stocks."""
        if not total_stocks:
            return "No stocks were analyzed. Please check your input data."

        strong_buy_percent = len(strong_buys) / total_stocks * 100
        buy_percent = len(buys) / total_stocks * 100
        
        if strong_buy_percent + buy_percent > 50:
            return f"The market appears bullish with {strong_buy_percent:.1f}% Strong Buys and {buy_percent:.1f}% Buys. Consider this a potentially good time for strategic investments."
        elif strong_buy_percent + buy_percent > 30:
            return f"The market shows mixed signals with {strong_buy_percent:.1f}% Strong Buys and {buy_percent:.1f}% Buys. Some good opportunities may be available, but caution is advised."
        else:
            return f"The market appears bearish with only {strong_buy_percent:.1f}% Strong Buys and {buy_percent:.1f}% Buys. It might be wise to be cautious with new investments and focus on capital preservation."

    def _generate_stock_analysis(self, symbol):
        """Generate a detailed analysis for a given stock."""
        data = self.symbol_data_cache.get(symbol, {})
        if not data:
            return [[f"No data available for {symbol}"]]

        analysis = [
            [f"Analysis for {symbol}:"],
            [f"Verdict: {data.get('verdict', 'N/A')}"],
            [f"Composite Score: {data.get('score', 'N/A')}"],
            [f"Current Price: ${data.get('price', 'N/A')}"],
            [f"Support: ${data.get('support', 'N/A')}"],
            [f"Resistance: ${data.get('resistance', 'N/A')}"],
            [f"RSI: {data.get('rsi', 'N/A')}"],
            [f"Sharpe Ratio: {data.get('sharpe', 'N/A')}"],
            [f"Beta: {data.get('beta', 'N/A')}"],
            [f"Explanation: {data.get('explanation', 'N/A')}"],
            [""]
        ]
        return analysis

    async def scan_additional_stocks(self):
        """Scan for additional stocks under $1 on NYSE or NASDAQ."""
        try:
            # This is a placeholder. In a real implementation, you would:
            # 1. Use an API or web scraping to get a list of stocks under $1 on NYSE or NASDAQ
            # 2. Fetch data for these stocks
            # 3. Analyze them using the same methods as the main stock list
            # 4. Add the best performers to the symbol_data_cache

            # For demonstration, we'll add some mock data
            mock_stocks = ['MOCK1', 'MOCK2', 'MOCK3', 'MOCK4', 'MOCK5']
            for symbol in mock_stocks:
                mock_data = await self.data_fetcher.fetch_all_data_for_symbol(symbol)
                processed_data = self._process_symbol_data(symbol, mock_data)
                self.symbol_data_cache[symbol] = processed_data

            logger.info(f"Scanned and added {len(mock_stocks)} additional stocks")
        except Exception as e:
            logger.error(f"Error scanning additional stocks: {str(e)}")

    def plot_top_pick(self):
        """
        Plot the top pick with support and resistance levels.
        
        This method creates a visualization of the top-performing stock (based on the highest score)
        using the plot_top_stock_pick function from the visualizations module. The plot includes
        historical price data, current price, and support/resistance levels. The resulting image
        is inserted into the Summary sheet of the Google Sheets document.
        """
        try:
            # Find the top pick (highest score)
            top_pick = max(self.symbol_data_cache.items(), key=lambda x: x[1].get('score', 0))
            symbol, data = top_pick

            # Fetch historical price data for the symbol
            historical_data = self.historical_data.get(symbol)
            if historical_data is None:
                logger.error(f"No historical data found for top pick {symbol}")
                return

            # Create the plot
            plot_buffer = plot_top_stock_pick(
                symbol,
                historical_data,
                data['price'],
                data['support'],
                data['resistance']
            )

            # Add the plot to the summary sheet
            summary_sheet = self.sheets_handler.get_worksheet("Summary")
            self.sheets_handler.insert_image(summary_sheet, 'A100', plot_buffer)

            # Add a text description
            plot_description = [
                [f"Top Pick: {symbol}"],
                [f"Current Price: ${data.get('price', 'N/A')}"],
                [f"Support: ${data.get('support', 'N/A')}"],
                [f"Resistance: ${data.get('resistance', 'N/A')}"],
                [f"Reason: {data.get('explanation', 'N/A')}"],
            ]
            self.sheets_handler.batch_update(summary_sheet, [{'range': 'A106', 'values': plot_description}])

            logger.info(f"Added top pick plot and description for {symbol}")
        except Exception as e:
            logger.error(f"Error plotting top pick: {str(e)}")

async def main():
    """Main entry point for the application."""
    try:
        app = StockUpdateApp()
        success = await app.run()
        
        if success:
            app.create_summary_tab()
            app.plot_top_pick()
            print("✅ Update completed successfully")
        else:
            print("❌ Update failed")
            
        return success
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {str(e)}", exc_info=True)
        print(f"❌ Critical error: {str(e)}")
        return False


if __name__ == '__main__':
    try:
        # Run the async main function
        if sys.version_info >= (3, 7):
            asyncio.run(main())
        else:
            # For Python 3.6
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n🛑 Update interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up any remaining threads
        import threading
        for thread in threading.enumerate():
            if thread is not threading.main_thread():
                try:
                    thread.join(timeout=1.0)
                except Exception as e:
                    logger.warning(f"Error joining thread {thread.name}: {str(e)}")