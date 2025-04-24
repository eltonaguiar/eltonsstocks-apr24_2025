import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from functools import lru_cache
from aiolimiter import AsyncLimiter
from config import API_ENDPOINTS, API_KEYS, RATE_LIMITS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_fetchers')

POSITIVE_WORDS = ['up', 'rise', 'gain', 'positive', 'growth', 'profit', 'success', 'bullish', 'optimistic']
NEGATIVE_WORDS = ['down', 'fall', 'loss', 'negative', 'decline', 'bearish', 'pessimistic', 'risk', 'concern']

class DataFetcher:
    def __init__(self):
        self.price_cache = {}
        self.api_endpoints = API_ENDPOINTS
        self.api_keys = API_KEYS
        self.rate_limits = RATE_LIMITS
        
        # Create per-API rate limiters
        self.rate_limiters = {
            api: AsyncLimiter(rate_limit, 60)  # requests per minute
            for api, rate_limit in self.rate_limits.items()
        }
        
        # Create per-symbol API state tracking
        self.symbol_api_state = {}
        
        self.retry_delay = 60  # seconds to wait before retrying after hitting rate limit
        self.max_retries = 5
        self.request_delay = 1  # seconds to wait between requests
        self.api_rotation = list(self.api_endpoints.keys())
        self.current_api_index = 0

        logger.info(f"DataFetcher initialized with APIs: {', '.join(self.api_rotation)}")

    def _get_next_api_for_symbol(self, symbol: str) -> str:
        """Get the next API to try for a specific symbol."""
        if symbol not in self.symbol_api_state:
            self.symbol_api_state[symbol] = {
                'current_index': 0,
                'rate_limited_apis': set(),
                'failed_apis': set()
            }
        
        state = self.symbol_api_state[symbol]
        
        # Try to find an API that's not rate limited or failed
        for _ in range(len(self.api_rotation)):
            api = self.api_rotation[state['current_index']]
            state['current_index'] = (state['current_index'] + 1) % len(self.api_rotation)
            
            if api not in state['rate_limited_apis'] and api not in state['failed_apis']:
                return api
        
        # If all APIs are rate limited or failed, try the least recently rate limited one
        if state['rate_limited_apis']:
            return next(iter(state['rate_limited_apis']))
        
        # If no APIs are rate limited but all failed, reset failed APIs and try again
        state['failed_apis'] = set()
        return self.api_rotation[0]

    def _mark_api_rate_limited(self, symbol: str, api: str):
        """Mark an API as rate limited for a specific symbol."""
        if symbol in self.symbol_api_state:
            self.symbol_api_state[symbol]['rate_limited_apis'].add(api)
            
            # Schedule removal of rate limit after delay
            asyncio.create_task(self._clear_rate_limit_after_delay(symbol, api))

    def _mark_api_failed(self, symbol: str, api: str):
        """Mark an API as failed for a specific symbol."""
        if symbol in self.symbol_api_state:
            self.symbol_api_state[symbol]['failed_apis'].add(api)

    async def _clear_rate_limit_after_delay(self, symbol: str, api: str):
        """Clear rate limit status after delay."""
        await asyncio.sleep(self.retry_delay)
        if symbol in self.symbol_api_state and api in self.symbol_api_state[symbol]['rate_limited_apis']:
            self.symbol_api_state[symbol]['rate_limited_apis'].remove(api)
            logger.debug(f"Cleared rate limit for {api} for symbol {symbol}")

    async def fetch_with_retry(self, api_func: Callable, symbol: str, max_retries: int = 3, **kwargs):
        """Fetch with automatic retry logic."""
        for attempt in range(max_retries):
            try:
                return await api_func(symbol, **kwargs)
            except aiohttp.ClientResponseError as e:
                if e.status == 429 and attempt < max_retries - 1:  # Rate limit hit
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Retrying {symbol} after {wait_time}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Error for {symbol}, retrying after {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    raise

    async def _fetch_from_api(self, symbol: str, api: str) -> Optional[Dict[str, Any]]:
        """Fetch historical data from a specific API."""
        try:
            url = f"{self.api_endpoints[api]['base_url']}/time_series"
            params = {
                "symbol": symbol,
                "interval": "1day",
                "outputsize": "5000",
                "apikey": self.api_keys[api]
            }
            
            async with self.rate_limiters[api]:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        response.raise_for_status()
                        data = await response.json()
                        
                        if isinstance(data, dict) and 'values' in data:
                            await asyncio.sleep(self.request_delay)  # Add delay between requests
                            return {
                                'symbol': symbol,
                                'historical_data': data['values']
                            }
                        else:
                            logger.error(f"Unexpected data structure for {symbol} from {api}: {data}")
                            return None
        except aiohttp.ClientResponseError as e:
            if e.status == 429:  # Too Many Requests
                self._mark_api_rate_limited(symbol, api)
                logger.warning(f"Rate limit hit for {symbol} on {api}. API marked as rate limited.")
            else:
                self._mark_api_failed(symbol, api)
                logger.error(f"Error fetching historical data for {symbol} from {api}: {str(e)}")
            raise
        except Exception as e:
            self._mark_api_failed(symbol, api)
            logger.error(f"Unexpected error fetching historical data for {symbol} from {api}: {str(e)}")
            raise

    async def fetch_historical_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch historical data for a given stock symbol using multiple APIs with smarter fallback.
        
        Args:
            symbol (str): Stock symbol to fetch data for.
        
        Returns:
            Dict[str, Any]: Historical data for the stock.
        """
        logger.info(f"Fetching historical data for {symbol}")
        
        # Try primary API first
        primary_api = self.api_rotation[0]
        try:
            result = await self._fetch_from_api(symbol, primary_api)
            if result:
                logger.info(f"Successfully fetched data for {symbol} from primary API {primary_api}")
                return result
        except Exception:
            logger.warning(f"Primary API {primary_api} failed for {symbol}, trying secondary APIs")
        
        # Try secondary APIs in sequence, with independent rate limiting
        for _ in range(len(self.api_rotation) * self.max_retries):
            api = self._get_next_api_for_symbol(symbol)
            
            try:
                if api in self.symbol_api_state[symbol]['rate_limited_apis']:
                    logger.debug(f"Skipping {api} for {symbol} due to rate limits")
                    continue
                    
                logger.info(f"Trying API {api} for {symbol}")
                result = await self._fetch_from_api(symbol, api)
                if result:
                    logger.info(f"Successfully fetched data for {symbol} from API {api}")
                    return result
            except aiohttp.ClientResponseError as e:
                if e.status == 429:  # Already marked as rate limited in _fetch_from_api
                    continue
            except Exception:
                continue

        logger.error(f"Failed to fetch historical data for {symbol} from all APIs")
        return {
            'symbol': symbol,
            'historical_data': []
        }

    @lru_cache(maxsize=1000)
    async def get_all_symbols(self) -> List[str]:
        """
        Fetch all stock symbols from NYSE and NASDAQ exchanges.
        
        Returns:
            List[str]: List of stock symbols
        """
        cache_key = "all_symbols"
        cached_symbols = self.price_cache.get(cache_key)
        
        if cached_symbols:
            return cached_symbols

        for api in self.api_rotation:
            try:
                url = f"{self.api_endpoints[api]['base_url']}/stocks"
                params = {
                    "exchange": "NYSE,NASDAQ",
                    "country": "United States",
                    "apikey": self.api_keys[api]
                }
                async with self.rate_limiters[api]:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params) as response:
                            response.raise_for_status()
                            data = await response.json()
                            symbols = [item['symbol'] for item in data.get('data', [])]
                            self.price_cache[cache_key] = symbols
                            return symbols
            except aiohttp.ClientError as e:
                logger.error(f"Error fetching symbols from {api}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error fetching symbols from {api}: {str(e)}")
        
        return []

    async def fetch_stock_data(self, session: aiohttp.ClientSession, symbol: str) -> Optional[Dict[str, Any]]:
        cache_key = f"stock_data_{symbol}"
        cached_data = self.price_cache.get(cache_key)
        
        if cached_data and (datetime.now() - cached_data['timestamp']).total_seconds() < 3600:
            return cached_data['data']

        for api in self.api_rotation:
            try:
                url = f"{self.api_endpoints[api]['base_url']}/quote"
                params = {
                    "symbol": symbol,
                    "apikey": self.api_keys[api]
                }
                async with self.rate_limiters[api]:
                    async with session.get(url, params=params) as response:
                        response.raise_for_status()
                        data = await response.json()
                        
                        if 'price' in data and isinstance(data['price'], (int, float)) and data['price'] < 1 and data['exchange'] in ['NYSE', 'NASDAQ']:
                            stock_data = {
                                'symbol': symbol,
                                'price': data['price'],
                                'exchange': data['exchange'],
                                'name': data.get('name', ''),
                                'volume': data.get('volume', 0)
                            }
                            self.price_cache[cache_key] = {
                                'data': stock_data,
                                'timestamp': datetime.now()
                            }
                            return stock_data
            except aiohttp.ClientError as e:
                logger.error(f"Error fetching data for {symbol} from {api}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error for {symbol} from {api}: {str(e)}")
        
        return None

    async def scan_stocks_under_one_dollar(self) -> List[Dict[str, Any]]:
        """
        Scan for stocks under $1 on NYSE and NASDAQ exchanges.
        
        Returns:
            List[Dict[str, Any]]: List of stocks under $1 with their details
        """
        async def fetch_all_stocks(symbols: List[str]) -> List[Dict[str, Any]]:
            async with aiohttp.ClientSession() as session:
                tasks = []
                for symbol in symbols:
                    task = asyncio.create_task(self.fetch_stock_data(session, symbol))
                    tasks.append(task)
                    # Add small delay between task creation to avoid overwhelming APIs
                    if len(tasks) % 10 == 0:
                        await asyncio.sleep(0.1)
                
                results = []
                for task in asyncio.as_completed(tasks):
                    result = await task
                    if result:
                        results.append(result)
                
                return results

        try:
            logger.info("Starting to scan stocks under $1")
            all_symbols = await self.get_all_symbols()
            logger.info(f"Retrieved {len(all_symbols)} symbols")
            
            stocks_under_one_dollar = await fetch_all_stocks(all_symbols)
            logger.info(f"Found {len(stocks_under_one_dollar)} stocks under $1")
            
            # Sort by volume in descending order
            stocks_under_one_dollar.sort(key=lambda x: x['volume'], reverse=True)
            
            return stocks_under_one_dollar
        except Exception as e:
            logger.error(f"Error in scan_stocks_under_one_dollar: {str(e)}")
            return []

    async def fetch_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch news headlines for a given stock symbol and perform sentiment analysis.
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            Dict[str, Any]: Dictionary containing sentiment score and headlines
        """
        for api in self.api_rotation:
            try:
                url = f"{self.api_endpoints[api]['base_url']}/company-news"
                params = {
                    "symbol": symbol,
                    "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                    "to": datetime.now().strftime("%Y-%m-%d"),
                    "apikey": self.api_keys[api]
                }
                async with self.rate_limiters[api]:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params) as response:
                            response.raise_for_status()
                            news_data = await response.json()
                
                headlines = [item['headline'] for item in news_data[:10]]  # Get the 10 most recent headlines
                sentiment_score = self.analyze_sentiment(headlines)
                
                return {
                    'sentiment_score': sentiment_score,
                    'headlines': headlines
                }
            except aiohttp.ClientError as e:
                logger.error(f"Error fetching news for {symbol} from {api}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error fetching news for {symbol} from {api}: {str(e)}")
        
        return {'sentiment_score': 0, 'headlines': []}

    def analyze_sentiment(self, headlines: List[str]) -> float:
        """
        Perform simple sentiment analysis on headlines.
        
        Args:
            headlines (List[str]): List of news headlines
        
        Returns:
            float: Sentiment score between -1 (very negative) and 1 (very positive)
        """
        if not headlines:
            return 0
            
        total_score = 0
        for headline in headlines:
            words = headline.lower().split()
            positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
            negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)
            total_score += positive_count - negative_count
        
        return total_score / len(headlines)

# Additional methods (fetch_price_history, fetch_fundamentals, etc.) remain unchanged