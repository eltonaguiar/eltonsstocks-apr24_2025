import unittest
import asyncio
from unittest.mock import patch, AsyncMock
from data_fetchers import DataFetcher

class TestDataFetcher(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.data_fetcher = DataFetcher()

    @patch('data_fetchers.aiohttp.ClientSession')
    async def test_get_all_symbols(self, mock_session):
        mock_response = AsyncMock()
        mock_response.json.return_value = [
            {'symbol': 'AAPL'},
            {'symbol': 'GOOGL'},
            {'symbol': 'MSFT'}
        ]
        mock_session.return_value.__aenter__.return_value.get.return_value = mock_response

        symbols = await self.data_fetcher.get_all_symbols()
        
        self.assertEqual(symbols, ['AAPL', 'GOOGL', 'MSFT'])
        self.assertEqual(self.data_fetcher.price_cache['all_symbols'], symbols)

    @patch('data_fetchers.aiohttp.ClientSession')
    async def test_fetch_stock_data(self, mock_session):
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            'symbol': 'AAPL',
            'price': 0.5,
            'exchange': 'NASDAQ',
            'name': 'Apple Inc.',
            'volume': 1000000
        }
        mock_session.return_value.__aenter__.return_value.get.return_value = mock_response

        stock_data = await self.data_fetcher.fetch_stock_data(AsyncMock(), 'AAPL')
        
        self.assertEqual(stock_data['symbol'], 'AAPL')
        self.assertEqual(stock_data['price'], 0.5)
        self.assertEqual(stock_data['exchange'], 'NASDAQ')
        self.assertEqual(stock_data['name'], 'Apple Inc.')
        self.assertEqual(stock_data['volume'], 1000000)

    @patch('data_fetchers.DataFetcher.get_all_symbols')
    @patch('data_fetchers.DataFetcher.fetch_stock_data')
    async def test_scan_stocks_under_one_dollar(self, mock_fetch_stock_data, mock_get_all_symbols):
        mock_get_all_symbols.return_value = ['AAPL', 'GOOGL', 'MSFT']
        mock_fetch_stock_data.side_effect = [
            {'symbol': 'AAPL', 'price': 0.5, 'volume': 1000000},
            {'symbol': 'GOOGL', 'price': 1.5, 'volume': 500000},
            {'symbol': 'MSFT', 'price': 0.8, 'volume': 750000}
        ]

        result = await self.data_fetcher.scan_stocks_under_one_dollar()
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['symbol'], 'AAPL')
        self.assertEqual(result[1]['symbol'], 'MSFT')
        self.assertEqual(result[0]['volume'], 1000000)
        self.assertEqual(result[1]['volume'], 750000)

    @patch('data_fetchers.aiohttp.ClientSession')
    async def test_fetch_price_history(self, mock_session):
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            'values': [
                {'datetime': '2023-04-23', 'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.05, 'volume': 1000000},
                {'datetime': '2023-04-22', 'open': 0.95, 'high': 1.0, 'low': 0.9, 'close': 0.98, 'volume': 900000}
            ]
        }
        mock_session.return_value.__aenter__.return_value.get.return_value = mock_response

        price_history = await self.data_fetcher.fetch_price_history('AAPL')
        
        self.assertEqual(len(price_history), 2)
        self.assertEqual(price_history.iloc[0]['Close'], 1.05)
        self.assertEqual(price_history.iloc[1]['Close'], 0.98)

if __name__ == '__main__':
    unittest.main()