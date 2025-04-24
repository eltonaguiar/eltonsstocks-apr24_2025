import unittest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import pandas as pd
import numpy as np
from data_fetchers import DataFetcher
from ml_backtesting import MLBacktesting
from scoring import ScoreCalculator
from main import fetch_and_process_stocks

class TestIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sample_symbols = ['AAPL', 'GOOGL', 'MSFT']
        self.sample_historical_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=100),
            'Open': np.random.rand(100) * 100 + 50,
            'High': np.random.rand(100) * 100 + 60,
            'Low': np.random.rand(100) * 100 + 40,
            'Close': np.random.rand(100) * 100 + 50,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }).set_index('Date')

    @patch('data_fetchers.DataFetcher.fetch_price_history')
    @patch('data_fetchers.DataFetcher.fetch_all_data_for_symbol')
    @patch('ml_backtesting.run_ml_backtesting')
    @patch('scoring.run_scoring')
    async def test_fetch_and_process_stocks(self, mock_run_scoring, mock_run_ml_backtesting, 
                                            mock_fetch_all_data, mock_fetch_price_history):
        # Mock the fetch_price_history method
        mock_fetch_price_history.return_value = self.sample_historical_data

        # Mock the fetch_all_data_for_symbol method
        mock_fetch_all_data.return_value = {
            'pe_ratio': 20,
            'sector_pe': 25,
            'pb_ratio': 3,
            'revenue_growth': 0.1,
            'earnings_growth': 0.15,
            'sentiment': 0.7,
            'volatility': 0.2,
            'beta': 1.1
        }

        # Mock the run_ml_backtesting function
        mock_run_ml_backtesting.return_value = {
            'prediction': np.random.rand(100) - 0.5,
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.1
        }

        # Mock the run_scoring function
        mock_run_scoring.return_value = {
            'score': 0.75,
            'verdict': 'Buy'
        }

        # Run the fetch_and_process_stocks function
        results = await fetch_and_process_stocks(self.sample_symbols)

        # Assertions
        self.assertEqual(len(results), len(self.sample_symbols))
        for result in results:
            self.assertIn(result['symbol'], self.sample_symbols)
            self.assertIsInstance(result['ml_prediction'], float)
            self.assertIsInstance(result['score'], float)
            self.assertIn(result['verdict'], ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'])
            self.assertIsInstance(result['total_return'], float)
            self.assertIsInstance(result['sharpe_ratio'], float)
            self.assertIsInstance(result['max_drawdown'], float)

        # Check if all mocked functions were called
        self.assertEqual(mock_fetch_price_history.call_count, len(self.sample_symbols))
        self.assertEqual(mock_fetch_all_data.call_count, len(self.sample_symbols))
        self.assertEqual(mock_run_ml_backtesting.call_count, len(self.sample_symbols))
        self.assertEqual(mock_run_scoring.call_count, len(self.sample_symbols))

    @patch('data_fetchers.DataFetcher.scan_stocks_under_one_dollar')
    @patch('main.fetch_and_process_stocks')
    async def test_main_function(self, mock_fetch_and_process, mock_scan_stocks):
        from main import main

        # Mock the scan_stocks_under_one_dollar method
        mock_scan_stocks.return_value = [
            {'symbol': 'AAPL', 'price': 0.5, 'volume': 1000000},
            {'symbol': 'GOOGL', 'price': 0.7, 'volume': 800000},
            {'symbol': 'MSFT', 'price': 0.9, 'volume': 1200000}
        ]

        # Mock the fetch_and_process_stocks function
        mock_fetch_and_process.return_value = [
            {'symbol': 'AAPL', 'score': 0.8, 'verdict': 'Buy', 'ml_prediction': 0.6, 'total_return': 0.2, 'sharpe_ratio': 1.5, 'max_drawdown': -0.1},
            {'symbol': 'GOOGL', 'score': 0.7, 'verdict': 'Hold', 'ml_prediction': 0.3, 'total_return': 0.15, 'sharpe_ratio': 1.2, 'max_drawdown': -0.15},
            {'symbol': 'MSFT', 'score': 0.9, 'verdict': 'Strong Buy', 'ml_prediction': 0.8, 'total_return': 0.25, 'sharpe_ratio': 1.8, 'max_drawdown': -0.05}
        ]

        # Run the main function
        await main()

        # Check if the mocked functions were called
        mock_scan_stocks.assert_called_once()
        mock_fetch_and_process.assert_called_once()

        # Check if the correct number of symbols were processed
        called_symbols = mock_fetch_and_process.call_args[0][0]
        self.assertEqual(len(called_symbols), 3)

if __name__ == '__main__':
    unittest.main()