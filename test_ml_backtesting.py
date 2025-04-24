import unittest
import pandas as pd
import numpy as np
from ml_backtesting import EnhancedBacktester

class TestEnhancedBacktester(unittest.TestCase):
    def setUp(self):
        self.backtester = EnhancedBacktester()
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', periods=100),
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
        self.sample_data.set_index('Date', inplace=True)

    def test_initialization(self):
        self.assertIsInstance(self.backtester, EnhancedBacktester)
        self.assertIsNotNone(self.backtester.models)

    def test_prepare_data(self):
        prepared_data = self.backtester.prepare_data(self.sample_data)
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns', 'volatility', 
                            'ma_5', 'ma_20', 'rsi', 'macd', 'signal', 'bollinger_upper', 'bollinger_lower', 
                            'atr', 'obv', 'cci', 'stoch_k', 'stoch_d', 'target']
        self.assertTrue(all(col in prepared_data.columns for col in expected_columns))
        self.assertGreater(len(prepared_data), 0)

    def test_calculate_position_size(self):
        volatility = 0.02
        risk_per_trade = 0.01
        position_size = self.backtester.calculate_position_size(volatility, risk_per_trade)
        self.assertEqual(position_size, 0.5)

    def test_detect_market_regime(self):
        regimes = self.backtester.detect_market_regime(self.sample_data)
        self.assertEqual(len(regimes), len(self.sample_data))
        self.assertTrue(all(regime in ['high_volatility', 'uptrend', 'downtrend', 'neutral'] for regime in regimes))

    def test_backtest(self):
        results = self.backtester.backtest(self.sample_data)
        expected_keys = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown', 
                         'win_rate', 'avg_win', 'avg_loss', 'cumulative_returns']
        self.assertTrue(all(key in results for key in expected_keys))
        self.assertIsInstance(results['total_return'], float)
        self.assertIsInstance(results['sharpe_ratio'], float)
        self.assertIsInstance(results['max_drawdown'], float)
        self.assertIsInstance(results['cumulative_returns'], pd.Series)

if __name__ == '__main__':
    unittest.main()