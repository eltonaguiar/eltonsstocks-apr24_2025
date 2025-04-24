import unittest
from unittest.mock import patch
from scoring import ScoreCalculator, run_scoring

class TestScoreCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = ScoreCalculator()
        self.sample_metrics = {
            'ml_prediction': 0.6,
            'rsi': 55,
            'macd': 0.5,
            'macd_signal': 0.3,
            'price': 10,
            'bb_lower': 9,
            'bb_upper': 11,
            'ma_50': 9.8,
            'ma_200': 9.5,
            'pe_ratio': 15,
            'sector_pe': 20,
            'pb_ratio': 2.5,
            'revenue_growth': 0.1,
            'earnings_growth': 0.15,
            'sentiment': 0.7,
            'volatility': 0.2,
            'beta': 1.1
        }

    def test_get_dynamic_weights(self):
        weights = self.calculator._get_dynamic_weights()
        self.assertEqual(sum(weights.values()), 1.0)
        
        bull_calculator = ScoreCalculator('bull')
        bear_calculator = ScoreCalculator('bear')
        
        self.assertNotEqual(weights, bull_calculator._get_dynamic_weights())
        self.assertNotEqual(weights, bear_calculator._get_dynamic_weights())

    def test_calculate_score(self):
        score = self.calculator.calculate_score(self.sample_metrics)
        self.assertTrue(0 <= score <= 1)

    def test_calculate_ml_score(self):
        ml_score = self.calculator._calculate_ml_score(0.5)
        self.assertEqual(ml_score, 0.75)

    def test_calculate_technical_score(self):
        technical_score = self.calculator._calculate_technical_score(self.sample_metrics)
        self.assertTrue(0 <= technical_score <= 1)

    def test_calculate_fundamental_score(self):
        fundamental_score = self.calculator._calculate_fundamental_score(self.sample_metrics)
        self.assertTrue(0 <= fundamental_score <= 1)

    def test_calculate_sentiment_score(self):
        sentiment_score = self.calculator._calculate_sentiment_score(0.7)
        self.assertEqual(sentiment_score, 0.7)

    def test_calculate_risk_score(self):
        risk_score = self.calculator._calculate_risk_score(self.sample_metrics)
        self.assertTrue(0 <= risk_score <= 1)

    def test_score_rsi(self):
        self.assertEqual(self.calculator._score_rsi(55), 1.0)
        self.assertEqual(self.calculator._score_rsi(35), 0.7)
        self.assertEqual(self.calculator._score_rsi(25), 0.4)
        self.assertEqual(self.calculator._score_rsi(15), 0.1)

    def test_score_macd(self):
        self.assertEqual(self.calculator._score_macd(0.5, 0.3), 1.0)
        self.assertEqual(self.calculator._score_macd(-0.1, -0.3), 0.7)
        self.assertEqual(self.calculator._score_macd(0.1, 0.3), 0.4)
        self.assertEqual(self.calculator._score_macd(-0.3, -0.1), 0.1)

    def test_score_bollinger_bands(self):
        self.assertEqual(self.calculator._score_bollinger_bands(10, 9, 11), 0.5)
        self.assertEqual(self.calculator._score_bollinger_bands(9.1, 9, 11), 0.9)
        self.assertEqual(self.calculator._score_bollinger_bands(9.3, 9, 11), 0.7)
        self.assertEqual(self.calculator._score_bollinger_bands(10.7, 9, 11), 0.3)
        self.assertEqual(self.calculator._score_bollinger_bands(10.9, 9, 11), 0.1)

    def test_score_moving_averages(self):
        self.assertEqual(self.calculator._score_moving_averages(105, 100), 1.0)
        self.assertEqual(self.calculator._score_moving_averages(103, 100), 0.7)
        self.assertEqual(self.calculator._score_moving_averages(98, 100), 0.4)
        self.assertEqual(self.calculator._score_moving_averages(94, 100), 0.1)

    def test_score_pe_ratio(self):
        self.assertEqual(self.calculator._score_pe_ratio(14, 20), 0.9)
        self.assertEqual(self.calculator._score_pe_ratio(18, 20), 0.7)
        self.assertEqual(self.calculator._score_pe_ratio(22, 20), 0.3)
        self.assertEqual(self.calculator._score_pe_ratio(30, 20), 0.1)

    def test_score_pb_ratio(self):
        self.assertEqual(self.calculator._score_pb_ratio(0.9), 0.9)
        self.assertEqual(self.calculator._score_pb_ratio(1.5), 0.7)
        self.assertEqual(self.calculator._score_pb_ratio(2.5), 0.5)
        self.assertEqual(self.calculator._score_pb_ratio(3.5), 0.3)
        self.assertEqual(self.calculator._score_pb_ratio(4.5), 0.1)

    def test_score_growth(self):
        self.assertAlmostEqual(self.calculator._score_growth(0.2, 0.2), 1.0)
        self.assertAlmostEqual(self.calculator._score_growth(0.1, 0.1), 0.75)
        self.assertAlmostEqual(self.calculator._score_growth(0, 0), 0.5)
        self.assertAlmostEqual(self.calculator._score_growth(-0.1, -0.1), 0.25)

    def test_get_verdict(self):
        self.assertEqual(self.calculator.get_verdict(0.9, 0.7), 'Strong Buy')
        self.assertEqual(self.calculator.get_verdict(0.7, 0.4), 'Buy')
        self.assertEqual(self.calculator.get_verdict(0.5, 0), 'Hold')
        self.assertEqual(self.calculator.get_verdict(0.3, -0.4), 'Sell')
        self.assertEqual(self.calculator.get_verdict(0.1, -0.7), 'Strong Sell')

    @patch('scoring.ScoreCalculator')
    def test_run_scoring(self, mock_score_calculator):
        mock_instance = mock_score_calculator.return_value
        mock_instance.calculate_score.return_value = 0.75
        mock_instance.get_verdict.return_value = 'Buy'

        result = run_scoring(self.sample_metrics)

        self.assertEqual(result['score'], 0.75)
        self.assertEqual(result['verdict'], 'Buy')
        mock_instance.calculate_score.assert_called_once_with(self.sample_metrics)
        mock_instance.get_verdict.assert_called_once_with(0.75, self.sample_metrics['ml_prediction'])

if __name__ == '__main__':
    unittest.main()