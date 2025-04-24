import numpy as np
from typing import Dict, Any, List, Tuple, Union
import logging
from scipy import stats
from config import TRANSACTION_COST, SLIPPAGE

logger = logging.getLogger(__name__)

class StatisticalArbitrage:
    def __init__(self, lookback_period: int = 60, correlation_threshold: float = 0.8, z_score_threshold: float = 2.0):
        self.lookback_period = lookback_period
        self.correlation_threshold = correlation_threshold
        self.z_score_threshold = z_score_threshold

    def find_correlated_pairs(self, price_data: Dict[str, np.array]) -> List[Tuple[str, str]]:
        """Find pairs of stocks with high correlation."""
        correlated_pairs = []
        symbols = list(price_data.keys())
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                correlation = np.corrcoef(price_data[symbols[i]], price_data[symbols[j]])[0, 1]
                if abs(correlation) > self.correlation_threshold:
                    correlated_pairs.append((symbols[i], symbols[j]))
        return correlated_pairs

    def calculate_spread(self, price_a: np.array, price_b: np.array) -> np.array:
        """Calculate the spread between two price series."""
        return price_a - price_b

    def calculate_z_score(self, spread: np.array) -> float:
        """Calculate the z-score of the spread."""
        return (spread[-1] - np.mean(spread)) / np.std(spread)

    def generate_trading_signal(self, z_score: float) -> int:
        """Generate trading signal based on z-score."""
        if z_score > self.z_score_threshold:
            return -1  # Sell signal
        elif z_score < -self.z_score_threshold:
            return 1  # Buy signal
        else:
            return 0  # No signal

    def backtest_pair(self, price_a: np.array, price_b: np.array) -> Dict[str, Any]:
        """Backtest a pair trading strategy."""
        spread = self.calculate_spread(price_a, price_b)
        z_scores = np.zeros(len(spread))
        positions = np.zeros(len(spread))
        returns = np.zeros(len(spread))

        for i in range(self.lookback_period, len(spread)):
            z_scores[i] = self.calculate_z_score(spread[i-self.lookback_period:i])
            positions[i] = self.generate_trading_signal(z_scores[i])
            if i > 0:
                returns[i] = positions[i-1] * (spread[i] - spread[i-1])

        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        total_return = np.sum(returns)

        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'positions': positions,
            'returns': returns
        }

class ScoreCalculator:
    def __init__(self, market_regime: str = 'normal'):
        self.market_regime = market_regime
        self.weights = self._get_dynamic_weights()

    def _get_dynamic_weights(self) -> Dict[str, float]:
        """Get dynamic weights based on the current market regime."""
        if self.market_regime == 'bull':
            return {
                'ml_prediction': 0.3,
                'technical': 0.25,
                'fundamental': 0.2,
                'sentiment': 0.15,
                'risk': 0.1
            }
        elif self.market_regime == 'bear':
            return {
                'ml_prediction': 0.25,
                'technical': 0.3,
                'fundamental': 0.2,
                'sentiment': 0.1,
                'risk': 0.15
            }
        else:  # normal
            return {
                'ml_prediction': 0.25,
                'technical': 0.25,
                'fundamental': 0.25,
                'sentiment': 0.15,
                'risk': 0.1
            }

    def calculate_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate the comprehensive score."""
        ml_score = self._calculate_ml_score(metrics.get('ml_prediction', 0))
        technical_score = self._calculate_technical_score(metrics)
        fundamental_score = self._calculate_fundamental_score(metrics)
        sentiment_score = self._calculate_sentiment_score(metrics.get('sentiment', 0.5))  # Default to neutral sentiment
        risk_score = self._calculate_risk_score(metrics)

        # Ensure weights are float values
        ml_weight = float(self.weights['ml_prediction'][0]) if isinstance(self.weights['ml_prediction'], (list, tuple)) else float(self.weights['ml_prediction'])
        technical_weight = float(self.weights['technical'][0]) if isinstance(self.weights['technical'], (list, tuple)) else float(self.weights['technical'])
        fundamental_weight = float(self.weights['fundamental'][0]) if isinstance(self.weights['fundamental'], (list, tuple)) else float(self.weights['fundamental'])
        sentiment_weight = float(self.weights['sentiment'][0]) if isinstance(self.weights['sentiment'], (list, tuple)) else float(self.weights['sentiment'])
        risk_weight = float(self.weights['risk'][0]) if isinstance(self.weights['risk'], (list, tuple)) else float(self.weights['risk'])

        weighted_score = (
            ml_weight * ml_score +
            technical_weight * technical_score +
            fundamental_weight * fundamental_score +
            sentiment_weight * sentiment_score +
            risk_weight * risk_score
        )

        return np.clip(weighted_score, 0, 1)

    def _calculate_ml_score(self, prediction: Union[float, List[float]]) -> float:
        """Calculate score based on ML prediction."""
        if isinstance(prediction, list):
            # If it's a list, take the average of the predictions
            return np.mean([(p + 1) / 2 for p in prediction])  # Assuming predictions are between -1 and 1
        else:
            return (prediction + 1) / 2  # Assuming prediction is between -1 and 1

    def _calculate_technical_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate technical score."""
        rsi_score = self._score_rsi(metrics['rsi'])
        macd_score = self._score_macd(metrics['macd'], metrics['macd_signal'])
        bb_score = self._score_bollinger_bands(metrics['price'], metrics['bb_lower'], metrics['bb_upper'])
        ma_score = self._score_moving_averages(metrics['ma_50'], metrics['ma_200'])

        return np.mean([rsi_score, macd_score, bb_score, ma_score])

    def _calculate_fundamental_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate fundamental score."""
        pe_score = self._score_pe_ratio(metrics.get('pe_ratio', 0), metrics.get('sector_pe', 0))
        pb_score = self._score_pb_ratio(metrics.get('pb_ratio', 0))
        growth_score = self._score_growth(metrics.get('revenue_growth', 0), metrics.get('earnings_growth', 0))

        return np.mean([pe_score, pb_score, growth_score])

    def _calculate_sentiment_score(self, sentiment: float) -> float:
        """Calculate sentiment score."""
        return sentiment  # Assuming sentiment is already between 0 and 1

    def _calculate_risk_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate risk-adjusted score."""
        volatility = metrics.get('volatility')
        beta = metrics.get('beta')
        
        if volatility is None and beta is None:
            return 0.5  # Return a neutral score if both volatility and beta are not available
        
        scores = []
        if volatility is not None:
            volatility_score = 1 - volatility  # Lower volatility is better
            scores.append(volatility_score)
        
        if beta is not None:
            beta_score = 1 - abs(beta - 1)  # Beta closer to 1 is better
            scores.append(beta_score)
        
        return np.mean(scores) if scores else 0.5

    def _score_rsi(self, rsi: float) -> float:
        """Score RSI."""
        if rsi is None:
            return 0.5  # Return a neutral score if RSI is not available
        if 40 <= rsi <= 60:
            return 1.0
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            return 0.7
        elif 20 <= rsi < 30 or 70 < rsi <= 80:
            return 0.4
        else:
            return 0.1

    def _score_macd(self, macd: float, signal: float) -> float:
        """Score MACD."""
        if macd is None or signal is None:
            return 0.5  # Return a neutral score if MACD or signal is not available
        if macd > signal and macd > 0:
            return 1.0
        elif macd > signal and macd <= 0:
            return 0.7
        elif macd <= signal and macd > 0:
            return 0.4
        else:
            return 0.1

    def _score_bollinger_bands(self, price: float, lower: float, upper: float) -> float:
        """Score Bollinger Bands position."""
        if lower == upper:
            return 0.5
        position = (price - lower) / (upper - lower)
        if 0.4 <= position <= 0.6:
            return 0.5
        elif position < 0.2:
            return 0.9
        elif position < 0.4:
            return 0.7
        elif position < 0.8:
            return 0.3
        else:
            return 0.1

    def _score_moving_averages(self, ma_50: float, ma_200: float) -> float:
        """Score Moving Averages."""
        if ma_50 is None or ma_200 is None or ma_200 == 0:
            return 0.5  # Return a neutral score if we don't have enough data
        ratio = ma_50 / ma_200
        if ratio > 1.05:
            return 1.0
        elif 1 < ratio <= 1.05:
            return 0.7
        elif 0.95 <= ratio < 1:
            return 0.4
        else:
            return 0.1

    def _score_pe_ratio(self, pe: float, sector_pe: float) -> float:
        """Score P/E ratio."""
        if pe <= 0:
            return 0.1
        relative_pe = pe / sector_pe
        if relative_pe < 0.7:
            return 0.9
        elif relative_pe < 0.9:
            return 0.7
        elif relative_pe < 1.1:
            return 0.5
        elif relative_pe < 1.3:
            return 0.3
        else:
            return 0.1

    def _score_pb_ratio(self, pb: float) -> float:
        """Score P/B ratio."""
        if pb <= 0:
            return 0.1
        elif pb < 1:
            return 0.9
        elif pb < 2:
            return 0.7
        elif pb < 3:
            return 0.5
        elif pb < 4:
            return 0.3
        else:
            return 0.1

    def _score_growth(self, revenue_growth: float, earnings_growth: float) -> float:
        """Score growth metrics."""
        avg_growth = (revenue_growth + earnings_growth) / 2
        return np.clip((avg_growth + 0.2) / 0.4, 0, 1)  # Normalize to 0-1

    def get_verdict(self, score: float, ml_prediction: Union[float, List[float]]) -> str:
        """Generate a trading verdict based on score and ML prediction."""
        if isinstance(ml_prediction, list):
            ml_prediction = np.mean(ml_prediction)  # Take the average if it's a list
        
        if score > 0.8 and ml_prediction > 0.6:
            return 'Strong Buy'
        elif score > 0.6 and ml_prediction > 0.3:
            return 'Buy'
        elif score < 0.4 and ml_prediction < -0.3:
            return 'Sell'
        elif score < 0.2 and ml_prediction < -0.6:
            return 'Strong Sell'
        else:
            return 'Hold'

class EnhancedPerformanceMetrics(ScoreCalculator):
    def __init__(self, market_regime: str = 'normal'):
        super().__init__(market_regime)

    def calculate_transaction_cost_impact(self, returns: np.array, positions: np.array) -> float:
        """Calculate the impact of transaction costs on returns."""
        transaction_costs = TRANSACTION_COST * np.abs(positions - np.roll(positions, 1))
        return np.mean(transaction_costs / np.abs(returns))

    def calculate_slippage_impact(self, returns: np.array) -> float:
        """Calculate the impact of slippage on returns."""
        return np.mean(SLIPPAGE / np.abs(returns))

    def calculate_cost_adjusted_sharpe_ratio(self, returns: np.array, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio adjusted for transaction costs and slippage."""
        cost_adjusted_returns = returns - TRANSACTION_COST * np.abs(returns) - SLIPPAGE * np.abs(returns)
        excess_returns = cost_adjusted_returns - risk_free_rate / 252  # Assuming daily returns
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def calculate_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate the comprehensive score including transaction costs and slippage."""
        base_score = super().calculate_score(metrics)
        
        returns = metrics.get('returns')
        positions = metrics.get('positions')
        
        if returns is None or positions is None:
            logger.warning("Missing 'returns' or 'positions' in metrics. Returning base score.")
            return base_score
        
        transaction_cost_impact = self.calculate_transaction_cost_impact(returns, positions)
        slippage_impact = self.calculate_slippage_impact(returns)
        cost_adjusted_sharpe = self.calculate_cost_adjusted_sharpe_ratio(returns)
        
        # Adjust the base score based on transaction costs and slippage
        cost_factor = 1 - (transaction_cost_impact + slippage_impact)
        adjusted_score = base_score * cost_factor
        
        # Incorporate cost-adjusted Sharpe ratio
        sharpe_weight = 0.2  # Weight for Sharpe ratio in the final score
        final_score = (1 - sharpe_weight) * adjusted_score + sharpe_weight * (cost_adjusted_sharpe / 3)  # Normalize Sharpe ratio
        
        return np.clip(final_score, 0, 1)

def run_scoring(metrics: Dict[str, Any], market_regime: str = 'normal') -> Dict[str, Any]:
    calculator = EnhancedPerformanceMetrics(market_regime)
    score = calculator.calculate_score(metrics)
    verdict = calculator.get_verdict(score, metrics['ml_prediction'])
    
    return {
        'score': score,
        'verdict': verdict,
        'transaction_cost_impact': calculator.calculate_transaction_cost_impact(metrics['returns'], metrics['positions']),
        'slippage_impact': calculator.calculate_slippage_impact(metrics['returns']),
        'cost_adjusted_sharpe_ratio': calculator.calculate_cost_adjusted_sharpe_ratio(metrics['returns'])
    }

def run_statistical_arbitrage(price_data: Dict[str, np.array]) -> Dict[str, Any]:
    stat_arb = StatisticalArbitrage()
    correlated_pairs = stat_arb.find_correlated_pairs(price_data)
    
    best_pair = None
    best_sharpe = -np.inf
    best_results = None

    for pair in correlated_pairs:
        results = stat_arb.backtest_pair(price_data[pair[0]], price_data[pair[1]])
        if results['sharpe_ratio'] > best_sharpe:
            best_sharpe = results['sharpe_ratio']
            best_pair = pair
            best_results = results

    if best_pair is not None:
        return {
            'best_pair': best_pair,
            'sharpe_ratio': best_results['sharpe_ratio'],
            'total_return': best_results['total_return'],
            'positions': best_results['positions'],
            'returns': best_results['returns']
        }
    else:
        return {'message': 'No suitable pairs found for statistical arbitrage'}

# Modify the run_scoring function to incorporate statistical arbitrage
def run_scoring(metrics: Dict[str, Any], price_data: Dict[str, np.array], market_regime: str = 'normal') -> Dict[str, Any]:
    calculator = EnhancedPerformanceMetrics(market_regime)
    score = calculator.calculate_score(metrics)
    verdict = calculator.get_verdict(score, metrics['ml_prediction'])
    
    stat_arb_results = run_statistical_arbitrage(price_data)
    
    return {
        'score': score,
        'verdict': verdict,
        'transaction_cost_impact': calculator.calculate_transaction_cost_impact(metrics['returns'], metrics['positions']),
        'slippage_impact': calculator.calculate_slippage_impact(metrics['returns']),
        'cost_adjusted_sharpe_ratio': calculator.calculate_cost_adjusted_sharpe_ratio(metrics['returns']),
        'statistical_arbitrage': stat_arb_results
    }