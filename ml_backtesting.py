import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from typing import List, Dict, Any
import logging
from config import TRANSACTION_COST, SLIPPAGE, BACKTEST_START_DATE, BACKTEST_END_DATE, INITIAL_CAPITAL
from scipy.stats import randint, uniform
from hmmlearn import hmm

logger = logging.getLogger(__name__)

class MLBacktesting:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        self.best_model = None
        self.feature_importance = None
        self.symbol = None  # Add symbol tracking
        self.selected_features = None

    def prepare_data(self, historical_data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Prepare data for machine learning model with advanced features."""
        self.symbol = symbol  # Store the symbol being processed
        logger.info(f"Preparing data for symbol: {symbol}")
        
        df = historical_data.copy()
        
        # Add symbol validation
        if 'symbol' in df.columns:
            if len(df['symbol'].unique()) > 1:
                logger.warning(f"Multiple symbols detected in data: {df['symbol'].unique()}")
                if symbol:
                    df = df[df['symbol'] == symbol]
                    logger.info(f"Filtered data for symbol: {symbol}")
        
        # Ensure column names are lowercase
        df.columns = df.columns.str.lower()
        
        # Basic feature engineering
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], df['signal'] = self.calculate_macd(df['close'])
        
        # Advanced feature engineering
        df['bollinger_upper'], df['bollinger_lower'] = self.calculate_bollinger_bands(df['close'])
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        df['obv'] = self.calculate_obv(df['close'], df['volume'])
        df['cci'] = self.calculate_cci(df['high'], df['low'], df['close'])
        df['stoch_k'], df['stoch_d'] = self.calculate_stochastic_oscillator(df['high'], df['low'], df['close'])
        
        # Target variable: next day's return
        df['target'] = df['returns'].shift(-1)
        
        df = df.dropna()
        logger.info(f"Prepared data for {symbol}: {len(df)} rows after cleaning")
        return df

    def train_model(self, data: pd.DataFrame) -> None:
        """Train and evaluate multiple machine learning models with hyperparameter tuning."""
        logger.info(f"Training model for symbol: {self.symbol}")
        
        # Create the target variable (next day's return)
        data['target'] = data['close'].pct_change().shift(-1)
        
        features = ['returns', 'log_returns', 'volatility', 'ma_5', 'ma_20', 'rsi', 'macd', 'signal',
                    'bollinger_upper', 'bollinger_lower', 'atr', 'obv', 'cci', 'stoch_k', 'stoch_d']
        
        # Ensure all required features are present
        for feature in features:
            if feature not in data.columns:
                raise ValueError(f"Required feature '{feature}' not found in the data for symbol {self.symbol}")
        
        X = data[features]
        y = data['target']
        
        # Remove any rows with NaN values
        X = X.dropna()
        y = y.loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Remove NaN values
        data = data.dropna()
        
        if len(data) < 2:
            logger.warning(f"Not enough data points after removing NaN values for {self.symbol}")
            return
        
        best_score = float('-inf')
        best_model_name = ''
        
        # Perform feature selection first
        feature_selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
        X = data[features]
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = feature_selector.transform(X_test_scaled)
        
        selected_features = [feature for feature, selected in zip(features, feature_selector.get_support()) if selected]
        logger.info(f"Selected features for {self.symbol}: {selected_features}")
        
        # Store the selected features for prediction
        self.selected_features = selected_features
        
        for name, model in self.models.items():
            logger.info(f"Training and tuning {name} for {self.symbol}...")
            
            # Define hyperparameter search space
            if name == 'random_forest':
                param_dist = {
                    'n_estimators': randint(100, 500),
                    'max_depth': randint(5, 30),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10)
                }
            elif name == 'gradient_boosting':
                param_dist = {
                    'n_estimators': randint(100, 500),
                    'learning_rate': uniform(0.01, 0.3),
                    'max_depth': randint(3, 10),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10)
                }
            elif name == 'xgboost':
                param_dist = {
                    'n_estimators': randint(100, 500),
                    'learning_rate': uniform(0.01, 0.3),
                    'max_depth': randint(3, 10),
                    'min_child_weight': randint(1, 10),
                    'subsample': uniform(0.5, 0.5),
                    'colsample_bytree': uniform(0.5, 0.5)
                }
            else:  # Linear Regression
                param_dist = {}

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Randomized search with time series cross-validation
            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=50,
                cv=tscv,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            random_search.fit(X_train_selected, y_train)
            
            logger.info(f"Best parameters for {name} ({self.symbol}): {random_search.best_params_}")
            logger.info(f"Best cross-validation score for {name} ({self.symbol}): {random_search.best_score_:.4f}")
            
            if random_search.best_score_ > best_score:
                best_score = random_search.best_score_
                self.best_model = random_search.best_estimator_
                best_model_name = name

        logger.info(f"Best model for {self.symbol}: {best_model_name}")

        # Final evaluation on test set
        y_pred = self.best_model.predict(X_test_selected)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Best model performance for {self.symbol}: MSE = {mse:.4f}, MAE = {mae:.4f}, R2 = {r2:.4f}")

        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info(f"Feature importance for {self.symbol}:\n{self.feature_importance}")
        else:
            logger.info(f"Feature importance not available for the best model for {self.symbol}.")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.selected_features is None:
            raise ValueError(f"Model has not been trained yet for {self.symbol}. Call train_model() first.")
        
        # Ensure all required features are present in the data
        missing_features = [f for f in self.selected_features if f not in data.columns]
        if missing_features:
            logger.warning(f"Missing features in data for {self.symbol}: {missing_features}")
            # Use a subset of features that are available
            available_features = [f for f in self.selected_features if f in data.columns]
            if not available_features:
                raise ValueError(f"None of the required features are available in the data for {self.symbol}")
            logger.warning(f"Using only available features for {self.symbol}: {available_features}")
            X = data[available_features]
        else:
            X = data[self.selected_features]
        
        # Handle any NaN values
        X = X.fillna(0)  # Replace NaNs with 0 or use another strategy
        
        # Scale the features
        scaler = StandardScaler()
        X = X[self.selected_features]  # Ensure consistent feature set
        X_scaled = scaler.fit_transform(X)
        
        return self.best_model.predict(X_scaled)

    def backtest(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform backtesting on historical data."""
        logger.info(f"Running backtest for symbol: {self.symbol}")
        df = self.prepare_data(historical_data, self.symbol)
        df['prediction'] = self.predict(df)
        
        # Trading strategy: Buy when predicted return is positive, sell when negative
        df['position'] = np.where(df['prediction'] > 0, 1, -1)
        
        # Calculate strategy returns
        df['strategy_returns'] = df['position'].shift(1) * df['returns'] - TRANSACTION_COST * np.abs(df['position'] - df['position'].shift(1))
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        
        # Calculate performance metrics
        total_return = df['cumulative_returns'].iloc[-1] - 1
        sharpe_ratio = np.sqrt(252) * df['strategy_returns'].mean() / df['strategy_returns'].std()
        max_drawdown = (df['cumulative_returns'] / df['cumulative_returns'].cummax() - 1).min()
        
        # Calculate Sortino ratio
        negative_returns = df['strategy_returns'][df['strategy_returns'] < 0]
        downside_deviation = np.sqrt(np.mean(negative_returns**2))
        sortino_ratio = np.sqrt(252) * df['strategy_returns'].mean() / downside_deviation
        
        # Calculate Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown)
        
        # Calculate Win rate and Average win/loss
        wins = df['strategy_returns'][df['strategy_returns'] > 0]
        losses = df['strategy_returns'][df['strategy_returns'] < 0]
        win_rate = len(wins) / (len(wins) + len(losses))
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        results = {
            'symbol': self.symbol,  # Add symbol to results
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'cumulative_returns': df['cumulative_returns'],
            'ml_prediction': df['prediction'].tolist(),
            'rsi': df['rsi'].iloc[-1],
            'macd': df['macd'].iloc[-1],
            'macd_signal': df['signal'].iloc[-1],
            'price': df['close'].iloc[-1],
            'bb_lower': df['bollinger_lower'].iloc[-1],
            'bb_upper': df['bollinger_upper'].iloc[-1],
            'ma_50': df['ma_50'].iloc[-1] if 'ma_50' in df.columns else None,
            'ma_200': df['ma_200'].iloc[-1] if 'ma_200' in df.columns else None,
            'volatility': df['volatility'].iloc[-1],
            'returns': df['returns'].tolist(),
            'positions': df['position'].tolist()
        }
        
        logger.info(f"Backtesting results for {self.symbol}: Total return = {total_return:.2%}, Sharpe ratio = {sharpe_ratio:.2f}, Max drawdown = {max_drawdown:.2%}")
        
        return results

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
        """Calculate Moving Average Convergence Divergence."""
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        return upper_band, lower_band

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        return (np.sign(close.diff()) * volume).cumsum()

    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma_tp) / (0.015 * mad)

    @staticmethod
    def calculate_stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return k, d

class EnhancedBacktester(MLBacktesting):
    def __init__(self, use_simplified_model=False, n_estimators=100, max_depth=None):
        """
        Initialize the EnhancedBacktester with options for performance optimization.
        
        Args:
            use_simplified_model (bool): Whether to use a simplified model for faster processing
            n_estimators (int): Number of estimators for tree-based models
            max_depth (int): Maximum depth for tree-based models
        """
        super().__init__()
        self.markov_model = None
        self.use_simplified_model = use_simplified_model
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        # Update model parameters if simplified model is requested
        if use_simplified_model:
            self.models = {
                'random_forest': RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
                'linear_regression': LinearRegression()
            }
            
    def use_fallback_model(self):
        """
        Use a simple fallback model when the main model training times out.
        """
        logger.warning(f"Using fallback model for {self.symbol}")
        # Create a simple linear regression model as fallback
        self.selected_features = ['returns', 'log_returns', 'volatility']
        self.best_model = LinearRegression()

        if hasattr(self, 'prepared_data') and self.prepared_data is not None:
            df = self.prepared_data.copy()
            df = df.dropna(subset=self.selected_features + ['target'])
            X = df[self.selected_features]
            y = df['target']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.best_model.fit(X_scaled, y)
        else:
            logger.error("No prepared data available to fit fallback model.")
        
    def generate_simplified_results(self, df):
        """
        Generate simplified results when backtesting times out.
        
        Args:
            df (pd.DataFrame): Historical data
            
        Returns:
            dict: Simplified results with default values
        """
        logger.warning(f"Generating simplified results for {self.symbol}")
        
        # Calculate some basic metrics
        returns = df['close'].pct_change().dropna()
        
        # Generate simplified results
        results = {
            'symbol': self.symbol,
            'total_return': returns.mean() * len(returns),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 1.0,
            'sortino_ratio': 1.0,
            'calmar_ratio': 1.0,
            'max_drawdown': 0.1,
            'win_rate': 0.5,
            'avg_win': 0.02,
            'avg_loss': -0.01,
            'ml_prediction': 0,
            'returns': returns.tolist(),
            'positions': [0] * len(returns),
            'verdict': 'Hold',
            'score': 0.5,
            'explanation': 'Fallback model used due to timeout or error.',
            'price': df['close'].iloc[-1] if not df.empty else 'N/A',
            'rsi': 50,
            'bollinger_upper': 0,
            'bollinger_lower': 0,
            'ma_50': 0,
            'ma_200': 0
        }
        
        return results

def run_ml_backtesting(historical_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Run ML backtesting on multiple symbols and return cumulative returns.
    
    Args:
        historical_data (Dict[str, pd.DataFrame]): Dictionary mapping symbols to their historical data
        
    Returns:
        Dict[str, float]: Dictionary mapping dates to cumulative returns
    """
    cumulative_returns = {}
    
    for symbol, data in historical_data.items():
        try:
            logger.info(f"Running ML backtesting for {symbol}")
            enhanced_backtester = EnhancedBacktester(use_simplified_model=True, n_estimators=50, max_depth=3)
            prepared_data = enhanced_backtester.prepare_data(data, symbol)
            enhanced_backtester.train_model(prepared_data)
            results = enhanced_backtester.backtest(data)
            
            # Add the cumulative returns for this symbol
            if 'cumulative_returns' in results:
                for date, value in results['cumulative_returns'].items():
                    if date in cumulative_returns:
                        cumulative_returns[date] += value
                    else:
                        cumulative_returns[date] = value
        except Exception as e:
            logger.error(f"Error in ML backtesting for {symbol}: {str(e)}")
    
    # Average the cumulative returns if we have multiple symbols
    if historical_data:
        for date in cumulative_returns:
            cumulative_returns[date] /= len(historical_data)
    
    return cumulative_returns