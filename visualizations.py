import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
from typing import List

def plot_top_stock_pick(symbol: str, historical_data: pd.DataFrame, current_price: float, support_level: float, resistance_level: float) -> BytesIO:
    """
    Create a visualization for the top stock pick.

    Args:
        symbol (str): The stock symbol
        historical_data (pd.DataFrame): Historical price data for the stock
        current_price (float): The current stock price
        support_level (float): The calculated support level
        resistance_level (float): The calculated resistance level

    Returns:
        BytesIO: A buffer containing the plot image in PNG format
    """
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data.index, historical_data['Close'], label='Close Price')
    plt.axhline(y=support_level, color='g', linestyle='--', label='Support Level')
    plt.axhline(y=resistance_level, color='r', linestyle='--', label='Resistance Level')
    plt.scatter(historical_data.index[-1], current_price, color='b', s=100, label='Current Price')

    plt.title(f'{symbol} Stock Price Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return buffer

def plot_transaction_cost_impact(returns: pd.Series, transaction_costs: pd.Series) -> BytesIO:
    """
    Create a visualization of the impact of transaction costs on returns over time.

    Args:
        returns (pd.Series): The series of returns
        transaction_costs (pd.Series): The series of transaction costs

    Returns:
        BytesIO: A buffer containing the plot image in PNG format
    """
    plt.figure(figsize=(12, 6))
    cumulative_returns = (1 + returns).cumprod()
    cumulative_returns_after_costs = (1 + returns - transaction_costs).cumprod()

    plt.plot(cumulative_returns.index, cumulative_returns, label='Returns without Transaction Costs')
    plt.plot(cumulative_returns_after_costs.index, cumulative_returns_after_costs, label='Returns with Transaction Costs')

    plt.title('Impact of Transaction Costs on Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return buffer

def plot_slippage_distribution(slippages: List[float]) -> BytesIO:
    """
    Create a visualization of the distribution of slippage.

    Args:
        slippages (List[float]): A list of slippage values

    Returns:
        BytesIO: A buffer containing the plot image in PNG format
    """
    plt.figure(figsize=(12, 6))
    plt.hist(slippages, bins=50, edgecolor='black')
    plt.title('Distribution of Slippage')
    plt.xlabel('Slippage')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Add mean and median lines
    mean_slippage = np.mean(slippages)
    median_slippage = np.median(slippages)
    plt.axvline(mean_slippage, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_slippage:.4f}')
    plt.axvline(median_slippage, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_slippage:.4f}')
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return buffer