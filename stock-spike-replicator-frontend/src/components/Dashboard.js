import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';

const Dashboard = () => {
  const { user } = useAuth();
  const [recentResults, setRecentResults] = useState([]);
  const [watchlistsSummary, setWatchlistsSummary] = useState(null);
  const [sentimentData, setSentimentData] = useState(null);
  const [statArbResults, setStatArbResults] = useState(null);

  useEffect(() => {
    const fetchRecentResults = async () => {
      try {
        const response = await api.get('/backtest/recent');
        setRecentResults(response.data.slice(0, 3)); // Get the 3 most recent results
      } catch (error) {
        console.error('Failed to fetch recent results:', error);
      }
    };

    const fetchWatchlistsSummary = async () => {
      try {
        const response = await api.getWatchlists();
        const totalWatchlists = response.data.length;
        const totalStocks = response.data.reduce((sum, watchlist) => sum + watchlist.stocks.length, 0);
        setWatchlistsSummary({ totalWatchlists, totalStocks });
      } catch (error) {
        console.error('Failed to fetch watchlists summary:', error);
      }
    };

    const fetchSentimentData = async () => {
      try {
        const response = await api.get('/stocks/AAPL/sentiment'); // Example with AAPL
        setSentimentData(response.data);
      } catch (error) {
        console.error('Failed to fetch sentiment data:', error);
      }
    };

    const fetchStatArbResults = async () => {
      try {
        const response = await api.post('/stocks/statistical-arbitrage', { symbols: ['AAPL', 'MSFT'] }); // Example with AAPL and MSFT
        setStatArbResults(response.data);
      } catch (error) {
        console.error('Failed to fetch statistical arbitrage results:', error);
      }
    };

    fetchRecentResults();
    fetchWatchlistsSummary();
    fetchSentimentData();
    fetchStatArbResults();
  }, []);

  return (
    <div className="dashboard">
      <h2>Welcome, {user?.name || 'User'}!</h2>
      <div className="dashboard-actions">
        <div className="action-card">
          <h3>Run Backtest</h3>
          <p>Test your trading strategies with historical data.</p>
          <Link to="/backtesting" className="btn btn-primary">Start Backtesting</Link>
        </div>
        <div className="action-card">
          <h3>View Profile</h3>
          <p>Manage your account settings and preferences.</p>
          <Link to="/profile" className="btn btn-secondary">Go to Profile</Link>
        </div>
        <div className="action-card">
          <h3>Recent Results</h3>
          <p>View your most recent backtest results.</p>
          {recentResults.length > 0 ? (
            <ul>
              {recentResults.map((result, index) => (
                <li key={index}>
                  <Link to={`/results/${result.id}`}>
                    {result.symbol} - {result.performance}
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <p>No recent results available.</p>
          )}
          <Link to="/results" className="btn btn-info">View All Results</Link>
        </div>
        <div className="action-card">
          <h3>Watchlists</h3>
          <p>Manage your stock watchlists.</p>
          {watchlistsSummary ? (
            <p>You have {watchlistsSummary.totalWatchlists} watchlist(s) with {watchlistsSummary.totalStocks} stocks.</p>
          ) : (
            <p>Loading watchlists summary...</p>
          )}
          <Link to="/watchlist" className="btn btn-success">Manage Watchlists</Link>
        </div>
        <div className="action-card">
          <h3>Sentiment Analysis</h3>
          <p>View sentiment analysis for stocks.</p>
          {sentimentData ? (
            <p>AAPL Sentiment Score: {sentimentData.sentiment_score.toFixed(2)}</p>
          ) : (
            <p>Loading sentiment data...</p>
          )}
          <Link to="/sentiment" className="btn btn-info">View Detailed Sentiment</Link>
        </div>
        <div className="action-card">
          <h3>Statistical Arbitrage</h3>
          <p>View statistical arbitrage opportunities.</p>
          {statArbResults ? (
            <p>Best Pair: {statArbResults.best_pair.join(' - ')}, Sharpe Ratio: {statArbResults.sharpe_ratio.toFixed(2)}</p>
          ) : (
            <p>Loading statistical arbitrage results...</p>
          )}
          <Link to="/statistical-arbitrage" className="btn btn-warning">Explore Arbitrage</Link>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;