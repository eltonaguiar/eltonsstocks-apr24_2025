import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';
import handleApiError from '../utils/errorHandler';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }

    const fetchResults = async () => {
      if (location.state?.backtestResults) {
        setResults(location.state.backtestResults);
        setLoading(false);
      } else {
        try {
          // Fetch the latest backtest results if not provided in location state
          const response = await api.get('/backtest/latest');
          setResults(response.data);
          setLoading(false);
        } catch (error) {
          const errorMessage = handleApiError(error);
          setError(errorMessage);
          setLoading(false);
        }
      }
    };

    fetchResults();
  }, [location.state, user, navigate]);

  const handleSaveResults = async () => {
    try {
      await api.post('/backtest/save', results);
      alert('Results saved successfully!');
    } catch (error) {
      const errorMessage = handleApiError(error);
      alert(`Failed to save results: ${errorMessage}`);
    }
  };

  const handleShareResults = () => {
    // Implement sharing functionality (e.g., generate a shareable link)
    alert('Sharing functionality not implemented yet.');
  };

  if (loading) {
    return <div className="text-center mt-5">Loading results...</div>;
  }

  if (error) {
    return <div className="alert alert-danger mt-5">{error}</div>;
  }

  if (!results) {
    return <div className="alert alert-warning mt-5">No results available. Please run a backtest first.</div>;
  }

  const chartData = {
    labels: results.trades.map(trade => trade.date),
    datasets: [
      {
        label: 'Portfolio Value',
        data: results.trades.map(trade => trade.portfolioValue),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      },
      {
        label: 'Stock Price',
        data: results.trades.map(trade => trade.price),
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: false,
        title: {
          display: true,
          text: 'Value ($)'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Date'
        }
      }
    },
    plugins: {
      tooltip: {
        mode: 'index',
        intersect: false,
      },
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Backtest Performance'
      }
    }
  };

  return (
    <div className="results container mt-5">
      <h2 className="mb-4">Backtesting Results</h2>
      <div className="row">
        <div className="col-md-6">
          <div className="card mb-4">
            <div className="card-header">Summary</div>
            <div className="card-body">
              <p><strong>Stock Symbol:</strong> {results.stockSymbol}</p>
              <p><strong>Period:</strong> {results.startDate} to {results.endDate}</p>
              <p><strong>Initial Capital:</strong> ${results.initialCapital.toFixed(2)}</p>
              <p><strong>Final Capital:</strong> ${results.finalCapital.toFixed(2)}</p>
              <p><strong>Total Return:</strong> {results.totalReturn.toFixed(2)}%</p>
              <p><strong>Annualized Return:</strong> {results.annualizedReturn.toFixed(2)}%</p>
              <p><strong>Sharpe Ratio:</strong> {results.sharpeRatio.toFixed(2)}</p>
              <p><strong>Max Drawdown:</strong> {results.maxDrawdown.toFixed(2)}%</p>
            </div>
          </div>
          <div className="mb-3">
            <button className="btn btn-primary me-2" onClick={handleSaveResults}>Save Results</button>
            <button className="btn btn-secondary" onClick={handleShareResults}>Share Results</button>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card mb-4">
            <div className="card-header">Performance Chart</div>
            <div className="card-body" style={{ height: '400px' }}>
              <Line data={chartData} options={chartOptions} />
            </div>
          </div>
        </div>
      </div>
      <div className="card">
        <div className="card-header">Trades</div>
        <div className="card-body">
          <div className="table-responsive">
            <table className="table table-striped">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Action</th>
                  <th>Price</th>
                  <th>Shares</th>
                  <th>Value</th>
                  <th>Portfolio Value</th>
                </tr>
              </thead>
              <tbody>
                {results.trades.map((trade, index) => (
                  <tr key={index}>
                    <td>{trade.date}</td>
                    <td>{trade.action}</td>
                    <td>${trade.price.toFixed(2)}</td>
                    <td>{trade.shares}</td>
                    <td>${(trade.price * trade.shares).toFixed(2)}</td>
                    <td>${trade.portfolioValue.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;