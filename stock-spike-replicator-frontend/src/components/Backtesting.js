import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { runBacktest } from '../services/api';
import handleApiError from '../utils/errorHandler';
import WebSocketHandler from './WebSocketHandler';
import { useAuth } from '../context/AuthContext';

const Backtesting = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [formData, setFormData] = useState({
    stockSymbol: '',
    startDate: '',
    endDate: '',
    initialCapital: '',
    strategy: 'movingAverage',
    movingAveragePeriod: 20,
  });
  const [errors, setErrors] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [backtestId, setBacktestId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [wsError, setWsError] = useState(null);

  useEffect(() => {
    return () => {
      // Cleanup function to close WebSocket connection
      if (backtestId) {
        // Implement a close method in WebSocketHandler if needed
        WebSocketHandler.close(backtestId);
      }
    };
  }, [backtestId]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value
    }));
  };

  const validateForm = () => {
    let formErrors = {};
    if (!formData.stockSymbol) formErrors.stockSymbol = 'Stock symbol is required';
    if (!formData.startDate) formErrors.startDate = 'Start date is required';
    if (!formData.endDate) formErrors.endDate = 'End date is required';
    if (!formData.initialCapital) formErrors.initialCapital = 'Initial capital is required';
    if (formData.initialCapital && isNaN(formData.initialCapital)) formErrors.initialCapital = 'Initial capital must be a number';
    if (formData.movingAveragePeriod && isNaN(formData.movingAveragePeriod)) formErrors.movingAveragePeriod = 'Moving average period must be a number';
    return formErrors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formErrors = validateForm();
    if (Object.keys(formErrors).length === 0) {
      setIsSubmitting(true);
      setErrors({});
      try {
        const response = await runBacktest(formData);
        setBacktestId(response.data.backtestId);
        setProgress(0);
        setIsLoading(true);
      } catch (error) {
        const errorMessage = handleApiError(error);
        setErrors({ submit: errorMessage });
      } finally {
        setIsSubmitting(false);
      }
    } else {
      setErrors(formErrors);
    }
  };

  const handleWebSocketUpdate = (data) => {
    if (data.type === 'progress') {
      setProgress(data.progress);
    } else if (data.type === 'complete') {
      setIsLoading(false);
      navigate('/results', { state: { backtestResults: data.results } });
    } else if (data.type === 'error') {
      setWsError(data.error);
      setIsLoading(false);
    }
  };

  if (!user) {
    return <div>Please log in to access this feature.</div>;
  }

  return (
    <div className="backtesting container mt-4">
      <h2 className="mb-4">Backtesting</h2>
      <form onSubmit={handleSubmit}>
        <div className="mb-3">
          <label htmlFor="stockSymbol" className="form-label">Stock Symbol:</label>
          <input
            type="text"
            className="form-control"
            id="stockSymbol"
            name="stockSymbol"
            value={formData.stockSymbol}
            onChange={handleChange}
          />
          {errors.stockSymbol && <div className="text-danger">{errors.stockSymbol}</div>}
        </div>
        <div className="mb-3">
          <label htmlFor="startDate" className="form-label">Start Date:</label>
          <input
            type="date"
            className="form-control"
            id="startDate"
            name="startDate"
            value={formData.startDate}
            onChange={handleChange}
          />
          {errors.startDate && <div className="text-danger">{errors.startDate}</div>}
        </div>
        <div className="mb-3">
          <label htmlFor="endDate" className="form-label">End Date:</label>
          <input
            type="date"
            className="form-control"
            id="endDate"
            name="endDate"
            value={formData.endDate}
            onChange={handleChange}
          />
          {errors.endDate && <div className="text-danger">{errors.endDate}</div>}
        </div>
        <div className="mb-3">
          <label htmlFor="initialCapital" className="form-label">Initial Capital:</label>
          <input
            type="number"
            className="form-control"
            id="initialCapital"
            name="initialCapital"
            value={formData.initialCapital}
            onChange={handleChange}
          />
          {errors.initialCapital && <div className="text-danger">{errors.initialCapital}</div>}
        </div>
        <div className="mb-3">
          <label htmlFor="strategy" className="form-label">Strategy:</label>
          <select
            className="form-select"
            id="strategy"
            name="strategy"
            value={formData.strategy}
            onChange={handleChange}
          >
            <option value="movingAverage">Moving Average</option>
            <option value="meanReversion">Mean Reversion</option>
            <option value="momentumTrading">Momentum Trading</option>
          </select>
        </div>
        {formData.strategy === 'movingAverage' && (
          <div className="mb-3">
            <label htmlFor="movingAveragePeriod" className="form-label">Moving Average Period:</label>
            <input
              type="number"
              className="form-control"
              id="movingAveragePeriod"
              name="movingAveragePeriod"
              value={formData.movingAveragePeriod}
              onChange={handleChange}
            />
            {errors.movingAveragePeriod && <div className="text-danger">{errors.movingAveragePeriod}</div>}
          </div>
        )}
        {errors.submit && <div className="alert alert-danger">{errors.submit}</div>}
        <button type="submit" className="btn btn-primary" disabled={isSubmitting || isLoading}>
          {isSubmitting ? 'Submitting...' : isLoading ? 'Running Backtest...' : 'Run Backtest'}
        </button>
      </form>
      {isLoading && (
        <div className="mt-4">
          <div className="progress">
            <div
              className="progress-bar"
              role="progressbar"
              style={{ width: `${progress}%` }}
              aria-valuenow={progress}
              aria-valuemin="0"
              aria-valuemax="100"
            >
              {progress}%
            </div>
          </div>
          <p className="mt-2">Backtest in progress...</p>
        </div>
      )}
      {wsError && <div className="alert alert-danger mt-3">{wsError}</div>}
      {backtestId && <WebSocketHandler backtestId={backtestId} onUpdate={handleWebSocketUpdate} />}
    </div>
  );
};

export default Backtesting;