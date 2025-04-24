import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';
import handleApiError from '../utils/errorHandler';

const Watchlist = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [watchlists, setWatchlists] = useState([]);
  const [newWatchlistName, setNewWatchlistName] = useState('');
  const [newStockSymbol, setNewStockSymbol] = useState('');
  const [selectedWatchlist, setSelectedWatchlist] = useState(null);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }

    fetchWatchlists();
  }, [user, navigate]);

  const fetchWatchlists = async () => {
    try {
      const response = await api.get('/watchlists');
      setWatchlists(response.data);
    } catch (err) {
      setError(handleApiError(err));
    }
  };

  const handleCreateWatchlist = async (e) => {
    e.preventDefault();
    try {
      const response = await api.post('/watchlists', { name: newWatchlistName });
      setWatchlists([...watchlists, response.data]);
      setNewWatchlistName('');
      setSuccessMessage('Watchlist created successfully');
    } catch (err) {
      setError(handleApiError(err));
    }
  };

  const handleAddStock = async (e) => {
    e.preventDefault();
    if (!selectedWatchlist) {
      setError('Please select a watchlist');
      return;
    }
    try {
      await api.addToWatchlist(selectedWatchlist.id, newStockSymbol);
      const updatedWatchlist = await api.getWatchlist(selectedWatchlist.id);
      setWatchlists(watchlists.map(w => w.id === selectedWatchlist.id ? updatedWatchlist.data : w));
      setNewStockSymbol('');
      setSuccessMessage('Stock added to watchlist successfully');
      setError(null);
    } catch (err) {
      const errorMessage = handleApiError(err);
      setError(errorMessage);
      if (errorMessage.includes('Invalid or unavailable stock symbol')) {
        setError('Invalid or unavailable stock symbol. Please check and try again.');
      }
    }
  };

  const handleRemoveStock = async (watchlistId, symbol) => {
    try {
      await api.removeFromWatchlist(watchlistId, symbol);
      const updatedWatchlist = await api.getWatchlist(watchlistId);
      setWatchlists(watchlists.map(w => w.id === watchlistId ? updatedWatchlist.data : w));
      setSuccessMessage('Stock removed from watchlist successfully');
      setError(null);
    } catch (err) {
      setError(handleApiError(err));
    }
  };

  const clearMessages = () => {
    setError(null);
    setSuccessMessage(null);
  };

  useEffect(() => {
    if (error || successMessage) {
      const timer = setTimeout(clearMessages, 5000);
      return () => clearTimeout(timer);
    }
  }, [error, successMessage]);

  return (
    <div className="watchlist container mt-5">
      <h2 className="mb-4">Watchlists</h2>
      {error && <div className="alert alert-danger">{error}</div>}
      {successMessage && <div className="alert alert-success">{successMessage}</div>}
      
      <div className="row">
        <div className="col-md-6">
          <h3>Create New Watchlist</h3>
          <form onSubmit={handleCreateWatchlist}>
            <div className="mb-3">
              <input
                type="text"
                className="form-control"
                value={newWatchlistName}
                onChange={(e) => setNewWatchlistName(e.target.value)}
                placeholder="Watchlist Name"
                required
              />
            </div>
            <button type="submit" className="btn btn-primary">Create Watchlist</button>
          </form>
        </div>
        
        <div className="col-md-6">
          <h3>Add Stock to Watchlist</h3>
          <form onSubmit={handleAddStock}>
            <div className="mb-3">
              <select
                className="form-control"
                value={selectedWatchlist ? selectedWatchlist.id : ''}
                onChange={(e) => setSelectedWatchlist(watchlists.find(w => w.id === e.target.value))}
                required
              >
                <option value="">Select Watchlist</option>
                {watchlists.map(watchlist => (
                  <option key={watchlist.id} value={watchlist.id}>{watchlist.name}</option>
                ))}
              </select>
            </div>
            <div className="mb-3">
              <input
                type="text"
                className="form-control"
                value={newStockSymbol}
                onChange={(e) => setNewStockSymbol(e.target.value)}
                placeholder="Stock Symbol"
                required
              />
            </div>
            <button type="submit" className="btn btn-primary">Add Stock</button>
          </form>
        </div>
      </div>

      <h3 className="mt-5">Your Watchlists</h3>
      {watchlists.map(watchlist => (
        <div key={watchlist.id} className="card mb-3">
          <div className="card-header">{watchlist.name}</div>
          <div className="card-body">
            <ul className="list-group">
              {watchlist.stocks.map(stock => (
                <li key={stock.symbol} className="list-group-item d-flex justify-content-between align-items-center">
                  {stock.symbol}
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => handleRemoveStock(watchlist.id, stock.symbol)}
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      ))}
    </div>
  );
};

export default Watchlist;