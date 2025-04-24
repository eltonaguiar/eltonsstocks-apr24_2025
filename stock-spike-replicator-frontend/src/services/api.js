import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add a response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Handle unauthorized access (e.g., redirect to login)
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth
export const login = (credentials) => api.post('/auth/login', credentials);
export const register = (userData) => api.post('/auth/register', userData);

// User
export const getUserProfile = () => api.get('/user/profile');
export const updateUserProfile = (userData) => api.put('/user/profile', userData);

// Stocks
export const getStockData = (symbol) => api.get(`/stocks/${symbol}`);
export const searchStocks = (query) => api.get(`/stocks/search?q=${query}`);

// Backtest
export const runBacktest = (backtestData) => api.post('/backtest', backtestData);
export const getBacktestResults = (backtestId) => api.get(`/backtest/${backtestId}`);

// Watchlists
export const getWatchlists = () => api.get('/watchlists');
export const createWatchlist = (watchlistData) => api.post('/watchlists', watchlistData);
export const updateWatchlist = (watchlistId, watchlistData) => api.put(`/watchlists/${watchlistId}`, watchlistData);
export const deleteWatchlist = (watchlistId) => api.delete(`/watchlists/${watchlistId}`);
export const addToWatchlist = (watchlistId, stockSymbol) => api.post(`/watchlists/${watchlistId}/stocks/${stockSymbol}`);
export const removeFromWatchlist = (watchlistId, stockSymbol) => api.delete(`/watchlists/${watchlistId}/stocks/${stockSymbol}`);
export const getWatchlist = (watchlistId) => api.get(`/watchlists/${watchlistId}`);

export default api;