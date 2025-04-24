import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate, Link } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Dashboard from './components/Dashboard';
import Backtesting from './components/Backtesting';
import Results from './components/Results';
import Watchlist from './components/Watchlist';
import Login from './components/Auth/Login';
import Register from './components/Auth/Register';
import './App.css';

function Navigation() {
  const { isAuthenticated, logout } = useAuth();

  return (
    <nav>
      <Link to="/dashboard">Dashboard</Link>
      <Link to="/backtesting">Backtesting</Link>
      <Link to="/results">Results</Link>
      <Link to="/watchlist">Watchlist</Link>
      {isAuthenticated ? (
        <button onClick={logout}>Logout</button>
      ) : (
        <>
          <Link to="/login">Login</Link>
          <Link to="/register">Register</Link>
        </>
      )}
    </nav>
  );
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="App">
          <header className="App-header">
            <h1>Stock Spike Replicator</h1>
          </header>
          <Navigation />
          <main>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              <Route element={<ProtectedRoute />}>
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/backtesting" element={<Backtesting />} />
                <Route path="/results" element={<Results />} />
                <Route path="/watchlist" element={<Watchlist />} />
              </Route>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </main>
          <footer>
            <p>&copy; 2025 Stock Spike Replicator</p>
          </footer>
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;
