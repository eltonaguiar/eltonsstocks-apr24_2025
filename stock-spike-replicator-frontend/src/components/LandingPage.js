import React from 'react';
import { Link } from 'react-router-dom';

const LandingPage = () => {
  return (
    <div className="landing-page">
      <h1>Welcome to Stock Spike Replicator</h1>
      <p>Discover and replicate stock market spikes with our advanced backtesting tools.</p>
      <div className="cta-buttons">
        <Link to="/login" className="btn btn-primary">Login</Link>
        <Link to="/register" className="btn btn-secondary">Register</Link>
      </div>
      <div className="features">
        <h2>Key Features</h2>
        <ul>
          <li>Advanced backtesting algorithms</li>
          <li>Real-time market data analysis</li>
          <li>Customizable trading strategies</li>
          <li>Comprehensive performance reports</li>
        </ul>
      </div>
    </div>
  );
};

export default LandingPage;