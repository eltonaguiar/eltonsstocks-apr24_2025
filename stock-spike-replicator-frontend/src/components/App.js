import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import { AppProvider } from '../context/AppContext';
import Navigation from './Navigation';
import LandingPage from './LandingPage';
import Login from './Auth/Login';
import Register from './Auth/Register';
import Dashboard from './Dashboard';
import Backtesting from './Backtesting';
import Results from './Results';
import UserProfile from './UserProfile';
import ProtectedRoute from './ProtectedRoute';
import '../App.css';

function App() {
  return (
    <AppProvider>
      <Router>
        <div className="App">
          <header className="App-header">
            <h1>Stock Spike Replicator</h1>
            <Navigation />
          </header>
          <main>
            <Switch>
              <Route exact path="/" component={LandingPage} />
              <Route path="/login" component={Login} />
              <Route path="/register" component={Register} />
              <ProtectedRoute path="/dashboard" component={Dashboard} />
              <ProtectedRoute path="/backtesting" component={Backtesting} />
              <ProtectedRoute path="/results" component={Results} />
              <ProtectedRoute path="/profile" component={UserProfile} />
            </Switch>
          </main>
          <footer>
            <p>&copy; 2025 Stock Spike Replicator</p>
          </footer>
        </div>
      </Router>
    </AppProvider>
  );
}

export default App;
