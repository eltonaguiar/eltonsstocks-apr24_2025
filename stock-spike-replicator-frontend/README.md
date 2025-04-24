# Stock Spike Replicator Frontend

This is the frontend application for the Stock Spike Replicator project. It provides a user interface for backtesting stock trading strategies and analyzing results.

## Prerequisites

- Node.js (v14 or later)
- npm (v6 or later)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/stock-spike-replicator.git
   cd stock-spike-replicator-frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

## Configuration

Create a `.env` file in the root directory of the project and add the following environment variables:

```
REACT_APP_API_BASE_URL=http://localhost:5000/api
```

Replace the URL with the appropriate backend API URL if different.

## Running the Application

To start the development server:

```
npm start
```

The application will be available at `http://localhost:3000`.

## Building for Production

To create a production build:

```
npm run build
```

The built files will be in the `build` directory.

## Features

- User authentication (login/register)
- Dashboard with quick access to main features
- Backtesting form for running trading strategy simulations
- Results page for viewing backtest outcomes
- User profile management

## Usage

1. Register a new account or log in with existing credentials.
2. Use the dashboard to navigate to different features.
3. To run a backtest:
   - Go to the Backtesting page
   - Enter the required parameters (stock symbol, date range, initial capital, etc.)
   - Select a trading strategy
   - Click "Run Backtest"
4. View the results on the Results page
5. Update your profile information on the User Profile page

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
