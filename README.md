# Stock Spike Replicator

## Overview

The Stock Spike Replicator is an advanced algorithmic trading system that uses machine learning techniques to analyze historical stock data, predict future performance, and provide stock recommendations. It incorporates enhanced backtesting, risk management, and market regime detection to optimize trading strategies. The system offers both a Google Sheets integration and a web-based user interface with a RESTful API for improved accessibility and integration.

## Features

- Machine learning-based stock analysis and prediction
- Advanced algorithmic techniques including:
  - Markov Models and hidden Markov models for state prediction
  - Statistical arbitrage for identifying market inefficiencies
  - Sentiment analysis and Natural Language Processing (NLP) for news and social media impact
- Enhanced backtesting with transaction costs and slippage simulation
- Risk management with position sizing and stop-loss mechanisms
- Market regime detection for adaptive trading strategies
- Multiple performance metrics for comprehensive strategy evaluation
- Integration with Google Sheets for easy data input and result visualization
- Automatic processing of user-provided symbols and cheap stocks under $1
- Web-based user interface for easy interaction
- RESTful API for programmatic access and integration
- Watchlist management for tracking favorite stocks
- Real-time stock data updates and alerts

## System Requirements

- Operating System: Windows 10 or later, macOS 10.14+, or Linux (Ubuntu 18.04+)
- CPU: Dual-core processor or better
- RAM: 8GB minimum, 16GB recommended
- Storage: 1GB of free disk space
- Internet Connection: Broadband internet connection for real-time data updates

## Software Requirements

- Python 3.7+
- Node.js 14+ and npm (for the web interface)
- PostgreSQL 12+ (for the database)

## Installation

### Backend Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock-spike-replicator.git
   cd stock-spike-replicator
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up Google Sheets API credentials (for Google Sheets integration):
   - Follow the [Google Sheets API Python Quickstart](https://developers.google.com/sheets/api/quickstart/python) to create a project and enable the API
   - Download the client configuration file and rename it to `credentials.json`
   - Place `credentials.json` in the project root directory

5. Set up the database:
   - Create a PostgreSQL database for the project
   - Update the `DATABASE_URL` in `api/core/config.py` with your database connection string
   - Run database migrations:
     ```
     alembic upgrade head
     ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd stock-spike-replicator-frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

## Usage

The Stock Spike Replicator offers two main ways to use the system: Google Sheets Integration and Web Interface. Before using either option, ensure that you have activated your virtual environment and installed all required dependencies.

### Activating the Virtual Environment and Installing Dependencies

1. Create and activate your virtual environment:

```
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS and Linux
python3 -m venv venv
source venv/bin/activate
```

2. Install the required packages:

```
pip install -r requirements.txt
```

### Option 1: Google Sheets Integration

1. Set up Google Sheets API credentials:
   - Follow the [Google Sheets API Python Quickstart](https://developers.google.com/sheets/api/quickstart/python) to create a project and enable the API
   - Download the client configuration file and rename it to `credentials.json`
   - Place `credentials.json` in the project root directory

2. Configure the `config.py` file with your Google Sheets information and other settings:
   - Set `SPREADSHEET_ID` to your Google Sheets document ID
   - Set `MAIN_SHEET` to the name of your main worksheet
   - Set `SERVICE_ACCOUNT_FILE` to the path of your `credentials.json` file

3. Add the stock symbols you want to analyze in the 'A' column of the main sheet in your Google Sheets document.

4. Ensure your virtual environment is activated, then run the main script:
   ```
   # On Windows
   python main.py

   # On macOS and Linux
   python3 main.py
   ```

   Note: The script uses asyncio, so make sure you're using a compatible Python version (3.7+).

5. Check the console output for the top 5 stock recommendations and further instructions.

6. View the full results in the connected Google Sheet.

7. If you encounter any issues related to asyncio or event loops, try running the script with the following command:
   ```
   # On Windows
   python -m asyncio main.py

   # On macOS and Linux
   python3 -m asyncio main.py
   ```

### Option 2: Web Interface (Recommended)

1. Start the backend server:
   ```
   cd stock-spike-replicator
   python -m uvicorn main:app --reload
   ```

2. In a new terminal, start the frontend development server:
   ```
   cd stock-spike-replicator-frontend
   npm start
   ```

3. Open your web browser and navigate to `http://localhost:3000` to access the web interface.

4. Use the following default credentials to log in:
   - Username: admin
   - Password: admin
   - Email: admin@admin.com

   Note: For security reasons, please change these default credentials after your first login.

5. Use the interactive dashboard to:
   - Manage your account and user profile
   - Create and manage watchlists of your favorite stocks
   - Run backtests on selected stocks or watchlists
   - View and analyze backtest results
   - Monitor real-time stock data and receive alerts
   - Access historical performance data and analytics

### Troubleshooting

If you encounter issues when running the script:

1. Ensure your virtual environment is activated and all dependencies are installed:
   ```
   pip install -r requirements.txt
   ```

2. If you get a `ModuleNotFoundError`, try installing the specific module:
   ```
   pip install <module_name>
   ```

3. For Google Sheets integration issues:
   - Verify that your `credentials.json` file is in the correct location and properly formatted
   - Ensure that the Google Sheets API is enabled for your project
   - Check that the `SPREADSHEET_ID`, `MAIN_SHEET`, and `SERVICE_ACCOUNT_FILE` in `config.py` are correctly set

4. For database-related issues, ensure that your database is properly set up and the `DATABASE_URL` in `api/core/config.py` is correct.

5. If the web interface is not loading, check that both the backend and frontend servers are running and that there are no console errors in your browser's developer tools.

6. If you're still having issues, check your Python path:
   ```
   python -c "import sys; print(sys.path)"
   ```
   Ensure that your virtual environment's site-packages directory is in the path.

7. For any other issues, please check the error messages in the console and refer to the project's documentation or seek help in the project's support channels.


### API

The Stock Spike Replicator provides a RESTful API for programmatic access. Here are some key endpoints:

- `/api/auth`: User authentication (signup, login, logout)
- `/api/user`: User profile management
- `/api/stocks`: Stock data and search
- `/api/backtest`: Run and manage backtests
- `/api/watchlists`: Manage stock watchlists

For full API documentation, visit `http://localhost:8000/docs` when the backend server is running.

## Configuration

Edit the `api/core/config.py` file to customize the following settings:

- `PROJECT_NAME`: Name of the project
- `PROJECT_VERSION`: Current version of the project
- `ALLOWED_ORIGINS`: List of allowed origins for CORS
- `DATABASE_URL`: URL for the database connection
- `SECRET_KEY`: Secret key for JWT token generation
- `ALGORITHM`: Algorithm used for JWT token generation
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Expiration time for access tokens

Additional settings from the original `config.py` file:

- `SPREADSHEET_ID`: ID of your Google Sheets document
- `MAIN_SHEET`: Name of the main worksheet
- `SERVICE_ACCOUNT_FILE`: Path to your Google Sheets API credentials file
- `TRANSACTION_COST`: Transaction cost per trade
- `SLIPPAGE`: Slippage percentage
- `BACKTEST_START_DATE`: Start date for backtesting
- `BACKTEST_END_DATE`: End date for backtesting
- `INITIAL_CAPITAL`: Initial capital for backtesting

## Algorithm Details

The Stock Spike Replicator employs a sophisticated ensemble of algorithms and techniques to analyze and predict stock performance:

1. Machine Learning Models: Utilizes various ML models including Random Forests, Gradient Boosting, and Neural Networks to predict stock price movements.

2. Markov Models and Hidden Markov Models: These are used to model and predict market states and regime changes, allowing the system to adapt its strategies based on current market conditions.

3. Statistical Arbitrage: Implements statistical arbitrage techniques to identify and exploit temporary mispricings in related securities.

4. Sentiment Analysis and NLP: Incorporates natural language processing to analyze news articles, social media posts, and other textual data sources to gauge market sentiment and its potential impact on stock prices.

5. Technical Indicators: Employs a wide range of technical indicators and oscillators to identify potential entry and exit points.

6. Fundamental Analysis: Considers key financial metrics and ratios to assess the underlying value and health of companies.

7. Risk Management: Implements advanced risk management techniques including dynamic position sizing and adaptive stop-loss mechanisms.

The exact combination and weighting of these techniques are continuously optimized based on market conditions and backtesting results.

## Performance Metrics

The system evaluates its performance using a comprehensive set of metrics:

- Total Return: Overall profitability of the strategy
- Sharpe Ratio: Risk-adjusted return metric
- Maximum Drawdown: Largest peak-to-trough decline
- Win Rate: Percentage of profitable trades
- Profit Factor: Ratio of gross profits to gross losses
- Alpha: Excess return compared to a benchmark index
- Beta: Measure of systematic risk relative to the market
- Information Ratio: Risk-adjusted excess returns relative to a benchmark
- Sortino Ratio: Variation of Sharpe ratio that only considers downside risk
- Calmar Ratio: Ratio of average annual return to maximum drawdown

These metrics are calculated and monitored in real-time to ensure the system's performance remains robust across various market conditions.

## Troubleshooting

- If you encounter issues with Google Sheets authentication, ensure that your `credentials.json` file is correctly set up and the API is enabled for your project.
- For any "ModuleNotFoundError", make sure all required packages are installed using `pip install -r requirements.txt`.
- If you experience rate limiting issues with data fetching, adjust the `RATE_LIMIT` parameter in `config.py`.
- For database-related issues, ensure that your database is properly set up and the `DATABASE_URL` in `api/core/config.py` is correct.
- If the web interface is not loading, check that both the backend and frontend servers are running and that there are no console errors in your browser's developer tools.

## Contributing

Contributions to the Stock Spike Replicator are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Implement your feature or bug fix
4. Add or update tests as necessary
5. Submit a pull request with a clear description of your changes

For more detailed guidelines, please refer to the CONTRIBUTING.md file.

## Post-Deployment Monitoring and Continuous Improvement

We have implemented a comprehensive post-deployment monitoring and continuous improvement process to ensure the Stock Spike Replicator remains effective and adapts to changing market conditions. This includes:

- Real-time performance monitoring
- Automated alerts for anomalies or performance degradation
- Regular backtesting and strategy optimization
- Continuous integration of new data sources and algorithmic improvements

For more details on our post-deployment processes, please refer to the [POST_DEPLOYMENT_GUIDE.md](POST_DEPLOYMENT_GUIDE.md) file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. It is not intended to be used as financial advice or a recommendation to buy or sell any securities. The use of advanced algorithmic trading techniques, including those rumored to be used by Renaissance Technologies, does not guarantee profits and may involve significant risks. Always do your own research and consult with a licensed financial advisor before making any investment decisions.
