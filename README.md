# Stock Spike Replicator

A Python-based tool for analyzing stocks using technical indicators and Google Sheets integration.

## Features

- Fetches stock data from Yahoo Finance
- Calculates technical indicators (RSI, Bollinger Bands, MACD, etc.)
- Integrates with Google Sheets for data visualization
- Identifies potential stock opportunities based on technical analysis
- Includes AI-powered stock recommendations (OpenAI and Gemini)

## Main Components

- `updatesheet.py`: Updates Google Sheets with stock data and technical indicators
- `test.py`: Tests API connectivity and runs stock scans

## Requirements

- Python 3.6+
- Required packages: gspread, google-auth, yfinance, pandas, numpy, requests

## Setup

1. Install required packages:
   ```
   pip install gspread google-auth yfinance pandas numpy requests
   ```

2. Set up Google Sheets API credentials:
   - Create a service account in Google Cloud Console
   - Download credentials as JSON file
   - Save as `credentials.json` in the project directory

3. Update the `SPREADSHEET_ID` in `updatesheet.py` to your Google Sheet ID

## Usage

Run the main update script:
```
python updatesheet.py
```

Run the test script:
```
python test.py
```

## Output Files

- `oversold_stocks.csv`: List of oversold stocks (RSI < 30)
- `squeeze_stocks.csv`: Stocks showing Bollinger Band squeeze pattern
- `top_candidates.csv`: Top stock candidates based on multiple criteria
