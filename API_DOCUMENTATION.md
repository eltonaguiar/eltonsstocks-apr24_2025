# Stock Spike Replicator API Documentation

This document provides detailed information about the Stock Spike Replicator API endpoints, including their parameters and expected responses.

## Base URL

All API requests should be made to: `https://api.stockspikereplicator.com/v1`

## Authentication

Most API endpoints require authentication. Include the JWT token in the Authorization header:

```
Authorization: Bearer YOUR_JWT_TOKEN
```

## Endpoints

### Authentication

#### POST /auth/signup

Create a new user account.

**Parameters:**
- `email` (string, required): User's email address
- `password` (string, required): User's password (min 8 characters)
- `full_name` (string, required): User's full name

**Response:**
```json
{
  "id": "user_id",
  "email": "user@example.com",
  "full_name": "John Doe"
}
```

#### POST /auth/login

Authenticate a user and receive a JWT token.

**Parameters:**
- `email` (string, required): User's email address
- `password` (string, required): User's password

**Response:**
```json
{
  "access_token": "JWT_TOKEN",
  "token_type": "bearer"
}
```

### User Management

#### GET /users/me

Get the current user's profile information.

**Response:**
```json
{
  "id": "user_id",
  "email": "user@example.com",
  "full_name": "John Doe",
  "created_at": "2023-04-24T00:00:00Z"
}
```

#### PUT /users/me

Update the current user's profile information.

**Parameters:**
- `full_name` (string, optional): New full name
- `password` (string, optional): New password

**Response:**
```json
{
  "id": "user_id",
  "email": "user@example.com",
  "full_name": "John Doe",
  "updated_at": "2023-04-24T00:00:00Z"
}
```

### Stocks

#### GET /stocks/search

Search for stocks by symbol or name.

**Parameters:**
- `query` (string, required): Search query
- `limit` (integer, optional, default=10): Maximum number of results to return

**Response:**
```json
[
  {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "exchange": "NASDAQ"
  },
  {
    "symbol": "GOOGL",
    "name": "Alphabet Inc.",
    "exchange": "NASDAQ"
  }
]
```

#### GET /stocks/{symbol}

Get detailed information for a specific stock.

**Parameters:**
- `symbol` (string, required): Stock symbol

**Response:**
```json
{
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "exchange": "NASDAQ",
  "current_price": 150.25,
  "change_percent": 1.5,
  "market_cap": 2500000000000,
  "volume": 50000000,
  "pe_ratio": 30.5
}
```

### Watchlists

#### GET /watchlists

Get all watchlists for the current user.

**Response:**
```json
[
  {
    "id": "watchlist_id_1",
    "name": "Tech Stocks",
    "description": "Top technology companies",
    "created_at": "2023-04-24T00:00:00Z",
    "stock_count": 5
  },
  {
    "id": "watchlist_id_2",
    "name": "Energy Sector",
    "description": "Oil and renewable energy stocks",
    "created_at": "2023-04-25T00:00:00Z",
    "stock_count": 8
  }
]
```

#### POST /watchlists

Create a new watchlist.

**Parameters:**
- `name` (string, required): Watchlist name
- `description` (string, optional): Watchlist description
- `symbols` (array of strings, required): List of stock symbols to add

**Response:**
```json
{
  "id": "new_watchlist_id",
  "name": "My New Watchlist",
  "description": "Description of my new watchlist",
  "created_at": "2023-04-26T00:00:00Z",
  "stock_count": 3
}
```

#### GET /watchlists/{watchlist_id}

Get detailed information for a specific watchlist.

**Parameters:**
- `watchlist_id` (string, required): Watchlist ID

**Response:**
```json
{
  "id": "watchlist_id",
  "name": "Tech Stocks",
  "description": "Top technology companies",
  "created_at": "2023-04-24T00:00:00Z",
  "stocks": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "current_price": 150.25,
      "change_percent": 1.5
    },
    {
      "symbol": "GOOGL",
      "name": "Alphabet Inc.",
      "current_price": 2800.75,
      "change_percent": -0.5
    }
  ]
}
```

#### PUT /watchlists/{watchlist_id}

Update a watchlist.

**Parameters:**
- `watchlist_id` (string, required): Watchlist ID
- `name` (string, optional): New watchlist name
- `description` (string, optional): New watchlist description
- `symbols` (array of strings, optional): Updated list of stock symbols

**Response:**
```json
{
  "id": "watchlist_id",
  "name": "Updated Watchlist Name",
  "description": "Updated description",
  "updated_at": "2023-04-26T00:00:00Z",
  "stock_count": 4
}
```

#### DELETE /watchlists/{watchlist_id}

Delete a watchlist.

**Parameters:**
- `watchlist_id` (string, required): Watchlist ID

**Response:**
```json
{
  "message": "Watchlist deleted successfully"
}
```

### Backtesting

#### POST /backtest

Run a new backtest.

**Parameters:**
- `watchlist_id` (string, optional): Watchlist ID to use for backtesting
- `symbols` (array of strings, optional): List of stock symbols (if not using a watchlist)
- `start_date` (string, required): Start date for backtesting (YYYY-MM-DD)
- `end_date` (string, required): End date for backtesting (YYYY-MM-DD)
- `initial_capital` (number, required): Initial capital for backtesting
- `strategy` (object, required): Strategy parameters (varies based on strategy type)

**Response:**
```json
{
  "backtest_id": "backtest_id",
  "status": "running",
  "estimated_completion_time": "2023-04-26T00:10:00Z"
}
```

#### GET /backtest/{backtest_id}

Get the results of a backtest.

**Parameters:**
- `backtest_id` (string, required): Backtest ID

**Response:**
```json
{
  "id": "backtest_id",
  "status": "completed",
  "start_date": "2023-01-01",
  "end_date": "2023-04-01",
  "initial_capital": 100000,
  "final_portfolio_value": 125000,
  "total_return": 25,
  "sharpe_ratio": 1.5,
  "max_drawdown": 10,
  "trades": [
    {
      "symbol": "AAPL",
      "entry_date": "2023-01-15",
      "entry_price": 150,
      "exit_date": "2023-02-28",
      "exit_price": 165,
      "profit_loss": 10
    }
  ]
}
```

### Alerts

#### POST /alerts

Create a new alert.

**Parameters:**
- `symbol` (string, required): Stock symbol
- `condition` (string, required): Alert condition (e.g., "price_above", "price_below")
- `value` (number, required): Threshold value for the alert
- `notification_method` (string, required): Notification method ("email", "sms", or "in_app")

**Response:**
```json
{
  "id": "alert_id",
  "symbol": "AAPL",
  "condition": "price_above",
  "value": 160,
  "notification_method": "email",
  "created_at": "2023-04-26T00:00:00Z"
}
```

#### GET /alerts

Get all alerts for the current user.

**Response:**
```json
[
  {
    "id": "alert_id_1",
    "symbol": "AAPL",
    "condition": "price_above",
    "value": 160,
    "notification_method": "email",
    "created_at": "2023-04-26T00:00:00Z"
  },
  {
    "id": "alert_id_2",
    "symbol": "GOOGL",
    "condition": "price_below",
    "value": 2700,
    "notification_method": "sms",
    "created_at": "2023-04-25T00:00:00Z"
  }
]
```

#### DELETE /alerts/{alert_id}

Delete an alert.

**Parameters:**
- `alert_id` (string, required): Alert ID

**Response:**
```json
{
  "message": "Alert deleted successfully"
}
```

## Error Handling

The API uses conventional HTTP response codes to indicate the success or failure of an API request. In general:

- 2xx codes indicate success
- 4xx codes indicate an error that failed given the information provided (e.g., a required parameter was omitted)
- 5xx codes indicate an error with our servers

All error responses will include a JSON object with an `error` key containing a human-readable error message.

Example error response:

```json
{
  "error": "Invalid API key provided"
}
```

## Rate Limiting

API requests are rate limited to prevent abuse. The current rate limit is 100 requests per minute per API key. If you exceed the rate limit, you'll receive a 429 Too Many Requests response.

## Versioning

The current API version is v1. We recommend specifying a version in the URL to ensure compatibility with your integration.

For any questions or issues regarding the API, please contact our support team at api-support@stockspikereplicator.com.