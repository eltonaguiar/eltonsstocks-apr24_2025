from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict

from api.core.database import get_db
from api.core.security import get_current_user
from api.models.user import User

# Import the necessary functions from existing modules
from data_fetchers import search_stocks as fetch_stocks, get_stock_data as fetch_stock_data, fetch_news_sentiment
from scoring import run_statistical_arbitrage

router = APIRouter()

@router.get("/search")
async def search_stocks(
    query: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        results = fetch_stocks(query)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while searching for stocks: {str(e)}"
        )

@router.get("/{symbol}/data")
async def get_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        data = fetch_stock_data(symbol, start_date, end_date)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching stock data: {str(e)}"
        )

@router.get("/{symbol}/sentiment")
async def get_stock_sentiment(
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        sentiment_data = fetch_news_sentiment(symbol)
        return sentiment_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching sentiment data: {str(e)}"
        )

@router.post("/statistical-arbitrage")
async def perform_statistical_arbitrage(
    symbols: List[str],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        price_data = {}
        for symbol in symbols:
            data = fetch_stock_data(symbol, "2023-01-01", "2023-12-31")  # Adjust date range as needed
            price_data[symbol] = data['close'].values  # Assuming 'close' is the column name for closing prices
        
        stat_arb_results = run_statistical_arbitrage(price_data)
        return stat_arb_results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while performing statistical arbitrage: {str(e)}"
        )

# Update existing endpoints to include new features
@router.get("/{symbol}/data")
async def get_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    include_sentiment: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        data = fetch_stock_data(symbol, start_date, end_date)
        if include_sentiment:
            sentiment_data = fetch_news_sentiment(symbol)
            data['sentiment'] = sentiment_data
        return data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching stock data: {str(e)}"
        )

# TODO: Implement additional stock-related endpoints as needed
# For example:
# - Get technical indicators
# - Get company information