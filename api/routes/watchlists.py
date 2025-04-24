from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from api.core.database import get_db
from api.core.security import get_current_user
from api.models.user import User
from api.models.watchlist import Watchlist, WatchlistCreate, WatchlistUpdate

router = APIRouter()

@router.get("/")
async def get_watchlists(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    watchlists = db.query(Watchlist).filter(Watchlist.user_id == current_user.id).all()
    return watchlists

@router.post("/")
async def create_watchlist(
    watchlist_data: WatchlistCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    new_watchlist = Watchlist(**watchlist_data.dict(), user_id=current_user.id)
    db.add(new_watchlist)
    db.commit()
    db.refresh(new_watchlist)
    return new_watchlist

@router.get("/{watchlist_id}")
async def get_watchlist(
    watchlist_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    watchlist = db.query(Watchlist).filter(Watchlist.id == watchlist_id, Watchlist.user_id == current_user.id).first()
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    return watchlist

@router.put("/{watchlist_id}")
async def update_watchlist(
    watchlist_id: int,
    watchlist_data: WatchlistUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    watchlist = db.query(Watchlist).filter(Watchlist.id == watchlist_id, Watchlist.user_id == current_user.id).first()
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    
    for key, value in watchlist_data.dict(exclude_unset=True).items():
        setattr(watchlist, key, value)
    
    db.commit()
    db.refresh(watchlist)
    return watchlist

@router.delete("/{watchlist_id}")
async def delete_watchlist(
    watchlist_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    watchlist = db.query(Watchlist).filter(Watchlist.id == watchlist_id, Watchlist.user_id == current_user.id).first()
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    
    db.delete(watchlist)
    db.commit()
    return {"message": "Watchlist deleted successfully"}

@router.post("/{watchlist_id}/stocks/{stock_symbol}")
async def add_stock_to_watchlist(
    watchlist_id: int,
    stock_symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    watchlist = db.query(Watchlist).filter(Watchlist.id == watchlist_id, Watchlist.user_id == current_user.id).first()
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    
    # TODO: Add validation for stock symbol existence and availability
    # This is a placeholder for actual stock validation logic
    if not is_valid_stock_symbol(stock_symbol):
        raise HTTPException(status_code=400, detail="Invalid or unavailable stock symbol")
    
    if stock_symbol not in watchlist.stocks:
        watchlist.stocks.append(stock_symbol)
        db.commit()
        db.refresh(watchlist)
    
    return watchlist

@router.delete("/{watchlist_id}/stocks/{stock_symbol}")
async def remove_stock_from_watchlist(
    watchlist_id: int,
    stock_symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    watchlist = db.query(Watchlist).filter(Watchlist.id == watchlist_id, Watchlist.user_id == current_user.id).first()
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    
    if stock_symbol in watchlist.stocks:
        watchlist.stocks.remove(stock_symbol)
        db.commit()
        db.refresh(watchlist)
    else:
        raise HTTPException(status_code=404, detail="Stock not found in watchlist")
    
    return watchlist

def is_valid_stock_symbol(symbol: str) -> bool:
    # TODO: Implement actual stock symbol validation
    # This could involve checking against a list of valid symbols or querying an external API
    return True  # Placeholder return, always considering the symbol as valid for now