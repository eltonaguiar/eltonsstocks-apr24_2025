from sqlalchemy import Column, Integer, String, ForeignKey, ARRAY
from sqlalchemy.orm import relationship
from pydantic import BaseModel

from api.core.database import Base

class Watchlist(Base):
    __tablename__ = "watchlists"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    stocks = Column(ARRAY(String))

    user = relationship("User", back_populates="watchlists")

class WatchlistCreate(BaseModel):
    name: str
    stocks: list[str] = []

class WatchlistUpdate(BaseModel):
    name: str
    stocks: list[str]

class WatchlistInDB(BaseModel):
    id: int
    name: str
    user_id: int
    stocks: list[str]

    class Config:
        orm_mode = True