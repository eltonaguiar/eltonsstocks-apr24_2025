from sqlalchemy import Column, Integer, String, JSON
from sqlalchemy.orm import relationship
from pydantic import BaseModel, EmailStr

from api.core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    preferences = Column(JSON)

    watchlists = relationship("Watchlist", back_populates="user")

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserUpdate(BaseModel):
    full_name: str

class UserPreferences(BaseModel):
    theme: str = "light"
    notification_enabled: bool = True
    default_chart_type: str = "candlestick"

class UserInDB(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    preferences: dict = {}

    class Config:
        orm_mode = True