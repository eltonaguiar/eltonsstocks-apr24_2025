# Backend Changes for Stock Spike Replicator Web UI

To support the new web-based user interface, we need to make several changes to the existing Python backend. These changes will involve creating new API endpoints, implementing authentication, and modifying existing functionality to work with the new frontend. We'll use FastAPI as our web framework due to its high performance and ease of use with async operations.

## 1. Project Structure Changes

1. Create a new `api` directory in the project root to house all API-related code.
2. Move relevant existing Python files into appropriate subdirectories within the `api` directory.
3. Create new files for API routes, authentication, and database models.

New project structure:

```
stock_spike_replicator/
├── api/
│   ├── routes/
│   │   ├── auth.py
│   │   ├── user.py
│   │   ├── stocks.py
│   │   ├── backtest.py
│   │   └── watchlists.py
│   ├── models/
│   │   ├── user.py
│   │   └── watchlist.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── database.py
│   └── main.py
├── ml_backtesting.py
├── data_fetchers.py
├── technical_indicators.py
├── scoring.py
├── visualizations.py
└── requirements.txt
```

## 2. New Dependencies

Add the following dependencies to `requirements.txt`:

```
fastapi
uvicorn
sqlalchemy
pydantic
python-jose[cryptography]
passlib[bcrypt]
python-multipart
```

## 3. API Implementation

### 3.1 Main FastAPI App (api/main.py)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.core.config import settings
from api.routes import auth, user, stocks, backtest, watchlists

app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(user.router, prefix="/api/user", tags=["user"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(watchlists.router, prefix="/api/watchlists", tags=["watchlists"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3.2 Authentication (api/routes/auth.py)

```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from api.core.security import create_access_token, get_password_hash, verify_password
from api.core.database import get_db
from api.models.user import User

router = APIRouter()

@router.post("/signup")
async def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    # Create new user
    # Return user data (excluding password)

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Authenticate user
    # Create access token
    # Return token

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    # Implement logout logic (e.g., token blacklisting if necessary)
    # Return success message
```

### 3.3 User Management (api/routes/user.py)

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api.core.database import get_db
from api.core.security import get_current_user
from api.models.user import User

router = APIRouter()

@router.get("/profile")
async def get_profile(current_user: User = Depends(get_current_user)):
    # Return user profile data

@router.put("/profile")
async def update_profile(user_data: UserUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Update user profile
    # Return updated user data

@router.get("/preferences")
async def get_preferences(current_user: User = Depends(get_current_user)):
    # Return user preferences

@router.put("/preferences")
async def update_preferences(preferences: UserPreferences, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Update user preferences
    # Return updated preferences
```

### 3.4 Stocks (api/routes/stocks.py)

```python
from fastapi import APIRouter, Depends
from api.core.security import get_current_user
from api.models.user import User

router = APIRouter()

@router.get("/search")
async def search_stocks(query: str, current_user: User = Depends(get_current_user)):
    # Implement stock search logic
    # Return matching stock symbols and names

@router.get("/{symbol}/data")
async def get_stock_data(symbol: str, start_date: str, end_date: str, current_user: User = Depends(get_current_user)):
    # Fetch stock data for the given symbol and date range
    # Return historical price data and any calculated indicators
```

### 3.5 Backtesting (api/routes/backtest.py)

```python
from fastapi import APIRouter, Depends, BackgroundTasks
from api.core.security import get_current_user
from api.models.user import User

router = APIRouter()

@router.post("/")
async def start_backtest(backtest_data: BacktestInput, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    # Validate input data
    # Start backtest as a background task
    # Return backtest ID for status tracking

@router.get("/{backtest_id}/status")
async def get_backtest_status(backtest_id: str, current_user: User = Depends(get_current_user)):
    # Check and return the status of the backtest

@router.get("/{backtest_id}/results")
async def get_backtest_results(backtest_id: str, current_user: User = Depends(get_current_user)):
    # Retrieve and return backtest results
```

### 3.6 Watchlists (api/routes/watchlists.py)

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api.core.database import get_db
from api.core.security import get_current_user
from api.models.user import User
from api.models.watchlist import Watchlist

router = APIRouter()

@router.get("/")
async def get_watchlists(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Retrieve user's watchlists
    # Return list of watchlists

@router.post("/")
async def create_watchlist(watchlist_data: WatchlistCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Create new watchlist
    # Return created watchlist

@router.put("/{watchlist_id}")
async def update_watchlist(watchlist_id: int, watchlist_data: WatchlistUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Update existing watchlist
    # Return updated watchlist

@router.delete("/{watchlist_id}")
async def delete_watchlist(watchlist_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Delete watchlist
    # Return success message
```

## 4. Database Models (api/models/)

Create SQLAlchemy models for User and Watchlist in separate files within the `api/models/` directory.

## 5. Core Functionality (api/core/)

### 5.1 Configuration (api/core/config.py)

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Stock Spike Replicator"
    PROJECT_VERSION: str = "1.0.0"
    ALLOWED_ORIGINS: list = ["http://localhost:3000"]  # Update with actual frontend URL
    DATABASE_URL: str = "sqlite:///./stock_spike_replicator.db"
    SECRET_KEY: str = "your-secret-key"  # Change this to a secure random string
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

settings = Settings()
```

### 5.2 Security (api/core/security.py)

```python
from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
from api.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# Implement get_current_user function for dependency injection
```

### 5.3 Database (api/core/database.py)

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from api.core.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## 6. Integration with Existing Functionality

1. Modify `ml_backtesting.py`, `data_fetchers.py`, `technical_indicators.py`, `scoring.py`, and `visualizations.py` to work as modules that can be imported and used by the API routes.
2. Ensure that these modules can handle asynchronous operations if necessary.
3. Update any hardcoded configurations to use environment variables or the new `settings` object.

## 7. WebSocket Implementation for Real-time Updates

Implement WebSocket functionality in the FastAPI app to provide real-time updates for long-running backtests:

```python
from fastapi import WebSocket

@app.websocket("/ws/backtest/{backtest_id}")
async def websocket_endpoint(websocket: WebSocket, backtest_id: str):
    await websocket.accept()
    try:
        while True:
            # Send backtest progress updates
            # Break the loop when backtest is complete
    except WebSocketDisconnect:
        # Handle disconnection
```

## 8. Testing

1. Create a new `tests` directory in the project root.
2. Implement unit tests for new API endpoints, database models, and core functionality.
3. Update existing tests to work with the new project structure and API endpoints.

## 9. Documentation

1. Use FastAPI's built-in Swagger UI for API documentation.
2. Update the project's README.md with information about the new web interface and API.

These changes will transform the existing Stock Spike Replicator into a web-based application with a RESTful API backend. The next steps would involve implementing these changes, testing thoroughly, and then integrating with the React frontend.