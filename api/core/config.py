from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Stock Spike Replicator"
    PROJECT_VERSION: str = "1.0.0"
    ALLOWED_ORIGINS: list = ["http://localhost:3000"]  # Update with actual frontend URL
    DATABASE_URL: str = "sqlite:///./stock_spike_replicator.db"
    SECRET_KEY: str = "your-secret-key"  # Change this to a secure random string
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"

settings = Settings()