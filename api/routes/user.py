from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.security import get_current_user
from api.models.user import User, UserUpdate, UserPreferences

router = APIRouter()

@router.get("/profile")
async def get_profile(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name
    }

@router.put("/profile")
async def update_profile(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    for key, value in user_data.dict(exclude_unset=True).items():
        setattr(current_user, key, value)
    
    db.commit()
    db.refresh(current_user)
    
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name
    }

@router.get("/preferences")
async def get_preferences(current_user: User = Depends(get_current_user)):
    return current_user.preferences if current_user.preferences else {}

@router.put("/preferences")
async def update_preferences(
    preferences: UserPreferences,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.preferences:
        current_user.preferences = {}
    
    current_user.preferences.update(preferences.dict())
    db.commit()
    db.refresh(current_user)
    
    return current_user.preferences