from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import ValidationError

from app.core.config import settings
from app.core.security import ALGORITHM
from app.schemas.token import TokenPayload
from app.schemas.user import User

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login"
)

TokenDep = Annotated[str, Depends(reusable_oauth2)]


def get_current_user(token: TokenDep) -> User:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    
    # Normally here you would fetch the user from the database.
    # We return a dummy user since there's no DB configured right now.
    user = User(
        id=1,
        username=token_data.sub,
        email=f"{token_data.sub}@example.com",
        is_active=True,
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]
