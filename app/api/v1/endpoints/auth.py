from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import CurrentUserDep
from app.core.config import settings
from app.core.security import create_access_token
from app.schemas.token import Token
from app.schemas.msg import Message
from app.schemas.user import ResetPassword, UpdatePassword, User, UserCreate

router = APIRouter()


@router.post("/login")
def login_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    # This is a dummy implementation since we don't have a DB yet.
    # Replace with actual user validation from database.
    if form_data.username != "admin" or form_data.password != "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect username or password",
        )
        
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=form_data.username, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me")
def read_users_me(current_user: CurrentUserDep) -> User:
    """
    Get current user.
    """
    return current_user


@router.post("/register")
def register_user(user_in: UserCreate) -> User:
    """
    Register a new user.
    """
    # Mock registration response
    return User(
        id=2,
        username=user_in.username,
        email=user_in.email,
        is_active=True,
    )


@router.post("/recover-password/{email}")
def recover_password(email: str) -> Message:
    """
    Password recovery email trigger.
    """
    # Mock password recovery response
    return Message(message=f"Password recovery email successfully sent to {email}")


@router.post("/reset-password")
def reset_password(body: ResetPassword) -> Message:
    """
    Reset password using recovery token.
    """
    # Mock password reset validation
    if not body.token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token",
        )
    return Message(message="Password updated successfully")


@router.post("/update-password")
def update_password(
    body: UpdatePassword, current_user: CurrentUserDep
) -> Message:
    """
    Update password for the logged-in user.
    """
    # Mock password verification
    # Normally check if verify_password(body.current_password, current_user.hashed_password)
    if body.current_password == body.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password",
        )
    return Message(message="Password successfully updated")


@router.post("/logout")
def logout(current_user: CurrentUserDep) -> Message:
    """
    Log out the current user (mock implementation).
    In a stateless JWT setup, logout is primarily handled on the client side by deleting the token.
    For a production-ready revoked token list, the token would be blacklisted in Redis/DB here.
    """
    return Message(message="Successfully logged out")


