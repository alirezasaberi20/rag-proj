"""
Authentication API routes.

Provides user registration, login, and profile endpoints.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.dependencies import CurrentUser
from app.core.exceptions import AppException
from app.core.logging import get_logger
from app.core.security import create_access_token
from app.models.user import Token, User, UserCreate
from app.services.user_service import user_service

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=User,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with username and password.",
)
async def register(user_create: UserCreate) -> User:
    """
    Register a new user.

    - **username**: Unique username (3-50 characters)
    - **password**: Password (6-100 characters)
    - **email**: Optional email address
    """
    try:
        user = user_service.create_user(user_create)
        logger.info(f"New user registered: {user.username}")
        return user

    except AppException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"error": e.message, "details": e.details},
        )


@router.post(
    "/login",
    response_model=Token,
    summary="Login and get access token",
    description="Authenticate with username and password to receive a JWT token.",
)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    """
    Login to get an access token.

    Use the token in the Authorization header:
    `Authorization: Bearer <token>`
    """
    user = user_service.authenticate_user(
        username=form_data.username,
        password=form_data.password,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={
            "sub": user.id,
            "username": user.username,
        }
    )

    logger.info(f"User logged in: {user.username}")

    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        username=user.username,
    )


@router.get(
    "/me",
    response_model=User,
    summary="Get current user",
    description="Get the profile of the currently authenticated user.",
)
async def get_current_user_profile(current_user: CurrentUser) -> User:
    """Get the current authenticated user's profile."""
    return current_user


@router.post(
    "/test-token",
    response_model=User,
    summary="Test access token",
    description="Verify that the access token is valid.",
)
async def test_token(current_user: CurrentUser) -> User:
    """Test if the current token is valid."""
    return current_user
