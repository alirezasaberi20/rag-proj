"""
FastAPI dependencies for authentication and common operations.

Provides reusable dependencies for route handlers.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.security import decode_access_token
from app.models.user import TokenData, User
from app.services.user_service import user_service

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> User:
    """
    Dependency to get the current authenticated user.

    Extracts user from JWT token in Authorization header.

    Args:
        token: JWT token from OAuth2 scheme

    Returns:
        Current user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        username: str = payload.get("username")

        if user_id is None:
            raise credentials_exception

        token_data = TokenData(user_id=user_id, username=username)

    except Exception:
        raise credentials_exception

    user = user_service.get_user_by_id(token_data.user_id)

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    return User(
        id=user.id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
        is_active=user.is_active,
    )


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_optional_user(
    token: Annotated[str | None, Depends(OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False))] = None,
) -> User | None:
    """
    Dependency to optionally get the current user.

    Returns None if no token provided (for public endpoints).
    """
    if token is None:
        return None

    try:
        return await get_current_user(token)
    except HTTPException:
        return None


OptionalUser = Annotated[User | None, Depends(get_optional_user)]
