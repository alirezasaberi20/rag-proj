"""
User service for authentication and user management.

Provides user CRUD operations with file-based storage.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.core.config import get_settings
from app.core.exceptions import AppException, ValidationError
from app.core.logging import get_logger
from app.core.security import get_password_hash, verify_password
from app.models.user import User, UserCreate, UserInDB

logger = get_logger(__name__)


class UserService:
    """
    User management service with file-based storage.

    For production, replace with PostgreSQL or another database.
    """

    def __init__(self, storage_path: Optional[str] = None):
        settings = get_settings()
        self.storage_path = storage_path or os.path.join(
            settings.vector_store_path, "users.json"
        )
        self._users: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load users from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    self._users = json.load(f)
                logger.info(f"Loaded {len(self._users)} users")
            except Exception as e:
                logger.warning(f"Failed to load users: {e}")
                self._users = {}

    def _save(self) -> None:
        """Save users to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self._users, f, indent=2, default=str)

    def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID."""
        user_data = self._users.get(user_id)
        if user_data:
            return UserInDB(**user_data)
        return None

    def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username."""
        for user_data in self._users.values():
            if user_data["username"] == username:
                return UserInDB(**user_data)
        return None

    def create_user(self, user_create: UserCreate) -> User:
        """
        Create a new user.

        Args:
            user_create: User creation data

        Returns:
            Created user (without password)

        Raises:
            ValidationError: If username already exists
        """
        if self.get_user_by_username(user_create.username):
            raise ValidationError(
                message="Username already registered",
                details={"username": user_create.username},
            )

        user_id = str(uuid4())
        now = datetime.now(timezone.utc)

        user_data = {
            "id": user_id,
            "username": user_create.username,
            "email": user_create.email,
            "hashed_password": get_password_hash(user_create.password),
            "created_at": now.isoformat(),
            "is_active": True,
        }

        self._users[user_id] = user_data
        self._save()

        logger.info(f"Created user: {user_create.username}")

        return User(
            id=user_id,
            username=user_create.username,
            email=user_create.email,
            created_at=now,
            is_active=True,
        )

    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """
        Authenticate user with username and password.

        Args:
            username: Username
            password: Plain password

        Returns:
            User if authenticated, None otherwise
        """
        user = self.get_user_by_username(username)

        if not user:
            return None

        if not verify_password(password, user.hashed_password):
            return None

        return user

    def get_all_users(self) -> list[User]:
        """Get all users (admin function)."""
        return [
            User(
                id=data["id"],
                username=data["username"],
                email=data.get("email"),
                created_at=datetime.fromisoformat(data["created_at"]),
                is_active=data.get("is_active", True),
            )
            for data in self._users.values()
        ]


user_service = UserService()
