"""
Step 16 — Authentication.

Simple, file-backed username/password login — enough to make the product
feel real (an actual account, not an open door), without pulling in a full
auth framework. Passwords are hashed with werkzeug's security helpers
(already a Flask dependency, so no new package needed).

Storage is a local JSON file (users.json). Fine for a demo/prototype;
swap for a real database before this goes anywhere near production.
"""

import json
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent.parent
USERS_PATH = BASE_DIR / "data" / "users.json"


def _load_users() -> dict:
    if not USERS_PATH.exists():
        return {}
    return json.loads(USERS_PATH.read_text())


def _save_users(users: dict):
    USERS_PATH.parent.mkdir(exist_ok=True)
    USERS_PATH.write_text(json.dumps(users, indent=2))


def user_exists(username: str) -> bool:
    return username in _load_users()


def register_user(username: str, password: str) -> tuple:
    """Returns (success: bool, message: str)."""
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()
    if username in users:
        return False, "That username is already taken."

    users[username] = {"password_hash": generate_password_hash(password)}
    _save_users(users)
    return True, "Account created."


def verify_login(username: str, password: str) -> bool:
    users = _load_users()
    record = users.get(username.strip())
    if not record:
        return False
    return check_password_hash(record["password_hash"], password)


if __name__ == "__main__":
    ok, msg = register_user("demo", "password123")
    print("register:", ok, msg)
    print("login correct pw:", verify_login("demo", "password123"))
    print("login wrong pw:", verify_login("demo", "wrongpass"))
    print("duplicate register:", register_user("demo", "password123"))
