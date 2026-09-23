"""Small, silent helpers for loading API credentials without exposing them."""

import os
from pathlib import Path


def get_openrouter_api_key():
    """Read the key from process environment or a local .env file.

    OPENROUTER_APIKEY is accepted as an alias for OPENROUTER_API_KEY.
    Values are returned to the caller and never printed or copied elsewhere.
    """
    names = ("OPENROUTER_API_KEY", "OPENROUTER_APIKEY")
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value

    candidates = (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env")
    seen = set()
    for env_path in candidates:
        try:
            resolved = env_path.resolve()
        except OSError:
            resolved = env_path
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            name, separator, value = stripped.partition("=")
            if not separator or name.strip() not in names:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                return value
    return None
