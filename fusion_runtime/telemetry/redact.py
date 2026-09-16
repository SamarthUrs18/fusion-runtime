"""Keeping secrets and conversation content out of logs."""
import re
from typing import Any

_SECRET_PATTERNS = [
    re.compile(r"\b(sk|rk|pk)-[A-Za-z0-9_\-]{12,}"),  # OpenAI-style keys
    re.compile(r"\bgsk_[A-Za-z0-9]{12,}"),  # Groq
    re.compile(r"\bhf_[A-Za-z0-9]{12,}"),  # Hugging Face
    re.compile(r"\bfsn_(live|test)_[A-Za-z0-9]{8,}"),  # fusion API keys
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[=:]\s*[^\s,;&\"']{6,}"),
]


def redact_secrets(value: Any) -> Any:
    """Replace anything that looks like a credential with [REDACTED]. Non-strings pass through."""
    if not isinstance(value, str):
        return value
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value
