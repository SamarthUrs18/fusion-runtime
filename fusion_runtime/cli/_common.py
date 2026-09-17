"""Small helpers shared by several `frun` commands. Must stay import-light."""
from enum import Enum
from pathlib import Path


class Profile(str, Enum):
    development = "development"
    production = "production"
    hybrid = "hybrid"  # local speech, LLM from an OpenAI-compatible endpoint


def profile_config(profile: Profile, apply_env: bool = True):
    """The profile's config; with FUSION_LLM_* overrides applied unless apply_env is False."""
    from fusion_runtime.config import PROFILES, load_profile

    return load_profile(profile.value) if apply_env else PROFILES[profile.value]


def short_path(path: Path) -> str:
    home = str(Path.home())
    return "~" + str(path)[len(home):] if str(path).startswith(home) else str(path)
