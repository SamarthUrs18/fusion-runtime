"""Small helpers shared by several `frun` commands. Must stay import-light."""
from enum import Enum
from pathlib import Path


class Profile(str, Enum):
    development = "development"
    production = "production"


def profile_config(profile: Profile):
    from fusion_runtime.config import DEVELOPMENT_CONFIG, PRODUCTION_CONFIG

    return {Profile.development: DEVELOPMENT_CONFIG, Profile.production: PRODUCTION_CONFIG}[profile]


def short_path(path: Path) -> str:
    home = str(Path.home())
    return "~" + str(path)[len(home):] if str(path).startswith(home) else str(path)
