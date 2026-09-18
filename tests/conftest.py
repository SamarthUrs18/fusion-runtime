"""Test-wide setup.

The one thing here matters: `frun` reads a `.env` from the directory it runs in,
which during a test run is this repository. A developer with their own `.env` —
a key they generated while trying the auth flow, say — would otherwise change
what the tests see, and the failure looks like a bug in the code rather than a
file on one machine. CI, with no `.env`, would disagree with the laptop.
"""
import pytest

# Blanked rather than deleted: env.py only fills in what isn't already set, so an
# empty value is what stops a real .env being read on top. A test that wants one
# of these sets it as usual, and that wins.
# Not FUSION_CONFIG: an empty profile name is an error, not "no profile", so
# blanking it would break every test that starts a server.
FROM_A_DEVELOPERS_ENV_FILE = (
    "FUSION_ACCEPTED_KEYS", "FUSION_ACCEPTED_KEYS_FILE", "FUSION_API_KEY", "FUSION_ALLOWED_ORIGINS",
    "FUSION_TRUSTED_PROXY", "FUSION_AGENT",
)


@pytest.fixture(autouse=True)
def _ignore_the_developers_env_file(monkeypatch):
    for name in FROM_A_DEVELOPERS_ENV_FILE:
        monkeypatch.setenv(name, "")
