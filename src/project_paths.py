from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the cloned repository root.

    Set HCLT2026_ROOT to override this when running scripts from a copied
    location or an external experiment directory.
    """
    override = os.environ.get("HCLT2026_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


ROOT = repo_root()
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"


def model_path(name: str, env_var: str | None = None) -> Path:
    if env_var and os.environ.get(env_var):
        return Path(os.environ[env_var]).expanduser().resolve()
    return MODELS / name
