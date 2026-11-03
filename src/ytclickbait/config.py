"""Load configs/config.yaml and resolve paths relative to the project root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, name: str) -> Path:
        """Directory from the `paths` section, e.g. cfg.path("processed")."""
        return self.root / self.raw["paths"][name]

    def file(self, name: str, folder: str = "processed") -> Path:
        """File from the `files` section inside a `paths` folder."""
        return self.path(folder) / self.raw["files"][name]


def load_config(path: Path | str = DEFAULT_CONFIG, root: Path | None = None) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, root=root or PROJECT_ROOT)
