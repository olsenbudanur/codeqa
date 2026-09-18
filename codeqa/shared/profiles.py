"""profiles.yaml -> EndpointProfile records. One place for every lane to resolve `--profile <name>`."""
from __future__ import annotations

from pathlib import Path

import yaml

from codeqa.shared import paths
from codeqa.shared.contracts import EndpointProfile


def load_profiles(path: Path | None = None) -> dict[str, EndpointProfile]:
    p = path or paths.PROFILES
    raw = yaml.safe_load(p.read_text()) or {}
    return {name: EndpointProfile(name=name, **cfg) for name, cfg in raw.items()}


def get_profile(name: str, path: Path | None = None) -> EndpointProfile:
    profiles = load_profiles(path)
    if name not in profiles:
        raise KeyError(f"profile {name!r} not in {path or paths.PROFILES}; have {sorted(profiles)}")
    return profiles[name]


def add_profile(profile: EndpointProfile, path: Path | None = None) -> None:
    """Append (or replace) a profile in profiles.yaml; used by the trainer for saved checkpoints."""
    p = path or paths.PROFILES
    raw = yaml.safe_load(p.read_text()) or {} if p.exists() else {}
    cfg = profile.model_dump(exclude={"name"}, exclude_none=True)
    raw[profile.name] = cfg
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
