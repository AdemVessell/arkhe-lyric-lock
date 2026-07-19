from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import FIXTURES


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    p = path or FIXTURES
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def list_fixtures(path: Path | None = None) -> list[dict[str, Any]]:
    data = load_catalog(path)
    fixtures = list(data.get("fixtures") or [])
    fixtures.sort(key=lambda x: (x.get("priority", 99), x.get("id", "")))
    return fixtures


def get_fixture(fixture_id: str, path: Path | None = None) -> dict[str, Any]:
    for fx in list_fixtures(path):
        if fx.get("id") == fixture_id:
            return fx
    known = ", ".join(f["id"] for f in list_fixtures(path))
    raise KeyError(f"Unknown fixture id {fixture_id!r}. Known: {known}")
