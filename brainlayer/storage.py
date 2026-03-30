from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .models import BrainLayerState
from .validation import validate_state_dict


def export_state_dict(state: BrainLayerState, validate: bool = True) -> Dict[str, Any]:
    payload = state.to_dict()
    if validate:
        validate_state_dict(payload)
    return payload


def save_state(state: BrainLayerState, path: str | Path, validate: bool = True) -> Path:
    target = Path(path)
    payload = export_state_dict(state, validate=validate)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return target


def load_state(path: str | Path, validate: bool = True) -> BrainLayerState:
    payload = json.loads(Path(path).read_text())
    if validate:
        validate_state_dict(payload)
    return BrainLayerState.from_dict(payload)


def load_state_dict(payload: Dict[str, Any], validate: bool = True) -> BrainLayerState:
    if validate:
        validate_state_dict(payload)
    return BrainLayerState.from_dict(payload)
