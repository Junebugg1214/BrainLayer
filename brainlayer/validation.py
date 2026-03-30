from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "schemas"


class BrainLayerValidationError(ValueError):
    """Raised when a payload does not satisfy the BrainLayer JSON schemas."""


def load_schema(name: str) -> Dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text())


def validate_state_dict(payload: Dict[str, Any]) -> None:
    schema = load_schema("brainlayer-state.schema.json")
    _validate_instance(payload, schema, "brainlayer-state.schema.json")


def _validate_instance(instance: Any, schema: Dict[str, Any], schema_name: str) -> None:
    if "$ref" in schema:
        ref_name = schema["$ref"]
        _validate_instance(instance, load_schema(ref_name), ref_name)
        return

    expected_type = schema.get("type")
    if expected_type is not None:
        _validate_type(instance, expected_type, schema_name)

    if "enum" in schema and instance not in schema["enum"]:
        raise BrainLayerValidationError(
            f"{schema_name}: expected one of {schema['enum']}, got {instance!r}"
        )

    if expected_type == "object":
        _validate_object(instance, schema, schema_name)
        return

    if expected_type == "array":
        _validate_array(instance, schema, schema_name)
        return

    if expected_type == "number":
        _validate_number(instance, schema, schema_name)
        return

    if expected_type == "string":
        _validate_string(instance, schema, schema_name)


def _validate_type(instance: Any, expected_type: str, schema_name: str) -> None:
    checks = {
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    }
    checker = checks.get(expected_type)
    if checker is None:
        raise BrainLayerValidationError(f"{schema_name}: unsupported schema type {expected_type}")
    if not checker(instance):
        raise BrainLayerValidationError(
            f"{schema_name}: expected {expected_type}, got {type(instance).__name__}"
        )


def _validate_object(instance: Dict[str, Any], schema: Dict[str, Any], schema_name: str) -> None:
    required = schema.get("required", [])
    for key in required:
        if key not in instance:
            raise BrainLayerValidationError(f"{schema_name}: missing required key {key!r}")

    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        unexpected = set(instance) - set(properties)
        if unexpected:
            extra = ", ".join(sorted(unexpected))
            raise BrainLayerValidationError(f"{schema_name}: unexpected properties: {extra}")

    for key, value in instance.items():
        if key in properties:
            _validate_instance(value, properties[key], schema_name)


def _validate_array(instance: Any, schema: Dict[str, Any], schema_name: str) -> None:
    min_items = schema.get("minItems")
    if min_items is not None and len(instance) < min_items:
        raise BrainLayerValidationError(
            f"{schema_name}: expected at least {min_items} items, got {len(instance)}"
        )

    items_schema = schema.get("items")
    if items_schema is None:
        return
    for item in instance:
        _validate_instance(item, items_schema, schema_name)


def _validate_number(instance: float, schema: Dict[str, Any], schema_name: str) -> None:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if minimum is not None and instance < minimum:
        raise BrainLayerValidationError(
            f"{schema_name}: number {instance} is below minimum {minimum}"
        )
    if maximum is not None and instance > maximum:
        raise BrainLayerValidationError(
            f"{schema_name}: number {instance} exceeds maximum {maximum}"
        )


def _validate_string(instance: str, schema: Dict[str, Any], schema_name: str) -> None:
    if schema.get("format") == "date-time":
        try:
            datetime.fromisoformat(instance.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BrainLayerValidationError(
                f"{schema_name}: invalid date-time value {instance!r}"
            ) from exc
