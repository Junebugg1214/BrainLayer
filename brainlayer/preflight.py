from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .llm import LLMAdapter, LLMError, ModelMessage
from .model_eval import build_live_model_eval_adapter
from .model_matrix import DEFAULT_MATRIX_CONFIG, ModelMatrixEntry, load_model_matrix_entries


@dataclass(frozen=True)
class ModelPreflightResult:
    entry_name: str
    provider_name: str
    requested_model: str
    response_model: str
    finish_reason: str
    latency_ms: float
    output_preview: str
    usage: dict[str, float]


def resolve_matrix_entry(
    config_path: str | Path,
    *,
    entry_name: str | None = None,
) -> ModelMatrixEntry:
    entries = load_model_matrix_entries(config_path)
    if not entries:
        raise ValueError("Matrix config did not produce any enabled entries.")
    if entry_name is None:
        return entries[0]
    for entry in entries:
        if entry.name == entry_name:
            return entry
    raise ValueError(f"Unknown matrix entry: {entry_name}")


def run_model_preflight(
    entry: ModelMatrixEntry,
    *,
    adapter: LLMAdapter | None = None,
    prompt: str = "Return exactly the word OK.",
    max_output_tokens: int = 32,
) -> ModelPreflightResult:
    active_adapter = adapter or build_live_model_eval_adapter(
        provider_name=entry.provider_name,
        api_key_env=entry.api_key_env,
        base_url=entry.base_url,
        request_path=entry.request_path,
        timeout_seconds=entry.timeout_seconds,
        max_output_tokens_field=entry.max_output_tokens_field,
    )
    messages = [
        ModelMessage(
            role="system",
            content="You are a provider preflight probe. Reply briefly and directly.",
        ),
        ModelMessage(role="user", content=prompt),
    ]
    started_at = time.perf_counter()
    response = active_adapter.generate(
        messages,
        model=entry.requested_model,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
    )
    latency_ms = (time.perf_counter() - started_at) * 1000.0
    preview = response.content.strip().replace("\n", " ")
    if len(preview) > 120:
        preview = preview[:117] + "..."
    usage = {}
    for key, value in response.usage.items():
        try:
            usage[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return ModelPreflightResult(
        entry_name=entry.name,
        provider_name=entry.provider_name,
        requested_model=entry.requested_model,
        response_model=response.model or entry.requested_model,
        finish_reason=response.finish_reason,
        latency_ms=latency_ms,
        output_preview=preview,
        usage=usage,
    )


def render_model_preflight(result: ModelPreflightResult) -> str:
    lines = [
        "BrainLayer Model Preflight",
        "==========================",
        "",
        f"Entry: {result.entry_name}",
        f"Provider: {result.provider_name}",
        f"Requested model: {result.requested_model}",
        f"Response model: {result.response_model}",
        f"Finish reason: {result.finish_reason or 'unknown'}",
        f"Latency: {result.latency_ms:.1f}ms",
        f"Output preview: {result.output_preview or '(empty)'}",
    ]
    if result.usage:
        lines.append(f"Usage: {json.dumps(result.usage, sort_keys=True)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a one-call BrainLayer provider preflight against a matrix entry."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_MATRIX_CONFIG),
        help="Path to a model matrix config JSON file.",
    )
    parser.add_argument(
        "--entry",
        default=None,
        help="Optional matrix entry name. Defaults to the first enabled entry.",
    )
    parser.add_argument(
        "--prompt",
        default="Return exactly the word OK.",
        help="Short prompt to send to the model.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=32,
        help="Max output tokens for the preflight call.",
    )
    args = parser.parse_args(argv)

    try:
        entry = resolve_matrix_entry(args.config, entry_name=args.entry)
        result = run_model_preflight(
            entry,
            prompt=args.prompt,
            max_output_tokens=args.max_output_tokens,
        )
    except (LLMError, ValueError, FileNotFoundError) as exc:
        print(f"BrainLayer model preflight failed: {exc}")
        return 1

    print(render_model_preflight(result))
    return 0


__all__ = [
    "ModelPreflightResult",
    "main",
    "render_model_preflight",
    "resolve_matrix_entry",
    "run_model_preflight",
]
