from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from brainlayer.llm import OpenAICompatibleChatAdapter, StaticLLMAdapter
from brainlayer.runtime import BrainLayerRuntime, BrainLayerRuntimeConfig
from brainlayer.scenarios import Observation
from brainlayer.session import BrainLayerSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a model-backed BrainLayer turn and persist updated state.",
    )
    parser.add_argument("--prompt", required=True, help="User task for the model-backed turn.")
    parser.add_argument(
        "--state",
        default="artifacts/live_state.json",
        help="Path to the BrainLayer state file to load and update.",
    )
    parser.add_argument(
        "--output-state",
        help="Optional output path. Defaults to overwriting --state.",
    )
    parser.add_argument(
        "--observe-file",
        help="Optional JSON file containing a list of observations to ingest before the turn.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("BRAINLAYER_MODEL", "gpt-4.1-mini"),
        help="Model name for the backing adapter.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="Chat-completions-compatible base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the backing model.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=900,
        help="Maximum output tokens requested from the model provider.",
    )
    parser.add_argument(
        "--top-k-per-layer",
        type=int,
        default=2,
        help="How many memories to keep from each BrainLayer layer.",
    )
    parser.add_argument(
        "--max-memories",
        type=int,
        default=8,
        help="Overall cap on retrieved memories included in the prompt.",
    )
    parser.add_argument(
        "--dry-run-response",
        help="Optional static JSON/plain-text response for local dry runs without network calls.",
    )
    parser.add_argument(
        "--no-auto-consolidate",
        action="store_true",
        help="Skip consolidation after the model turn.",
    )
    return parser.parse_args()


def load_observations(path: str | Path) -> List[Observation]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("Observation file must contain a JSON array.")

    observations: List[Observation] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each observation must be a JSON object.")
        observations.append(
            Observation(
                text=str(item["text"]),
                memory_type=str(item["memory_type"]),
                payload={str(key): str(value) for key, value in dict(item["payload"]).items()},
                salience=float(item.get("salience", 0.5)),
            )
        )
    return observations


def main() -> int:
    args = parse_args()
    state_path = Path(args.state)
    output_state = Path(args.output_state) if args.output_state else state_path
    observations = load_observations(args.observe_file) if args.observe_file else []

    if state_path.exists():
        session = BrainLayerSession.from_file(state_path)
    else:
        session = BrainLayerSession()

    if args.dry_run_response is not None:
        adapter = StaticLLMAdapter(response=args.dry_run_response)
    else:
        adapter = OpenAICompatibleChatAdapter(
            api_key=os.environ.get(args.api_key_env),
            base_url=args.base_url,
        )

    runtime = BrainLayerRuntime(
        adapter,
        session=session,
        model=args.model,
        config=BrainLayerRuntimeConfig(
            response_temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            top_k_per_layer=args.top_k_per_layer,
            max_memories=args.max_memories,
            auto_consolidate_after_turn=not args.no_auto_consolidate,
        ),
    )
    result = runtime.run_turn(args.prompt, observations=observations)
    session.save(output_state)

    print("Assistant response:")
    print(result.assistant_response)
    print()
    print("Retrieved BrainLayer context:")
    print(runtime.render_retrieved_context(result.retrieved_memories))
    print()
    print(f"Saved state: {output_state}")
    if result.applied_observations:
        print("Applied model observations:")
        for observation in result.applied_observations:
            print(
                f"- {observation.memory_type}: {observation.text} "
                f"(salience={observation.salience:.2f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
