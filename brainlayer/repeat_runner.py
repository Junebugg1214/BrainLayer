from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NATURAL_EVAL_SCRIPT = ROOT / "scripts" / "run_natural_model_evals.py"
DEFAULT_MODEL_EVAL_SCRIPT = ROOT / "scripts" / "run_model_evals.py"
EXPORT_DIR_RE = re.compile(r"Natural-eval exports written to (?P<path>.+)$")
MODEL_EXPORT_DIR_RE = re.compile(r"Model-loop exports written to (?P<path>.+)$")


@dataclass(frozen=True)
class RepeatRunResult:
    repeat_index: int
    label: str
    status: str
    duration_seconds: float
    exit_code: int | None
    timed_out: bool
    log_path: str
    run_dir: str = ""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_natural_eval_command(
    *,
    script_path: Path = DEFAULT_NATURAL_EVAL_SCRIPT,
    mode: str,
    model: str,
    provider_name: str,
    base_url: str,
    api_key_env: str,
    request_path: str,
    timeout_seconds: float,
    max_output_tokens_field: str,
    temperature: float,
    max_output_tokens: int,
    scenario_pack: str,
    runtime_profile: str,
    export_results: Path,
    label: str,
    scenario_slugs: Sequence[str] = (),
    runtime_names: Sequence[str] = (),
    score_exact: bool = False,
) -> List[str]:
    command = [
        sys.executable,
        str(script_path),
        "--mode",
        mode,
        "--model",
        model,
        "--provider-name",
        provider_name,
        "--base-url",
        base_url,
        "--api-key-env",
        api_key_env,
        "--request-path",
        request_path,
        "--timeout-seconds",
        str(timeout_seconds),
        "--max-output-tokens-field",
        max_output_tokens_field,
        "--temperature",
        str(temperature),
        "--max-output-tokens",
        str(max_output_tokens),
        "--scenario-pack",
        scenario_pack,
        "--runtime-profile",
        runtime_profile,
        "--export-results",
        str(export_results),
        "--label",
        label,
    ]
    if score_exact:
        command.append("--score-exact")
    for slug in scenario_slugs:
        command.extend(["--scenario-slug", slug])
    for runtime_name in runtime_names:
        command.extend(["--runtime-name", runtime_name])
    return command


def build_model_eval_command(
    *,
    script_path: Path = DEFAULT_MODEL_EVAL_SCRIPT,
    mode: str,
    model: str,
    provider_name: str,
    base_url: str,
    api_key_env: str,
    request_path: str,
    timeout_seconds: float,
    max_output_tokens_field: str,
    temperature: float,
    max_output_tokens: int,
    scenario_pack: str,
    runtime_profile: str,
    export_results: Path,
    label: str,
    scenario_slugs: Sequence[str] = (),
    runtime_names: Sequence[str] = (),
    score_exact: bool = False,
) -> List[str]:
    command = [
        sys.executable,
        str(script_path),
        "--mode",
        mode,
        "--model",
        model,
        "--provider-name",
        provider_name,
        "--base-url",
        base_url,
        "--api-key-env",
        api_key_env,
        "--request-path",
        request_path,
        "--timeout-seconds",
        str(timeout_seconds),
        "--max-output-tokens-field",
        max_output_tokens_field,
        "--temperature",
        str(temperature),
        "--max-output-tokens",
        str(max_output_tokens),
        "--scenario-pack",
        scenario_pack,
        "--runtime-profile",
        runtime_profile,
        "--export-results",
        str(export_results),
        "--label",
        label,
    ]
    if score_exact:
        command.append("--score-exact")
    for slug in scenario_slugs:
        command.extend(["--scenario-slug", slug])
    for runtime_name in runtime_names:
        command.extend(["--runtime-name", runtime_name])
    return command


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=5)


def run_logged_command(
    command: Sequence[str],
    *,
    log_path: Path,
    wall_clock_seconds: float,
    cwd: Path | None = None,
    export_dir_regex: re.Pattern[str] = EXPORT_DIR_RE,
) -> RepeatRunResult:
    cwd = cwd or ROOT
    log_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    selector.register(process.stdout, selectors.EVENT_READ)

    started_at = time.monotonic()
    run_dir = ""
    timed_out = False
    buffered_lines: List[str] = []

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[{_timestamp()}] command: {' '.join(command)}\n")
        handle.flush()
        try:
            while True:
                elapsed = time.monotonic() - started_at
                if elapsed >= wall_clock_seconds:
                    timed_out = True
                    handle.write(
                        f"[{_timestamp()}] timeout after {wall_clock_seconds:.1f}s; terminating process\n"
                    )
                    handle.flush()
                    _terminate_process(process)
                    break

                events = selector.select(timeout=1.0)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    buffered_lines.append(line.rstrip("\n"))
                    handle.write(line)
                    handle.flush()
                    match = export_dir_regex.search(line.strip())
                    if match:
                        run_dir = match.group("path")

                if process.poll() is not None:
                    remainder = process.stdout.read()
                    if remainder:
                        handle.write(remainder)
                        handle.flush()
                        buffered_lines.extend(remainder.splitlines())
                        for line in remainder.splitlines():
                            match = export_dir_regex.search(line.strip())
                            if match:
                                run_dir = match.group("path")
                    break
        finally:
            selector.close()
            process.stdout.close()

    duration_seconds = time.monotonic() - started_at
    exit_code = process.poll()
    status = "timeout" if timed_out else "completed"
    if not timed_out and exit_code not in (0, None):
        status = "failed"
    return RepeatRunResult(
        repeat_index=0,
        label="",
        status=status,
        duration_seconds=duration_seconds,
        exit_code=exit_code,
        timed_out=timed_out,
        log_path=str(log_path),
        run_dir=run_dir,
    )


def render_repeat_summary_markdown(results: Sequence[RepeatRunResult]) -> str:
    lines = ["# Natural Eval Repeats", ""]
    for result in results:
        exit_value = "timeout" if result.timed_out else str(result.exit_code)
        lines.append(
            f"- repeat {result.repeat_index}: `{result.label}` -> {result.status} "
            f"({result.duration_seconds:.1f}s, exit={exit_value})"
        )
        if result.run_dir:
            lines.append(f"  run dir: {result.run_dir}")
        lines.append(f"  log: {result.log_path}")
    return "\n".join(lines) + "\n"


def run_natural_eval_repeats(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run repeated natural-eval live comparisons with hard wall-clock cutoffs."
    )
    parser.add_argument("--mode", default="live")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider-name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--request-path", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-output-tokens-field", default="none")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--scenario-pack", default="forgetting_stress")
    parser.add_argument("--runtime-profile", default="study_v2")
    parser.add_argument("--export-results", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--wall-clock-seconds", type=float, default=300.0)
    parser.add_argument("--progress-log-dir", type=Path)
    parser.add_argument("--score-exact", action="store_true")
    parser.add_argument("--scenario-slug", action="append", default=[])
    parser.add_argument("--runtime-name", action="append", default=[])
    args = parser.parse_args(argv)

    progress_log_dir = args.progress_log_dir or (args.export_results / "repeat_logs")
    progress_log_dir.mkdir(parents=True, exist_ok=True)

    results: List[RepeatRunResult] = []
    for repeat_index in range(1, args.repeats + 1):
        label = f"{args.label}-repeat{repeat_index}"
        log_path = progress_log_dir / f"{label}.log"
        print(
            f"[repeat {repeat_index}/{args.repeats}] starting {label} "
            f"(wall_clock={args.wall_clock_seconds:.0f}s)"
        )
        command = build_natural_eval_command(
            mode=args.mode,
            model=args.model,
            provider_name=args.provider_name,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            request_path=args.request_path,
            timeout_seconds=args.timeout_seconds,
            max_output_tokens_field=args.max_output_tokens_field,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            scenario_pack=args.scenario_pack,
            runtime_profile=args.runtime_profile,
            export_results=args.export_results,
            label=label,
            scenario_slugs=args.scenario_slug,
            runtime_names=args.runtime_name,
            score_exact=args.score_exact,
        )
        attempt = run_logged_command(
            command,
            log_path=log_path,
            wall_clock_seconds=args.wall_clock_seconds,
        )
        result = RepeatRunResult(
            repeat_index=repeat_index,
            label=label,
            status=attempt.status,
            duration_seconds=attempt.duration_seconds,
            exit_code=attempt.exit_code,
            timed_out=attempt.timed_out,
            log_path=attempt.log_path,
            run_dir=attempt.run_dir,
        )
        results.append(result)
        print(
            f"[repeat {repeat_index}/{args.repeats}] {result.status} "
            f"after {result.duration_seconds:.1f}s"
        )
        if result.run_dir:
            print(f"[repeat {repeat_index}/{args.repeats}] run dir: {result.run_dir}")
        print(f"[repeat {repeat_index}/{args.repeats}] log: {result.log_path}")

    summary_path = progress_log_dir / f"{args.label}-summary.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = progress_log_dir / f"{args.label}-summary.md"
    markdown_path.write_text(render_repeat_summary_markdown(results), encoding="utf-8")
    print(f"Repeat summary written to {summary_path}")
    print(f"Repeat summary markdown written to {markdown_path}")
    return 0 if all(result.status == "completed" for result in results) else 1


def run_model_eval_repeats(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run repeated contradiction-eval live comparisons with hard wall-clock cutoffs."
    )
    parser.add_argument("--mode", default="live")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider-name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--request-path", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-output-tokens-field", default="none")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--scenario-pack", default="consolidation_stress")
    parser.add_argument("--runtime-profile", default="study_v2")
    parser.add_argument("--export-results", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--wall-clock-seconds", type=float, default=300.0)
    parser.add_argument("--progress-log-dir", type=Path)
    parser.add_argument("--score-exact", action="store_true")
    parser.add_argument("--scenario-slug", action="append", default=[])
    parser.add_argument("--runtime-name", action="append", default=[])
    args = parser.parse_args(argv)

    progress_log_dir = args.progress_log_dir or (args.export_results / "repeat_logs")
    progress_log_dir.mkdir(parents=True, exist_ok=True)

    results: List[RepeatRunResult] = []
    for repeat_index in range(1, args.repeats + 1):
        label = f"{args.label}-repeat{repeat_index}"
        log_path = progress_log_dir / f"{label}.log"
        print(
            f"[repeat {repeat_index}/{args.repeats}] starting {label} "
            f"(wall_clock={args.wall_clock_seconds:.0f}s)"
        )
        command = build_model_eval_command(
            mode=args.mode,
            model=args.model,
            provider_name=args.provider_name,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            request_path=args.request_path,
            timeout_seconds=args.timeout_seconds,
            max_output_tokens_field=args.max_output_tokens_field,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            scenario_pack=args.scenario_pack,
            runtime_profile=args.runtime_profile,
            export_results=args.export_results,
            label=label,
            scenario_slugs=args.scenario_slug,
            runtime_names=args.runtime_name,
            score_exact=args.score_exact,
        )
        attempt = run_logged_command(
            command,
            log_path=log_path,
            wall_clock_seconds=args.wall_clock_seconds,
            export_dir_regex=MODEL_EXPORT_DIR_RE,
        )
        result = RepeatRunResult(
            repeat_index=repeat_index,
            label=label,
            status=attempt.status,
            duration_seconds=attempt.duration_seconds,
            exit_code=attempt.exit_code,
            timed_out=attempt.timed_out,
            log_path=attempt.log_path,
            run_dir=attempt.run_dir,
        )
        results.append(result)
        print(
            f"[repeat {repeat_index}/{args.repeats}] {result.status} "
            f"after {result.duration_seconds:.1f}s"
        )
        if result.run_dir:
            print(f"[repeat {repeat_index}/{args.repeats}] run dir: {result.run_dir}")
        print(f"[repeat {repeat_index}/{args.repeats}] log: {result.log_path}")

    summary_path = progress_log_dir / f"{args.label}-summary.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = progress_log_dir / f"{args.label}-summary.md"
    markdown_path.write_text(render_repeat_summary_markdown(results), encoding="utf-8")
    print(f"Repeat summary written to {summary_path}")
    print(f"Repeat summary markdown written to {markdown_path}")
    return 0 if all(result.status == "completed" for result in results) else 1


__all__ = [
    "RepeatRunResult",
    "build_model_eval_command",
    "build_natural_eval_command",
    "render_repeat_summary_markdown",
    "run_logged_command",
    "run_model_eval_repeats",
    "run_natural_eval_repeats",
]
