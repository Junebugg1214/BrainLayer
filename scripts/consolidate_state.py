import argparse

from brainlayer.session import BrainLayerSession


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run consolidation and forgetting on a saved BrainLayer state file."
    )
    parser.add_argument("path", help="Path to the input BrainLayer JSON state file.")
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to overwriting the input file.",
    )
    args = parser.parse_args()

    session = BrainLayerSession.from_file(args.path, validate=True)
    report = session.consolidate()
    target = args.output or args.path
    session.save(target, validate=True)

    print(f"Consolidated state written to {target}")
    print(
        "Report:"
        f" beliefs={len(report.promoted_belief_keys)},"
        f" procedures={len(report.promoted_procedure_triggers)},"
        f" working={len(report.updated_working_keys)},"
        f" autobio={len(report.updated_autobio_keys)},"
        f" forgotten_episodes={len(report.forgotten_episode_ids)},"
        f" paused_working_items={len(report.paused_working_item_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
