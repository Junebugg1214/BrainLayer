import argparse

from brainlayer.storage import load_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and load a BrainLayer state file.")
    parser.add_argument("path", help="Path to the BrainLayer JSON state file.")
    args = parser.parse_args()

    state = load_state(args.path, validate=True)
    print(
        "Valid BrainLayer state:"
        f" {len(state.working_state)} working items,"
        f" {len(state.episodes)} episodes,"
        f" {len(state.beliefs)} beliefs,"
        f" {len(state.autobiographical_state)} autobiographical notes,"
        f" {len(state.procedures)} procedures."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
