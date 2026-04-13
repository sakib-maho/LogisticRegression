"""Backward-compatible entrypoint for CLI training run."""

from cli import main


if __name__ == "__main__":
    raise SystemExit(main())