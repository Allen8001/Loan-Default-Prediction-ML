#!/usr/bin/env python3
"""
Unified CLI: install dependencies, train the default model, launch Streamlit.

Usage (from project root)::

    python run.py install
    python run.py train [--sample N]
    python run.py app [--port 8505]
    python run.py all [--sample N] [--port 8505]

``all`` runs install, then train, then starts the app (blocking).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
APP_ENTRY = ROOT / "app" / "app_real_data.py"


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run a subprocess with project root as working directory."""
    merged = {**os.environ, **(env or {})}
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=merged)


def cmd_install() -> None:
    """Install Python dependencies from requirements.txt."""
    _run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def cmd_train(sample: int | None) -> None:
    """Train and save models/loan_default_model_real.pkl."""
    train_cmd = [sys.executable, "-m", "src.train_default_model"]
    if sample is not None:
        train_cmd.extend(["--sample", str(sample)])
    _run(train_cmd)


def cmd_app(port: int) -> None:
    """Start the Streamlit dashboard (blocks until stopped)."""
    _run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_ENTRY),
            f"--server.port={port}",
        ]
    )


def cmd_all(sample: int | None, port: int) -> None:
    """Install, train, then launch the app."""
    cmd_install()
    cmd_train(sample)
    cmd_app(port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Clarity in Credit — install, train, and run the Streamlit app.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("install", help="pip install -r requirements.txt")

    p_train = sub.add_parser("train", help="Train XGBoost model to models/")
    p_train.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Train on N random rows (faster smoke test).",
    )

    p_app = sub.add_parser("app", help="Run Streamlit app")
    p_app.add_argument("--port", type=int, default=8505)

    p_all = sub.add_parser("all", help="install + train + app")
    p_all.add_argument("--sample", type=int, default=None, metavar="N")
    p_all.add_argument("--port", type=int, default=8505)

    args = parser.parse_args(argv)

    if not REQUIREMENTS.exists():
        print(f"Missing {REQUIREMENTS}", file=sys.stderr)
        return 1
    if not APP_ENTRY.exists():
        print(f"Missing {APP_ENTRY}", file=sys.stderr)
        return 1

    try:
        if args.command == "install":
            cmd_install()
        elif args.command == "train":
            cmd_train(args.sample)
        elif args.command == "app":
            cmd_app(args.port)
        elif args.command == "all":
            cmd_all(args.sample, args.port)
        else:
            parser.error("Unknown command")
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
