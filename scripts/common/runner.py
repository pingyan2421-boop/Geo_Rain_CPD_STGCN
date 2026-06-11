import argparse
import os
import runpy
import sys
import warnings
from typing import Dict, List

from scripts.common.paths import ensure_project_on_path


def run_module(module_name: str, argv: List[str]) -> None:
    """Run a workflow module as if it were invoked directly."""
    ensure_project_on_path()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    previous_argv = sys.argv[:]
    sys.argv = [module_name] + list(argv)
    try:
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = previous_argv


def dispatch(title: str, commands: Dict[str, str], argv: List[str] = None) -> None:
    """Dispatch a domain command to an implementation module."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("command", choices=sorted(commands), help="Workflow command to run.")
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
        return
    command = argv[0]
    if command not in commands:
        parser.error(f"unknown command: {command}")
    run_module(commands[command], argv[1:])
