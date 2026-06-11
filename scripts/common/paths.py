from pathlib import Path


def project_root() -> Path:
    """Return the repository root from any script location."""
    return Path(__file__).resolve().parents[2]


def ensure_project_on_path() -> Path:
    """Add the repository root to sys.path and return it."""
    import sys

    root = project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root

