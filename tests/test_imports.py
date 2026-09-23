"""Smoke-test that every src module imports without error.

Modules whose only failure is a missing optional dependency (torch,
transformers, etc.) are skipped so the test suite stays useful in the
lightweight CI environment.
"""

import importlib
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"

OPTIONAL_PACKAGES = frozenset({
    "torch",
    "torchvision",
    "torchcodec",
    "transformers",
    "accelerate",
    "lightning",
    "supervision",
    "ultralytics",
    "groundingdino",
    "sam2",
    "jax",
    "videoprism",
    "shap",
    "omegaconf",
    "motmetrics",
    "kmedoids",
})


def _discover_modules():
    """Yield dotted module paths for every .py file under src/."""
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        relative = py_file.relative_to(SRC_ROOT.parent)
        yield str(relative.with_suffix("")).replace("/", ".")


def _is_optional_dep_missing(exc: ModuleNotFoundError) -> bool:
    missing = getattr(exc, "name", "") or ""
    top_level = missing.split(".")[0]
    return top_level in OPTIONAL_PACKAGES


@pytest.mark.parametrize("module", list(_discover_modules()))
def test_import(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if _is_optional_dep_missing(exc):
            pytest.skip(f"optional dependency not installed: {exc.name}")
        raise
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"data file not available: {exc}")
