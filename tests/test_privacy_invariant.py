"""Privacy invariant guard for the edge service (mirrors the backend guard).

The edge must never ship a cloud SDK that could move frames/embeddings
off-premise. Fails the build if a known cloud LLM/data SDK is declared.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

FORBIDDEN_PREFIXES = (
    "openai",
    "anthropic",
    "cohere",
    "google-generativeai",
    "google-cloud",
    "mistralai",
    "replicate",
    "together",
    "boto3",
    "botocore",
    "azure-",
)

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _all_declared_dependencies() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    deps: list[str] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        deps.extend(extra)
    return deps


def _dist_name(requirement: str) -> str:
    name = requirement.strip()
    for sep in (" ", "[", "=", ">", "<", "!", "~", ";"):
        name = name.split(sep, 1)[0]
    return name.lower()


def test_no_offpremise_sdks_declared() -> None:
    offenders = [
        dep
        for dep in _all_declared_dependencies()
        if _dist_name(dep).startswith(FORBIDDEN_PREFIXES)
    ]
    assert not offenders, f"Off-premise cloud SDK(s) declared in the edge: {offenders}."
