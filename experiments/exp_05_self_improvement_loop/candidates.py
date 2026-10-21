"""A candidate is the only thing the loop may change: the system prompt and tool descriptions.

On disk: <root>/<name>/prompt.md, tools.json and (for generated candidates) meta.json with the
parent, critique and rationale. The hash covers prompt + tool descriptions. The baseline
profile (v1) exposes only run_query and get_sample_rows, so get_schema is not editable.
The loop writes its proposals to the workspace (Workspace.candidates_dir). The starting point,
BASELINE, is not stored: Workspace.candidate builds it from the profile's agent prompt and the
original tool descriptions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

BASELINE = "c000_baseline"
ENABLED_TOOLS = ("run_query", "get_sample_rows")

# Original descriptions from agent/tools.py (Toolbox.langchain_tools), copied verbatim.
ORIGINAL_DESCRIPTIONS = {
    "run_query": (
        "Execute a read-only SELECT query. Returns columns and rows, "
        "or an 'error' string if the query failed. DML is blocked."
    ),
    "get_sample_rows": (
        "Return a few sample rows from a table -- useful for "
        "discovering enum values and how values co-occur across columns."
    ),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    prompt: str
    descriptions: dict = field(default_factory=lambda: dict(ORIGINAL_DESCRIPTIONS))

    @property
    def sha256(self) -> str:
        blob = json.dumps({"prompt": self.prompt, "descriptions": self.descriptions}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def save(self, root: Path, meta: dict | None = None) -> Path:
        folder = root / self.name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "prompt.md").write_text(self.prompt, encoding="utf-8", newline="\n")
        (folder / "tools.json").write_text(json.dumps(self.descriptions, indent=2) + "\n", encoding="utf-8")
        if meta is not None:
            (folder / "meta.json").write_text(
                json.dumps({**meta, "sha256": self.sha256}, indent=2) + "\n", encoding="utf-8")
        return folder


def load(name: str, root: Path) -> Candidate:
    """Load a saved candidate from `root` (the workspace's proposals)."""
    folder = root / name
    if not (folder / "prompt.md").exists():
        raise SystemExit(f"candidate {name!r} not found in {root}")
    tools_path = folder / "tools.json"
    descriptions = (json.loads(tools_path.read_text(encoding="utf-8")) if tools_path.exists()
                    else dict(ORIGINAL_DESCRIPTIONS))
    if set(descriptions) != set(ENABLED_TOOLS):
        raise ValueError(f"{name}: tools.json must describe exactly {ENABLED_TOOLS}")
    return Candidate(name, (folder / "prompt.md").read_text(encoding="utf-8"), descriptions)
