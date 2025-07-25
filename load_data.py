from __future__ import annotations
import json, uuid
from pathlib import Path
from typing import List, Dict, Any

from rollout import MDScenario, TaskType   # keep the import path you use


def _collect_pair_dirs(root: Path):
    """Yield every dir that holds the trio of JSONs."""
    for base_json in root.rglob("base_memory.json"):
        pair = base_json.parent
        if (pair / "retrieval_questions.json").is_file() and (pair / "update_queries.json").is_file():
            yield pair


def _mk_scenarios(
    mem_snapshot: Dict[str, str],
    qs: Dict[str, Any],
    task: TaskType,
    field_query: str,
    field_ref: str,
) -> List[MDScenario]:
    out: list[MDScenario] = []
    for hop_items in qs.values():
        if isinstance(hop_items, dict):
            hop_items = [hop_items]  # ensure it's a list of dicts
        for it in hop_items:
            q = it[field_query]
            # some Q’s are list[str]
            q = q if isinstance(q, str) else q[0]
            ref = it.get(field_ref)
            out.append(
                MDScenario(
                    id=str(uuid.uuid4()),
                    step=0,
                    task=task,
                    query=q,
                    answer=ref if task == TaskType.RETRIEVAL else None,
                    diff=ref if task == TaskType.UPDATE else None,
                    memory=mem_snapshot.copy(),
                )
            )
    return out


def load_scenarios(root: str | Path) -> List[MDScenario]:
    """Loads **all** base_memory / question / diff sets under `root`."""
    root = Path(root)
    scenarios: list[MDScenario] = []

    for pair_dir in _collect_pair_dirs(root):
        mem = json.loads((pair_dir / "base_memory.json").read_text())

        # retrieval ----------------------------------------------------------
        rqs = json.loads((pair_dir / "retrieval_questions.json").read_text())
        scenarios += _mk_scenarios(mem, rqs, TaskType.RETRIEVAL, "q", "a")

        # updates ------------------------------------------------------------
        uqs = json.loads((pair_dir / "update_queries.json").read_text())
        scenarios += _mk_scenarios(mem, uqs, TaskType.UPDATE, "query", "diff")

    return scenarios

if __name__ == "__main__":
    # Example usage
    scenarios = load_scenarios("instances")
    for scenario in scenarios:
        print(f"Scenario ID: {scenario.id}, Query: {scenario.query}, Type: {scenario.task}")