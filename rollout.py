"""
rollout.py  – ART rollout for a markdown-knowledge-base agent
=============================================================

The agent answers *retrieval* questions or produces *update* diffs against an
in-memory Obsidian-style vault.  A Scenario supplies:

• snapshot of the vault (dict: path → content)
• user query (+ gold answer *or* gold diff)
• task type (retrieval / update)

During a rollout the LLM may call these tools:

    read_file(path)          → markdown string
    list_files(dir_path="")  → recursive listing
    write_diff(diff)         → apply / store patch in memory
    return_final_answer(...) → ends episode

The episode is scored:
    • retrieval → LLM judge on answer correctness
    • update    → string-equality on unified diff (swap in smarter judge later)

The returned `ProjectTrajectory` carries `traj.final_answer` and
`traj.reward ∈ {0,1}`.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Callable, Dict, List

import art
import weave
from langchain_core.utils.function_calling import convert_to_openai_tool
from litellm import acompletion
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt
from enum import Enum

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MAX_TURNS = 10  # roll-forward budget
JUDGE_MODEL = "openai/o4-mini"


# -----------------------------------------------------------------------------
# Scenario & answer containers
# -----------------------------------------------------------------------------
class TaskType(str, Enum):
    RETRIEVAL = "retrieval"
    UPDATE    = "update"


class MDScenario(BaseModel):
    id: str
    step: int
    task: TaskType
    query: str
    answer: str | None = None  # retrieval
    diff: str | None = None  # update
    memory: Dict[str, Any]  # path → file content snapshot


class FinalAnswer(BaseModel):
    answer: str
    source_ids: List[str] | None = None


# -----------------------------------------------------------------------------
# LLM-based correctness judge (retrieval)
# -----------------------------------------------------------------------------
class CorrectnessJudgeResponse(BaseModel):
    reasoning: str = Field(description="Why accept / reject")
    accept: bool = Field(description="True ↔︎ matches reference answer")


class UpdateJudgeResponse(BaseModel):
    reasoning: str = Field(description="Why the diff should / shouldn’t be accepted")
    accept: bool = Field(description="True if the diff correctly applies the reference")


@retry(stop=stop_after_attempt(3))
async def judge_correctness(scenario: MDScenario, model_answer: str) -> bool:
    system = dedent(
        """
        You are given a reference answer and an answer written by an AI.
        Accept if the AI answer includes the key facts of the reference;
        reject if facts are missing or contradicted.
        """
    )

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Question  : {scenario.query}\n"
                f"Reference : {scenario.answer}\n"
                f"AI answer : {model_answer}"
            ),
        },
    ]
    resp = await acompletion(
        model=JUDGE_MODEL,
        messages=messages,
        response_format=CorrectnessJudgeResponse,
    )

    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = CorrectnessJudgeResponse.model_validate_json(raw)
        return parsed.accept
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Simple diff judge (update)
# -----------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3))
async def judge_update(scenario: MDScenario, model_diff: str) -> bool:
    """
    Accept when the AI-supplied unified diff produces exactly the same
    change (semantics + paths + hunks) as the gold `scenario.diff`.
    """
    system = dedent(
        """
        You are given a *reference* unified diff (ground truth) and an
        *AI-generated* diff.  Accept the AI diff only if it:
        • targets the same file(s) and
        • makes the same edits (after ignoring whitespace) and
        • introduces no extra changes.

        Reject if any file, hunk, or edit is missing, wrong, or extraneous.
        """
    )

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Reference diff:\n```\n{scenario.diff}\n```\n"
                f"AI diff:\n```\n{model_diff}\n```"
            ),
        },
    ]

    resp = await acompletion(
        model=JUDGE_MODEL,
        messages=messages,
        response_format=UpdateJudgeResponse,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = UpdateJudgeResponse.model_validate_json(raw)
        return parsed.accept
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Tools factory (each rollout gets fresh closures bound to its memory)
# -----------------------------------------------------------------------------
def build_tools(memory: Dict[str, str]) -> List[Callable[..., Any]]:
    def read_file(path: str) -> str:
        """Return file contents or error string."""
        return memory.get(path, f"ERROR: {path} not found")

    def list_files(dir_path: str | None = None) -> List[str]:
        """Recursive listing, relative to dir_path (default vault root)."""
        if dir_path is None:
            dir_path = ""
        prefix = dir_path.rstrip("/") + "/"
        return [p for p in memory if p.startswith(prefix)]

    def write_diff(diff: str) -> str:
        """Store diff text for later evaluation (no real patching)."""
        memory["__last_patch__"] = diff
        return diff

    def return_final_answer(answer: str, reference_paths: list[str] | None = None):
        return FinalAnswer(answer=answer, source_ids=reference_paths or [])

    return [read_file, list_files, write_diff, return_final_answer]


# -----------------------------------------------------------------------------
# Custom Trajectory
# -----------------------------------------------------------------------------
class ProjectTrajectory(art.Trajectory):
    """Extends ART’s Trajectory with a parsed `final_answer` field."""

    final_answer: FinalAnswer | None = None


# -----------------------------------------------------------------------------
# Main rollout op
# -----------------------------------------------------------------------------
@weave.op
async def rollout(
    model: art.Model, scenario: MDScenario
) -> ProjectTrajectory:  # noqa: C901
    """Single episode execution."""
    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": scenario.id,
            "task": scenario.task,
            "step": scenario.step,
        },
    )

    # ------------------------------------------------------------------ prompts
    system_prompt = dedent(
        f"""
        You are a **Markdown KB agent**.

        A snapshot of the user’s Obsidian vault is available through tools:
        • read_file(path)          – read markdown
        • list_files(dir="")       – list all files under dir
        • write_diff(diff)         – stage unified diff for an update task
        • return_final_answer(..)  – finish and output answer

        Take ≤ {MAX_TURNS} tool calls.  Always call return_final_answer when done.
        """
    )
    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.query},
    ]

    # ------------------------------------------------------------------ tools
    tools = build_tools(scenario.memory)
    tools_by = {t.__name__: t for t in tools}
    traj.tools = [convert_to_openai_tool(t) for t in tools]

    llm_name = f"hosted_vllm/{model.name}" if model.trainable else model.name

    # ------------------------------------------------------------------ loop
    for _ in range(MAX_TURNS):
        resp = await acompletion(
            model=llm_name,
            base_url=model.inference_base_url,
            api_key=model.inference_api_key,
            messages=traj.messages(),
            temperature=1.0,
            caching=False,
            tools=traj.tools,
        )
        choice = resp.choices[0]
        traj.messages_and_choices.append(
            art.utils.litellm.convert_litellm_choice_to_openai(choice)
        )

        # LLM didn’t request a tool → terminate unsuccessfully
        if not choice.message.tool_calls:
            return traj

        # Execute requested tools
        try:
            for tc in choice.message.tool_calls:
                tname = tc.function.name
                if tname not in tools_by:
                    continue
                result = tools_by[tname](**json.loads(tc.function.arguments or "{}"))
                traj.messages_and_choices.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tname,
                        "content": str(result),
                    }
                )

                # Finished?
                if tname == "return_final_answer":
                    traj.final_answer = result  # type: ignore

                    if traj.final_answer:
                        if scenario.task == TaskType.RETRIEVAL:
                            correct = await judge_correctness(
                                scenario, traj.final_answer.answer
                            )
                        else:  # UPDATE
                            correct = await judge_update(
                                scenario, traj.final_answer.answer
                            )
                        traj.metrics["correct"] = float(correct)
                        traj.reward = float(correct)
                    return traj
        except Exception as exc:  # noqa: BLE001
            # Tool crash – end trajectory silently (reward 0)
            print("TOOL-ERROR:", exc)
            return traj

    return traj  # max-turn fallback
