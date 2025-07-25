"""
train.py – Tiny end-to-end fine-tuning loop with ART, RULER, and our rollout.
----------------------------------------------------------------------------
• Uses in-memory markdown dataset from JSON files.
• Creates one MDScenario per question (retrieval only in this demo).
• For each global step:
      – sample N rollouts per scenario
      – score them with RULER
      – update model with ART PPO style

Swap out the toy dataset / model with real ones as needed.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import List

import art
from art.rewards import ruler_score_group
from art.utils import iterate_dataset
from art.local import LocalBackend
from load_data import load_scenarios
from dotenv import load_dotenv

load_dotenv()
scenarios = load_scenarios("data")

from rollout import MDScenario, TaskType, rollout

# --------------------------------------------------------------------------- #
#                               Training config                               #
# --------------------------------------------------------------------------- #
cfg = {
    "groups_per_step": 2,
    "num_epochs": 3,
    "rollouts_per_group": 4,
    "learning_rate": 1e-5,
    "max_steps": 12,
}

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
model = art.TrainableModel(base_model=MODEL_ID, name="MarkdownAgent", project="markdown-agent-art",trainable=True)


# --------------------------------------------------------------------------- #
#                                  Main loop                                  #
# --------------------------------------------------------------------------- #
async def main() -> None:

    backend = LocalBackend()

    # Register the model with the local backend (sets up logging, inference, and training)
    await model.register(backend)

    iterator = iterate_dataset(
        scenarios,
        groups_per_step=cfg["groups_per_step"],
        num_epochs=cfg["num_epochs"],
        initial_step=await model.get_step(),
    )

    for batch, epoch, g_step, epoch_step in iterator:
        print(f"\n=== Global {g_step}  (epoch {epoch} / {epoch_step})  ===")

        # 1️⃣  Rollout groups
        groups = [
            art.TrajectoryGroup(
                rollout(model, sc.model_copy(update={"step": g_step}))  # type: ignore
                for _ in range(cfg["rollouts_per_group"])
            )
            for sc in batch
        ]

        finished = await art.gather_trajectory_groups(
            groups,
            pbar_desc="rollouts",
            max_exceptions=cfg["rollouts_per_group"] * len(batch),
        )

        # 2️⃣  RULER – relative ranking per group
        judged = [
            await ruler_score_group(gr, "openai/o4-mini", debug=False)
            for gr in finished
        ]

        # 3️⃣  Gradient step
        await model.delete_checkpoints()
        await model.train(
            judged,
            config=art.TrainConfig(learning_rate=cfg["learning_rate"]),
            _config={"logprob_calculation_chunk_size": 8},
        )

        if g_step >= cfg["max_steps"]:
            break

    # Optional: push final checkpoint
    # await model.push_to_hub("your-hf-org/markdown-kb-agent")


if __name__ == "__main__":
    asyncio.run(main())
