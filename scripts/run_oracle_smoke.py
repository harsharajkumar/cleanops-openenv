"""Deterministic smoke baseline that applies each task's known solution sequence."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleanops_env.local_env import LocalCleanOpsEnv
from cleanops_env.models import DataCleaningAction
from cleanops_env.tasks import get_task_spec, list_task_ids


def run_oracle() -> dict[str, object]:
    env = LocalCleanOpsEnv()
    task_results = []
    for task_id in list_task_ids():
        task_spec = get_task_spec(task_id)
        observation = env.reset(task_id=task_id, seed=7)
        total_reward = 0.0
        done = observation.done
        step_count = 0
        for operation_id in task_spec.solution_operation_ids:
            observation, reward, done, _ = env.step(
                DataCleaningAction(action_type="apply_operation", operation_id=operation_id, reasoning=f"Apply known-cleaning operation {operation_id}.")
            )
            total_reward += reward
            step_count += 1
            if done:
                break
        if not done:
            observation, reward, done, _ = env.step(DataCleaningAction(action_type="submit", reasoning="Submit deterministic oracle solution."))
            total_reward += reward
            step_count += 1
        task_results.append(
            {
                "task_id": task_id,
                "difficulty": task_spec.difficulty,
                "final_score": observation.quality_score,
                "grader": observation.grader.model_dump(),
                "steps": step_count,
                "total_reward": round(total_reward, 4),
                "done": done,
            }
        )
    return {
        "agent": "oracle_solution_sequence",
        "tasks": task_results,
        "mean_score": round(sum(item["final_score"] for item in task_results) / len(task_results), 4),
    }


def main() -> None:
    print(json.dumps(run_oracle(), indent=2))


if __name__ == "__main__":
    main()

