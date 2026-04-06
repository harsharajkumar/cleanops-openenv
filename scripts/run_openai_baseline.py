"""OpenAI baseline agent for CleanOps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleanops_env.local_env import LocalCleanOpsEnv
from cleanops_env.models import DataCleaningAction, DataCleaningObservation
from cleanops_env.tasks import list_task_ids

SYSTEM_PROMPT = """You are a careful data-cleaning operations agent.
Your job is to improve the current task score by choosing one JSON action at a time.
Use only this JSON schema:
{
  "action_type": "inspect_table" | "inspect_operation" | "apply_operation" | "submit",
  "table_name": string | null,
  "operation_id": string | null,
  "reasoning": string
}
Rules:
- Prefer safe/review operations that directly address unresolved validation issues.
- Avoid destructive operations unless the objective explicitly asks for row deletion.
- Call submit only when the data looks clean or there is 1 step left.
- Return a single JSON object and no extra text."""


def compact_observation(observation: DataCleaningObservation) -> dict[str, Any]:
    return {
        "task_id": observation.task_id,
        "task_title": observation.task_title,
        "difficulty": observation.difficulty,
        "objective": observation.objective,
        "dataset_context": observation.dataset_context,
        "quality_score": observation.quality_score,
        "remaining_steps": observation.remaining_steps,
        "last_action_status": observation.last_action_status,
        "recent_history": observation.recent_history[-5:],
        "table_summaries": [summary.model_dump() for summary in observation.table_summaries],
        "focus_table": observation.focus_table.model_dump() if observation.focus_table else None,
        "focus_operation": observation.focus_operation.model_dump() if observation.focus_operation else None,
        "available_operations": [operation.model_dump() for operation in observation.available_operations],
        "validation_issues": [issue.model_dump() for issue in observation.validation_issues],
        "issue_cards": [issue_card.model_dump() for issue_card in observation.issue_cards],
        "grader": observation.grader.model_dump(),
    }


def choose_action(client: OpenAI, model: str, seed: int, observation: DataCleaningObservation) -> DataCleaningAction:
    payload = compact_observation(observation)
    request_kwargs = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
        "response_format": {"type": "json_object"},
    }
    try:
        request_kwargs["seed"] = seed
        response = client.chat.completions.create(**request_kwargs)
    except TypeError:
        request_kwargs.pop("seed", None)
        response = client.chat.completions.create(**request_kwargs)

    content = response.choices[0].message.content or "{}"
    try:
        action_payload = json.loads(content)
        return DataCleaningAction.model_validate(action_payload)
    except Exception:
        fallback_operation = next((op.operation_id for op in observation.available_operations if not op.already_applied and op.risk != "destructive"), None)
        if observation.remaining_steps <= 1 or fallback_operation is None:
            return DataCleaningAction(action_type="submit", reasoning="Fallback submit because model output could not be parsed.")
        return DataCleaningAction(action_type="apply_operation", operation_id=fallback_operation, reasoning="Fallback safe operation after parse failure.")


def run_baseline(model: str, seed: int) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running this baseline.")

    openai_client = OpenAI()
    env = LocalCleanOpsEnv()
    results = []
    for task_id in list_task_ids():
        observation = env.reset(task_id=task_id, seed=seed)
        done = observation.done
        total_reward = 0.0
        step_count = 0
        trajectory = []
        while not done:
            action = choose_action(openai_client, model, seed + step_count, observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            trajectory.append(
                {
                    "action": action.model_dump(),
                    "reward": reward,
                    "score": observation.quality_score,
                    "done": done,
                    "status": info["last_action_status"],
                }
            )
            if step_count >= 32:
                break
        results.append(
            {
                "task_id": task_id,
                "final_score": observation.quality_score,
                "grader": observation.grader.model_dump(),
                "steps": step_count,
                "total_reward": round(total_reward, 4),
                "trajectory": trajectory,
            }
        )
    return {
        "agent": "openai_chat_completions",
        "model": model,
        "seed": seed,
        "tasks": results,
        "mean_score": round(sum(item["final_score"] for item in results) / len(results), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("OPENAI_SEED", "7")))
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    report = run_baseline(model=args.model, seed=args.seed)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

