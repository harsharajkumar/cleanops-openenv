"""Submission inference runner for CleanOps OpenEnv."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import textwrap
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleanops_env import CleanOpsEnvClient, DataCleaningAction, LocalCleanOpsEnv
from cleanops_env.models import DataCleaningObservation
from cleanops_env.tasks import list_task_ids

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAME = os.getenv("TASK_NAME", "all")
BENCHMARK = os.getenv("BENCHMARK", "cleanops_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "18"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.95"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a data-cleaning operations agent working in the CleanOps OpenEnv benchmark.
    Choose exactly one JSON action per turn using this schema:
    {
      "action_type": "inspect_table" | "inspect_operation" | "apply_operation" | "submit",
      "table_name": string | null,
      "operation_id": string | null,
      "reasoning": string
    }
    Prefer safe/review operations that directly resolve current validation issues.
    Avoid destructive operations unless the task objective explicitly asks for deletions.
    Submit once quality_score is high and remaining validation issues are gone.
    Return only a single JSON object.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    safe_action = action.replace("\n", " ").replace("\r", " ").strip()
    safe_error = error.replace("\n", " ").replace("\r", " ").strip() if error else "null"
    print(f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={str(done).lower()} error={safe_error}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def build_observation_prompt(observation: DataCleaningObservation) -> str:
    payload = {
        "task_id": observation.task_id,
        "difficulty": observation.difficulty,
        "objective": observation.objective,
        "quality_score": observation.quality_score,
        "remaining_steps": observation.remaining_steps,
        "table_summaries": [summary.model_dump() for summary in observation.table_summaries],
        "focus_table": observation.focus_table.model_dump() if observation.focus_table else None,
        "focus_operation": observation.focus_operation.model_dump() if observation.focus_operation else None,
        "available_operations": [operation.model_dump() for operation in observation.available_operations],
        "validation_issues": [issue.model_dump() for issue in observation.validation_issues],
        "issue_cards": [issue_card.model_dump() for issue_card in observation.issue_cards],
        "recent_history": observation.recent_history,
        "last_action_status": observation.last_action_status,
        "last_action_error": observation.last_action_error,
        "grader": observation.grader.model_dump(),
    }
    return json.dumps(payload, separators=(",", ":"))


def fallback_action(observation: DataCleaningObservation) -> DataCleaningAction:
    for issue_card in observation.issue_cards:
        for operation_id in issue_card.recommended_operation_ids:
            operation = next((candidate for candidate in observation.available_operations if candidate.operation_id == operation_id), None)
            if operation and not operation.already_applied and operation.risk != "destructive":
                return DataCleaningAction(action_type="apply_operation", operation_id=operation.operation_id, reasoning=f"Apply recommended operation {operation.operation_id}.")
    for operation in observation.available_operations:
        if not operation.already_applied and operation.risk != "destructive":
            return DataCleaningAction(action_type="apply_operation", operation_id=operation.operation_id, reasoning=f"Apply next safe operation {operation.operation_id}.")
    return DataCleaningAction(action_type="submit", reasoning="Submit after exhausting all safe non-destructive operations.")


def choose_action(client: OpenAI | None, observation: DataCleaningObservation) -> DataCleaningAction:
    if observation.remaining_steps <= 1 and not observation.validation_issues:
        return DataCleaningAction(action_type="submit", reasoning="Submit on final clean step.")
    if client is None:
        return fallback_action(observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_observation_prompt(observation)}],
            temperature=0.0,
            max_tokens=256,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        action_payload = json.loads(content)
        return DataCleaningAction.model_validate(action_payload)
    except Exception:
        return fallback_action(observation)


def action_to_string(action: DataCleaningAction) -> str:
    if action.action_type == "inspect_table":
        return f"inspect_table({action.table_name})"
    if action.action_type == "inspect_operation":
        return f"inspect_operation({action.operation_id})"
    if action.action_type == "apply_operation":
        return f"apply_operation({action.operation_id})"
    return "submit()"


def create_env() -> Any:
    if LOCAL_IMAGE_NAME:
        return CleanOpsEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
    return LocalCleanOpsEnv()


def run_episode(task_name: str) -> None:
    env = None
    rewards: list[float] = []
    steps_taken = 0
    success = False
    final_score = 0.0
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    try:
        env = create_env()
        result = env.reset(task_id=task_name, seed=7)
        observation = result.observation if hasattr(result, "observation") else result
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "EMPTY", timeout=30.0) if HF_TOKEN else None
        for step in range(1, MAX_STEPS + 1):
            if observation.done:
                break
            action = choose_action(client, observation)
            step_result = env.step(action)
            if isinstance(step_result, tuple):
                observation, reward, done, info = step_result
                error = info.get("last_action_error")
            else:
                observation = step_result.observation
                reward = float(step_result.reward or 0.0)
                done = bool(step_result.done)
                error = observation.last_action_error
            rewards.append(float(reward))
            steps_taken = step
            log_step(step=step, action=action_to_string(action), reward=float(reward), done=bool(done), error=error)
            if done:
                break
        final_score = float(observation.quality_score)
        success = final_score >= SUCCESS_SCORE_THRESHOLD and observation.done
    except Exception as exc:
        log_step(step=max(1, steps_taken + 1), action="submit()", reward=0.0, done=True, error=str(exc))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


def main() -> None:
    task_names = list_task_ids() if TASK_NAME == "all" else [TASK_NAME]
    for task_name in task_names:
        run_episode(task_name)


if __name__ == "__main__":
    main()
