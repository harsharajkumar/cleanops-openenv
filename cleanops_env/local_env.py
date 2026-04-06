"""Local in-process wrapper with a Gym-style step/reset/state interface."""

from __future__ import annotations

from typing import Any

from cleanops_env.environment import CleanOpsEnvironment
from cleanops_env.models import DataCleaningAction, DataCleaningObservation, DataCleaningState
from cleanops_env.tasks import list_task_ids


class LocalCleanOpsEnv:
    """Direct local environment wrapper used by tests and baseline scripts."""

    def __init__(self) -> None:
        self._env = CleanOpsEnvironment()

    @property
    def task_ids(self) -> list[str]:
        return list_task_ids()

    def reset(self, task_id: str | None = None, seed: int | None = None, episode_id: str | None = None) -> DataCleaningObservation:
        return self._env.reset(seed=seed, episode_id=episode_id, task_id=task_id)

    def step(self, action: DataCleaningAction, **kwargs: Any) -> tuple[DataCleaningObservation, float, bool, dict[str, Any]]:
        observation = self._env.step(action, **kwargs)
        info = {
            "state": self._env.state.model_dump(),
            "grader": observation.grader.model_dump(),
            "reward_breakdown": observation.reward_breakdown.model_dump(),
            "last_action_status": observation.last_action_status,
            "last_action_error": observation.last_action_error,
        }
        return observation, float(observation.reward or 0.0), bool(observation.done), info

    def state(self) -> DataCleaningState:
        return self._env.state

    def close(self) -> None:
        self._env.close()
