"""OpenEnv WebSocket client for CleanOps."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from cleanops_env.models import DataCleaningAction, DataCleaningObservation, DataCleaningState


class CleanOpsEnvClient(EnvClient[DataCleaningAction, DataCleaningObservation, DataCleaningState]):
    """Typed client for interacting with a running CleanOps server."""

    def _step_payload(self, action: DataCleaningAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[DataCleaningObservation]:
        obs_payload = payload.get("observation", {})
        observation = DataCleaningObservation.model_validate(
            {
                **obs_payload,
                "reward": payload.get("reward", obs_payload.get("reward")),
                "done": payload.get("done", obs_payload.get("done", False)),
            }
        )
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: dict[str, Any]) -> DataCleaningState:
        return DataCleaningState.model_validate(payload)

    def step_tuple(self, action: DataCleaningAction, **kwargs: Any) -> tuple[DataCleaningObservation, float, bool, dict[str, Any]]:
        """Convenience adapter that returns (observation, reward, done, info)."""

        result = self.step(action, **kwargs)
        info = {
            "state": self.state().model_dump(),
            "grader": result.observation.grader.model_dump(),
            "reward_breakdown": result.observation.reward_breakdown.model_dump(),
            "last_action_status": result.observation.last_action_status,
            "last_action_error": result.observation.last_action_error,
        }
        return result.observation, float(result.reward or 0.0), bool(result.done), info

