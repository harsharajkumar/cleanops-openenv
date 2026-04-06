"""OpenEnv server-side environment for operational data cleaning tasks."""

from __future__ import annotations

import copy
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from cleanops_env.graders import build_table_summary, grade_tables
from cleanops_env.models import (
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
    OperationDetail,
    OperationSummary,
    RewardBreakdown,
    RowChange,
    TableView,
)
from cleanops_env.tasks import (
    TaskSpec,
    apply_operation_to_tables,
    clone_tables,
    first_table_name,
    get_task_spec,
    list_task_ids,
    normalize_whitespace,
    sorted_rows,
)


class CleanOpsEnvironment(Environment[DataCleaningAction, DataCleaningObservation, DataCleaningState]):
    """A realistic data-cleaning workflow environment with deterministic graders."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._task_order = list_task_ids()
        self._task_spec = get_task_spec(self._task_order[0])
        self._grade = grade_tables(self._task_spec, self._task_spec.dirty_tables)
        self._focus_table_name = first_table_name(self._task_spec)
        self._focus_operation_detail: OperationDetail | None = None
        self._done = False
        self._initial_issue_count = max(1, len(self._grade.validation_issues))
        self._state = DataCleaningState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self._task_spec.task_id,
            task_title=self._task_spec.title,
            difficulty=self._task_spec.difficulty,
            requested_seed=None,
            max_steps=self._task_spec.max_steps,
            submitted=False,
            current_score=self._grade.score,
            best_score=self._grade.score,
            outstanding_issue_count=len(self._grade.validation_issues),
            tables=clone_tables(self._task_spec.dirty_tables),
            applied_operation_ids=[],
            inspected_tables=[self._focus_table_name],
            inspected_operations=[],
            recent_history=[],
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: object,
    ) -> DataCleaningObservation:
        del kwargs
        selected_task_id = task_id or self._task_order[0]
        self._task_spec = get_task_spec(selected_task_id)
        normalized_seed = seed if seed is None else max(0, int(seed))
        self._focus_table_name = self._choose_initial_focus_table(self._task_spec, normalized_seed)
        self._focus_operation_detail = None
        self._done = False
        self._grade = grade_tables(self._task_spec, self._task_spec.dirty_tables)
        self._initial_issue_count = max(1, len(self._grade.validation_issues))
        self._state = DataCleaningState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task_spec.task_id,
            task_title=self._task_spec.title,
            difficulty=self._task_spec.difficulty,
            requested_seed=normalized_seed,
            max_steps=self._task_spec.max_steps,
            submitted=False,
            current_score=self._grade.score,
            best_score=self._grade.score,
            outstanding_issue_count=len(self._grade.validation_issues),
            tables=clone_tables(self._task_spec.dirty_tables),
            applied_operation_ids=[],
            inspected_tables=[self._focus_table_name],
            inspected_operations=[],
            recent_history=[f"reset -> loaded task {self._task_spec.task_id} ({self._task_spec.difficulty}) seed={normalized_seed}"],
        )
        return self._build_observation(
            reward_breakdown=RewardBreakdown(total=0.0),
            reward=0.0,
            done=False,
            last_action_status=f"Environment reset to task {self._task_spec.task_id}.",
            last_action_error=None,
        )

    def step(
        self,
        action: DataCleaningAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> DataCleaningObservation:
        del timeout_s, kwargs
        if self._done:
            penalty = RewardBreakdown(invalid_action_penalty=-0.25, total=-0.25)
            return self._build_observation(
                reward_breakdown=penalty,
                reward=penalty.total,
                done=True,
                last_action_status="Episode already finished. Call reset() to start a new task.",
                last_action_error="Episode already finished. Call reset() to start a new task.",
            )

        self._state.step_count += 1
        previous_score = self._state.current_score
        previous_issue_count = self._state.outstanding_issue_count

        invalid_action_penalty = 0.0
        noop_penalty = 0.0
        insight_bonus = 0.0
        submit_bonus = 0.0
        status_message = ""
        action_error: str | None = None

        if action.action_type == "inspect_table":
            table_name = normalize_whitespace(action.table_name or "")
            if table_name not in self._state.tables:
                invalid_action_penalty = -0.25
                status_message = f"Unknown table '{table_name}'."
                action_error = status_message
            else:
                self._focus_table_name = table_name
                if table_name not in self._state.inspected_tables:
                    self._state.inspected_tables.append(table_name)
                    insight_bonus = 0.01
                    status_message = f"Inspected table '{table_name}'."
                else:
                    noop_penalty = -0.02
                    status_message = f"Table '{table_name}' was already inspected."
        elif action.action_type == "inspect_operation":
            operation_id = normalize_whitespace(action.operation_id or "")
            if operation_id not in self._task_spec.operations:
                invalid_action_penalty = -0.25
                status_message = f"Unknown operation '{operation_id}'."
                action_error = status_message
            else:
                self._focus_operation_detail = self._build_operation_detail(self._task_spec, operation_id, self._state.tables, None)
                if operation_id not in self._state.inspected_operations:
                    self._state.inspected_operations.append(operation_id)
                    insight_bonus = 0.01
                    status_message = f"Inspected operation '{operation_id}'."
                else:
                    noop_penalty = -0.02
                    status_message = f"Operation '{operation_id}' was already inspected."
        elif action.action_type == "apply_operation":
            operation_id = normalize_whitespace(action.operation_id or "")
            if operation_id not in self._task_spec.operations:
                invalid_action_penalty = -0.25
                status_message = f"Unknown operation '{operation_id}'."
                action_error = status_message
            elif operation_id in self._state.applied_operation_ids:
                noop_penalty = -0.12
                self._focus_operation_detail = self._build_operation_detail(self._task_spec, operation_id, self._state.tables, self._state.tables)
                status_message = f"Operation '{operation_id}' was already applied."
            else:
                before_tables = clone_tables(self._state.tables)
                after_tables = apply_operation_to_tables(self._task_spec, before_tables, operation_id)
                self._focus_operation_detail = self._build_operation_detail(self._task_spec, operation_id, before_tables, after_tables)
                if after_tables == before_tables:
                    noop_penalty = -0.08
                    status_message = f"Operation '{operation_id}' produced no table changes."
                else:
                    self._state.tables = clone_tables(after_tables)
                    self._state.applied_operation_ids.append(operation_id)
                    affected_tables = ", ".join(self._task_spec.operations[operation_id].tables_affected)
                    if self._task_spec.operations[operation_id].tables_affected:
                        self._focus_table_name = self._task_spec.operations[operation_id].tables_affected[0]
                    status_message = f"Applied '{operation_id}' to {affected_tables or 'current tables'}."
        elif action.action_type == "submit":
            self._state.submitted = True
            self._done = True
            status_message = "Submitted cleaned tables for grading."

        self._grade = grade_tables(self._task_spec, self._state.tables)
        self._state.current_score = self._grade.score
        self._state.best_score = max(self._state.best_score, self._grade.score)
        self._state.outstanding_issue_count = len(self._grade.validation_issues)

        quality_delta = round(self._state.current_score - previous_score, 4)
        issue_delta = round((previous_issue_count - self._state.outstanding_issue_count) / self._initial_issue_count, 4)
        efficiency_penalty = -0.01

        if action.action_type == "submit":
            submit_bonus = round(0.4 * self._state.current_score, 4) if self._state.current_score >= 0.8 else round(-0.2 * (1.0 - self._state.current_score), 4)

        if self._state.step_count >= self._state.max_steps and not self._done:
            self._done = True
            self._state.submitted = False
            status_message = f"{status_message} Step budget exhausted; episode truncated.".strip()

        reward_total = round(1.25 * quality_delta + 0.35 * issue_delta + insight_bonus + efficiency_penalty + invalid_action_penalty + noop_penalty + submit_bonus, 4)
        reward_breakdown = RewardBreakdown(
            quality_delta=quality_delta,
            issue_delta=issue_delta,
            insight_bonus=insight_bonus,
            efficiency_penalty=efficiency_penalty,
            invalid_action_penalty=invalid_action_penalty,
            noop_penalty=noop_penalty,
            submit_bonus=submit_bonus,
            total=reward_total,
        )

        action_descriptor = action.action_type
        if action.operation_id:
            action_descriptor += f"[{action.operation_id}]"
        if action.table_name:
            action_descriptor += f"[{action.table_name}]"
        self._state.recent_history.append(f"step {self._state.step_count}: {action_descriptor} -> score={self._state.current_score:.4f}")
        self._state.recent_history = self._state.recent_history[-10:]

        return self._build_observation(
            reward_breakdown=reward_breakdown,
            reward=reward_total,
            done=self._done,
            last_action_status=status_message or "Action processed.",
            last_action_error=action_error,
        )

    @property
    def state(self) -> DataCleaningState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="CleanOpsEnvironment",
            description="A realistic OpenEnv benchmark where an agent cleans operational customer, order, subscription, and payment tables using a curated data-cleaning toolkit.",
            version="0.1.0",
            author="OpenEnv CleanOps",
        )

    def _build_observation(
        self,
        *,
        reward_breakdown: RewardBreakdown,
        reward: float,
        done: bool,
        last_action_status: str,
        last_action_error: str | None,
    ) -> DataCleaningObservation:
        summaries = [build_table_summary(self._task_spec, table_name, self._state.tables) for table_name in self._task_spec.dirty_tables]
        focus_table = self._build_table_view(self._task_spec, self._focus_table_name)
        available_operations = [
            OperationSummary(
                operation_id=operation.operation_id,
                title=operation.title,
                category=operation.category,
                risk=operation.risk,
                tables_affected=list(operation.tables_affected),
                description=operation.description,
                already_applied=operation.operation_id in self._state.applied_operation_ids,
            )
            for operation in sorted(self._task_spec.operations.values(), key=lambda op: op.operation_id)
        ]
        return DataCleaningObservation(
            task_id=self._task_spec.task_id,
            task_title=self._task_spec.title,
            difficulty=self._task_spec.difficulty,
            requested_seed=self._state.requested_seed,
            objective=self._task_spec.objective,
            dataset_context=self._task_spec.dataset_context,
            quality_score=self._state.current_score,
            best_score=self._state.best_score,
            remaining_steps=max(0, self._state.max_steps - self._state.step_count),
            table_summaries=summaries,
            focus_table=focus_table,
            available_operations=available_operations,
            focus_operation=self._focus_operation_detail,
            validation_issues=self._grade.validation_issues,
            issue_cards=list(self._task_spec.issue_cards),
            recent_history=list(self._state.recent_history),
            grader=self._grade.breakdown,
            reward_breakdown=reward_breakdown,
            last_action_status=last_action_status,
            last_action_error=last_action_error,
            reward=reward,
            done=done,
            metadata={
                "episode_id": self._state.episode_id,
                "requested_seed": self._state.requested_seed,
                "applied_operation_ids": list(self._state.applied_operation_ids),
                "submitted": self._state.submitted,
            },
        )

    def _build_table_view(self, task_spec: TaskSpec, table_name: str) -> TableView:
        primary_key = task_spec.primary_keys[table_name]
        rows = self._preview_rows(task_spec, table_name, self._state.tables.get(table_name, []))
        columns = sorted({column_name for row in rows for column_name in row})
        return TableView(name=table_name, primary_key=primary_key, columns=columns, rows=rows)

    def _choose_initial_focus_table(self, task_spec: TaskSpec, seed: int | None) -> str:
        table_names = sorted(task_spec.dirty_tables)
        if not table_names:
            return first_table_name(task_spec)
        if seed is None:
            return table_names[0]
        return table_names[seed % len(table_names)]

    def _preview_rows(
        self,
        task_spec: TaskSpec,
        table_name: str,
        rows: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        primary_key = task_spec.primary_keys[table_name]
        ordered_rows = sorted_rows(rows, primary_key)
        seed = self._state.requested_seed
        if seed is None or len(ordered_rows) <= 1:
            return ordered_rows
        shuffled_rows = copy.deepcopy(ordered_rows)
        random.Random(seed + sum(ord(char) for char in table_name)).shuffle(shuffled_rows)
        return shuffled_rows

    def _build_operation_detail(
        self,
        task_spec: TaskSpec,
        operation_id: str,
        before_tables: dict[str, list[dict[str, str]]],
        after_tables: dict[str, list[dict[str, str]]] | None,
    ) -> OperationDetail:
        operation = task_spec.operations[operation_id]
        simulated_after = after_tables
        if simulated_after is None:
            simulated_after = apply_operation_to_tables(task_spec, before_tables, operation_id)

        preview: list[RowChange] = []
        for table_name in operation.tables_affected:
            primary_key = task_spec.primary_keys[table_name]
            before_rows = {normalize_whitespace(row.get(primary_key, "")): dict(row) for row in before_tables.get(table_name, [])}
            after_rows = {normalize_whitespace(row.get(primary_key, "")): dict(row) for row in simulated_after.get(table_name, [])}
            changed_keys = sorted(set(before_rows) | set(after_rows))
            for row_key in changed_keys:
                if before_rows.get(row_key) == after_rows.get(row_key):
                    continue
                preview.append(RowChange(primary_key_value=row_key, before=before_rows.get(row_key), after=after_rows.get(row_key)))
                if len(preview) >= 12:
                    break
            if len(preview) >= 12:
                break

        return OperationDetail(
            operation_id=operation.operation_id,
            title=operation.title,
            category=operation.category,
            risk=operation.risk,
            tables_affected=list(operation.tables_affected),
            description=operation.description,
            already_applied=operation.operation_id in self._state.applied_operation_ids,
            why_it_matters=operation.why_it_matters,
            change_preview=preview,
        )
