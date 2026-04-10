"""OpenEnv server-side environment for operational data cleaning tasks."""

from __future__ import annotations

import copy
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from cleanops_env.graders import build_table_summary, count_duplicate_groups, grade_tables
from cleanops_env.models import (
    ActionCostEntry,
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
    DownstreamHealth,
    DryRunFinding,
    DryRunReport,
    OperationDetail,
    OperationSummary,
    PendingReview,
    ReviewResolution,
    ReviewTarget,
    RiskCard,
    RewardBreakdown,
    RowChange,
    TableView,
)
from cleanops_env.tasks import (
    ReviewCaseSpec,
    TaskSpec,
    apply_operation_to_tables,
    clone_tables,
    first_table_name,
    get_task_spec,
    list_task_ids,
    normalize_whitespace,
    sorted_rows,
)

ACTION_COSTS: dict[str, float] = {
    "inspect_table": 0.005,
    "inspect_operation": 0.005,
    "apply_operation:safe": 0.01,
    "apply_operation:review": 0.015,
    "apply_operation:destructive": 0.03,
    "request_review": 0.025,
    "run_sync_dry_run": 0.02,
    "submit": 0.005,
}

ACTION_COST_DESCRIPTIONS: dict[str, str] = {
    "inspect_table": "Low-cost inspection to understand current records.",
    "inspect_operation": "Low-cost preview to inspect an operation before applying it.",
    "apply_operation:safe": "Safe automated cleanup with low operational risk.",
    "apply_operation:review": "Review-sensitive cleanup that should be used more deliberately.",
    "apply_operation:destructive": "Destructive cleanup with higher business risk if applied incorrectly.",
    "request_review": "Consumes limited human-review budget to resolve ambiguity safely.",
    "run_sync_dry_run": "Runs a deterministic downstream system simulation before submit.",
    "submit": "Low-cost finalization step after cleanup is complete.",
}


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
        initial_tables = clone_tables(self._task_spec.dirty_tables)
        initial_downstream_health = self._compute_downstream_health(self._task_spec, initial_tables, self._grade.validation_issues)
        self._state = DataCleaningState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self._task_spec.task_id,
            task_title=self._task_spec.title,
            difficulty=self._task_spec.difficulty,
            requested_seed=None,
            max_steps=self._task_spec.max_steps,
            review_budget_total=self._task_spec.review_budget,
            review_budget_remaining=self._task_spec.review_budget,
            submitted=False,
            current_score=self._grade.score,
            best_score=self._grade.score,
            outstanding_issue_count=len(self._grade.validation_issues),
            downstream_health=initial_downstream_health,
            last_dry_run=None,
            tables=initial_tables,
            applied_operation_ids=[],
            inspected_tables=[self._focus_table_name],
            inspected_operations=[],
            requested_review_ids=[],
            pending_reviews=[],
            resolved_reviews=[],
            dry_run_targets=[],
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
        initial_tables = clone_tables(self._task_spec.dirty_tables)
        initial_downstream_health = self._compute_downstream_health(self._task_spec, initial_tables, self._grade.validation_issues)
        self._state = DataCleaningState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task_spec.task_id,
            task_title=self._task_spec.title,
            difficulty=self._task_spec.difficulty,
            requested_seed=normalized_seed,
            max_steps=self._task_spec.max_steps,
            review_budget_total=self._task_spec.review_budget,
            review_budget_remaining=self._task_spec.review_budget,
            submitted=False,
            current_score=self._grade.score,
            best_score=self._grade.score,
            outstanding_issue_count=len(self._grade.validation_issues),
            downstream_health=initial_downstream_health,
            last_dry_run=None,
            tables=initial_tables,
            applied_operation_ids=[],
            inspected_tables=[self._focus_table_name],
            inspected_operations=[],
            requested_review_ids=[],
            pending_reviews=[],
            resolved_reviews=[],
            dry_run_targets=[],
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
        previous_downstream_score = self._state.downstream_health.overall_health_score

        invalid_action_penalty = 0.0
        noop_penalty = 0.0
        insight_bonus = 0.0
        review_bonus = 0.0
        review_cost_penalty = 0.0
        action_cost_penalty = 0.0
        submit_bonus = 0.0
        status_message = ""
        action_error: str | None = None
        released_reviews = self._release_ready_reviews()
        if released_reviews:
            review_bonus = round(0.04 * len(released_reviews), 4)

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
        elif action.action_type == "request_review":
            entity_type = normalize_whitespace(action.entity_type or "").lower()
            entity_id = normalize_whitespace(action.entity_id or "")
            reason_code = normalize_whitespace(action.reason_code or "")
            review_case = self._find_review_case(entity_type, entity_id, reason_code)
            if not entity_type or not entity_id or not reason_code:
                invalid_action_penalty = -0.25
                status_message = "request_review requires entity_type, entity_id, and reason_code."
                action_error = status_message
            elif review_case is None:
                invalid_action_penalty = -0.2
                status_message = f"No deterministic review case exists for {entity_type}:{entity_id} ({reason_code})."
                action_error = status_message
            elif review_case.review_id in self._state.requested_review_ids:
                noop_penalty = -0.05
                status_message = f"Review '{review_case.review_id}' was already requested."
            elif self._state.review_budget_remaining <= 0:
                invalid_action_penalty = -0.18
                status_message = "No review budget remaining for this episode."
                action_error = status_message
            else:
                self._state.review_budget_remaining -= 1
                self._state.requested_review_ids.append(review_case.review_id)
                self._state.pending_reviews.append(
                    PendingReview(
                        review_id=review_case.review_id,
                        entity_type=review_case.entity_type,
                        entity_id=review_case.entity_id,
                        reason_code=review_case.reason_code,
                        title=review_case.title,
                        requested_at_step=self._state.step_count,
                        ready_at_step=self._state.step_count + 1,
                    )
                )
                review_cost_penalty = -0.02
                status_message = (
                    f"Queued review '{review_case.review_id}' for {review_case.entity_type} {review_case.entity_id}; "
                    "response will be available on the next step."
                )
        elif action.action_type == "run_sync_dry_run":
            target_system = action.target_system
            if target_system is None:
                invalid_action_penalty = -0.2
                status_message = "run_sync_dry_run requires target_system."
                action_error = status_message
            elif target_system not in self._task_spec.sync_targets:
                invalid_action_penalty = -0.2
                status_message = f"Task '{self._task_spec.task_id}' does not support dry-run target '{target_system}'."
                action_error = status_message
            else:
                self._state.last_dry_run = self._build_dry_run_report(target_system)
                if target_system not in self._state.dry_run_targets:
                    self._state.dry_run_targets.append(target_system)
                    insight_bonus = max(insight_bonus, 0.01)
                else:
                    noop_penalty = min(noop_penalty, -0.01)
                status_message = self._state.last_dry_run.summary
        elif action.action_type == "submit":
            self._state.submitted = True
            self._done = True
            status_message = "Submitted cleaned tables for grading."

        action_cost_penalty = -self._estimate_action_cost(action)

        self._grade = grade_tables(self._task_spec, self._state.tables)
        self._state.current_score = self._grade.score
        self._state.best_score = max(self._state.best_score, self._grade.score)
        self._state.outstanding_issue_count = len(self._grade.validation_issues)
        self._state.downstream_health = self._compute_downstream_health(self._task_spec, self._state.tables, self._grade.validation_issues)

        quality_delta = round(self._state.current_score - previous_score, 4)
        issue_delta = round((previous_issue_count - self._state.outstanding_issue_count) / self._initial_issue_count, 4)
        downstream_health_delta = round(self._state.downstream_health.overall_health_score - previous_downstream_score, 4)
        efficiency_penalty = -0.01

        if action.action_type == "submit":
            submission_health = round(0.65 * self._state.current_score + 0.35 * self._state.downstream_health.overall_health_score, 4)
            submit_bonus = round(0.4 * submission_health, 4) if submission_health >= 0.82 else round(-0.2 * (1.0 - submission_health), 4)

        if self._state.step_count >= self._state.max_steps and not self._done:
            self._done = True
            self._state.submitted = False
            status_message = f"{status_message} Step budget exhausted; episode truncated.".strip()

        if released_reviews:
            release_note = ", ".join(review.review_id for review in released_reviews)
            status_message = f"{status_message} Review response available: {release_note}.".strip()

        reward_total = round(
            1.0 * quality_delta
            + 0.35 * issue_delta
            + 0.55 * downstream_health_delta
            + insight_bonus
            + review_bonus
            + efficiency_penalty
            + invalid_action_penalty
            + noop_penalty
            + review_cost_penalty
            + action_cost_penalty
            + submit_bonus,
            4,
        )
        reward_breakdown = RewardBreakdown(
            quality_delta=quality_delta,
            issue_delta=issue_delta,
            downstream_health_delta=downstream_health_delta,
            insight_bonus=insight_bonus,
            review_bonus=review_bonus,
            efficiency_penalty=efficiency_penalty,
            invalid_action_penalty=invalid_action_penalty,
            noop_penalty=noop_penalty,
            review_cost_penalty=review_cost_penalty,
            action_cost_penalty=action_cost_penalty,
            submit_bonus=submit_bonus,
            total=reward_total,
        )

        action_descriptor = action.action_type
        if action.operation_id:
            action_descriptor += f"[{action.operation_id}]"
        if action.table_name:
            action_descriptor += f"[{action.table_name}]"
        if action.entity_id:
            action_descriptor += f"[{action.entity_id}]"
        if action.target_system:
            action_descriptor += f"[{action.target_system}]"
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
        available_review_targets = [
            ReviewTarget(
                review_id=review_case.review_id,
                entity_type=review_case.entity_type,
                entity_id=review_case.entity_id,
                reason_code=review_case.reason_code,
                title=review_case.title,
                detail=review_case.detail,
                recommended_operation_ids=list(review_case.recommended_operation_ids),
            )
            for review_case in sorted(self._task_spec.review_cases.values(), key=lambda case: case.review_id)
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
            review_budget_remaining=self._state.review_budget_remaining,
            supported_sync_targets=list(self._task_spec.sync_targets),
            downstream_health=self._state.downstream_health,
            risk_cards=self._build_risk_cards(),
            last_dry_run=self._state.last_dry_run,
            action_costs=self._build_action_cost_entries(),
            table_summaries=summaries,
            focus_table=focus_table,
            available_operations=available_operations,
            available_review_targets=available_review_targets,
            pending_reviews=list(self._state.pending_reviews),
            resolved_reviews=list(self._state.resolved_reviews),
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
                "review_budget_remaining": self._state.review_budget_remaining,
                "requested_review_ids": list(self._state.requested_review_ids),
                "dry_run_targets": list(self._state.dry_run_targets),
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

    def _find_review_case(self, entity_type: str, entity_id: str, reason_code: str) -> ReviewCaseSpec | None:
        for review_case in self._task_spec.review_cases.values():
            if (
                review_case.entity_type == entity_type
                and review_case.entity_id == entity_id
                and review_case.reason_code == reason_code
            ):
                return review_case
        return None

    def _release_ready_reviews(self) -> list[ReviewResolution]:
        if not self._state.pending_reviews:
            return []

        still_pending: list[PendingReview] = []
        released: list[ReviewResolution] = []
        for pending_review in self._state.pending_reviews:
            if pending_review.ready_at_step > self._state.step_count:
                still_pending.append(pending_review)
                continue
            review_case = self._task_spec.review_cases[pending_review.review_id]
            released_review = ReviewResolution(
                review_id=review_case.review_id,
                entity_type=review_case.entity_type,
                entity_id=review_case.entity_id,
                reason_code=review_case.reason_code,
                title=review_case.title,
                resolution=review_case.resolution,
                response_summary=review_case.response_summary,
                evidence_summary=review_case.evidence_summary,
                recommended_operation_ids=list(review_case.recommended_operation_ids),
            )
            self._state.resolved_reviews.append(released_review)
            released.append(released_review)
        self._state.pending_reviews = still_pending
        return released

    def _estimate_action_cost(self, action: DataCleaningAction) -> float:
        if action.action_type == "apply_operation":
            operation = self._task_spec.operations.get(normalize_whitespace(action.operation_id or ""))
            if operation is None:
                return ACTION_COSTS["apply_operation:safe"]
            if operation.risk == "review":
                return ACTION_COSTS["apply_operation:review"]
            if operation.risk == "destructive":
                return ACTION_COSTS["apply_operation:destructive"]
            return ACTION_COSTS["apply_operation:safe"]
        return ACTION_COSTS.get(action.action_type, 0.01)

    def _build_action_cost_entries(self) -> list[ActionCostEntry]:
        return [
            ActionCostEntry(action_key=action_key, estimated_cost=estimated_cost, description=ACTION_COST_DESCRIPTIONS[action_key])
            for action_key, estimated_cost in ACTION_COSTS.items()
        ]

    @staticmethod
    def _open_metric(value: float) -> float:
        return round(min(0.99, max(0.01, value)), 4)

    def _compute_downstream_health(
        self,
        task_spec: TaskSpec,
        tables: dict[str, list[dict[str, str]]],
        validation_issues: list,
    ) -> DownstreamHealth:
        customers = tables.get("customers", [])
        orders = tables.get("orders", [])
        subscriptions = tables.get("subscriptions", [])
        payments = tables.get("payments", [])

        crm_rows = max(1, len(customers) + len(subscriptions))
        billing_rows = max(1, len(orders) + len(subscriptions) + len(payments))
        payment_rows = max(1, len(orders) + len(payments))

        crm_issue_weight = sum(max(1, len(issue.row_ids)) for issue in validation_issues if issue.table_name in {"customers", "subscriptions"})
        billing_issue_weight = sum(
            max(1, len(issue.row_ids))
            for issue in validation_issues
            if issue.table_name in {"orders", "payments", "subscriptions"}
            and (issue.code.startswith("foreign_key:") or issue.code.startswith("required:") or issue.code.startswith("unique:"))
        )
        payment_issue_weight = sum(
            max(1, len(issue.row_ids))
            for issue in validation_issues
            if issue.table_name in {"orders", "payments"}
        )

        customer_duplicate_groups = count_duplicate_groups(task_spec, "customers", customers) if "customers" in task_spec.duplicate_identity_columns else 0
        customer_rows = max(1, len(customers))
        payment_duplicate_groups = count_duplicate_groups(task_spec, "payments", payments) if "payments" in task_spec.duplicate_identity_columns else 0

        crm_sync_success_rate = self._open_metric(1.0 - (crm_issue_weight / max(2, crm_rows * 2)))
        if not orders and not payments:
            billing_link_integrity = 0.99
            revenue_reporting_risk = 0.01
        else:
            billing_link_integrity = self._open_metric(1.0 - (billing_issue_weight / max(2, billing_rows * 2)))
            revenue_reporting_risk = self._open_metric(min(0.99, (payment_issue_weight / max(2, payment_rows * 2)) + (payment_duplicate_groups / max(1, payment_rows))))

        duplicate_contact_risk = self._open_metric(min(0.99, (customer_duplicate_groups / customer_rows) + 0.06 * sum(1 for issue in validation_issues if issue.code.startswith("unique:customers"))))
        overall_health_score = self._open_metric(
            (
                crm_sync_success_rate
                + billing_link_integrity
                + (1.0 - duplicate_contact_risk)
                + (1.0 - revenue_reporting_risk)
            )
            / 4.0
        )

        return DownstreamHealth(
            crm_sync_success_rate=crm_sync_success_rate,
            billing_link_integrity=billing_link_integrity,
            duplicate_contact_risk=duplicate_contact_risk,
            revenue_reporting_risk=revenue_reporting_risk,
            overall_health_score=overall_health_score,
        )

    def _build_risk_cards(self) -> list[RiskCard]:
        health = self._state.downstream_health
        cards = [
            RiskCard(
                title="CRM import risk",
                detail="Customer and subscription issues can block CRM migration syncs.",
                severity="high" if health.crm_sync_success_rate < 0.8 else "medium" if health.crm_sync_success_rate < 0.92 else "low",
                metric_name="crm_sync_success_rate",
                current_value=health.crm_sync_success_rate,
                recommended_action_ids=[op_id for op_id in self._recommended_operation_ids_for_tables({"customers", "subscriptions"})],
            ),
            RiskCard(
                title="Billing linkage risk",
                detail="Broken foreign keys or missing IDs can mislink orders, subscriptions, and payments.",
                severity="high" if health.billing_link_integrity < 0.8 else "medium" if health.billing_link_integrity < 0.92 else "low",
                metric_name="billing_link_integrity",
                current_value=health.billing_link_integrity,
                recommended_action_ids=[op_id for op_id in self._recommended_operation_ids_for_tables({"orders", "subscriptions", "payments"})],
            ),
            RiskCard(
                title="Duplicate contact risk",
                detail="Remaining duplicate customer identities can create bad merges downstream.",
                severity="high" if health.duplicate_contact_risk > 0.3 else "medium" if health.duplicate_contact_risk > 0.12 else "low",
                metric_name="duplicate_contact_risk",
                current_value=health.duplicate_contact_risk,
                recommended_action_ids=[op_id for op_id in self._recommended_operation_ids_for_keyword("merge")],
            ),
            RiskCard(
                title="Revenue reporting risk",
                detail="Duplicate or mislinked payment and order facts can distort downstream reporting.",
                severity="high" if health.revenue_reporting_risk > 0.3 else "medium" if health.revenue_reporting_risk > 0.12 else "low",
                metric_name="revenue_reporting_risk",
                current_value=health.revenue_reporting_risk,
                recommended_action_ids=[op_id for op_id in self._recommended_operation_ids_for_tables({"orders", "payments"})],
            ),
        ]
        return cards

    def _recommended_operation_ids_for_tables(self, table_names: set[str]) -> list[str]:
        return [
            operation.operation_id
            for operation in sorted(self._task_spec.operations.values(), key=lambda op: op.operation_id)
            if set(operation.tables_affected) & table_names
        ][:4]

    def _recommended_operation_ids_for_keyword(self, keyword: str) -> list[str]:
        lowered = keyword.lower()
        return [
            operation.operation_id
            for operation in sorted(self._task_spec.operations.values(), key=lambda op: op.operation_id)
            if lowered in operation.operation_id.lower() or lowered in operation.title.lower()
        ][:4]

    def _build_dry_run_report(self, target_system: str) -> DryRunReport:
        findings: list[DryRunFinding] = []
        for issue in self._grade.validation_issues:
            if target_system == "crm" and issue.table_name not in {"customers", "subscriptions"}:
                continue
            if target_system == "billing" and issue.table_name not in {"orders", "subscriptions", "payments"}:
                continue
            findings.append(
                DryRunFinding(
                    code=issue.code,
                    severity=issue.severity,
                    table_name=issue.table_name,
                    row_ids=list(issue.row_ids),
                    message=issue.message,
                )
            )

        health = self._state.downstream_health
        success_rate = health.crm_sync_success_rate if target_system == "crm" else health.billing_link_integrity

        if target_system == "crm" and health.duplicate_contact_risk > 0.12:
            findings.append(
                DryRunFinding(
                    code="risk:duplicate_contacts",
                    severity="medium" if health.duplicate_contact_risk <= 0.3 else "high",
                    table_name="customers",
                    message="CRM dry run predicts duplicate-contact collisions after import.",
                )
            )
        if target_system == "billing" and health.revenue_reporting_risk > 0.12:
            findings.append(
                DryRunFinding(
                    code="risk:revenue_reporting",
                    severity="medium" if health.revenue_reporting_risk <= 0.3 else "high",
                    table_name="payments" if "payments" in self._state.tables else "orders",
                    message="Billing dry run predicts mislinked or duplicated revenue facts.",
                )
            )

        summary = (
            f"Dry run for {target_system.upper()} found {len(findings)} blocking or risky findings; "
            f"estimated success rate is {success_rate:.2f}."
        )
        return DryRunReport(
            target_system=target_system,
            success_rate=success_rate,
            finding_count=len(findings),
            findings=findings,
            summary=summary,
            generated_at_step=self._state.step_count,
        )

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
