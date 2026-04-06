"""Typed models for the CleanOps environment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


class RewardBreakdown(BaseModel):
    """Explains how a scalar reward was produced."""

    quality_delta: float = Field(default=0.0, description="Change in overall grader score after the action.")
    issue_delta: float = Field(default=0.0, description="Normalized change in outstanding validation issues.")
    insight_bonus: float = Field(default=0.0, description="Small positive reward for inspecting new assets.")
    efficiency_penalty: float = Field(default=0.0, description="Per-step penalty to discourage long episodes.")
    invalid_action_penalty: float = Field(default=0.0, description="Penalty for malformed or unsupported actions.")
    noop_penalty: float = Field(default=0.0, description="Penalty for no-op or repeated actions.")
    submit_bonus: float = Field(default=0.0, description="End-of-episode bonus based on final score.")
    total: float = Field(default=0.0, description="Final scalar reward returned.")


class ValidationIssue(BaseModel):
    """A concrete validation problem the agent should resolve."""

    code: str = Field(..., description="Stable machine-readable issue code.")
    severity: Literal["low", "medium", "high"] = Field(..., description="Issue severity.")
    table_name: str = Field(..., description="Table containing the issue.")
    column_name: str | None = Field(default=None, description="Column containing the issue, if applicable.")
    row_ids: list[str] = Field(default_factory=list, description="Affected primary-key values.")
    message: str = Field(..., description="Human-readable issue summary.")


class IssueCard(BaseModel):
    """Aggregated issue summary paired with likely remediation operations."""

    title: str = Field(..., description="Short issue title.")
    detail: str = Field(..., description="Why the issue matters in this task.")
    issue_codes: list[str] = Field(default_factory=list, description="Validation codes represented by this card.")
    recommended_operation_ids: list[str] = Field(default_factory=list, description="Operations likely to address the issue.")


class TableSummary(BaseModel):
    """Compact summary of a table."""

    name: str = Field(..., description="Table name.")
    primary_key: str = Field(..., description="Primary key column.")
    row_count: int = Field(..., description="Number of rows in the current table.")
    columns: list[str] = Field(default_factory=list, description="Column names.")
    missing_cells: int = Field(default=0, description="Count of blank required or optional cells.")
    duplicate_groups: int = Field(default=0, description="Count of duplicate identity groups.")
    preview_rows: list[dict[str, str]] = Field(default_factory=list, description="Small row preview for quick inspection.")


class TableView(BaseModel):
    """Full table contents for one focused table."""

    name: str = Field(..., description="Table name.")
    primary_key: str = Field(..., description="Primary key column.")
    columns: list[str] = Field(default_factory=list, description="Column names.")
    rows: list[dict[str, str]] = Field(default_factory=list, description="Current table rows.")


class RowChange(BaseModel):
    """Before/after preview for a changed row."""

    primary_key_value: str = Field(..., description="Changed row identifier.")
    before: dict[str, str] | None = Field(default=None, description="Row values before applying an operation.")
    after: dict[str, str] | None = Field(default=None, description="Row values after applying an operation.")


class OperationSummary(BaseModel):
    """A cleaning operation the agent can choose."""

    operation_id: str = Field(..., description="Stable operation identifier.")
    title: str = Field(..., description="Short action title.")
    category: str = Field(..., description="Operation category.")
    risk: Literal["safe", "review", "destructive"] = Field(..., description="Risk level for the operation.")
    tables_affected: list[str] = Field(default_factory=list, description="Tables changed by the operation.")
    description: str = Field(..., description="What the operation does.")
    already_applied: bool = Field(default=False, description="Whether this operation has already been applied.")


class OperationDetail(OperationSummary):
    """Extra context for one operation."""

    why_it_matters: str = Field(default="", description="Business-oriented explanation of the operation.")
    change_preview: list[RowChange] = Field(default_factory=list, description="Predicted row changes if the operation were applied now.")


class GradeBreakdown(BaseModel):
    """Deterministic grader components."""

    cell_match_score: float = Field(default=0.0, description="Fraction of gold cells matched.")
    key_recall_score: float = Field(default=0.0, description="Row identity and deduplication quality.")
    validation_score: float = Field(default=0.0, description="How well the current tables satisfy constraints.")
    final_score: float = Field(default=0.0, description="Weighted final task score.")


class DataCleaningAction(Action):
    """Action model for the environment."""

    action_type: Literal["inspect_table", "inspect_operation", "apply_operation", "submit"] = Field(..., description="Type of action to perform.")
    table_name: str | None = Field(default=None, description="Table to inspect when action_type=inspect_table.")
    operation_id: str | None = Field(default=None, description="Operation to inspect or apply when action_type is inspect_operation or apply_operation.")
    reasoning: str = Field(default="", description="Optional natural-language reasoning for debugging baselines.")


class DataCleaningObservation(Observation):
    """Observation returned after each environment interaction."""

    task_id: str = Field(..., description="Current task identifier.")
    task_title: str = Field(..., description="Human-readable task title.")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Task difficulty.")
    requested_seed: int | None = Field(default=None, description="Seed used when resetting the current episode.")
    objective: str = Field(..., description="Concrete task objective.")
    dataset_context: str = Field(..., description="Why this dataset exists in the real world.")
    quality_score: float = Field(default=0.0, description="Current deterministic grader score.")
    best_score: float = Field(default=0.0, description="Best score seen in the current episode.")
    remaining_steps: int = Field(default=0, description="How many actions remain before truncation.")
    table_summaries: list[TableSummary] = Field(default_factory=list, description="Compact summaries of all tables.")
    focus_table: TableView | None = Field(default=None, description="Detailed contents for the currently inspected table.")
    available_operations: list[OperationSummary] = Field(default_factory=list, description="Available cleaning actions.")
    focus_operation: OperationDetail | None = Field(default=None, description="Detailed preview for the currently inspected operation.")
    validation_issues: list[ValidationIssue] = Field(default_factory=list, description="Current unresolved validation issues.")
    issue_cards: list[IssueCard] = Field(default_factory=list, description="Aggregated issue cards with suggested next actions.")
    recent_history: list[str] = Field(default_factory=list, description="Recent action log.")
    grader: GradeBreakdown = Field(default_factory=GradeBreakdown, description="Deterministic score components.")
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown, description="How the last reward was computed.")
    last_action_status: str = Field(default="", description="Outcome message for the most recent action.")
    last_action_error: str | None = Field(default=None, description="Raw error string for the last action, or null when no error occurred.")


class DataCleaningState(State):
    """Full server-side state for the current episode."""

    task_id: str = Field(..., description="Current task identifier.")
    task_title: str = Field(..., description="Current task title.")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Current task difficulty.")
    requested_seed: int | None = Field(default=None, description="Seed used when resetting the current episode.")
    max_steps: int = Field(..., description="Task step budget.")
    submitted: bool = Field(default=False, description="Whether submit was called.")
    current_score: float = Field(default=0.0, description="Current deterministic grader score.")
    best_score: float = Field(default=0.0, description="Best score achieved this episode.")
    outstanding_issue_count: int = Field(default=0, description="Number of unresolved validation issues.")
    tables: dict[str, list[dict[str, str]]] = Field(default_factory=dict, description="Current mutable table contents.")
    applied_operation_ids: list[str] = Field(default_factory=list, description="Operations already applied.")
    inspected_tables: list[str] = Field(default_factory=list, description="Tables inspected so far.")
    inspected_operations: list[str] = Field(default_factory=list, description="Operations inspected so far.")
    recent_history: list[str] = Field(default_factory=list, description="Recent action log.")
