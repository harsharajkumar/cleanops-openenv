"""Typed models for the CleanOps environment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


class RewardBreakdown(BaseModel):
    """Explains how a scalar reward was produced."""

    quality_delta: float = Field(default=0.0, description="Change in overall grader score after the action.")
    issue_delta: float = Field(default=0.0, description="Normalized change in outstanding validation issues.")
    downstream_health_delta: float = Field(default=0.0, description="Change in downstream operational health after the action.")
    insight_bonus: float = Field(default=0.0, description="Small positive reward for inspecting new assets.")
    efficiency_penalty: float = Field(default=0.0, description="Per-step penalty to discourage long episodes.")
    invalid_action_penalty: float = Field(default=0.0, description="Penalty for malformed or unsupported actions.")
    noop_penalty: float = Field(default=0.0, description="Penalty for no-op or repeated actions.")
    review_bonus: float = Field(default=0.0, description="Positive reward when a queued review response becomes available.")
    review_cost_penalty: float = Field(default=0.0, description="Small cost for consuming limited human-review budget.")
    action_cost_penalty: float = Field(default=0.0, description="Cost-aware penalty attached to the chosen action.")
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


class ReviewTarget(BaseModel):
    """A reviewable entity that can be escalated to a human reviewer."""

    review_id: str = Field(..., description="Stable review case identifier.")
    entity_type: str = Field(..., description="Type of entity under review.")
    entity_id: str = Field(..., description="Primary identifier for the reviewed entity.")
    reason_code: str = Field(..., description="Why the review would be requested.")
    title: str = Field(..., description="Short human-readable review title.")
    detail: str = Field(..., description="Why this review matters.")
    recommended_operation_ids: list[str] = Field(default_factory=list, description="Operations likely to be safe once review resolves.")


class PendingReview(BaseModel):
    """A queued review request awaiting a deterministic response."""

    review_id: str = Field(..., description="Stable review case identifier.")
    entity_type: str = Field(..., description="Type of entity under review.")
    entity_id: str = Field(..., description="Primary identifier for the reviewed entity.")
    reason_code: str = Field(..., description="Why the review was requested.")
    title: str = Field(..., description="Short human-readable review title.")
    requested_at_step: int = Field(..., description="Step index when the review was requested.")
    ready_at_step: int = Field(..., description="First step on which the deterministic response becomes available.")


class ReviewResolution(BaseModel):
    """A resolved human-review response surfaced back to the agent."""

    review_id: str = Field(..., description="Stable review case identifier.")
    entity_type: str = Field(..., description="Type of entity under review.")
    entity_id: str = Field(..., description="Primary identifier for the reviewed entity.")
    reason_code: str = Field(..., description="Why the review was requested.")
    title: str = Field(..., description="Short human-readable review title.")
    resolution: str = Field(..., description="Deterministic review outcome label.")
    response_summary: str = Field(..., description="What the reviewer concluded.")
    evidence_summary: str = Field(..., description="Short explanation for the decision.")
    recommended_operation_ids: list[str] = Field(default_factory=list, description="Operations that become safer after the review response.")


class DryRunFinding(BaseModel):
    """A deterministic downstream issue surfaced by a dry-run sync."""

    code: str = Field(..., description="Stable machine-readable issue code.")
    severity: Literal["low", "medium", "high"] = Field(..., description="Issue severity.")
    table_name: str | None = Field(default=None, description="Table implicated by the dry-run finding.")
    row_ids: list[str] = Field(default_factory=list, description="Primary-key values implicated by the finding.")
    message: str = Field(..., description="Human-readable dry-run explanation.")


class DryRunReport(BaseModel):
    """A dry-run simulation result for a downstream business system."""

    target_system: Literal["crm", "billing"] = Field(..., description="Which downstream system was tested.")
    success_rate: float = Field(default=0.0, description="Deterministic estimate of how many records would import successfully.")
    finding_count: int = Field(default=0, description="How many concrete blockers or risks were found.")
    findings: list[DryRunFinding] = Field(default_factory=list, description="Structured findings from the simulated sync.")
    summary: str = Field(default="", description="Short narrative summary of the dry-run result.")
    generated_at_step: int = Field(default=0, description="Step on which the report was generated.")


class DownstreamHealth(BaseModel):
    """Operational health estimates for downstream systems."""

    crm_sync_success_rate: float = Field(default=0.0, description="Estimated CRM import success rate.")
    billing_link_integrity: float = Field(default=0.0, description="Estimated correctness of billing/customer linkages.")
    duplicate_contact_risk: float = Field(default=0.0, description="Estimated risk that duplicate contacts still remain.")
    revenue_reporting_risk: float = Field(default=0.0, description="Estimated risk of duplicate or mislinked revenue facts.")
    overall_health_score: float = Field(default=0.0, description="Composite downstream health score used for reward shaping.")


class RiskCard(BaseModel):
    """A compact operational risk summary derived from downstream health."""

    title: str = Field(..., description="Short risk title.")
    detail: str = Field(..., description="Why this risk matters operationally.")
    severity: Literal["low", "medium", "high"] = Field(..., description="Severity for UI and agent prioritization.")
    metric_name: str = Field(..., description="Downstream metric represented by this card.")
    current_value: float = Field(default=0.0, description="Current metric or risk value in [0, 1].")
    recommended_action_ids: list[str] = Field(default_factory=list, description="Operations likely to improve this risk.")


class ActionCostEntry(BaseModel):
    """Estimated operational cost of taking an action."""

    action_key: str = Field(..., description="Stable action or risk key.")
    estimated_cost: float = Field(default=0.0, description="Relative action cost used in reward shaping.")
    description: str = Field(default="", description="Why this action costs reviewer or system capacity.")


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

    action_type: Literal["inspect_table", "inspect_operation", "apply_operation", "request_review", "run_sync_dry_run", "submit"] = Field(..., description="Type of action to perform.")
    table_name: str | None = Field(default=None, description="Table to inspect when action_type=inspect_table.")
    operation_id: str | None = Field(default=None, description="Operation to inspect or apply when action_type is inspect_operation or apply_operation.")
    entity_type: str | None = Field(default=None, description="Entity type to review when action_type=request_review.")
    entity_id: str | None = Field(default=None, description="Entity identifier to review when action_type=request_review.")
    target_system: Literal["crm", "billing"] | None = Field(default=None, description="Downstream system to simulate when action_type=run_sync_dry_run.")
    reason_code: str | None = Field(default=None, description="Reason for escalating a review request.")
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
    review_budget_remaining: int = Field(default=0, description="How many human-review requests remain in the current episode.")
    supported_sync_targets: list[str] = Field(default_factory=list, description="Downstream systems that can be tested with run_sync_dry_run.")
    downstream_health: DownstreamHealth = Field(default_factory=DownstreamHealth, description="Current operational health estimates for downstream systems.")
    risk_cards: list[RiskCard] = Field(default_factory=list, description="Operational risk summaries derived from downstream health.")
    last_dry_run: DryRunReport | None = Field(default=None, description="Most recent downstream dry-run result, if any.")
    action_costs: list[ActionCostEntry] = Field(default_factory=list, description="Estimated cost of each action family.")
    table_summaries: list[TableSummary] = Field(default_factory=list, description="Compact summaries of all tables.")
    focus_table: TableView | None = Field(default=None, description="Detailed contents for the currently inspected table.")
    available_operations: list[OperationSummary] = Field(default_factory=list, description="Available cleaning actions.")
    available_review_targets: list[ReviewTarget] = Field(default_factory=list, description="Entities that can be escalated for deterministic review.")
    pending_reviews: list[PendingReview] = Field(default_factory=list, description="Review requests that have been queued but not yet resolved.")
    resolved_reviews: list[ReviewResolution] = Field(default_factory=list, description="Resolved review responses available to the agent.")
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
    review_budget_total: int = Field(default=0, description="Total number of review requests available in this task.")
    review_budget_remaining: int = Field(default=0, description="Remaining number of review requests available in this task.")
    submitted: bool = Field(default=False, description="Whether submit was called.")
    current_score: float = Field(default=0.0, description="Current deterministic grader score.")
    best_score: float = Field(default=0.0, description="Best score achieved this episode.")
    outstanding_issue_count: int = Field(default=0, description="Number of unresolved validation issues.")
    downstream_health: DownstreamHealth = Field(default_factory=DownstreamHealth, description="Current downstream operational health.")
    last_dry_run: DryRunReport | None = Field(default=None, description="Most recent downstream dry-run result.")
    tables: dict[str, list[dict[str, str]]] = Field(default_factory=dict, description="Current mutable table contents.")
    applied_operation_ids: list[str] = Field(default_factory=list, description="Operations already applied.")
    inspected_tables: list[str] = Field(default_factory=list, description="Tables inspected so far.")
    inspected_operations: list[str] = Field(default_factory=list, description="Operations inspected so far.")
    requested_review_ids: list[str] = Field(default_factory=list, description="Review cases already requested in this episode.")
    pending_reviews: list[PendingReview] = Field(default_factory=list, description="Queued review requests awaiting deterministic responses.")
    resolved_reviews: list[ReviewResolution] = Field(default_factory=list, description="Resolved review responses available to the agent.")
    dry_run_targets: list[str] = Field(default_factory=list, description="Downstream targets that have already been dry-run in this episode.")
    recent_history: list[str] = Field(default_factory=list, description="Recent action log.")
