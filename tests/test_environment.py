from __future__ import annotations

from cleanops_env.graders import grade_tables
from cleanops_env.local_env import LocalCleanOpsEnv
from cleanops_env.models import DataCleaningAction
from cleanops_env.tasks import TASK_CATALOG, clone_tables


def test_reset_step_state_api() -> None:
    env = LocalCleanOpsEnv()
    observation = env.reset(task_id="customer_contacts_easy", seed=7)
    assert observation.task_id == "customer_contacts_easy"
    assert observation.requested_seed == 7
    assert observation.review_budget_remaining == 1
    assert observation.supported_sync_targets == ["crm"]
    assert len(observation.available_review_targets) == 1
    assert 0.0 < observation.downstream_health.overall_health_score < 1.0
    assert observation.done is False
    assert 0.0 < observation.quality_score < 1.0

    observation, reward, done, info = env.step(
        DataCleaningAction(action_type="inspect_table", table_name="customers", reasoning="Inspect the main table first.")
    )
    assert isinstance(reward, float)
    assert done is False
    assert info["state"]["task_id"] == "customer_contacts_easy"
    assert env.state().task_id == "customer_contacts_easy"
    assert observation.focus_table is not None
    assert observation.focus_table.name == "customers"


def test_oracle_solution_scores_strictly_inside_open_interval_for_all_tasks() -> None:
    env = LocalCleanOpsEnv()
    for task_id, task_spec in TASK_CATALOG.items():
        observation = env.reset(task_id=task_id, seed=7)
        for operation_id in task_spec.solution_operation_ids:
            observation, _, done, _ = env.step(DataCleaningAction(action_type="apply_operation", operation_id=operation_id, reasoning=f"Apply {operation_id}"))
            assert done is False
        observation, _, done, _ = env.step(DataCleaningAction(action_type="submit", reasoning="Submit cleaned tables."))
        assert done is True
        assert 0.0 < observation.quality_score < 1.0
        assert observation.quality_score == 0.99
        assert observation.grader.final_score == 0.99


def test_decoy_operation_lowers_easy_task_quality() -> None:
    task_spec = TASK_CATALOG["customer_contacts_easy"]
    clean_tables = clone_tables(task_spec.gold_tables)
    damaged_tables = task_spec.operations["easy_drop_inactive_customers"].transform(clone_tables(clean_tables))
    clean_grade = grade_tables(task_spec, clean_tables)
    damaged_grade = grade_tables(task_spec, damaged_tables)
    assert clean_grade.score == 0.99
    assert damaged_grade.score < clean_grade.score


def test_seed_changes_visible_preview_rows() -> None:
    env = LocalCleanOpsEnv()
    observation_seed_2 = env.reset(task_id="customer_contacts_easy", seed=2)
    preview_seed_2 = [row["customer_id"] for row in observation_seed_2.focus_table.rows[:4]]

    observation_seed_7 = env.reset(task_id="customer_contacts_easy", seed=7)
    preview_seed_7 = [row["customer_id"] for row in observation_seed_7.focus_table.rows[:4]]

    assert observation_seed_2.requested_seed == 2
    assert observation_seed_7.requested_seed == 7
    assert preview_seed_2 != preview_seed_7


def test_request_review_queues_and_releases_deterministic_response() -> None:
    env = LocalCleanOpsEnv()
    observation = env.reset(task_id="crm_migration_hard", seed=7)
    assert observation.review_budget_remaining == 2
    assert len(observation.pending_reviews) == 0
    assert len(observation.resolved_reviews) == 0

    observation, reward, done, info = env.step(
        DataCleaningAction(
            action_type="request_review",
            entity_type="customer",
            entity_id="CU101",
            reason_code="possible_duplicate",
            reasoning="Escalate the ambiguous Ana Lopez duplicate before merging.",
        )
    )
    assert done is False
    assert reward < 0.0
    assert observation.review_budget_remaining == 1
    assert len(observation.pending_reviews) == 1
    assert len(observation.resolved_reviews) == 0
    assert "response will be available on the next step" in observation.last_action_status
    assert info["state"]["requested_review_ids"] == ["hard_customer_merge_review"]

    observation, reward, done, _ = env.step(
        DataCleaningAction(
            action_type="inspect_table",
            table_name="customers",
            reasoning="Read the customer table again after the review response arrives.",
        )
    )
    assert done is False
    assert reward > 0.0
    assert len(observation.pending_reviews) == 0
    assert len(observation.resolved_reviews) == 1
    resolved_review = observation.resolved_reviews[0]
    assert resolved_review.review_id == "hard_customer_merge_review"
    assert "hard_merge_customers_by_email" in resolved_review.recommended_operation_ids
    assert "Review response available" in observation.last_action_status


def test_run_sync_dry_run_surfaces_downstream_findings() -> None:
    env = LocalCleanOpsEnv()
    observation = env.reset(task_id="crm_migration_hard", seed=7)
    starting_health = observation.downstream_health.overall_health_score

    observation, reward, done, info = env.step(
        DataCleaningAction(
            action_type="run_sync_dry_run",
            target_system="billing",
            reasoning="Check whether the current migration state would break downstream billing.",
        )
    )
    assert done is False
    assert observation.last_dry_run is not None
    assert observation.last_dry_run.target_system == "billing"
    assert observation.last_dry_run.finding_count > 0
    assert observation.last_dry_run.success_rate == observation.downstream_health.billing_link_integrity
    assert "billing" in info["state"]["dry_run_targets"]
    assert observation.downstream_health.overall_health_score == starting_health


def test_duplicate_review_request_is_penalized() -> None:
    env = LocalCleanOpsEnv()
    env.reset(task_id="customer_contacts_easy", seed=7)
    env.step(
        DataCleaningAction(
            action_type="request_review",
            entity_type="customer",
            entity_id="C005",
            reason_code="possible_duplicate",
            reasoning="Ask for confirmation once.",
        )
    )
    observation, reward, done, _ = env.step(
        DataCleaningAction(
            action_type="request_review",
            entity_type="customer",
            entity_id="C005",
            reason_code="possible_duplicate",
            reasoning="Repeat the same review request.",
        )
    )
    assert done is False
    assert reward < 0.0
    assert observation.review_budget_remaining == 0
    assert len(observation.pending_reviews) == 0
    assert len(observation.resolved_reviews) == 1
    assert "already requested" in observation.last_action_status
