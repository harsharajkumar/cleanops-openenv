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
