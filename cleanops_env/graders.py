"""Deterministic graders and validators for CleanOps tasks."""

from __future__ import annotations

from dataclasses import dataclass

from cleanops_env.models import GradeBreakdown, TableSummary, ValidationIssue
from cleanops_env.tasks import (
    EnumRule,
    ForeignKeyRule,
    PatternRule,
    RequiredRule,
    TaskSpec,
    Tables,
    UniqueRule,
    normalize_amount,
    normalize_currency,
    normalize_date,
    normalize_email,
    normalize_phone_us,
    normalize_state,
    normalize_whitespace,
    sorted_rows,
)

MIN_OPEN_SCORE = 0.01
MAX_OPEN_SCORE = 0.99


@dataclass(frozen=True)
class GraderResult:
    """Full grader output for one environment state."""

    breakdown: GradeBreakdown
    validation_issues: list[ValidationIssue]

    @property
    def score(self) -> float:
        return self.breakdown.final_score


def _open_interval_score(raw_score: float) -> float:
    """Normalize task-facing scores into the validator-safe open interval (0, 1)."""

    return round(min(MAX_OPEN_SCORE, max(MIN_OPEN_SCORE, raw_score)), 4)


def _canonical_cell(column_name: str, value: object) -> str:
    if "email" in column_name:
        return normalize_email(value)
    if column_name in {"phone"}:
        return normalize_phone_us(value)
    if column_name in {"state", "shipping_state"}:
        return normalize_state(value)
    if column_name in {"currency"}:
        return normalize_currency(value)
    if column_name in {"amount", "total_amount"}:
        return normalize_amount(value)
    if "date" in column_name or column_name.endswith("_at"):
        return normalize_date(value)
    if column_name in {"status", "order_status", "payment_status", "lifecycle_stage", "plan_code"}:
        return normalize_whitespace(value).upper().replace(" ", "_")
    return normalize_whitespace(value)


def _issue_code(rule: object) -> str:
    if isinstance(rule, RequiredRule):
        return f"required:{rule.table_name}.{rule.column_name}"
    if isinstance(rule, PatternRule):
        return f"pattern:{rule.table_name}.{rule.column_name}"
    if isinstance(rule, EnumRule):
        return f"enum:{rule.table_name}.{rule.column_name}"
    if isinstance(rule, UniqueRule):
        return f"unique:{rule.table_name}.{'+'.join(rule.columns)}"
    if isinstance(rule, ForeignKeyRule):
        return f"foreign_key:{rule.table_name}.{rule.column_name}"
    return "unknown"


def validate_tables(task_spec: TaskSpec, tables: Tables) -> list[ValidationIssue]:
    """Run deterministic validation rules and return structured issues."""

    issues: list[ValidationIssue] = []
    for rule in task_spec.validation_rules:
        if isinstance(rule, RequiredRule):
            primary_key = task_spec.primary_keys[rule.table_name]
            failing_ids = [normalize_whitespace(row.get(primary_key, "")) for row in tables[rule.table_name] if not normalize_whitespace(row.get(rule.column_name, ""))]
            if failing_ids:
                issues.append(ValidationIssue(code=_issue_code(rule), severity=rule.severity, table_name=rule.table_name, column_name=rule.column_name, row_ids=failing_ids, message=f"{len(failing_ids)} rows have blank {rule.column_name}."))
        elif isinstance(rule, PatternRule):
            primary_key = task_spec.primary_keys[rule.table_name]
            failing_ids = [normalize_whitespace(row.get(primary_key, "")) for row in tables[rule.table_name] if not rule.pattern.fullmatch(normalize_whitespace(row.get(rule.column_name, "")))]
            if failing_ids:
                issues.append(ValidationIssue(code=_issue_code(rule), severity=rule.severity, table_name=rule.table_name, column_name=rule.column_name, row_ids=failing_ids, message=f"{len(failing_ids)} rows violate {rule.message}"))
        elif isinstance(rule, EnumRule):
            primary_key = task_spec.primary_keys[rule.table_name]
            allowed = set(rule.allowed)
            failing_ids = [normalize_whitespace(row.get(primary_key, "")) for row in tables[rule.table_name] if normalize_whitespace(row.get(rule.column_name, "")) not in allowed]
            if failing_ids:
                issues.append(ValidationIssue(code=_issue_code(rule), severity=rule.severity, table_name=rule.table_name, column_name=rule.column_name, row_ids=failing_ids, message=f"{len(failing_ids)} rows contain non-canonical values in {rule.column_name}."))
        elif isinstance(rule, UniqueRule):
            primary_key = task_spec.primary_keys[rule.table_name]
            seen: dict[tuple[str, ...], list[str]] = {}
            for row in tables[rule.table_name]:
                identity = tuple(_canonical_cell(column_name, row.get(column_name, "")) for column_name in rule.columns)
                seen.setdefault(identity, []).append(normalize_whitespace(row.get(primary_key, "")))
            duplicate_ids = sorted(row_id for row_ids in seen.values() if len(row_ids) > 1 for row_id in row_ids)
            if duplicate_ids:
                issues.append(ValidationIssue(code=_issue_code(rule), severity=rule.severity, table_name=rule.table_name, column_name="+".join(rule.columns), row_ids=duplicate_ids, message=f"{len(duplicate_ids)} rows belong to duplicate identity groups."))
        elif isinstance(rule, ForeignKeyRule):
            child_pk = task_spec.primary_keys[rule.table_name]
            valid_parent_ids = {_canonical_cell(rule.ref_column_name, row.get(rule.ref_column_name, "")) for row in tables[rule.ref_table_name]}
            failing_ids = [normalize_whitespace(row.get(child_pk, "")) for row in tables[rule.table_name] if _canonical_cell(rule.column_name, row.get(rule.column_name, "")) not in valid_parent_ids]
            if failing_ids:
                issues.append(ValidationIssue(code=_issue_code(rule), severity=rule.severity, table_name=rule.table_name, column_name=rule.column_name, row_ids=failing_ids, message=f"{len(failing_ids)} rows reference unknown {rule.ref_table_name}.{rule.ref_column_name} values."))
    return issues


def _cell_match_score(task_spec: TaskSpec, tables: Tables) -> float:
    matched = 0
    total = 0
    for table_name, gold_rows in task_spec.gold_tables.items():
        primary_key = task_spec.primary_keys[table_name]
        current_index = {_canonical_cell(primary_key, row.get(primary_key, "")): row for row in tables.get(table_name, [])}
        for gold_row in gold_rows:
            row_key = _canonical_cell(primary_key, gold_row.get(primary_key, ""))
            current_row = current_index.get(row_key)
            for column_name, expected_value in gold_row.items():
                total += 1
                if current_row is None:
                    continue
                if _canonical_cell(column_name, current_row.get(column_name, "")) == _canonical_cell(column_name, expected_value):
                    matched += 1
    return 1.0 if total == 0 else matched / total


def _key_recall_score(task_spec: TaskSpec, tables: Tables) -> float:
    scores: list[float] = []
    for table_name, gold_rows in task_spec.gold_tables.items():
        primary_key = task_spec.primary_keys[table_name]
        current_keys = {_canonical_cell(primary_key, row.get(primary_key, "")) for row in tables.get(table_name, [])}
        gold_keys = {_canonical_cell(primary_key, row.get(primary_key, "")) for row in gold_rows}
        if not current_keys and not gold_keys:
            scores.append(1.0)
            continue
        denom = len(current_keys) + len(gold_keys)
        scores.append((2.0 * len(current_keys & gold_keys)) / denom if denom else 1.0)
    return sum(scores) / len(scores) if scores else 1.0


def _validation_score(task_spec: TaskSpec, current_issues: list[ValidationIssue]) -> float:
    initial_issues = validate_tables(task_spec, task_spec.dirty_tables)
    if not initial_issues:
        return 1.0 if not current_issues else 0.0
    return max(0.0, 1.0 - (len(current_issues) / len(initial_issues)))


def grade_tables(task_spec: TaskSpec, tables: Tables) -> GraderResult:
    """Compute a deterministic validator-safe score against the task's gold tables."""

    validation_issues = validate_tables(task_spec, tables)
    cell_match = _open_interval_score(_cell_match_score(task_spec, tables))
    key_recall = _open_interval_score(_key_recall_score(task_spec, tables))
    validation = _open_interval_score(_validation_score(task_spec, validation_issues))
    final_score = _open_interval_score(0.55 * cell_match + 0.20 * key_recall + 0.25 * validation)
    return GraderResult(
        breakdown=GradeBreakdown(
            cell_match_score=cell_match,
            key_recall_score=key_recall,
            validation_score=validation,
            final_score=final_score,
        ),
        validation_issues=validation_issues,
    )


def count_duplicate_groups(task_spec: TaskSpec, table_name: str, rows: list[dict[str, str]]) -> int:
    identity_columns = task_spec.duplicate_identity_columns[table_name]
    groups: dict[tuple[str, ...], int] = {}
    for row in rows:
        identity = tuple(_canonical_cell(column_name, row.get(column_name, "")) for column_name in identity_columns)
        groups[identity] = groups.get(identity, 0) + 1
    return sum(1 for count in groups.values() if count > 1)


def build_table_summary(task_spec: TaskSpec, table_name: str, tables: Tables) -> TableSummary:
    rows = sorted_rows(tables.get(table_name, []), task_spec.primary_keys[table_name])
    columns = sorted({column_name for row in rows for column_name in row})
    missing_cells = sum(1 for row in rows for column_name in columns if not normalize_whitespace(row.get(column_name, "")))
    return TableSummary(
        name=table_name,
        primary_key=task_spec.primary_keys[table_name],
        row_count=len(rows),
        columns=columns,
        missing_cells=missing_cells,
        duplicate_groups=count_duplicate_groups(task_spec, table_name, rows),
        preview_rows=rows[:3],
    )
