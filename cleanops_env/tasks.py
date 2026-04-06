"""Task specs and deterministic cleaning operations for CleanOps."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from cleanops_env.models import IssueCard

Tables = dict[str, list[dict[str, str]]]
TransformFn = Callable[[Tables], Tables]

EMAIL_RE = re.compile(r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$")
PHONE_RE = re.compile(r"^\(\d{3}\) \d{3}-\d{4}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
AMOUNT_RE = re.compile(r"^\d+\.\d{2}$")
STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}
STATE_ALIASES = {
    "tennessee": "TN",
    "tn": "TN",
    "texas": "TX",
    "tx": "TX",
    "california": "CA",
    "ca": "CA",
}
STATUS_RANK = {
    "active": 5,
    "trial": 4,
    "pending": 3,
    "past_due": 2,
    "returned": 2,
    "churn_risk": 1,
    "inactive": 0,
    "cancelled": 0,
}


@dataclass(frozen=True)
class RequiredRule:
    table_name: str
    column_name: str
    severity: str = "high"


@dataclass(frozen=True)
class PatternRule:
    table_name: str
    column_name: str
    pattern: re.Pattern[str]
    message: str
    severity: str = "medium"


@dataclass(frozen=True)
class EnumRule:
    table_name: str
    column_name: str
    allowed: tuple[str, ...]
    severity: str = "medium"


@dataclass(frozen=True)
class UniqueRule:
    table_name: str
    columns: tuple[str, ...]
    severity: str = "high"


@dataclass(frozen=True)
class ForeignKeyRule:
    table_name: str
    column_name: str
    ref_table_name: str
    ref_column_name: str
    severity: str = "high"


ValidationRule = RequiredRule | PatternRule | EnumRule | UniqueRule | ForeignKeyRule


@dataclass(frozen=True)
class OperationSpec:
    operation_id: str
    title: str
    category: str
    risk: str
    tables_affected: tuple[str, ...]
    description: str
    why_it_matters: str
    transform: TransformFn


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    difficulty: str
    objective: str
    dataset_context: str
    max_steps: int
    primary_keys: dict[str, str]
    duplicate_identity_columns: dict[str, tuple[str, ...]]
    dirty_tables: Tables
    gold_tables: Tables
    validation_rules: tuple[ValidationRule, ...]
    operations: dict[str, OperationSpec]
    solution_operation_ids: tuple[str, ...]
    issue_cards: tuple[IssueCard, ...]


def clone_tables(tables: Tables) -> Tables:
    return {table_name: [dict(row) for row in rows] for table_name, rows in tables.items()}


def normalize_whitespace(value: object) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split())


def normalize_name(value: object) -> str:
    cleaned = normalize_whitespace(value)
    return " ".join(part.capitalize() for part in cleaned.split())


def normalize_email(value: object) -> str:
    return normalize_whitespace(value).lower()


def normalize_phone_us(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 7:
        digits = "615" + digits
    if len(digits) != 10:
        return normalize_whitespace(value)
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def normalize_state(value: object) -> str:
    cleaned = normalize_whitespace(value).lower()
    return STATE_ALIASES.get(cleaned, cleaned.upper())


def fill_state_from_city(city: object, state: object) -> str:
    current = normalize_state(state)
    if current:
        return current
    city_lookup = {"nashville": "TN", "austin": "TX", "san jose": "CA"}
    return city_lookup.get(normalize_whitespace(city).lower(), "")


def normalize_status(value: object, mapping: dict[str, str]) -> str:
    cleaned = normalize_whitespace(value).lower().replace(" ", "_")
    return mapping.get(cleaned, cleaned.upper())


def normalize_date(value: object) -> str:
    cleaned = normalize_whitespace(value)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y", "%Y.%m.%d"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return cleaned


def normalize_currency(value: object) -> str:
    cleaned = normalize_whitespace(value).replace("$", "").upper()
    if cleaned in {"", "USD", "US"}:
        return "USD"
    return cleaned


def normalize_amount(value: object) -> str:
    cleaned = normalize_whitespace(value)
    cleaned = cleaned.replace("$", "").replace(",", "").replace("O", "0").replace("o", "0")
    try:
        return f"{float(cleaned):.2f}"
    except ValueError:
        return normalize_whitespace(value)


def rank_value(field_name: str, value: str) -> tuple[int, int, str]:
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return (0, 0, "")
    if field_name in {"status", "order_status", "payment_status", "lifecycle_stage"}:
        return (2, STATUS_RANK.get(cleaned.lower(), 1), cleaned)
    if field_name.endswith("id") or field_name.endswith("_id"):
        return (1, len(cleaned), cleaned)
    return (1, len(cleaned), cleaned)


def choose_preferred_value(field_name: str, values: list[str]) -> str:
    candidates = [normalize_whitespace(value) for value in values if normalize_whitespace(value)]
    if not candidates:
        return ""
    return sorted(candidates, key=lambda item: rank_value(field_name, item), reverse=True)[0]


def dedupe_rows(rows: list[dict[str, str]], primary_key: str, identity_columns: tuple[str, ...]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        identity = tuple(normalize_whitespace(row.get(column_name, "")).lower() for column_name in identity_columns)
        groups.setdefault(identity, []).append(dict(row))

    merged_rows: list[dict[str, str]] = []
    for group_rows in groups.values():
        canonical = sorted(group_rows, key=lambda row: normalize_whitespace(row.get(primary_key, "")))[0]
        merged_row = dict(canonical)
        all_columns = sorted({column_name for row in group_rows for column_name in row})
        for column_name in all_columns:
            if column_name == primary_key:
                merged_row[column_name] = normalize_whitespace(canonical.get(column_name, ""))
                continue
            merged_row[column_name] = choose_preferred_value(column_name, [row.get(column_name, "") for row in group_rows])
        merged_rows.append(merged_row)
    return sorted(merged_rows, key=lambda row: normalize_whitespace(row.get(primary_key, "")))


def remap_foreign_keys_from_email(
    tables: Tables,
    child_table: str,
    fk_column: str,
    email_column: str,
    parent_table: str,
    parent_key_column: str,
    parent_email_column: str,
) -> Tables:
    updated = clone_tables(tables)
    parent_lookup: dict[str, str] = {}
    for row in sorted(updated[parent_table], key=lambda item: normalize_whitespace(item.get(parent_key_column, ""))):
        email = normalize_email(row.get(parent_email_column, ""))
        if email and email not in parent_lookup:
            parent_lookup[email] = normalize_whitespace(row.get(parent_key_column, ""))
    for row in updated[child_table]:
        email = normalize_email(row.get(email_column, ""))
        if email in parent_lookup:
            row[fk_column] = parent_lookup[email]
    return updated


def normalize_columns(tables: Tables, table_name: str, column_transforms: dict[str, Callable[[object], str]]) -> Tables:
    updated = clone_tables(tables)
    for row in updated[table_name]:
        for column_name, transform in column_transforms.items():
            row[column_name] = transform(row.get(column_name, ""))
    return updated


def fill_customer_state_from_city(tables: Tables) -> Tables:
    updated = clone_tables(tables)
    for row in updated["customers"]:
        row["state"] = fill_state_from_city(row.get("city", ""), row.get("state", ""))
    return updated


def merge_easy_customers(tables: Tables) -> Tables:
    updated = clone_tables(tables)
    updated["customers"] = dedupe_rows(updated["customers"], "customer_id", ("email",))
    return updated


def drop_inactive_customers(tables: Tables) -> Tables:
    updated = clone_tables(tables)
    updated["customers"] = [row for row in updated["customers"] if normalize_whitespace(row.get("status", "")).lower() != "inactive"]
    return updated


def dedupe_orders_by_source_id(tables: Tables) -> Tables:
    updated = clone_tables(tables)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in updated["orders"]:
        identity = normalize_whitespace(row.get("source_order_id", "")) or normalize_whitespace(row.get("order_id", ""))
        groups.setdefault(identity, []).append(dict(row))

    merged_rows: list[dict[str, str]] = []
    for identity, group_rows in groups.items():
        canonical = sorted(group_rows, key=lambda row: normalize_whitespace(row.get("order_id", "")))[0]
        merged = dict(canonical)
        merged["order_id"] = identity
        all_columns = sorted({column for row in group_rows for column in row})
        for column_name in all_columns:
            if column_name in {"order_id", "source_order_id"}:
                continue
            merged[column_name] = choose_preferred_value(column_name, [row.get(column_name, "") for row in group_rows])
        merged.pop("source_order_id", None)
        merged_rows.append(merged)
    updated["orders"] = sorted(merged_rows, key=lambda row: normalize_whitespace(row.get("order_id", "")))
    return updated


def drop_cancelled_orders(tables: Tables) -> Tables:
    updated = clone_tables(tables)
    updated["orders"] = [row for row in updated["orders"] if normalize_whitespace(row.get("order_status", "")).lower() not in {"cancelled", "canceled"}]
    return updated


def round_order_amounts_to_int(tables: Tables) -> Tables:
    updated = clone_tables(tables)
    for row in updated["orders"]:
        normalized = normalize_amount(row.get("total_amount", ""))
        try:
            row["total_amount"] = str(int(round(float(normalized))))
        except ValueError:
            row["total_amount"] = normalized
    return updated


def merge_hard_customers_by_email(tables: Tables) -> Tables:
    updated = clone_tables(tables)
    updated["customers"] = dedupe_rows(updated["customers"], "customer_id", ("email",))
    return updated


def remove_duplicate_payments(tables: Tables) -> Tables:
    updated = clone_tables(tables)
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in updated["payments"]:
        identity = (
            normalize_email(row.get("customer_email", "")),
            normalize_whitespace(row.get("subscription_id", "")),
            normalize_amount(row.get("amount", "")),
            normalize_date(row.get("paid_at", "")),
        )
        groups.setdefault(identity, []).append(dict(row))
    deduped = [sorted(group_rows, key=lambda row: normalize_whitespace(row.get("payment_id", "")))[0] for group_rows in groups.values()]
    updated["payments"] = sorted(deduped, key=lambda row: normalize_whitespace(row.get("payment_id", "")))
    return updated


def drop_orphaned_subscriptions_and_payments(tables: Tables) -> Tables:
    updated = clone_tables(tables)
    valid_customer_ids = {normalize_whitespace(row.get("customer_id", "")) for row in updated["customers"]}
    updated["subscriptions"] = [row for row in updated["subscriptions"] if normalize_whitespace(row.get("customer_id", "")) in valid_customer_ids]
    updated["payments"] = [row for row in updated["payments"] if normalize_whitespace(row.get("customer_id", "")) in valid_customer_ids]
    return updated


def _task_from_solution(
    *,
    task_id: str,
    title: str,
    difficulty: str,
    objective: str,
    dataset_context: str,
    max_steps: int,
    primary_keys: dict[str, str],
    duplicate_identity_columns: dict[str, tuple[str, ...]],
    dirty_tables: Tables,
    validation_rules: tuple[ValidationRule, ...],
    operations: dict[str, OperationSpec],
    solution_operation_ids: tuple[str, ...],
    issue_cards: tuple[IssueCard, ...],
) -> TaskSpec:
    gold_tables = clone_tables(dirty_tables)
    for operation_id in solution_operation_ids:
        gold_tables = operations[operation_id].transform(gold_tables)
    return TaskSpec(
        task_id=task_id,
        title=title,
        difficulty=difficulty,
        objective=objective,
        dataset_context=dataset_context,
        max_steps=max_steps,
        primary_keys=primary_keys,
        duplicate_identity_columns=duplicate_identity_columns,
        dirty_tables=dirty_tables,
        gold_tables=gold_tables,
        validation_rules=validation_rules,
        operations=operations,
        solution_operation_ids=solution_operation_ids,
        issue_cards=issue_cards,
    )


def _build_easy_task() -> TaskSpec:
    dirty_tables = {
        "customers": [
            {"customer_id": "C001", "full_name": " alice johnson ", "email": "ALICE@example.com ", "phone": "615.555.0101", "state": "tn", "city": "Nashville", "status": "pending"},
            {"customer_id": "C002", "full_name": "Bob smith", "email": " bob.smith@example.com", "phone": "(615) 555-0102", "state": "Tennessee", "city": "Nashville", "status": "active"},
            {"customer_id": "C003", "full_name": "Carla Gomez", "email": "carla.gomez@example.com", "phone": "5550103", "state": "", "city": "Austin", "status": "pending"},
            {"customer_id": "C004", "full_name": "Dan wu", "email": "DAN.WU@example.com", "phone": "+1 615-555-0104", "state": " TX ", "city": "Austin", "status": "inactive"},
            {"customer_id": "C005", "full_name": "Alice Johnson", "email": "alice@example.com", "phone": "6155550101", "state": "TN", "city": "Nashville", "status": "active"},
        ]
    }
    operations = {
        "easy_normalize_names": OperationSpec("easy_normalize_names", "Normalize customer names", "standardize_text", "safe", ("customers",), "Trim whitespace and convert full_name to title case.", "Customer-facing names should be consistently formatted before CRM import.", lambda tables: normalize_columns(tables, "customers", {"full_name": normalize_name})),
        "easy_normalize_emails": OperationSpec("easy_normalize_emails", "Normalize customer emails", "standardize_text", "safe", ("customers",), "Lowercase and trim email addresses.", "Case and whitespace inconsistencies hide duplicate customer records.", lambda tables: normalize_columns(tables, "customers", {"email": normalize_email})),
        "easy_normalize_phones": OperationSpec("easy_normalize_phones", "Normalize US phone numbers", "standardize_contact", "safe", ("customers",), "Convert phone numbers into '(AAA) BBB-CCCC' format and infer the missing local area code.", "Support and sales workflows depend on consistent contact fields.", lambda tables: normalize_columns(tables, "customers", {"phone": normalize_phone_us})),
        "easy_normalize_states": OperationSpec("easy_normalize_states", "Normalize state codes", "standardize_geo", "safe", ("customers",), "Convert state names and lower-case abbreviations to two-letter US state codes.", "Downstream tax and routing systems expect canonical state codes.", lambda tables: normalize_columns(tables, "customers", {"state": normalize_state})),
        "easy_fill_state_from_city": OperationSpec("easy_fill_state_from_city", "Fill missing state from city", "impute_missing", "review", ("customers",), "Fill blank state values using a deterministic city-to-state lookup.", "Missing geo fields prevent territory assignment and validation checks.", fill_customer_state_from_city),
        "easy_merge_customers_by_email": OperationSpec("easy_merge_customers_by_email", "Merge duplicate customers by email", "deduplicate", "review", ("customers",), "Collapse rows with the same normalized email into one canonical customer record.", "Duplicate contacts inflate outreach counts and break customer analytics.", merge_easy_customers),
        "easy_drop_inactive_customers": OperationSpec("easy_drop_inactive_customers", "Drop inactive customers", "destructive_filter", "destructive", ("customers",), "Delete all rows where status is inactive.", "This can destroy valid historical records and should be avoided for this task.", drop_inactive_customers),
    }
    validation_rules = (
        RequiredRule("customers", "customer_id", "high"),
        RequiredRule("customers", "full_name", "high"),
        RequiredRule("customers", "email", "high"),
        RequiredRule("customers", "phone", "medium"),
        RequiredRule("customers", "state", "medium"),
        PatternRule("customers", "email", EMAIL_RE, "Email must be lowercase and valid.", "high"),
        PatternRule("customers", "phone", PHONE_RE, "Phone must use '(AAA) BBB-CCCC' format.", "medium"),
        EnumRule("customers", "state", tuple(sorted(STATE_CODES)), "medium"),
        EnumRule("customers", "status", ("active", "inactive", "pending"), "low"),
        UniqueRule("customers", ("email",), "high"),
    )
    issue_cards = (
        IssueCard(title="Contact formatting is inconsistent", detail="Names, emails, phones, and state values need canonical formatting before handoff.", issue_codes=["pattern:customers.email", "pattern:customers.phone", "enum:customers.state"], recommended_operation_ids=["easy_normalize_names", "easy_normalize_emails", "easy_normalize_phones", "easy_normalize_states"]),
        IssueCard(title="A missing state value blocks validation", detail="One customer record has city information but no state code.", issue_codes=["required:customers.state"], recommended_operation_ids=["easy_fill_state_from_city"]),
        IssueCard(title="Duplicate customer identities exist", detail="Two rows refer to the same customer once emails are normalized.", issue_codes=["unique:customers.email"], recommended_operation_ids=["easy_merge_customers_by_email"]),
    )
    return _task_from_solution(
        task_id="customer_contacts_easy",
        title="Customer Contacts Standardization",
        difficulty="easy",
        objective="Prepare a customer-contact export for CRM import by standardizing contact fields, filling one missing state, and merging duplicate customer rows without deleting valid inactive accounts.",
        dataset_context="This table simulates a weekly B2B CRM export that sales ops cleans before loading into a customer system.",
        max_steps=10,
        primary_keys={"customers": "customer_id"},
        duplicate_identity_columns={"customers": ("email",)},
        dirty_tables=dirty_tables,
        validation_rules=validation_rules,
        operations=operations,
        solution_operation_ids=("easy_normalize_names", "easy_normalize_emails", "easy_normalize_phones", "easy_normalize_states", "easy_fill_state_from_city", "easy_merge_customers_by_email"),
        issue_cards=issue_cards,
    )


def _build_medium_task() -> TaskSpec:
    dirty_tables = {
        "orders": [
            {"order_id": "O1001", "customer_email": "alice@example.com", "order_date": "2025/01/05", "currency": "usd", "total_amount": "1,200.00", "order_status": "Shipped", "shipping_state": "tn", "source_order_id": ""},
            {"order_id": "O1002", "customer_email": "bob.smith@example.com", "order_date": "01-06-2025", "currency": "$", "total_amount": "45.50", "order_status": "pending ", "shipping_state": "Tennessee", "source_order_id": ""},
            {"order_id": "O1003", "customer_email": "carla.gomez@example.com", "order_date": "2025-01-07", "currency": "USD ", "total_amount": "12O.00", "order_status": "cancelled", "shipping_state": "TX", "source_order_id": ""},
            {"order_id": "O1004", "customer_email": "dan.wu@example.com", "order_date": "2025.01.08", "currency": "usd", "total_amount": "89", "order_status": "Ship", "shipping_state": " tx ", "source_order_id": ""},
            {"order_id": "O1002", "customer_email": "bob.smith@example.com", "order_date": "2025-01-06", "currency": "USD", "total_amount": "45.50", "order_status": "PENDING", "shipping_state": "TN", "source_order_id": "O1002"},
            {"order_id": "O1005", "customer_email": "enterprise@example.com", "order_date": "2025/01/09", "currency": "usd", "total_amount": "2500", "order_status": "Returned", "shipping_state": "CA", "source_order_id": ""},
        ]
    }
    order_status_map = {"shipped": "SHIPPED", "ship": "SHIPPED", "pending": "PENDING", "cancelled": "CANCELLED", "returned": "RETURNED"}
    operations = {
        "med_normalize_dates": OperationSpec("med_normalize_dates", "Normalize order dates", "standardize_dates", "safe", ("orders",), "Convert order_date values into ISO 8601 format (YYYY-MM-DD).", "Finance systems reject mixed date formats during settlement reconciliation.", lambda tables: normalize_columns(tables, "orders", {"order_date": normalize_date})),
        "med_normalize_currency_amounts": OperationSpec("med_normalize_currency_amounts", "Normalize currency and amounts", "standardize_money", "review", ("orders",), "Standardize currency to USD and normalize total_amount to two-decimal strings.", "Revenue aggregation fails when currency and amount encodings are inconsistent.", lambda tables: normalize_columns(tables, "orders", {"currency": normalize_currency, "total_amount": normalize_amount})),
        "med_normalize_order_statuses": OperationSpec("med_normalize_order_statuses", "Canonicalize order statuses", "standardize_status", "safe", ("orders",), "Map free-text order status values to SHIPPED, PENDING, CANCELLED, or RETURNED.", "Operational dashboards and SLAs depend on normalized state machines.", lambda tables: normalize_columns(tables, "orders", {"order_status": lambda value: normalize_status(value, order_status_map)})),
        "med_normalize_shipping_states": OperationSpec("med_normalize_shipping_states", "Normalize shipping state codes", "standardize_geo", "safe", ("orders",), "Convert shipping_state values into two-letter US state codes.", "Warehouse routing rules require canonical geography fields.", lambda tables: normalize_columns(tables, "orders", {"shipping_state": normalize_state})),
        "med_dedupe_orders": OperationSpec("med_dedupe_orders", "Remove duplicated order exports", "deduplicate", "review", ("orders",), "Merge duplicate order rows using source_order_id when present.", "Duplicate transactions overstate revenue and shipment volume.", dedupe_orders_by_source_id),
        "med_drop_cancelled_orders": OperationSpec("med_drop_cancelled_orders", "Drop cancelled orders", "destructive_filter", "destructive", ("orders",), "Delete all cancelled rows from the table.", "Cancelled orders are still valid operational records and should not be removed here.", drop_cancelled_orders),
        "med_round_amounts_to_int": OperationSpec("med_round_amounts_to_int", "Round all order totals to whole dollars", "destructive_transform", "destructive", ("orders",), "Round every total_amount to an integer string.", "This destroys cents precision and should be rejected.", round_order_amounts_to_int),
    }
    validation_rules = (
        RequiredRule("orders", "order_id", "high"),
        RequiredRule("orders", "customer_email", "high"),
        RequiredRule("orders", "order_date", "high"),
        RequiredRule("orders", "currency", "high"),
        RequiredRule("orders", "total_amount", "high"),
        RequiredRule("orders", "order_status", "high"),
        RequiredRule("orders", "shipping_state", "medium"),
        PatternRule("orders", "customer_email", EMAIL_RE, "Customer email must be canonical.", "high"),
        PatternRule("orders", "order_date", ISO_DATE_RE, "Order date must be YYYY-MM-DD.", "high"),
        PatternRule("orders", "total_amount", AMOUNT_RE, "Amount must have exactly two decimals.", "high"),
        EnumRule("orders", "currency", ("USD",), "high"),
        EnumRule("orders", "order_status", ("SHIPPED", "PENDING", "CANCELLED", "RETURNED"), "medium"),
        EnumRule("orders", "shipping_state", tuple(sorted(STATE_CODES)), "medium"),
        UniqueRule("orders", ("order_id",), "high"),
    )
    issue_cards = (
        IssueCard(title="Dates, money, and statuses use mixed encodings", detail="The order export mixes separators, symbols, and ad hoc status spellings.", issue_codes=["pattern:orders.order_date", "pattern:orders.total_amount", "enum:orders.order_status", "enum:orders.currency"], recommended_operation_ids=["med_normalize_dates", "med_normalize_currency_amounts", "med_normalize_order_statuses"]),
        IssueCard(title="Shipping state labels are not canonical", detail="Downstream warehouse tools require two-letter state abbreviations.", issue_codes=["enum:orders.shipping_state"], recommended_operation_ids=["med_normalize_shipping_states"]),
        IssueCard(title="A duplicated order row exists", detail="One record is a second export copy of another order.", issue_codes=["unique:orders.order_id"], recommended_operation_ids=["med_dedupe_orders"]),
    )
    return _task_from_solution(
        task_id="orders_reconciliation_medium",
        title="E-commerce Order Reconciliation",
        difficulty="medium",
        objective="Clean a transactional orders export by normalizing dates, money, statuses, and shipping states while deduplicating repeated order exports without deleting legitimate cancelled orders.",
        dataset_context="This table simulates a daily order extract from an e-commerce platform that revenue ops must reconcile before BI ingestion.",
        max_steps=12,
        primary_keys={"orders": "order_id"},
        duplicate_identity_columns={"orders": ("order_id",)},
        dirty_tables=dirty_tables,
        validation_rules=validation_rules,
        operations=operations,
        solution_operation_ids=("med_normalize_dates", "med_normalize_currency_amounts", "med_normalize_order_statuses", "med_normalize_shipping_states", "med_dedupe_orders"),
        issue_cards=issue_cards,
    )


def _build_hard_task() -> TaskSpec:
    dirty_tables = {
        "customers": [
            {"customer_id": "CU100", "full_name": "Ana Lopez", "email": "ana.lopez@example.com ", "lifecycle_stage": "Active"},
            {"customer_id": "CU101", "full_name": "Ana  Lopez", "email": "ANA.LOPEZ@example.com", "lifecycle_stage": "active"},
            {"customer_id": "CU102", "full_name": "Ben Carter", "email": "ben.carter@example.com", "lifecycle_stage": "trial"},
            {"customer_id": "CU103", "full_name": "Mia Chen", "email": "mia.chen@example.com", "lifecycle_stage": "churn_risk"},
        ],
        "subscriptions": [
            {"subscription_id": "S900", "customer_id": "CU101", "customer_email": "ana.lopez@example.com", "plan_code": "BASIC monthly", "status": "active", "renewal_date": "2025/02/01"},
            {"subscription_id": "S901", "customer_id": "CU102", "customer_email": "ben.carter@example.com", "plan_code": "pro annual", "status": "trial ", "renewal_date": "02-15-2025"},
            {"subscription_id": "S902", "customer_id": "CU999", "customer_email": "mia.chen@example.com", "plan_code": "BASIC", "status": "past_due", "renewal_date": "2025.03.01"},
        ],
        "payments": [
            {"payment_id": "P500", "customer_id": "CU100", "customer_email": "ana.lopez@example.com", "subscription_id": "S900", "amount": "29", "currency": "usd", "payment_status": "Paid", "paid_at": "2025/01/01"},
            {"payment_id": "P501", "customer_id": "", "customer_email": "ben.carter@example.com", "subscription_id": "S901", "amount": "299.0", "currency": "USD ", "payment_status": "settled", "paid_at": "01-15-2025"},
            {"payment_id": "P502", "customer_id": "CU999", "customer_email": "mia.chen@example.com", "subscription_id": "S902", "amount": "29.00", "currency": "usd", "payment_status": "FAILED ", "paid_at": "2025.02.01"},
            {"payment_id": "P503", "customer_id": "CU101", "customer_email": "ana.lopez@example.com", "subscription_id": "S900", "amount": "29.00", "currency": "usd", "payment_status": "paid", "paid_at": "2025-01-01"},
        ],
    }
    lifecycle_map = {"active": "ACTIVE", "trial": "TRIAL", "churn_risk": "CHURN_RISK"}
    subscription_status_map = {"active": "ACTIVE", "trial": "TRIAL", "past_due": "PAST_DUE"}
    payment_status_map = {"paid": "PAID", "settled": "PAID", "failed": "FAILED"}
    plan_map = {"basic_monthly": "BASIC_MONTHLY", "basic": "BASIC_MONTHLY", "pro_annual": "PRO_ANNUAL"}
    operations = {
        "hard_normalize_customer_fields": OperationSpec("hard_normalize_customer_fields", "Normalize customer master records", "standardize_master_data", "safe", ("customers",), "Trim customer names/emails and standardize lifecycle_stage values.", "Canonical customer keys and lifecycle labels are the backbone of the migration.", lambda tables: normalize_columns(tables, "customers", {"full_name": normalize_name, "email": normalize_email, "lifecycle_stage": lambda value: normalize_status(value, lifecycle_map)})),
        "hard_merge_customers_by_email": OperationSpec("hard_merge_customers_by_email", "Merge duplicated CRM customers", "deduplicate_master_data", "review", ("customers",), "Merge customer rows with the same normalized email and keep the lowest customer_id as the canonical ID.", "Duplicate customer IDs fragment subscriptions and payments across child tables.", merge_hard_customers_by_email),
        "hard_normalize_subscriptions": OperationSpec("hard_normalize_subscriptions", "Normalize subscription records", "standardize_subscriptions", "safe", ("subscriptions",), "Normalize subscription plan codes, statuses, renewal dates, and customer emails.", "Subscription analytics and renewals automation rely on stable plan/status values.", lambda tables: normalize_columns(tables, "subscriptions", {"customer_email": normalize_email, "plan_code": lambda value: normalize_status(value, plan_map), "status": lambda value: normalize_status(value, subscription_status_map), "renewal_date": normalize_date})),
        "hard_repair_subscription_customer_refs": OperationSpec("hard_repair_subscription_customer_refs", "Repair subscription customer references", "repair_foreign_keys", "review", ("subscriptions",), "Rewrite subscriptions.customer_id by matching customer_email against the current customers table.", "Migration import will reject subscriptions that reference unknown customer IDs.", lambda tables: remap_foreign_keys_from_email(tables, "subscriptions", "customer_id", "customer_email", "customers", "customer_id", "email")),
        "hard_normalize_payments": OperationSpec("hard_normalize_payments", "Normalize payment ledger rows", "standardize_payments", "safe", ("payments",), "Normalize payment customer emails, amounts, currency, statuses, and paid_at dates.", "Consistent payment values are required for revenue reconciliation.", lambda tables: normalize_columns(tables, "payments", {"customer_email": normalize_email, "amount": normalize_amount, "currency": normalize_currency, "payment_status": lambda value: normalize_status(value, payment_status_map), "paid_at": normalize_date})),
        "hard_repair_payment_customer_refs": OperationSpec("hard_repair_payment_customer_refs", "Repair payment customer references", "repair_foreign_keys", "review", ("payments",), "Rewrite payments.customer_id using the payment customer_email and the current customers table.", "Payment facts must link to customer dimensions after the dedupe migration.", lambda tables: remap_foreign_keys_from_email(tables, "payments", "customer_id", "customer_email", "customers", "customer_id", "email")),
        "hard_remove_duplicate_payments": OperationSpec("hard_remove_duplicate_payments", "Remove duplicate payment facts", "deduplicate_facts", "review", ("payments",), "Collapse duplicated payment rows sharing customer, subscription, amount, and paid_at values.", "A duplicated ledger entry double-counts revenue.", remove_duplicate_payments),
        "hard_drop_orphaned_rows": OperationSpec("hard_drop_orphaned_rows", "Drop orphaned subscriptions and payments", "destructive_filter", "destructive", ("subscriptions", "payments"), "Delete child rows whose customer_id is not present in customers.", "Dropping business records hides data-quality issues and loses legitimate revenue facts.", drop_orphaned_subscriptions_and_payments),
    }
    validation_rules = (
        RequiredRule("customers", "customer_id", "high"),
        RequiredRule("customers", "full_name", "high"),
        RequiredRule("customers", "email", "high"),
        RequiredRule("customers", "lifecycle_stage", "medium"),
        PatternRule("customers", "email", EMAIL_RE, "Customer email must be canonical.", "high"),
        EnumRule("customers", "lifecycle_stage", ("ACTIVE", "TRIAL", "CHURN_RISK"), "medium"),
        UniqueRule("customers", ("email",), "high"),
        RequiredRule("subscriptions", "subscription_id", "high"),
        RequiredRule("subscriptions", "customer_id", "high"),
        RequiredRule("subscriptions", "customer_email", "medium"),
        RequiredRule("subscriptions", "plan_code", "high"),
        RequiredRule("subscriptions", "status", "medium"),
        RequiredRule("subscriptions", "renewal_date", "medium"),
        PatternRule("subscriptions", "customer_email", EMAIL_RE, "Subscription email must be canonical.", "medium"),
        PatternRule("subscriptions", "renewal_date", ISO_DATE_RE, "Renewal date must be YYYY-MM-DD.", "medium"),
        EnumRule("subscriptions", "plan_code", ("BASIC_MONTHLY", "PRO_ANNUAL"), "medium"),
        EnumRule("subscriptions", "status", ("ACTIVE", "TRIAL", "PAST_DUE"), "medium"),
        UniqueRule("subscriptions", ("subscription_id",), "high"),
        ForeignKeyRule("subscriptions", "customer_id", "customers", "customer_id", "high"),
        RequiredRule("payments", "payment_id", "high"),
        RequiredRule("payments", "customer_id", "high"),
        RequiredRule("payments", "customer_email", "medium"),
        RequiredRule("payments", "subscription_id", "high"),
        RequiredRule("payments", "amount", "high"),
        RequiredRule("payments", "currency", "high"),
        RequiredRule("payments", "payment_status", "high"),
        RequiredRule("payments", "paid_at", "high"),
        PatternRule("payments", "customer_email", EMAIL_RE, "Payment email must be canonical.", "medium"),
        PatternRule("payments", "amount", AMOUNT_RE, "Payment amount must have two decimals.", "high"),
        PatternRule("payments", "paid_at", ISO_DATE_RE, "Payment date must be YYYY-MM-DD.", "medium"),
        EnumRule("payments", "currency", ("USD",), "medium"),
        EnumRule("payments", "payment_status", ("PAID", "FAILED"), "medium"),
        UniqueRule("payments", ("customer_email", "subscription_id", "amount", "paid_at"), "high"),
        ForeignKeyRule("payments", "customer_id", "customers", "customer_id", "high"),
        ForeignKeyRule("payments", "subscription_id", "subscriptions", "subscription_id", "high"),
    )
    issue_cards = (
        IssueCard(title="Customer master data contains duplicated identities", detail="Two Ana Lopez records use different customer IDs but the same email after normalization.", issue_codes=["unique:customers.email", "pattern:customers.email"], recommended_operation_ids=["hard_normalize_customer_fields", "hard_merge_customers_by_email"]),
        IssueCard(title="Child tables contain invalid customer references", detail="Subscription and payment rows reference stale or blank customer IDs that must be repaired from email joins.", issue_codes=["foreign_key:subscriptions.customer_id", "foreign_key:payments.customer_id", "required:payments.customer_id"], recommended_operation_ids=["hard_repair_subscription_customer_refs", "hard_repair_payment_customer_refs"]),
        IssueCard(title="Subscription and payment facts use inconsistent formats", detail="Plans, statuses, dates, amounts, and currency values need canonicalization before loading.", issue_codes=["enum:subscriptions.plan_code", "enum:subscriptions.status", "pattern:subscriptions.renewal_date", "pattern:payments.amount", "enum:payments.payment_status", "pattern:payments.paid_at"], recommended_operation_ids=["hard_normalize_subscriptions", "hard_normalize_payments"]),
        IssueCard(title="Duplicate payment facts are present", detail="Two payment rows represent the same invoice settlement and one should be removed.", issue_codes=["unique:payments.customer_email+subscription_id+amount+paid_at"], recommended_operation_ids=["hard_remove_duplicate_payments"]),
    )
    return _task_from_solution(
        task_id="crm_migration_hard",
        title="CRM Migration Referential Cleanup",
        difficulty="hard",
        objective="Repair a three-table CRM migration extract by standardizing customer, subscription, and payment data; merging duplicate customers; fixing foreign keys from email joins; and removing duplicate payment facts without dropping legitimate orphan-like child rows.",
        dataset_context="This dataset simulates a SaaS CRM and billing migration where a team must clean customer master data and child ledger references before import.",
        max_steps=18,
        primary_keys={"customers": "customer_id", "subscriptions": "subscription_id", "payments": "payment_id"},
        duplicate_identity_columns={"customers": ("email",), "subscriptions": ("subscription_id",), "payments": ("customer_email", "subscription_id", "amount", "paid_at")},
        dirty_tables=dirty_tables,
        validation_rules=validation_rules,
        operations=operations,
        solution_operation_ids=("hard_normalize_customer_fields", "hard_merge_customers_by_email", "hard_normalize_subscriptions", "hard_repair_subscription_customer_refs", "hard_normalize_payments", "hard_repair_payment_customer_refs", "hard_remove_duplicate_payments"),
        issue_cards=issue_cards,
    )


TASK_CATALOG: dict[str, TaskSpec] = {spec.task_id: spec for spec in (_build_easy_task(), _build_medium_task(), _build_hard_task())}


def list_task_ids() -> list[str]:
    return list(TASK_CATALOG.keys())


def get_task_spec(task_id: str) -> TaskSpec:
    if task_id not in TASK_CATALOG:
        raise KeyError(f"Unknown task_id '{task_id}'. Available tasks: {list_task_ids()}")
    return TASK_CATALOG[task_id]


def first_table_name(task_spec: TaskSpec) -> str:
    return next(iter(task_spec.dirty_tables))


def apply_operation_to_tables(task_spec: TaskSpec, tables: Tables, operation_id: str) -> Tables:
    if operation_id not in task_spec.operations:
        raise KeyError(f"Unknown operation_id '{operation_id}' for task '{task_spec.task_id}'.")
    transform = task_spec.operations[operation_id].transform
    return transform(clone_tables(tables))


def sorted_rows(rows: list[dict[str, str]], primary_key: str) -> list[dict[str, str]]:
    return sorted(copy.deepcopy(rows), key=lambda row: normalize_whitespace(row.get(primary_key, "")))
