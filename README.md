---
title: CleanOps Env
emoji: "🧹"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - data-cleaning
  - reinforcement-learning
---

# CleanOps OpenEnv

CleanOps is a real-world OpenEnv benchmark for evaluating AI agents on
operational data-cleaning workflows. Instead of solving a game or toy puzzle,
the agent has to inspect messy business tables, choose safe remediation actions,
escalate ambiguous records for human review, dry-run downstream syncs, avoid
destructive shortcuts, and submit a cleaned dataset scored by deterministic
graders.

The benchmark models the kind of cleanup work that sales ops, RevOps, support
ops, and data platform teams perform before loading data into CRMs, billing
systems, and analytics warehouses.

## Live Links

- Hugging Face Space: [harsharajkumar273/cleanops-openenv](https://huggingface.co/spaces/harsharajkumar273/cleanops-openenv)
- Live App: [harsharajkumar273-cleanops-openenv.hf.space](https://harsharajkumar273-cleanops-openenv.hf.space/)
- GitHub Repository: [harsharajkumar/cleanops-openenv](https://github.com/harsharajkumar/cleanops-openenv)

## Portfolio Highlights

- Real-world benchmark: evaluates agents on CRM, orders, subscriptions, and
  payments data-cleaning tasks rather than synthetic game mechanics.
- Full OpenEnv implementation: typed `Action`, `Observation`, and `State`
  models plus `reset()`, `step()`, and `state()` APIs.
- Human-in-the-loop realism: review queues and downstream dry-run simulation
  make the benchmark feel closer to RevOps / DataOps work than static CSV cleanup.
- Deterministic evaluation: three graded tasks with reproducible `(0.0, 1.0)`
  scoring and interpretable grader components.
- Dense reward shaping: partial progress signals reward useful cleanup while
  penalizing invalid, repeated, or premature actions.
- Production-style delivery: shipped with a Dockerfile, a live Hugging Face
  Space, baseline inference scripts, and tests.

## Why This Environment Is Useful

- Realistic domain: tabular standardization, missing-value repair,
  deduplication, and referential-integrity fixes.
- Reproducible evaluation: every task returns a deterministic `(0.0, 1.0)` score
  with interpretable components.
- Curriculum structure: one easy, one medium, and one hard task with increasing
  schema complexity and cross-table dependencies.
- Agent-friendly observations: the environment surfaces validation issues,
  table previews, review targets, downstream health, and reward breakdowns
  that make policy learning and debugging tractable.

## What The Agent Actually Does

On each episode, the agent:

1. inspects noisy business tables and validation issues
2. chooses from a typed catalog of cleaning operations
3. requests human review for ambiguous records when automation would be risky
4. runs deterministic dry-run syncs against downstream CRM / billing systems
5. applies targeted fixes while avoiding destructive shortcuts
6. submits the cleaned dataset for deterministic scoring

This creates a realistic benchmark loop for evaluating whether an agent can
reason about messy structured data over multiple steps.

## Task Suite

| Task ID | Difficulty | Description |
|---|---|---|
| `customer_contacts_easy` | Easy | Clean a CRM contacts export by normalizing names/emails/phones/states, handling one reviewable duplicate, and preparing the table for CRM import. |
| `orders_reconciliation_medium` | Medium | Clean an e-commerce order extract by standardizing dates, currency, amounts, statuses, and shipping states while preserving returned orders and checking downstream billing readiness. |
| `crm_migration_hard` | Hard | Repair a 3-table CRM migration extract with duplicate customers, broken foreign keys, ambiguous payment/customer linkages, review escalation, and CRM/billing dry-run checks. |

## API

### Local Python API

```python
from cleanops_env import DataCleaningAction, LocalCleanOpsEnv

env = LocalCleanOpsEnv()
observation = env.reset(task_id="customer_contacts_easy", seed=7)

observation, reward, done, info = env.step(
    DataCleaningAction(
        action_type="apply_operation",
        operation_id="easy_normalize_emails",
        reasoning="Normalize emails before customer deduplication.",
    )
)

state = env.state()
```

### OpenEnv Server API

```bash
PYTHONPATH="$PWD" python -m server.app --host 0.0.0.0 --port 8000
```

Then use the typed WebSocket client:

```python
from cleanops_env import CleanOpsEnvClient, DataCleaningAction

with CleanOpsEnvClient(base_url="http://127.0.0.1:8000") as env:
    result = env.reset(task_id="orders_reconciliation_medium", seed=7)
    result = env.step(
        DataCleaningAction(
            action_type="inspect_table",
            table_name="orders",
            reasoning="Review order rows before cleaning.",
        )
    )
    state = env.state()
```

## Action Space

`DataCleaningAction`

| Field | Type | Meaning |
|---|---|---|
| `action_type` | `"inspect_table" \| "inspect_operation" \| "apply_operation" \| "request_review" \| "run_sync_dry_run" \| "submit"` | Selects the action family. |
| `table_name` | `str \| null` | Table to inspect when `action_type="inspect_table"`. |
| `operation_id` | `str \| null` | Cleaning operation to inspect/apply. |
| `entity_type`, `entity_id`, `reason_code` | `str \| null` | Structured review request fields for ambiguous entities. |
| `target_system` | `"crm" \| "billing" \| null` | Downstream system to test with a dry run. |
| `reasoning` | `str` | Optional trace text used by baseline scripts. |
| `metadata` | `dict` | OpenEnv metadata channel. |

## Observation Space

`DataCleaningObservation` extends OpenEnv's typed `Observation` model and includes:

| Field | Meaning |
|---|---|
| `task_id`, `task_title`, `difficulty`, `objective`, `dataset_context` | Task metadata and objective. |
| `quality_score`, `best_score`, `grader` | Deterministic score and score decomposition. |
| `remaining_steps`, `done`, `reward`, `reward_breakdown` | Episode and reward state. |
| `review_budget_remaining`, `available_review_targets`, `pending_reviews`, `resolved_reviews` | Human-review queue state for ambiguous records. |
| `supported_sync_targets`, `downstream_health`, `risk_cards`, `last_dry_run` | Downstream business-system simulation state. |
| `action_costs` | Estimated cost profile for the action families available in this benchmark. |
| `table_summaries` | Compact per-table statistics and previews. |
| `focus_table` | Full rows for the currently inspected table. |
| `available_operations` | Typed catalog of cleaning actions and risk labels. |
| `focus_operation` | Predicted row-level before/after diff for an inspected operation. |
| `validation_issues`, `issue_cards` | Current rule failures and remediation hints. |
| `recent_history`, `last_action_status`, `last_action_error`, `metadata` | Interaction trace and episode metadata. |

`DataCleaningState` returns the current mutable tables, applied operations,
inspection history, step count, and score state.

## Reward Function

Each step computes:

```text
reward =
  1.00 * score_delta
+ 0.35 * issue_count_delta
+ 0.55 * downstream_health_delta
+ inspection_bonus
+ review_bonus
+ step_penalty
+ invalid_action_penalty
+ no_op_penalty
+ review_cost_penalty
+ action_cost_penalty
+ submit_bonus
```

This gives partial progress credit throughout the trajectory and penalizes
repeat/no-op actions, invalid operations, unnecessary review spend, risky
destructive behavior, and low-quality premature submission.

## System Design

- `cleanops_env/tasks.py`: task definitions, gold tables, and operation catalog
- `cleanops_env/graders.py`: deterministic scoring logic and validation checks
- `cleanops_env/environment.py`: OpenEnv episode state, reward shaping, and
  typed step/reset/state implementation with review queues and dry-run simulation
- `server/app.py`: FastAPI/OpenEnv server plus the Hugging Face demo UI
- `inference.py`: submission-ready baseline runner with structured logs

The design intentionally separates task data, grading logic, and runtime state
so the benchmark is easier to extend and easier to reason about during
evaluation.

## Grading

Each task uses a deterministic grader that outputs a final score in `(0.0, 1.0)`
from three components:

- `cell_match_score`: exact canonicalized cell match against gold cleaned tables.
- `key_recall_score`: entity/row identity quality after dedupe and row retention.
- `validation_score`: fraction of unresolved data-quality checks eliminated.

Final score:

```text
0.55 * cell_match_score + 0.20 * key_recall_score + 0.25 * validation_score
```

## Setup

```bash
git clone https://github.com/harsharajkumar/cleanops-openenv.git
cd cleanops-openenv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Validate

```bash
openenv validate --verbose
pytest -q
```

## Submission Inference Script

`inference.py` lives at the project root and follows the required stdout
contract:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
```

Environment variables:

| Variable | Purpose |
|---|---|
| `API_BASE_URL` | OpenAI-compatible inference endpoint. Defaults to `https://router.huggingface.co/v1`. |
| `MODEL_NAME` | Model identifier. Defaults to `Qwen/Qwen2.5-72B-Instruct`. |
| `HF_TOKEN` | API key for the inference endpoint. |
| `LOCAL_IMAGE_NAME` | Optional local Docker image name used with `CleanOpsEnvClient.from_docker_image()`. |
| `TASK_NAME` | Task to run, or `all` for all tasks. Defaults to `all`. |

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="..."
PYTHONPATH="$PWD" python inference.py
```

## Baselines

### Deterministic Oracle Smoke Baseline

```bash
PYTHONPATH="$PWD" python scripts/run_oracle_smoke.py
```

Expected scores measured locally:

| Task ID | Score | Steps | Total Reward |
|---|---:|---:|---:|
| `customer_contacts_easy` | 0.9900 | 7 | 1.1280 |
| `orders_reconciliation_medium` | 0.9900 | 6 | 1.0325 |
| `crm_migration_hard` | 0.9900 | 8 | 1.2568 |
| Mean | 0.9900 | - | - |

### OpenAI Baseline Agent

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"
export OPENAI_SEED=7
PYTHONPATH="$PWD" python scripts/run_openai_baseline.py --output openai_baseline.json
```

The OpenAI runner uses temperature `0`, fixed seed values, and the typed
`DataCleaningAction` schema to produce reproducible rollouts.

## Why This Is A Strong Portfolio Project

- It shows environment design, not just model prompting.
- It combines backend engineering, evaluation design, reward shaping, and
  deployment.
- It demonstrates agent tooling with typed APIs, deterministic graders, and a
  live hosted demo.
- It reflects an applied ML systems problem that maps to real business
  workflows.

## Docker

```bash
docker build -t cleanops-env:latest .
docker run --rm -p 8000:8000 cleanops-env:latest
curl http://127.0.0.1:8000/health
```

## Hugging Face Spaces Deployment

1. Create a new Docker Space.
2. Upload this directory as the Space repo contents.
3. Keep the README metadata frontmatter and `Dockerfile` at repo root.
4. Ensure the Space has the `openenv` tag.
5. If needed, push with the OpenEnv CLI:

```bash
openenv push
```

## Project Structure

```text
cleanops-openenv/
├── cleanops_env/
│   ├── client.py
│   ├── environment.py
│   ├── graders.py
│   ├── local_env.py
│   ├── models.py
│   └── tasks.py
├── scripts/
│   ├── run_openai_baseline.py
│   └── run_oracle_smoke.py
├── server/
│   ├── app.py
│   └── Dockerfile
├── tests/
│   └── test_environment.py
├── Dockerfile
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```
