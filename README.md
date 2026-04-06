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

CleanOps is a real-world OpenEnv benchmark where an agent cleans operational
tabular data from CRM, order, subscription, and payment pipelines. The agent
must inspect tables, choose remediation operations, avoid destructive shortcuts,
and submit a cleaned dataset scored by deterministic graders.

This is intentionally not a game or toy task. It models the kind of operational
data cleanup that sales ops, RevOps, support ops, and data platform teams perform
before loading systems-of-record or analytics warehouses.

## Why This Environment Is Useful

- Realistic domain: tabular data standardization, missing-value repair,
  deduplication, and referential integrity fixes.
- Deterministic programmatic graders: every task returns a reproducible
  `0.0-1.0` score with interpretable components.
- Dense reward shaping: reward is driven by score deltas, issue-count reduction,
  inspection bonuses, step costs, no-op penalties, and submission bonuses.
- Curriculum-ready tasks: one easy, one medium, and one hard task with increasing
  schema complexity and cross-table dependencies.

## Task Suite

| Task ID | Difficulty | Description |
|---|---|---|
| `customer_contacts_easy` | Easy | Clean a CRM contacts export by normalizing names/emails/phones/states, filling one missing state, and merging duplicate customers without dropping inactive accounts. |
| `orders_reconciliation_medium` | Medium | Clean an e-commerce order extract by standardizing dates, currency, amounts, statuses, and shipping states while deduplicating repeated exports and preserving cancelled orders. |
| `crm_migration_hard` | Hard | Repair a 3-table CRM migration extract by normalizing customer/subscription/payment fields, merging duplicate customer IDs, fixing foreign keys from email joins, and removing duplicate payment facts. |

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
cd /Users/harsharajkumar/Downloads/research_paper_simplifier-main/meta
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
| `action_type` | `"inspect_table" \| "inspect_operation" \| "apply_operation" \| "submit"` | Selects the action family. |
| `table_name` | `str \| null` | Table to inspect when `action_type="inspect_table"`. |
| `operation_id` | `str \| null` | Cleaning operation to inspect/apply. |
| `reasoning` | `str` | Optional trace text used by baseline scripts. |
| `metadata` | `dict` | OpenEnv metadata channel. |

## Observation Space

`DataCleaningObservation` extends OpenEnv's typed `Observation` model and includes:

| Field | Meaning |
|---|---|
| `task_id`, `task_title`, `difficulty`, `objective`, `dataset_context` | Task metadata and objective. |
| `quality_score`, `best_score`, `grader` | Deterministic score and score decomposition. |
| `remaining_steps`, `done`, `reward`, `reward_breakdown` | Episode and reward state. |
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
  1.25 * score_delta
+ 0.35 * issue_count_delta
+ inspection_bonus
+ step_penalty
+ invalid_action_penalty
+ no_op_penalty
+ submit_bonus
```

This gives partial progress credit throughout the trajectory and penalizes
repeat/no-op actions, invalid operations, and low-quality premature submission.

## Grading

Each task uses a deterministic grader that outputs a final score in `[0.0, 1.0]`
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
cd /Users/harsharajkumar/Downloads/research_paper_simplifier-main/meta
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Validate

```bash
cd /Users/harsharajkumar/Downloads/research_paper_simplifier-main/meta
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
cd /Users/harsharajkumar/Downloads/research_paper_simplifier-main/meta
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="..."
PYTHONPATH="$PWD" python inference.py
```

## Baselines

### Deterministic Oracle Smoke Baseline

```bash
cd /Users/harsharajkumar/Downloads/research_paper_simplifier-main/meta
PYTHONPATH="$PWD" python scripts/run_oracle_smoke.py
```

Expected scores measured locally:

| Task ID | Score | Steps | Total Reward |
|---|---:|---:|---:|
| `customer_contacts_easy` | 1.0000 | 7 | 1.1430 |
| `orders_reconciliation_medium` | 1.0000 | 6 | 1.0222 |
| `crm_migration_hard` | 1.0000 | 8 | 1.0827 |
| Mean | 1.0000 | - | - |

### OpenAI Baseline Agent

```bash
cd /Users/harsharajkumar/Downloads/research_paper_simplifier-main/meta
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"
export OPENAI_SEED=7
PYTHONPATH="$PWD" python scripts/run_openai_baseline.py --output openai_baseline.json
```

The OpenAI runner uses temperature `0`, fixed seed values, and the typed
`DataCleaningAction` schema to produce reproducible rollouts.

## Docker

```bash
cd /Users/harsharajkumar/Downloads/research_paper_simplifier-main/meta
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
cd /Users/harsharajkumar/Downloads/research_paper_simplifier-main/meta
openenv push
```

## Project Structure

```text
meta/
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
