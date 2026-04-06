"""FastAPI application for the CleanOps data-cleaning environment."""

from __future__ import annotations

import copy
import random

from openenv.core import create_app
from fastapi.responses import HTMLResponse, JSONResponse

from cleanops_env.environment import CleanOpsEnvironment
from cleanops_env.models import DataCleaningAction, DataCleaningObservation
from cleanops_env.tasks import first_table_name, get_task_spec, sorted_rows


app = create_app(
    CleanOpsEnvironment,
    DataCleaningAction,
    DataCleaningObservation,
    env_name="cleanops_env",
    max_concurrent_envs=4,
)


@app.get("/demo/compare", include_in_schema=False)
def demo_compare(task_id: str = "customer_contacts_easy", table_name: str | None = None, seed: int | None = None) -> JSONResponse:
    task_spec = get_task_spec(task_id)
    selected_table = table_name if table_name in task_spec.dirty_tables else first_table_name(task_spec)
    primary_key = task_spec.primary_keys[selected_table]
    before_rows = _seed_preview_rows(task_spec.dirty_tables[selected_table], primary_key, selected_table, seed)
    after_rows = _seed_preview_rows(task_spec.gold_tables[selected_table], primary_key, selected_table, seed)
    columns = sorted({column_name for row in before_rows + after_rows for column_name in row})
    return JSONResponse(
        {
            "task_id": task_spec.task_id,
            "task_title": task_spec.title,
            "table_name": selected_table,
            "requested_seed": seed,
            "available_tables": list(task_spec.dirty_tables.keys()),
            "columns": columns,
            "before_rows": before_rows[:4],
            "after_rows": after_rows[:4],
            "before_row_count": len(before_rows),
            "after_row_count": len(after_rows),
            "solution_operation_ids": list(task_spec.solution_operation_ids),
        }
    )


def _seed_preview_rows(
    rows: list[dict[str, str]],
    primary_key: str,
    table_name: str,
    seed: int | None,
) -> list[dict[str, str]]:
    ordered_rows = sorted_rows(rows, primary_key)
    if seed is None or len(ordered_rows) <= 1:
        return ordered_rows
    shuffled_rows = copy.deepcopy(ordered_rows)
    random.Random(max(0, int(seed)) + sum(ord(char) for char in table_name)).shuffle(shuffled_rows)
    return shuffled_rows


@app.get("/", include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>CleanOps OpenEnv</title>
            <style>
              :root {
                color-scheme: light;
                --bg: #f8fbff;
                --bg-2: #eef6ff;
                --panel: rgba(255, 255, 255, 0.92);
                --panel-strong: #ffffff;
                --panel-soft: #f4f8ff;
                --line: rgba(148, 163, 184, 0.24);
                --line-strong: rgba(148, 163, 184, 0.34);
                --text: #22314d;
                --muted: #64748b;
                --accent: #4f9cf9;
                --accent-2: #f59ac2;
                --success: #22c55e;
                --accent-3: #7dd3fc;
                --danger: #dc6b8a;
                --warning: #d9a441;
                --shadow: 0 24px 80px rgba(113, 140, 177, 0.20);
                --radius: 22px;
              }
              html,
              body {
                font-family: Inter, Arial, sans-serif;
                margin: 0;
                min-height: 100vh;
                background:
                  radial-gradient(circle at top left, rgba(79, 156, 249, 0.18), transparent 28%),
                  radial-gradient(circle at top right, rgba(245, 154, 194, 0.16), transparent 24%),
                  linear-gradient(180deg, #fdfcff 0%, #f4f9ff 48%, #eef6ff 100%);
                color: var(--text);
                overflow-x: hidden;
              }
              * {
                box-sizing: border-box;
              }
              .shell {
                width: min(1180px, 100%);
                max-width: 1180px;
                margin: 0 auto;
                padding: 28px 20px 56px;
              }
              .topbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 18px;
                margin-bottom: 18px;
              }
              .brand {
                display: flex;
                align-items: center;
                gap: 14px;
              }
              .brand-mark {
                width: 42px;
                height: 42px;
                border-radius: 14px;
                display: grid;
                place-items: center;
                font-size: 20px;
                background: linear-gradient(135deg, rgba(79, 156, 249, 0.18), rgba(245, 154, 194, 0.18));
                border: 1px solid rgba(148, 163, 184, 0.20);
              }
              .brand-copy strong {
                display: block;
                font-size: 15px;
                letter-spacing: 0.01em;
              }
              .brand-copy span {
                color: var(--muted);
                font-size: 13px;
              }
              .health {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                padding: 10px 14px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid var(--line);
                color: var(--muted);
                font-size: 13px;
                box-shadow: 0 8px 24px rgba(148, 163, 184, 0.10);
              }
              .health-dot {
                width: 10px;
                height: 10px;
                border-radius: 999px;
                background: var(--warning);
                box-shadow: 0 0 0 5px rgba(251, 191, 36, 0.10);
              }
              .health.ready .health-dot {
                background: var(--success);
                box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.12);
              }
              .health.error .health-dot {
                background: var(--danger);
                box-shadow: 0 0 0 5px rgba(248, 113, 113, 0.10);
              }
              .hero {
                position: relative;
                overflow: hidden;
                border-radius: 30px;
                padding: 34px;
                background:
                  linear-gradient(135deg, rgba(79, 156, 249, 0.20), rgba(245, 154, 194, 0.16)),
                  rgba(255, 255, 255, 0.82);
                border: 1px solid var(--line);
                box-shadow: var(--shadow);
                display: grid;
                grid-template-columns: 1.15fr 0.85fr;
                gap: 26px;
              }
              .hero::after {
                content: "";
                position: absolute;
                inset: auto -10% -35% auto;
                width: 360px;
                height: 360px;
                border-radius: 999px;
                background: radial-gradient(circle, rgba(245, 154, 194, 0.20), transparent 60%);
                pointer-events: none;
              }
              .badge-row {
                display: flex;
                gap: 9px;
                flex-wrap: wrap;
                margin-bottom: 18px;
              }
              .badge {
                display: inline-flex;
                align-items: center;
                gap: 7px;
                font-size: 12px;
                padding: 7px 11px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.18);
                color: #35527d;
              }
              h1 {
                margin: 0 0 14px;
                font-size: 46px;
                line-height: 1.02;
                letter-spacing: -0.03em;
              }
              .hero p {
                margin: 0;
                font-size: 17px;
                color: #5a708d;
                max-width: 58ch;
              }
              .hero-actions {
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                margin-top: 22px;
              }
              .demo-controls {
                margin-top: 18px;
                display: grid;
                grid-template-columns: 1fr auto auto;
                gap: 10px;
                align-items: end;
              }
              .field {
                display: grid;
                gap: 6px;
              }
              .field label {
                font-size: 12px;
                color: var(--muted);
                font-weight: 600;
              }
              .field input,
              .field select {
                width: 100%;
                appearance: none;
                border: 1px solid var(--line);
                border-radius: 14px;
                background: rgba(255, 255, 255, 0.88);
                color: #294668;
                padding: 12px 14px;
                font-size: 14px;
                font-family: inherit;
              }
              .field input:focus,
              .field select:focus {
                outline: 2px solid rgba(79, 156, 249, 0.20);
                border-color: rgba(79, 156, 249, 0.35);
              }
              .run-feedback {
                margin-top: 12px;
                display: inline-flex;
                align-items: center;
                gap: 10px;
                min-height: 20px;
                color: #426284;
                font-size: 13px;
                font-weight: 600;
              }
              .run-feedback.loading::before,
              .run-feedback.success::before,
              .run-feedback.error::before {
                content: "";
                width: 10px;
                height: 10px;
                border-radius: 999px;
                display: inline-block;
              }
              .run-feedback.loading::before {
                background: var(--accent);
                box-shadow: 0 0 0 5px rgba(79, 156, 249, 0.10);
              }
              .run-feedback.success::before {
                background: #4fb58d;
                box-shadow: 0 0 0 5px rgba(79, 181, 141, 0.10);
              }
              .run-feedback.error::before {
                background: var(--danger);
                box-shadow: 0 0 0 5px rgba(220, 107, 138, 0.10);
              }
              .hero-note {
                margin-top: 18px;
                color: var(--muted);
                font-size: 13px;
              }
              .hero-card {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid var(--line);
                border-radius: 24px;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 16px;
              }
              .hero-card h2 {
                margin: 0;
                font-size: 16px;
                color: #26436b;
              }
              .hero-card p,
              .hero-card li {
                font-size: 14px;
                color: var(--muted);
              }
              .stat-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 12px;
              }
              .stat {
                background: rgba(244, 248, 255, 0.92);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 14px;
              }
              .stat span {
                display: block;
                font-size: 12px;
                color: var(--muted);
                margin-bottom: 6px;
              }
              .stat strong {
                font-size: 18px;
                color: #2b466e;
              }
              .grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 22px;
                margin-top: 22px;
                align-items: start;
              }
              .hero > *,
              .grid > *,
              .subgrid > *,
              .compare-grid > *,
              .panel,
              .mini-panel,
              .table-wrap,
              .compare-card {
                min-width: 0;
              }
              .panel {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 24px;
                box-shadow: var(--shadow);
                padding: 24px;
                backdrop-filter: blur(10px);
              }
              .panel h2 {
                margin: 0 0 10px;
                font-size: 21px;
                letter-spacing: -0.02em;
              }
              .panel p {
                margin: 0 0 14px;
                color: var(--muted);
              }
              .subgrid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 16px;
                margin-top: 18px;
                align-items: start;
              }
              .button-row {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin: 18px 0 18px;
              }
              button,
              .link-btn {
                appearance: none;
                border: 1px solid transparent;
                border-radius: 14px;
                padding: 11px 15px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                text-decoration: none;
                transition: transform 0.15s ease, opacity 0.15s ease, box-shadow 0.15s ease;
                display: inline-flex;
                align-items: center;
                justify-content: center;
              }
              button:hover,
              .link-btn:hover {
                transform: translateY(-1px);
              }
              button:disabled {
                cursor: not-allowed;
                opacity: 0.7;
                transform: none;
              }
              .primary {
                background: linear-gradient(135deg, rgba(79, 156, 249, 0.18), rgba(245, 154, 194, 0.18));
                color: #244267;
                border-color: rgba(79, 156, 249, 0.22);
                box-shadow: 0 8px 24px rgba(79, 156, 249, 0.12);
              }
              .primary.is-loading {
                background: linear-gradient(135deg, rgba(79, 156, 249, 0.24), rgba(245, 154, 194, 0.24));
                border-color: rgba(79, 156, 249, 0.32);
              }
              .secondary {
                background: rgba(244, 248, 255, 0.92);
                color: #32527d;
                border-color: var(--line);
              }
              .button-row .primary.active {
                background: linear-gradient(135deg, rgba(79, 156, 249, 0.24), rgba(245, 154, 194, 0.24));
                border-color: rgba(79, 156, 249, 0.32);
              }
              .status {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: rgba(255, 255, 255, 0.94);
                color: #2d4a73;
                border: 1px solid var(--line);
                padding: 10px 12px;
                border-radius: 12px;
                font-size: 14px;
                font-weight: 600;
              }
              .status.error {
                color: var(--danger);
                border-color: rgba(248, 113, 113, 0.28);
              }
              .kpis {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
                margin-top: 16px;
              }
              .kpi {
                background: rgba(244, 248, 255, 0.92);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 14px;
              }
              .kpi span {
                display: block;
                font-size: 12px;
                color: var(--muted);
                margin-bottom: 6px;
              }
              .kpi strong {
                font-size: 24px;
                letter-spacing: -0.02em;
                display: block;
                line-height: 1.05;
                word-break: break-word;
                overflow-wrap: anywhere;
              }
              .kpi.task-card strong {
                font-size: clamp(18px, 2vw, 28px);
                line-height: 1.15;
              }
              .kpi small {
                display: block;
                margin-top: 8px;
                color: var(--muted);
                font-size: 12px;
                word-break: break-word;
                overflow-wrap: anywhere;
              }
              pre {
                margin: 0;
                background: #f5f8ff;
                color: #29405e;
                border-radius: 18px;
                padding: 16px;
                overflow-x: auto;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                font-size: 13px;
                line-height: 1.5;
                border: 1px solid var(--line);
              }
              code {
                background: rgba(79, 156, 249, 0.10);
                color: #35527d;
                border-radius: 8px;
                padding: 2px 7px;
              }
              .mini-panel {
                background: rgba(255, 255, 255, 0.80);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 16px;
              }
              .mini-panel h3 {
                margin: 0 0 10px;
                font-size: 14px;
                color: #294668;
              }
              .chips {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
              }
              .chip {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 8px 10px;
                border-radius: 999px;
                background: rgba(79, 156, 249, 0.08);
                border: 1px solid var(--line);
                color: #35527d;
                font-size: 12px;
              }
              .chip.warn {
                color: #9b5a72;
                border-color: rgba(245, 154, 194, 0.24);
                background: rgba(245, 154, 194, 0.10);
              }
              .table-wrap {
                overflow: auto;
                border: 1px solid var(--line);
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.84);
                width: 100%;
                max-width: 100%;
              }
              .compare-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 12px;
              }
              .compare-card {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid var(--line);
                border-radius: 16px;
                overflow: hidden;
              }
              .compare-card header {
                display: flex;
                justify-content: space-between;
                gap: 10px;
                align-items: center;
                padding: 12px 14px;
                border-bottom: 1px solid var(--line);
                background: rgba(244, 248, 255, 0.92);
              }
              .compare-card strong {
                color: #294668;
                font-size: 13px;
              }
              .compare-card span {
                color: var(--muted);
                font-size: 12px;
              }
              table {
                width: max-content;
                min-width: 100%;
                border-collapse: collapse;
              }
              th, td {
                padding: 12px 14px;
                text-align: left;
                border-bottom: 1px solid rgba(148, 163, 184, 0.12);
                font-size: 13px;
                vertical-align: top;
              }
              th {
                position: sticky;
                top: 0;
                background: rgba(244, 248, 255, 0.98);
                color: #294668;
                font-weight: 600;
              }
              td {
                color: #49617f;
              }
              .endpoint-list {
                display: grid;
                gap: 10px;
              }
              .endpoint-list {
                display: grid;
                gap: 10px;
              }
              .endpoint {
                display: flex;
                justify-content: space-between;
                gap: 10px;
                align-items: center;
                background: rgba(244, 248, 255, 0.92);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 12px 14px;
              }
              .endpoint small {
                color: var(--muted);
              }
              a {
                color: var(--accent);
              }
              .footer {
                margin-top: 20px;
                color: var(--muted);
                font-size: 13px;
              }
              .stack {
                display: grid;
                gap: 16px;
              }
              @media (max-width: 860px) {
                .topbar,
                .hero,
                .grid,
                .subgrid {
                  grid-template-columns: 1fr;
                }
                .demo-controls {
                  grid-template-columns: 1fr;
                }
                .kpis,
                .stat-grid {
                  grid-template-columns: 1fr;
                }
                .compare-grid {
                  grid-template-columns: 1fr;
                }
                h1 {
                  font-size: 34px;
                }
              }
            </style>
          </head>
          <body>
            <div class="shell">
              <div class="topbar">
                <div class="brand">
                  <div class="brand-mark">🧹</div>
                  <div class="brand-copy">
                    <strong>CleanOps OpenEnv</strong>
                    <span>Operational data cleaning benchmark</span>
                  </div>
                </div>
                <div id="health" class="health">
                  <span class="health-dot"></span>
                  <span id="healthText">Checking live API status...</span>
                </div>
              </div>

              <section class="hero">
                <div>
                  <div class="badge-row">
                    <div class="badge">OpenEnv Benchmark</div>
                    <div class="badge">Real-world Data Cleaning</div>
                    <div class="badge">Deterministic Graders</div>
                  </div>
                  <h1>See real data cleaning tasks working live.</h1>
                  <p>
                    CleanOps simulates the kind of operational cleanup analysts
                    actually do before data reaches a CRM, warehouse, or billing
                    system. The UI below runs the same hosted benchmark API used
                    by the evaluator.
                  </p>
                  <div class="hero-actions">
                    <button class="primary active" data-task="customer_contacts_easy">Try Easy Task</button>
                    <button class="primary" data-task="orders_reconciliation_medium">Try Medium Task</button>
                    <button class="primary" data-task="crm_migration_hard">Try Hard Task</button>
                  </div>
                  <div class="demo-controls">
                    <div class="field">
                      <label for="taskSelect">Choose task</label>
                      <select id="taskSelect">
                        <option value="customer_contacts_easy">customer_contacts_easy</option>
                        <option value="orders_reconciliation_medium">orders_reconciliation_medium</option>
                        <option value="crm_migration_hard">crm_migration_hard</option>
                      </select>
                    </div>
                    <div class="field">
                      <label for="seedInput">Seed</label>
                      <input id="seedInput" type="number" min="0" step="1" value="7" />
                    </div>
                    <button id="runCustomTask" class="secondary" type="button">Run Selected Task</button>
                  </div>
                  <div id="runFeedback" class="run-feedback">Ready to run a live benchmark task.</div>
                  <div class="hero-note">
                    Changing the seed changes the visible preview ordering and compare view. It does not change the task score itself.
                  </div>
                  <div class="hero-note">
                    Fixed tasks, typed actions, shaped rewards, and reproducible graders.
                  </div>
                </div>
                <div class="hero-card">
                  <h2>At a glance</h2>
                  <div class="stat-grid">
                    <div class="stat">
                      <span>Task ladder</span>
                      <strong>Easy → Hard</strong>
                    </div>
                    <div class="stat">
                      <span>Core API</span>
                      <strong>/reset /step /state</strong>
                    </div>
                    <div class="stat">
                      <span>Domain</span>
                      <strong>CRM + Orders + Billing</strong>
                    </div>
                    <div class="stat">
                      <span>Reward signal</span>
                      <strong>Dense + partial progress</strong>
                    </div>
                  </div>
                  <p>
                    This homepage is a thin demo over the live environment. It
                    doesn’t fake results: every task button calls the deployed API.
                  </p>
                </div>
              </section>

              <section class="grid">
                <div class="panel">
                  <h2>Live Task Snapshot</h2>
                  <p>
                    The cards and table below are populated from a real
                    <code>POST /reset</code> response. Use the task buttons above to
                    switch between benchmark scenarios, or choose your own task and seed.
                  </p>

                  <div class="kpis">
                    <div class="kpi task-card">
                      <span>Task</span>
                      <strong id="taskId">-</strong>
                      <small id="taskMeta">-</small>
                    </div>
                    <div class="kpi">
                      <span>Seed Used</span>
                      <strong id="seedUsed">-</strong>
                    </div>
                    <div class="kpi">
                      <span>Initial Score</span>
                      <strong id="score">-</strong>
                    </div>
                    <div class="kpi">
                      <span>Validation Issues</span>
                      <strong id="issues">-</strong>
                    </div>
                    <div class="kpi">
                      <span>Focus Table Rows</span>
                      <strong id="rowCount">-</strong>
                    </div>
                  </div>

                  <div class="subgrid">
                    <div class="stack">
                      <div class="mini-panel">
                        <h3>Objective</h3>
                        <div id="objective" style="color: var(--text); line-height: 1.55;">
                          Loading...
                        </div>
                      </div>
                      <div class="mini-panel">
                        <h3>Validation Issues</h3>
                        <div id="issueChips" class="chips"></div>
                      </div>
                      <div class="mini-panel">
                        <h3>Available Operations</h3>
                        <div id="operationChips" class="chips"></div>
                      </div>
                    </div>
                    <div class="stack">
                      <div class="mini-panel">
                        <h3>Before / After Cleaning</h3>
                        <div id="compareMeta" style="color: var(--muted); margin-bottom: 12px;">
                          Loading compare view...
                        </div>
                        <div id="solutionChips" class="chips" style="margin-bottom: 12px;"></div>
                        <div class="compare-grid">
                          <div class="compare-card">
                            <header>
                              <strong>Dirty input</strong>
                              <span id="beforeMeta">-</span>
                            </header>
                            <div class="table-wrap" style="border: none; border-radius: 0; background: transparent;">
                              <table>
                                <thead>
                                  <tr id="beforeHeadRow"></tr>
                                </thead>
                                <tbody id="beforeBody"></tbody>
                              </table>
                            </div>
                          </div>
                          <div class="compare-card">
                            <header>
                              <strong>Expected clean output</strong>
                              <span id="afterMeta">-</span>
                            </header>
                            <div class="table-wrap" style="border: none; border-radius: 0; background: transparent;">
                              <table>
                                <thead>
                                  <tr id="afterHeadRow"></tr>
                                </thead>
                                <tbody id="afterBody"></tbody>
                              </table>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div class="mini-panel">
                        <h3>Focus Table Preview</h3>
                        <div class="table-wrap">
                          <table>
                            <thead>
                              <tr id="tableHeadRow"></tr>
                            </thead>
                            <tbody id="tableBody"></tbody>
                          </table>
                        </div>
                      </div>
                      <div class="mini-panel">
                        <h3>Raw Demo Payload</h3>
                        <pre id="output">Loading live task data...</pre>
                      </div>
                    </div>
                  </div>
                </div>

                <div class="panel">
                  <h2>API & Submission Notes</h2>
                  <p>
                    The evaluator checks these endpoints directly. This page exists
                    to make the environment easier to inspect visually.
                  </p>
                  <div class="endpoint-list">
                    <div class="endpoint">
                      <div>
                        <strong>GET /health</strong><br />
                        <small>Service liveness check</small>
                      </div>
                      <a href="/health">Open</a>
                    </div>
                    <div class="endpoint">
                      <div>
                        <strong>GET /schema</strong><br />
                        <small>Typed OpenEnv schema</small>
                      </div>
                      <a href="/schema">Open</a>
                    </div>
                    <div class="endpoint">
                      <div>
                        <strong>GET /docs</strong><br />
                        <small>Interactive FastAPI docs</small>
                      </div>
                      <a href="/docs">Open</a>
                    </div>
                    <div class="endpoint">
                      <div>
                        <strong>POST /reset</strong><br />
                        <small>Start a task episode</small>
                      </div>
                      <code>live</code>
                    </div>
                    <div class="endpoint">
                      <div>
                        <strong>POST /step</strong><br />
                        <small>Apply a typed action</small>
                      </div>
                      <code>live</code>
                    </div>
                    <div class="endpoint">
                      <div>
                        <strong>GET /state</strong><br />
                        <small>Inspect current environment state</small>
                      </div>
                      <code>live</code>
                    </div>
                  </div>

                  <div class="mini-panel" style="margin-top: 18px;">
                    <h3>Sample curl</h3>
                    <pre>curl -X POST /reset -H "Content-Type: application/json" -d '{"task_id":"customer_contacts_easy","seed":7}'</pre>
                  </div>

                  <div class="footer">
                    Fixed tasks plus deterministic graders keep evaluation reproducible.
                  </div>
                </div>
              </section>
            </div>

            <script>
              const healthEl = document.getElementById("health");
              const healthTextEl = document.getElementById("healthText");
              const outputEl = document.getElementById("output");
              const taskEl = document.getElementById("taskId");
              const taskMetaEl = document.getElementById("taskMeta");
              const seedUsedEl = document.getElementById("seedUsed");
              const taskSelectEl = document.getElementById("taskSelect");
              const seedInputEl = document.getElementById("seedInput");
              const runCustomTaskEl = document.getElementById("runCustomTask");
              const runFeedbackEl = document.getElementById("runFeedback");
              const scoreEl = document.getElementById("score");
              const issuesEl = document.getElementById("issues");
              const rowCountEl = document.getElementById("rowCount");
              const objectiveEl = document.getElementById("objective");
              const issueChipsEl = document.getElementById("issueChips");
              const operationChipsEl = document.getElementById("operationChips");
              const tableHeadRowEl = document.getElementById("tableHeadRow");
              const tableBodyEl = document.getElementById("tableBody");
              const compareMetaEl = document.getElementById("compareMeta");
              const solutionChipsEl = document.getElementById("solutionChips");
              const beforeMetaEl = document.getElementById("beforeMeta");
              const afterMetaEl = document.getElementById("afterMeta");
              const beforeHeadRowEl = document.getElementById("beforeHeadRow");
              const beforeBodyEl = document.getElementById("beforeBody");
              const afterHeadRowEl = document.getElementById("afterHeadRow");
              const afterBodyEl = document.getElementById("afterBody");
              const taskButtons = Array.from(document.querySelectorAll("button[data-task]"));
              let isRunning = false;

              function setHealth(kind, message) {
                healthEl.className = `health ${kind}`;
                healthTextEl.textContent = message;
              }

              function setRunFeedback(kind, message) {
                runFeedbackEl.className = `run-feedback ${kind}`;
                runFeedbackEl.textContent = message;
              }

              function clearChildren(node) {
                while (node.firstChild) {
                  node.removeChild(node.firstChild);
                }
              }

              function chip(text, className = "chip") {
                const el = document.createElement("div");
                el.className = className;
                el.textContent = text;
                return el;
              }

              function renderTableTo(headEl, bodyEl, columns, rows) {
                clearChildren(headEl);
                clearChildren(bodyEl);
                columns.forEach((column) => {
                  const th = document.createElement("th");
                  th.textContent = column;
                  headEl.appendChild(th);
                });
                rows.forEach((row) => {
                  const tr = document.createElement("tr");
                  columns.forEach((column) => {
                    const td = document.createElement("td");
                    td.textContent = row[column] ?? "";
                    tr.appendChild(td);
                  });
                  bodyEl.appendChild(tr);
                });
              }

              function renderTable(columns, rows) {
                renderTableTo(tableHeadRowEl, tableBodyEl, columns, rows);
              }

              function setActiveTask(taskId) {
                taskSelectEl.value = taskId;
                taskButtons.forEach((button) => {
                  button.classList.toggle("active", button.dataset.task === taskId);
                });
              }

              function setRunningState(taskId, running) {
                isRunning = running;
                taskButtons.forEach((button) => {
                  const isSelected = button.dataset.task === taskId;
                  button.disabled = running;
                  button.classList.toggle("is-loading", running && isSelected);
                  button.textContent = running && isSelected
                    ? "Loading..."
                    : button.dataset.task === "customer_contacts_easy"
                      ? "Try Easy Task"
                      : button.dataset.task === "orders_reconciliation_medium"
                        ? "Try Medium Task"
                        : "Try Hard Task";
                });
                taskSelectEl.disabled = running;
                seedInputEl.disabled = running;
                runCustomTaskEl.disabled = running;
                runCustomTaskEl.textContent = running ? "Running..." : "Run Selected Task";
              }

              async function loadHealth() {
                try {
                  const response = await fetch("/health");
                  if (!response.ok) throw new Error(`HTTP ${response.status}`);
                  const data = await response.json();
                  setHealth("ready", `API healthy: ${data.status}`);
                } catch (error) {
                  setHealth("error", `API check failed: ${error.message}`);
                }
              }

              async function runTask(taskId, seed = 7) {
                if (isRunning) {
                  return;
                }
                setActiveTask(taskId);
                setRunningState(taskId, true);
                setRunFeedback("loading", `Running ${taskId} with seed ${seed}...`);
                outputEl.textContent = "Loading...";
                objectiveEl.textContent = "Loading...";
                clearChildren(issueChipsEl);
                clearChildren(operationChipsEl);
                clearChildren(solutionChipsEl);
                try {
                  const response = await fetch("/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ task_id: taskId, seed }),
                  });
                  if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                  }
                  const payload = await response.json();
                  const observation = payload.observation || {};
                  const usedSeed = observation.requested_seed ?? seed;
                  taskEl.textContent = observation.task_title || observation.task_id || taskId;
                  taskMetaEl.textContent = observation.task_id || taskId;
                  seedUsedEl.textContent = String(usedSeed);
                  scoreEl.textContent = String(observation.quality_score ?? "-");
                  issuesEl.textContent = String((observation.validation_issues || []).length);
                  rowCountEl.textContent = String((observation.focus_table?.rows || []).length);
                  objectiveEl.textContent = observation.objective || "-";

                  const validationIssues = observation.validation_issues || [];
                  if (validationIssues.length === 0) {
                    issueChipsEl.appendChild(chip("No validation issues", "chip"));
                  } else {
                    validationIssues.slice(0, 6).forEach((issue) => {
                      issueChipsEl.appendChild(chip(`${issue.table_name}.${issue.column_name}: ${issue.row_ids.length}`, "chip warn"));
                    });
                  }

                  const operations = observation.available_operations || [];
                  operations.slice(0, 8).forEach((operation) => {
                    operationChipsEl.appendChild(chip(operation.operation_id));
                  });

                  const columns = observation.focus_table?.columns || [];
                  const rows = (observation.focus_table?.rows || []).slice(0, 4);
                  renderTable(columns, rows);

                  const compareResponse = await fetch(`/demo/compare?task_id=${encodeURIComponent(taskId)}&table_name=${encodeURIComponent(observation.focus_table?.name || "")}&seed=${encodeURIComponent(String(seed))}`);
                  if (!compareResponse.ok) {
                    throw new Error(`Compare HTTP ${compareResponse.status}`);
                  }
                  const comparePayload = await compareResponse.json();
                  compareMetaEl.textContent = `${comparePayload.task_title} • table: ${comparePayload.table_name} • seed: ${comparePayload.requested_seed ?? usedSeed}`;
                  beforeMetaEl.textContent = `${comparePayload.before_row_count} rows`;
                  afterMetaEl.textContent = `${comparePayload.after_row_count} rows`;
                  renderTableTo(beforeHeadRowEl, beforeBodyEl, comparePayload.columns || [], comparePayload.before_rows || []);
                  renderTableTo(afterHeadRowEl, afterBodyEl, comparePayload.columns || [], comparePayload.after_rows || []);
                  (comparePayload.solution_operation_ids || []).forEach((operationId) => {
                    solutionChipsEl.appendChild(chip(operationId));
                  });

                  outputEl.textContent = JSON.stringify(
                    {
                      task_id: observation.task_id,
                      requested_seed: usedSeed,
                      difficulty: observation.difficulty,
                      objective: observation.objective,
                      quality_score: observation.quality_score,
                      remaining_steps: observation.remaining_steps,
                      validation_issue_count: (observation.validation_issues || []).length,
                      focus_table: observation.focus_table?.name,
                      available_operations: (observation.available_operations || []).map((item) => item.operation_id).slice(0, 8),
                    },
                    null,
                    2
                  );
                  setRunFeedback("success", `Loaded ${observation.task_title || taskId} successfully with seed ${usedSeed}.`);
                } catch (error) {
                  outputEl.textContent = `Request failed: ${error.message}`;
                  objectiveEl.textContent = "Request failed.";
                  taskEl.textContent = "Unavailable";
                  taskMetaEl.textContent = taskId;
                  seedUsedEl.textContent = "-";
                  clearChildren(tableHeadRowEl);
                  clearChildren(tableBodyEl);
                  clearChildren(beforeHeadRowEl);
                  clearChildren(beforeBodyEl);
                  clearChildren(afterHeadRowEl);
                  clearChildren(afterBodyEl);
                  compareMetaEl.textContent = "Compare view unavailable.";
                  beforeMetaEl.textContent = "-";
                  afterMetaEl.textContent = "-";
                  setRunFeedback("error", `Run failed: ${error.message}`);
                } finally {
                  setRunningState(taskId, false);
                }
              }

              taskButtons.forEach((button) => {
                button.addEventListener("click", () => {
                  const seed = Number.parseInt(seedInputEl.value || "7", 10);
                  runTask(button.dataset.task, Number.isNaN(seed) ? 7 : seed);
                });
              });

              runCustomTaskEl.addEventListener("click", () => {
                const seed = Number.parseInt(seedInputEl.value || "7", 10);
                runTask(taskSelectEl.value, Number.isNaN(seed) ? 7 : seed);
              });

              loadHealth();
              runTask("customer_contacts_easy", 7);
            </script>
          </body>
        </html>
        """
    )


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.host == "0.0.0.0" and args.port == 8000:
        main()
    else:
        main(host=args.host, port=args.port)
