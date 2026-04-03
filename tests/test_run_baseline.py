"""Baseline-runner helper tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

from latentgoalops.baseline.run_baseline import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_PERSONA_MODEL,
    _logged_episode_payloads,
    _parse_budget_cap_arg,
    run,
)


def test_logged_episode_payloads_handles_missing_file(tmp_path):
    assert _logged_episode_payloads(tmp_path / "missing.jsonl") == []


def test_logged_episode_payloads_filters_non_episode_rows(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step_index": 0, "reward": 0.1}),
                json.dumps({"task_id": "task1_feedback_triage", "score": 0.8, "total_steps": 1}),
            ]
        ),
        encoding="utf-8",
    )
    assert _logged_episode_payloads(path) == [
        {"task_id": "task1_feedback_triage", "score": 0.8, "total_steps": 1}
    ]


def test_parse_budget_cap_arg_supports_uncapped_mode():
    assert _parse_budget_cap_arg(None) is None
    assert _parse_budget_cap_arg("none") is None
    assert _parse_budget_cap_arg("uncapped") is None
    assert _parse_budget_cap_arg("2.5") == 2.5


def test_run_accepts_uncapped_budget(tmp_path):
    means = run(
        SimpleNamespace(
            policy="random",
            model=DEFAULT_AGENT_MODEL,
            persona_model=DEFAULT_PERSONA_MODEL,
            operator_style="auto",
            tasks="task1_feedback_triage",
            seeds="100:1",
            output_dir=str(tmp_path),
            run_id="random-uncapped-smoke",
            budget_cap=None,
            temperature=0.0,
            disable_hidden_shift=False,
            disable_delayed_effects=False,
            hide_decision_ledger=False,
            reward_mode="shaped",
            task3_horizon_override=None,
            scenario_split="core",
            disable_parse_repair=False,
            disable_heuristic_rescue=False,
            enable_task2_visible_floor=False,
            wandb=False,
            wandb_project="latentgoalops",
            wandb_entity=None,
            wandb_group=None,
            wandb_tags="",
        )
    )

    assert "task1_feedback_triage" in means
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["episode_count"] == 1
