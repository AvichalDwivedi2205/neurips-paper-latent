"""Report-generation slice tests."""

from __future__ import annotations

import pandas as pd

from latentgoalops.analysis.report import _primary_episode_slice, _primary_step_slice


def test_report_prefers_strict_episode_rows_when_available():
    episodes = pd.DataFrame(
        [
            {"task_id": "task2_roadmap_priority", "score": 0.9, "strict_episode": 1},
            {"task_id": "task2_roadmap_priority", "score": 0.4, "strict_episode": 0},
        ]
    )
    primary, mode = _primary_episode_slice(episodes)
    assert mode == "strict_only"
    assert len(primary) == 1
    assert float(primary.iloc[0]["score"]) == 0.9


def test_report_prefers_strict_step_rows_when_available():
    steps = pd.DataFrame(
        [
            {"task_id": "task3_startup_week", "step_index": 0, "reward": 0.2, "strict_step": 1},
            {"task_id": "task3_startup_week", "step_index": 0, "reward": 0.8, "strict_step": 0},
        ]
    )
    primary, mode = _primary_step_slice(steps)
    assert mode == "strict_only"
    assert len(primary) == 1
    assert float(primary.iloc[0]["reward"]) == 0.2
