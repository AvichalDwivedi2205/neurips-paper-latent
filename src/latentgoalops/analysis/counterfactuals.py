"""Counterfactual consistency checks for multi-step tasks."""

from __future__ import annotations

import argparse
import json
import random

from latentgoalops.experiment import ExperimentConfig
from latentgoalops.models import TaskId
from latentgoalops.server.environment import LatentGoalOpsEnvironment
from latentgoalops.server.hidden_goals import active_weights
from latentgoalops.server.tasks.task3_startup_week import evaluate_task3_action_value, solve_task3_oracle_action
from latentgoalops.server.tasks.task7_quarterly_headcount_plan import evaluate_task7_action_value, solve_task7_oracle_action


def _advance_to_step(env: LatentGoalOpsEnvironment, target_step: int) -> None:
    observation = env.reset(seed=env._seed, task_id=env._task_id.value)  # type: ignore[attr-defined]
    while observation.step_index < target_step and not observation.done:
        observation = env.step(env.sample_heuristic_action())


def counterfactual_report(task_id: str, seed: int, step_index: int, scenario_split: str = "core") -> dict:
    """Return a lightweight counterfactual comparison around one state."""
    config = ExperimentConfig(scenario_split=scenario_split)
    env = LatentGoalOpsEnvironment(experiment_config=config)
    observation = env.reset(seed=seed, task_id=task_id)
    while observation.step_index < step_index and not observation.done:
        observation = env.step(env.sample_heuristic_action())
    if observation.done:
        raise ValueError("Requested step is beyond episode horizon.")
    assert env._episode is not None  # noqa: SLF001
    assert env._hidden_goal is not None  # noqa: SLF001
    weights = active_weights(env._hidden_goal, env._episode["step_index"])  # noqa: SLF001

    heuristic = env.sample_heuristic_action()
    random_action = env.sample_random_action()
    if TaskId(task_id) in {TaskId.TASK3, TaskId.TASK6}:
        oracle, oracle_value = solve_task3_oracle_action(env._episode, weights)  # noqa: SLF001
        heuristic_value = evaluate_task3_action_value(env._episode, heuristic, weights)  # noqa: SLF001
        random_value = evaluate_task3_action_value(env._episode, random_action, weights)  # noqa: SLF001
    elif TaskId(task_id) == TaskId.TASK7:
        oracle, oracle_value = solve_task7_oracle_action(env._episode, weights)  # noqa: SLF001
        heuristic_value = evaluate_task7_action_value(env._episode, heuristic, weights)  # noqa: SLF001
        random_value = evaluate_task7_action_value(env._episode, random_action, weights)  # noqa: SLF001
    else:
        raise ValueError("Counterfactual checks currently target task3/task6/task7.")

    return {
        "task_id": task_id,
        "seed": seed,
        "step_index": step_index,
        "scenario_split": scenario_split,
        "oracle_action": oracle.model_dump(mode="json", exclude_none=True),
        "heuristic_action": heuristic.model_dump(mode="json", exclude_none=True),
        "random_action": random_action.model_dump(mode="json", exclude_none=True),
        "oracle_value": oracle_value,
        "heuristic_value": heuristic_value,
        "random_value": random_value,
        "oracle_margin_over_heuristic": oracle_value - heuristic_value,
        "oracle_margin_over_random": oracle_value - random_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=[TaskId.TASK3.value, TaskId.TASK6.value, TaskId.TASK7.value], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--scenario-split", choices=["core", "heldout"], default="core")
    args = parser.parse_args()
    report = counterfactual_report(args.task, args.seed, args.step_index, scenario_split=args.scenario_split)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
