"""CLI to generate a paper-friendly plot bundle from run logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latentgoalops.analysis.aggregate import load_run_records
from latentgoalops.analysis.plots import (
    plot_adaptation_distribution,
    plot_coherence_vs_score,
    plot_cost_vs_score,
    plot_heatmap,
    plot_oracle_gap,
    plot_parse_fallbacks,
    plot_score_distributions,
    plot_step_reward_trajectories,
    plot_task_scores,
    plot_token_usage,
)
from latentgoalops.analysis.tables import write_summary_tables


def _primary_episode_slice(episodes):
    if episodes.empty or "strict_episode" not in episodes.columns:
        return episodes, "all"
    strict = episodes[episodes["strict_episode"] == 1]
    if strict.empty:
        return episodes, "all"
    return strict, "strict_only"


def _primary_step_slice(steps):
    if steps.empty or "strict_step" not in steps.columns:
        return steps, "all"
    strict = steps[steps["strict_step"] == 1]
    if strict.empty:
        return steps, "all"
    return strict, "strict_only"


def generate_report(input_path: str, output_dir: str) -> None:
    """Generate a plot bundle from one log file or a directory tree of runs."""
    steps, episodes = load_run_records(input_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    primary_episodes, episode_slice = _primary_episode_slice(episodes)
    primary_steps, step_slice = _primary_step_slice(steps)

    plot_task_scores(primary_episodes, output_root / "task_scores_bar.png")
    plot_score_distributions(primary_episodes, output_root / "score_distributions_box.png")
    plot_heatmap(primary_episodes, output_root / "score_heatmap.png")
    plot_cost_vs_score(primary_episodes, output_root / "cost_vs_score.png")
    plot_step_reward_trajectories(primary_steps, output_root / "reward_trajectories.png")
    plot_token_usage(primary_episodes, output_root / "token_usage.png")
    plot_parse_fallbacks(episodes, output_root / "parse_fallbacks.png")
    plot_adaptation_distribution(primary_episodes, output_root / "adaptation_distribution.png")
    plot_coherence_vs_score(primary_episodes, output_root / "coherence_vs_score.png")
    plot_oracle_gap(primary_episodes, output_root / "oracle_gap.png")
    write_summary_tables(primary_episodes, output_root)
    (output_root / "report_metadata.json").write_text(
        json.dumps(
            {
                "episode_slice": episode_slice,
                "step_slice": step_slice,
                "total_episode_rows": int(len(episodes)),
                "primary_episode_rows": int(len(primary_episodes)),
                "total_step_rows": int(len(steps)),
                "primary_step_rows": int(len(primary_steps)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs", help="A runs.jsonl file or a directory containing many run outputs.")
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()
    generate_report(args.input, args.output_dir)
    print(f"Generated plots under {args.output_dir}")


if __name__ == "__main__":
    main()
