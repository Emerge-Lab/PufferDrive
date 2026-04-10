from __future__ import annotations

import os
import tempfile

cache_dir = os.path.join(tempfile.gettempdir(), "mfpbt-matplotlib-cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", cache_dir)
os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


POPULATION_PALETTE = [
    "#0B8F72",
    "#1F78B4",
    "#E3B505",
    "#D1495B",
    "#2A9D8F",
    "#4F86C6",
    "#F2C14E",
    "#E76F51",
]


def _population_color(population_id: int) -> str:
    return POPULATION_PALETTE[population_id % len(POPULATION_PALETTE)]


def _read_logs(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs_dir = os.path.join(run_dir, "logs")
    agent_history_path = os.path.join(logs_dir, "agent_history.csv")
    round_summary_path = os.path.join(logs_dir, "round_summary.csv")

    if not os.path.exists(agent_history_path):
        raise FileNotFoundError(f"Missing agent history csv: {agent_history_path}")
    if not os.path.exists(round_summary_path):
        raise FileNotFoundError(f"Missing round summary csv: {round_summary_path}")

    return pd.read_csv(agent_history_path), pd.read_csv(round_summary_path)


def _resolve_step_series(round_summary: pd.DataFrame) -> pd.Series:
    if "max_env_steps" in round_summary.columns:
        return pd.to_numeric(round_summary["max_env_steps"], errors="coerce")
    if "mean_env_steps" in round_summary.columns:
        return pd.to_numeric(round_summary["mean_env_steps"], errors="coerce")
    if "round_index" in round_summary.columns:
        return pd.to_numeric(round_summary["round_index"], errors="coerce") + 1
    raise KeyError("Could not resolve a step axis from round_summary.csv")


def _save_best_score_plot(round_summary: pd.DataFrame, output_dir: str) -> str:
    steps = _resolve_step_series(round_summary)
    scores = pd.to_numeric(round_summary["best_selection_score"], errors="coerce")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(steps, scores, color="#1F78B4", linewidth=2.5)
    ax.scatter(steps, scores, color="#0B8F72", s=18, alpha=0.9)
    ax.set_title("Best Score vs Steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Best score")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = os.path.join(output_dir, "best_score_vs_steps.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_hyperparameter_distribution_plot(
    agent_history: pd.DataFrame,
    output_dir: str,
    hyperparameter_name: str,
) -> str:
    if hyperparameter_name not in agent_history.columns:
        raise KeyError(f"Hyperparameter '{hyperparameter_name}' not found in agent_history.csv")

    steps = pd.to_numeric(agent_history["env_steps"], errors="coerce")
    values = pd.to_numeric(agent_history[hyperparameter_name], errors="coerce")
    populations = pd.to_numeric(agent_history["population_id"], errors="coerce").astype("Int64")

    plot_df = pd.DataFrame(
        {
            "env_steps": steps,
            hyperparameter_name: values,
            "population_id": populations,
        }
    ).dropna()
    plot_df = plot_df[plot_df[hyperparameter_name] > 0]

    fig, ax = plt.subplots(figsize=(11, 6))
    for population_id in sorted(plot_df["population_id"].unique()):
        population_rows = plot_df[plot_df["population_id"] == population_id]
        ax.scatter(
            population_rows["env_steps"],
            population_rows[hyperparameter_name],
            label=f"Population {int(population_id)}",
            color=_population_color(int(population_id)),
            s=24,
            alpha=0.7,
            edgecolors="none",
        )

    ax.set_yscale("log")
    ax.set_title(f"{hyperparameter_name} distribution vs Steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel(hyperparameter_name)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = os.path.join(output_dir, f"{hyperparameter_name}_distribution_vs_steps.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_analysis_plots(run_dir: str, hyperparameter_name: str = "learning_rate") -> list[str]:
    agent_history, round_summary = _read_logs(run_dir)
    output_dir = os.path.join(run_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    outputs = [
        _save_best_score_plot(round_summary, output_dir),
        _save_hyperparameter_distribution_plot(agent_history, output_dir, hyperparameter_name),
    ]
    return outputs
