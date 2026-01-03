import time
import random
import argparse
import configparser
import ast

import numpy as np
import torch

from rich_argparse import RichHelpFormatter
from rich.traceback import install

install(show_locals=False)  # Rich tracebacks

import pufferlib
import pufferlib.sweep

from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.plotting import figure, show
from bokeh.palettes import Turbo256


def synthetic_basic_task(args):
    train_args = args["train"]
    learning_rate = train_args["learning_rate"]
    total_timesteps = train_args["total_timesteps"]
    score = np.exp(-((np.log10(learning_rate) + 3) ** 2))
    cost = total_timesteps / 50_000_000
    return score, cost


def synthetic_linear_task(args):
    score, cost = synthetic_basic_task(args)
    return score * cost, cost


def synthetic_log_task(args):
    score, cost = synthetic_basic_task(args)
    noise_cost = cost + 0.20 * np.random.randn() * cost
    noise_cost = min(noise_cost, 200)
    noise_cost = max(noise_cost, 1)
    return score * np.log10(noise_cost), cost


def synthetic_percentile_task(args):
    score, cost = synthetic_basic_task(args)
    noise_cost = cost - 0.20 * abs(np.random.randn()) * cost
    noise_cost = min(noise_cost, 200)
    noise_cost = max(noise_cost, 1)
    return score / (1 + np.exp(-noise_cost / 10)), cost


def synthetic_cutoff_task(args):
    score, cost = synthetic_basic_task(args)
    return score * min(2, np.log10(cost)), cost


def test_sweep(args):
    method = args["sweep"]["method"]
    if method == "Random":
        sweep = pufferlib.sweep.Random(args["sweep"])
    elif method == "ParetoGenetic":
        sweep = pufferlib.sweep.ParetoGenetic(args["sweep"])
    elif method == "Protein":
        sweep = pufferlib.sweep.Protein(
            args["sweep"],
            expansion_rate=1.0,
        )
    else:
        raise ValueError(f"Invalid sweep method {method} (random/pareto_genetic/protein)")

    task = args["task"]
    if task == "linear":
        synthetic_task = synthetic_linear_task
    elif task == "log":
        synthetic_task = synthetic_log_task
    elif task == "percentile":
        synthetic_task = synthetic_percentile_task
    else:
        raise ValueError(f"Invalid task {task}")

    target_metric = args["sweep"]["metric"]
    scores, costs = [], []
    for i in range(args["max_runs"]):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        try:
            _, info = sweep.suggest(args)
        except:
            break

        total_timesteps = args["train"]["total_timesteps"]
        for i in range(1, 6):
            args["train"]["total_timesteps"] = i * total_timesteps / 5
            score, cost = synthetic_task(args)
            sweep.observe(args, score, cost)
            print("Score:", score, "Cost:", cost)

        scores.append(score)
        costs.append(cost)

    pareto, pareto_idx = pufferlib.sweep.pareto_points(sweep.success_observations)

    np.save(args["data_path"] + ".npy", {"scores": scores, "costs": costs})

    # pareto_scores = np.array(scores)[pareto_idx].tolist()
    # pareto_costs = np.array(costs)[pareto_idx].tolist()
    # np.save(args['data_path']+'_pareto.npy', {'scores': pareto_scores, 'costs': pareto_costs})


def visualize(args):
    data = np.load(args["vis_path"] + ".npy", allow_pickle=True).item()
    costs = data["costs"]
    scores = data["scores"]

    sorted_costs = np.sort(costs)
    aoc = np.max(scores) * np.cumsum(sorted_costs) / np.sum(costs)

    # Create a ColumnDataSource that includes the 'order' for each point
    source = ColumnDataSource(
        data=dict(
            x=costs,
            y=scores,
            order=list(range(len(scores))),  # index/order for each point
        )
    )

    curve = ColumnDataSource(
        data=dict(
            x=sorted_costs,
            y=aoc,
            order=list(range(len(scores))),  # index/order for each point
        )
    )

    # Define a color mapper across the range of point indices
    mapper = LinearColorMapper(palette=Turbo256, low=0, high=len(scores))

    # Set up the figure
    p = figure(title="Synthetic Hyperparam Test", x_axis_label="Cost", y_axis_label="Score")

    # Use the 'order' field for color -> mapped by 'mapper'
    p.scatter(x="x", y="y", color={"field": "order", "transform": mapper}, size=10, source=source)

    p.line(x="x", y="y", color="purple", source=curve)

    show(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f":blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]"
        " demo options. Shows valid args for your env and policy",
        formatter_class=RichHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--task", type=str, default="linear", help="Task to optimize")
    parser.add_argument("--vis-path", type=str, default="", help="Set to visualize a saved sweep")
    parser.add_argument("--data-path", type=str, default="sweep", help="Used for testing hparam algorithms")
    parser.add_argument("--max-runs", type=int, default=100, help="Max number of sweep runs")
    parser.add_argument("--tag", type=str, default=None, help="Tag for experiment")
    parser.add_argument("--wandb", action="store_true", help="Track on WandB")
    parser.add_argument("--neptune", action="store_true", help="Track on Neptune")
    args = parser.parse_known_args()[0]

    p = configparser.ConfigParser()
    p.read("config/default.ini")
    for section in p.sections():
        for key in p[section]:
            argparse_key = f"--{section}.{key}".replace("_", "-")
            parser.add_argument(argparse_key, default=p[section][key])

    # Late add help so you get a dynamic menu based on the env
    parser.add_argument(
        "-h", "--help", default=argparse.SUPPRESS, action="help", help="Show this help message and exit"
    )

    parsed = parser.parse_args().__dict__
    args = {"env": {}, "policy": {}, "rnn": {}}
    for key, value in parsed.items():
        next = args
        for subkey in key.split("."):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:
            prev[subkey] = value

    if args["vis_path"]:
        visualize(args)
        exit(0)

    test_sweep(args)
