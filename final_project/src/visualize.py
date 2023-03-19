import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import numpy.typing as npt
import matplotlib.pyplot as plt

import graphs


FIG_SIZE = (4, 4)
DPI = 200

METHOD_COLORS = {
    "hill_climb": "blue",
    "tabu_walk": "yellow",
    "random_restart": "red"
}


def visualize_reg_n_params(n_params_per_reg: npt.NDArray[np.float32], out_file: str = "plots/n_params.png"):
    plt.figure(figsize=FIG_SIZE)
    plt.plot(n_params_per_reg[:, 0], n_params_per_reg[:, 1])
    plt.scatter(n_params_per_reg[:, 0], n_params_per_reg[:, 1], marker="x")
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xscale("symlog")
    # plt.title("influence of $\lambda$")
    plt.xlabel("$\lambda$")
    plt.ylabel("#params")
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI)

def visualize_times(in_file: str, out_file: str):
    small_lambdas = [25, 30, 50, 60, 80, 100, 120, 150, 194, 240, 280, 350, 360, 370]
    medium_lambdas = [4.5, 5, 5.2, 5.5, 6, 8, 10, 15, 21, 22, 23.15, 23.1]
    big_lambdas = [0, 0.2, 0.5, 1, 2, 3]
    times = np.load(in_file)

    print("calculating scores")
    x = small_lambdas + medium_lambdas + big_lambdas
    y = times
    order = np.argsort(x)
    x = np.array(x)[order]
    y = np.array(y)[order]
    print("plotting")
    plt.figure(figsize=FIG_SIZE)
    plt.plot(x, y)
    plt.scatter(x, y, marker='x')
    plt.axvline(4, c="red")
    plt.axvline(24, c="red")
    plt.text(1, 100, "big")
    plt.text(5, 100, "medium")
    plt.text(70, 100, "small")
    plt.xlim(left=-0.05)
    plt.xscale("symlog")
    plt.xlabel("$\lambda$")
    plt.ylabel("time in seconds")
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI)


def visualize_score_history(score_history: list[float], method_starts: list[tuple[str, int]], out_file: str = "plots/scores.png"):
    plt.figure(figsize=FIG_SIZE)
    plt.plot(score_history)
    time_intervals = [(method_name, method_start, method_starts[i + 1][1] - 1 if i + 1 < len(method_starts) else len(score_history) - 1) for i, (method_name, method_start) in enumerate(method_starts)]
    already_labelled = {
        "hill_climb": False,
        "tabu_walk": False,
        "random_restart": False
    }
    for method_name, method_start, method_end in time_intervals:
        # maybe don't colour hill climb
        if method_name == "hill_climb":
            continue
        plt.axvspan(method_start, method_end,
                    # only display the label once
                    label=method_name if not already_labelled[method_name] else f"_{method_name}",
                    color=METHOD_COLORS[method_name],
                    alpha=0.3)
        already_labelled[method_name] = True
    # plt.title("the whole search")
    plt.xlabel("iteration")
    plt.ylabel("objective function")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI)
    plt.cla()


def visualize_tabu_walk(score_history: list[float], method_starts: list[tuple[str, int]], out_file: str = "plots/tabu_walk.png"):
    first_hill_climb_length = method_starts[1][1]

    time_intervals = [(method_name, method_start, method_starts[i + 1][1] - 1 if i + 1 < len(method_starts) else len(score_history) - 1) for i, (method_name, method_start) in enumerate(method_starts)]
    all_tabu_walk_values = [score for method_name, method_start, method_end in time_intervals if method_name == "tabu_walk" for score in score_history[method_start:method_end]]
    plt.figure(figsize=FIG_SIZE)
    plt.plot(range(first_hill_climb_length - 1, len(score_history)), score_history[first_hill_climb_length - 1:])
    bottom = min(all_tabu_walk_values)
    top = max(all_tabu_walk_values)
    extra_space = 0.25 * (top - bottom)
    plt.ylim(bottom - extra_space, top + extra_space)
    already_labelled = {
        "hill_climb": False,
        "tabu_walk": False,
        "random_restart": False
    }
    for method_name, method_start, method_end in time_intervals:
        # maybe don't colour hill climb
        if method_name == "hill_climb":
            continue
        plt.axvspan(method_start, method_end,
                    # only display the label once
                    label=method_name if not already_labelled[method_name] else f"_{method_name}",
                    color=METHOD_COLORS[method_name],
                    alpha=0.3)
        already_labelled[method_name] = True
    # plt.title("the tabu walks")
    plt.xlabel("iteration")
    plt.ylabel("objective function")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI)
    plt.cla()


def visualize_adjacency_matrix(adjacency_matrix: npt.NDArray[np.bool_], labels: list[str] = None, out_file: str = "dot_encodings/graph.dot"):
    n_nodes = graphs.n_nodes(adjacency_matrix)
    if labels is None:
        labels = [str(i) for i in range(n_nodes)]
    else:
        assert len(labels) == n_nodes, f"Number of labels has to be the same as the number of nodes ({len(labels)} labels vs {n_nodes} nodes)."
    dot_str = "strict digraph {\n"
    # add all nodes
    for node in range(n_nodes):
        dot_str += f"\t{node} [label=\"{labels[node]}\"]\n"
    # add all edges
    for from_node, to_node in graphs.all_edges(adjacency_matrix):
        dot_str += f"\t{from_node} -> {to_node}\n"
    # close encoding
    dot_str += "}"
    with open(out_file, 'w') as f:
        f.write(dot_str)


def main():
    parser = argparse.ArgumentParser(description="Reads a file with context and visualizes the data.")
    parser.add_argument("filename",
                        help="File with the encoded adjacency matrix, logs, or parameter data.")

    args = parser.parse_args()
    dirname = os.path.dirname(args.filename)
    if dirname == "graphs":
        # assuming they are graphs
        print("visualizing adjacency matrix")
        adjacency_matrix: npt.NDArray[np.bool_] = np.load(args.filename)
        out_file = f"dot_encodings/{Path(args.filename).stem}.dot"
        # if used from the terminal, these labels are assumed
        LABELS = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
        visualize_adjacency_matrix(adjacency_matrix, LABELS, out_file)
        print(f"output in {out_file}")
    elif dirname == "logs":
        # assuming they are logs
        print("visualizing log")
        score_history, method_starts = np.load(args.filename, allow_pickle=True)
        stem = Path(args.filename).stem
        out_scores = f"plots/scores_{stem}.png"
        out_tabu = f"plots/tabu_{stem}.png"
        visualize_score_history(score_history, method_starts, out_file=out_scores)
        visualize_tabu_walk(score_history, method_starts, out_file=out_tabu)
        print(f"output in {out_scores} and {out_tabu}")
    elif dirname == "results":
        stem = Path(args.filename).stem
        # assuming n_params.csv
        if stem == "n_params":
            print("visualizing params")
            n_params_per_reg = pd.read_csv(args.filename).to_numpy()
            out_file = "plots/n_params.png"
            visualize_reg_n_params(n_params_per_reg, out_file)
        elif "times" in stem:
            print("visualizing times")
            out_file = f"plots/{stem}.png"
            visualize_times(args.filename, out_file)
        print(f"output in {out_file}")
    else:
        print(f"Unknown dir: {dirname}")


if __name__ == "__main__":
    main()
