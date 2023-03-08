import argparse
import numpy as np
from pathlib import Path
import numpy.typing as npt
import matplotlib.pyplot as plt

import graphs


def visualize_graph(adjacency_matrix: npt.NDArray[np.bool_], labels: list[str] = None, out_file: str = "dot_encodings/graph.dot"):
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
    parser = argparse.ArgumentParser(description="Reads an adjacency matrix from a .npy file and saves the corresponding graph as a dot encoding.")
    parser.add_argument("filename",
                        help="File with the encoded adjacency matrix.")

    args = parser.parse_args()
    adjacency_matrix: npt.NDArray[np.bool_] = np.load(args.filename)
    out_file = f"dot_encodings/{Path(args.filename).stem}.dot"
    # if used from the terminal, these labels are assumed
    LABELS = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    visualize_graph(adjacency_matrix, LABELS, out_file)


if __name__ == "__main__":
    main()
