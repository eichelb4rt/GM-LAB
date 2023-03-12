import numpy as np
import pandas as pd

import graphs
import structure
import clock
from test_likelihood import print_report


def main():
    dataset = pd.read_csv("trainset.csv").to_numpy()
    n = dataset.shape[1]

    full_adjacency_matrix = np.full((n, n), False)
    for i in range(n):
        for j in range(i + 1, n):
            full_adjacency_matrix[i, j:] = True
    graphs.save(full_adjacency_matrix, name="full")

    empty_adjacency_matrix = np.full((n, n), False)
    graphs.save(empty_adjacency_matrix, name="empty")

    clock.start("hill climb")
    top_adjacency_matrix, node_scores = structure.hill_climb(empty_adjacency_matrix, dataset, regularization_constant=5)
    clock.stop("hill climb")
    clock.total("hill climb")
    print_report(top_adjacency_matrix, dataset)
    print(f"scores: {np.sum(node_scores)}")
    graphs.save(top_adjacency_matrix, name="top")
    print(top_adjacency_matrix)

    # test_arr = np.array([])
    # test_arr = np.insert(test_arr, 0, 1)
    # print(test_arr)


if __name__ == "__main__":
    main()
