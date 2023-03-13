import numpy as np
import pandas as pd

import graphs
from structure import GreedySearcher
import clock
from test_likelihood import print_report, cross_validate_detective


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

    # 90: 0
    # 82: 0.2
    # 78: 0.5
    # 76: 1
    # 72: 2
    # 67: 3
    # 63: 4.5
    # 62: 5
    # 61: 5
    # 60: 5.2
    # 58: 5.5
    # 57: 6
    # 55: 8
    # 54: 10
    # 49: 15
    # 48: 21
    # 47: 22
    # 46: 23.15 (??? lol)
    # 44: 23.1
    # 43: 25
    # 40: 30
    # 38: 50
    # 36: 60
    # 35: 80
    # 34: 100
    # 33: 120
    # 32: 150
    # 30: 194
    # 29: 240
    # 28: 280
    # 27: 350
    # 26: 360
    # 25: 370
    detective = GreedySearcher(empty_adjacency_matrix,
                               regularization_constant=4.5,
                               n_tabu_walks=5,
                               max_tabu_list_size=10,
                               tabu_walk_length=5,
                               n_random_restarts=5,
                               random_walk_length=3)
    clock.start("hill climb")
    top_adjacency_matrix = detective.fit(dataset)
    clock.stop("hill climb")
    clock.total("hill climb")
    print_report(top_adjacency_matrix, dataset)
    graphs.save(top_adjacency_matrix, name=f"top_{graphs.n_params(top_adjacency_matrix)}")

    # log_likelihood = cross_validate_detective(detective, dataset)
    # print(f"log likelihood: {log_likelihood}")

    # test_arr = np.array([])
    # test_arr = np.insert(test_arr, 0, 1)
    # print(test_arr)


if __name__ == "__main__":
    main()
