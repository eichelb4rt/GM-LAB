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

    # 66: 0
    # 58: 0.2
    # 54: 0.5
    # 52: 1
    # 48: 2
    # 43: 3
    # 37: 5
    # 31: 8
    # 30: 10
    # 25: 15
    # 19: 25
    # 16: 30
    # 14: 50
    # 12: 60
    # 11: 80
    # 10: 100
    # 9: 120
    # 8: 150
    # 6: 194
    # 5: 240
    # 4: 280
    # 3: 350
    # 2: 360
    # 1: 370
    detective = GreedySearcher(empty_adjacency_matrix,
                               regularization_constant=0,
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
    graphs.save(top_adjacency_matrix, name=f"top_{graphs.n_edges(top_adjacency_matrix)}")

    # log_likelihood = cross_validate_detective(detective, dataset)
    # print(f"log likelihood: {log_likelihood}")

    # test_arr = np.array([])
    # test_arr = np.insert(test_arr, 0, 1)
    # print(test_arr)


if __name__ == "__main__":
    main()
