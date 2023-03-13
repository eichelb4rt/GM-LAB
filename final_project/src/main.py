import numpy as np
import pandas as pd

import graphs
from structure import GreedySearcher
import clock
from test_likelihood import print_report, cross_validate_detective
from visualize import visualize_score_history, visualize_tabu_walk


def main():
    dataset = pd.read_csv("trainset.csv").to_numpy()
    n = dataset.shape[1]
    empty_adjacency_matrix = np.full((n, n), False)

    # NOTE: turn off logging for a small performance boost
    # NOTE: small number of parameters (43)
    # detective = GreedySearcher(empty_adjacency_matrix,
    #                            regularization_constant=25,

    #                            n_tabu_walks=3,
    #                            max_tabu_list_size=10,
    #                            tabu_walk_length=5,

    #                            n_random_restarts=5,
    #                            random_walk_length=5,

    #                            logging_enabled=True)
    # NOTE: small 2
    # detective = GreedySearcher(empty_adjacency_matrix,
    #                            regularization_constant=25,

    #                            n_tabu_walks=3,
    #                            max_tabu_list_size=100,
    #                            tabu_walk_length=20,

    #                            n_random_restarts=5,
    #                            random_walk_length=5,

    #                            logging_enabled=True)
    # NOTE: medium number of parameters (54)
    # detective = GreedySearcher(empty_adjacency_matrix,
    #                            regularization_constant=8,

    #                            n_tabu_walks=3,
    #                            max_tabu_list_size=150,
    #                            tabu_walk_length=30,

    #                            n_random_restarts=5,
    #                            random_walk_length=5,

    #                            logging_enabled=True)
    # NOTE: medium 2
    # detective = GreedySearcher(empty_adjacency_matrix,
    #                            regularization_constant=8,

    #                            n_tabu_walks=3,
    #                            max_tabu_list_size=400,
    #                            tabu_walk_length=80,

    #                            n_random_restarts=5,
    #                            random_walk_length=5,

    #                            logging_enabled=True)
    # NOTE: big number of parameters (67)
    # detective = GreedySearcher(empty_adjacency_matrix,
    #                            regularization_constant=3,

    #                            n_tabu_walks=3,
    #                            max_tabu_list_size=300,
    #                            tabu_walk_length=70,

    #                            n_random_restarts=5,
    #                            random_walk_length=10,

    #                            logging_enabled=True)
    # NOTE: big 2
    detective = GreedySearcher(empty_adjacency_matrix,
                               regularization_constant=3,

                               n_tabu_walks=3,
                               max_tabu_list_size=2000,
                               tabu_walk_length=150,

                               n_random_restarts=5,
                               random_walk_length=10,

                               logging_enabled=True)
    clock.start("hill climb")
    print("climbing...")
    top_adjacency_matrix = detective.fit(dataset)
    clock.stop("hill climb")
    clock.total("hill climb")
    print_report(top_adjacency_matrix, dataset)
    graphs.save(top_adjacency_matrix, name=f"top_{graphs.n_params(top_adjacency_matrix)}")
    np.save("logs/big_2a.npy", np.array((detective.score_history, detective.method_starts), dtype=object))

    # NOTE: uncomment to cross validate stuff
    # log_likelihood = cross_validate_detective(detective, dataset)
    # print(f"log likelihood: {log_likelihood}")


if __name__ == "__main__":
    main()
