import numpy as np
import pandas as pd

import clock
import graphs
from structure import GreedySearcher

n = 12
empty_adjacency_matrix = np.full((n, n), False)

small_lambdas = [25, 30, 50, 60, 80, 100, 120, 150, 194, 240, 280, 350, 360, 370]
medium_lambdas = [4.5, 5, 5.2, 5.5, 6, 8, 10, 15, 21, 22, 23.15, 23.1]
big_lambdas = [0, 0.2, 0.5, 1, 2, 3]
train_set = pd.read_csv("my_train.csv").to_numpy()
times = []
print("starting runs")
for lambda_reg in small_lambdas + medium_lambdas + big_lambdas:
    detective = GreedySearcher(empty_adjacency_matrix,
                                regularization_constant=lambda_reg,

                                n_tabu_walks=3,
                                max_tabu_list_size=100,
                                tabu_walk_length=20,

                                n_random_restarts=5,
                                random_walk_length=5)
    clock.start("greedy search")
    top_adjacency_matrix = detective.fit(train_set)
    clock.stop("greedy search")
    out_file = f"train_small_top_{graphs.n_params(top_adjacency_matrix)}"
    time_needed = clock.total("hill climb")
    print(f"done: lambda={lambda_reg}\ntime needed: {round(time_needed, 2)}s\noutput: {out_file}")
    times.append(time_needed)
    clock.clear_all()
    graphs.save(top_adjacency_matrix, name=out_file)

print(times)
np.save("small_times.npy", np.array(times))
