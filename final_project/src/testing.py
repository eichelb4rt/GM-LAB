import numpy as np
import numpy.typing as npt
from queue import Queue
import glob
import pandas as pd
import matplotlib.pyplot as plt

import clock
import graphs
from structure import GreedySearcher
from visualize import visualize_reg_n_params, FIG_SIZE, DPI
from test_likelihood import cross_validate_structure, print_report

# def hash_adj(adjacency_matrix: npt.NDArray[np.bool_]) -> int:
#     """Hashable byte representation of the adjacency matrix."""

#     return int.from_bytes(bytes(np.packbits(adjacency_matrix)), "little")


n = 12
# full_adjacency_matrix = np.full((n, n), False)
# for i in range(n):
#     for j in range(i + 1, n):
#         full_adjacency_matrix[i, j:] = True
empty_adjacency_matrix = np.full((n, n), False)
# graphs.save(full_adjacency_matrix, name="full")
# graphs.save(empty_adjacency_matrix, name="empty")


# taboo_list: Queue[int] = Queue()
# taboo_list.put(hash_adj(full_adjacency_matrix))
# current_taboo_states: set[int] = set(taboo_list.queue)

# print(hash_adj(full_adjacency_matrix) in current_taboo_states)
# print(hash_adj(empty_adjacency_matrix) in current_taboo_states)
# print(hash_adj(full_adjacency_matrix))

# print(-3078.8979333084617 + -4827.041815833941)
# print(-3431.50537109375 + -4474.4345703125)

# np.save("test.npy", np.array(([1, 2, 3, 4], [0, 5]), dtype=object))
# print(np.array(([1, 2, 3, 4], [0, 5]), dtype=object))
# a, b = np.load("test.npy", allow_pickle=True)
# print(a)
# print(b)

small_lambdas = [25, 30, 50, 60, 80, 100, 120, 150, 194, 240, 280, 350, 360, 370]
medium_lambdas = [4.5, 5, 5.2, 5.5, 6, 8, 10, 15, 21, 22, 23.15, 23.1]
big_lambdas = [0, 0.2, 0.5, 1, 2, 3]
train_set = pd.read_csv("my_train.csv").to_numpy()
test_set = pd.read_csv("my_test.csv").to_numpy()
for lambda_reg in small_lambdas + medium_lambdas + big_lambdas:
    if lambda_reg in small_lambdas:
        detective = GreedySearcher(empty_adjacency_matrix,
                                   regularization_constant=lambda_reg,

                                   n_tabu_walks=3,
                                   max_tabu_list_size=100,
                                   tabu_walk_length=20,

                                   n_random_restarts=5,
                                   random_walk_length=5)
    elif lambda_reg in medium_lambdas:
        detective = GreedySearcher(empty_adjacency_matrix,
                                   regularization_constant=lambda_reg,

                                   n_tabu_walks=3,
                                   max_tabu_list_size=400,
                                   tabu_walk_length=80,

                                   n_random_restarts=5,
                                   random_walk_length=5)
    elif lambda_reg in big_lambdas:
        detective = GreedySearcher(empty_adjacency_matrix,
                                   regularization_constant=lambda_reg,

                                   n_tabu_walks=3,
                                   max_tabu_list_size=2000,
                                   tabu_walk_length=150,

                                   n_random_restarts=5,
                                   random_walk_length=10)
    clock.start("hill climb")
    print("climbing...")
    top_adjacency_matrix = detective.fit(train_set)
    clock.stop("hill climb")
    clock.total("hill climb")
    print_report(top_adjacency_matrix, test_set)
    graphs.save(top_adjacency_matrix, name=f"train_top_{graphs.n_params(top_adjacency_matrix)}")


print("calculating scores")
x = []
y = []
for adjacency_matrix_file in glob.glob("graphs/train_top_*.npy"):
    adjacency_matrix = np.load(adjacency_matrix_file)
    n_params = graphs.n_params(adjacency_matrix)
    log_likelihood = cross_validate_structure(adjacency_matrix, train_set, print_time=False)
    x.append(n_params)
    y.append(log_likelihood)
order = np.argsort(x)
x = np.array(x)[order]
y = np.array(y)[order]
print("plotting")
plt.figure(figsize=FIG_SIZE)
plt.plot(x, y)
plt.scatter(x, y, marker='x')
plt.xlabel("#params")
plt.ylabel("log likelihood")
plt.savefig("plots/likelihood.png", dpi=DPI)
