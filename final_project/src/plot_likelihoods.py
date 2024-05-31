import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import graph_utils
from test_likelihood import cross_validate_structure
from visualize import DPI, FIG_SIZE

test_set = pd.read_csv("my_test.csv").to_numpy()

print("calculating scores")
x = []
y = []
small_x = []
small_y = []
for adjacency_matrix_file in glob.glob("graphs/train_top_*.npy"):
    adjacency_matrix = np.load(adjacency_matrix_file)
    n_params = graph_utils.n_params(adjacency_matrix)
    log_likelihood = cross_validate_structure(adjacency_matrix, test_set, print_time=False)
    x.append(n_params)
    y.append(log_likelihood)
for adjacency_matrix_file in glob.glob("graphs/train_small_top_*.npy"):
    adjacency_matrix = np.load(adjacency_matrix_file)
    n_params = graph_utils.n_params(adjacency_matrix)
    log_likelihood = cross_validate_structure(adjacency_matrix, test_set, print_time=False)
    small_x.append(n_params)
    small_y.append(log_likelihood)
order = np.argsort(x)
x = np.array(x)[order]
y = np.array(y)[order]
small_order = np.argsort(small_x)
small_x = np.array(small_x)[small_order]
small_y = np.array(small_y)[small_order]

print("plotting")
plt.figure(figsize=FIG_SIZE)
SIMILAR_UNTIL = 12
plt.scatter(small_x[SIMILAR_UNTIL:], small_y[SIMILAR_UNTIL:], marker='o', c="orange", label="small models")
plt.plot(x, y)
plt.scatter(x, y, marker='x', label="all models")
plt.xlim(left=0, right=100)
# plt.axvline(45, c="red")
plt.xticks([0, 20, 40, 45, 60, 80, 100])
plt.xlabel("#params")
plt.ylabel("log likelihood")
plt.legend()
plt.tight_layout()
plt.savefig("plots/both_likelihoods.png", dpi=DPI)
