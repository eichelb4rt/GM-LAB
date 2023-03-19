import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

import graphs
from visualize import FIG_SIZE, DPI
from test_likelihood import cross_validate_structure


test_set = pd.read_csv("my_test.csv").to_numpy()

print("calculating scores")
x = []
y = []
for adjacency_matrix_file in glob.glob("graphs/train_top_*.npy"):
    adjacency_matrix = np.load(adjacency_matrix_file)
    n_params = graphs.n_params(adjacency_matrix)
    log_likelihood = cross_validate_structure(adjacency_matrix, test_set, print_time=False)
    x.append(n_params)
    y.append(log_likelihood)
order = np.argsort(x)
x = np.array(x)[order]
y = np.array(y)[order]

print("plotting")
plt.figure(figsize=FIG_SIZE)
plt.plot(x, y)
plt.scatter(x, y, marker='x')
plt.xlim(left=0, right=100)
plt.axvline(45, c="red")
plt.xticks([0, 20, 40, 45, 60, 80, 100])
plt.xlabel("#params")
plt.ylabel("log likelihood")
plt.tight_layout()
plt.savefig("plots/likelihood.png", dpi=DPI)
