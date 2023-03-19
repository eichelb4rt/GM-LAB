import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

import graphs
from visualize import FIG_SIZE, DPI
from test_likelihood import cross_validate_structure

small_lambdas = [25, 30, 50, 60, 80, 100, 120, 150, 194, 240, 280, 350, 360, 370]
medium_lambdas = [4.5, 5, 5.2, 5.5, 6, 8, 10, 15, 21, 22, 23.15, 23.1]
big_lambdas = [0, 0.2, 0.5, 1, 2, 3]
times = np.load("results/times.npy")
small_times = np.load("results/small_times.npy")

print("calculating scores")
lambdas = small_lambdas + medium_lambdas + big_lambdas
order = np.argsort(lambdas)
lambdas = np.array(lambdas)[order]
times = np.array(times)[order]
small_times = np.array(small_times)[order]

print("plotting")
plt.figure(figsize=FIG_SIZE)
plt.plot(lambdas, small_times, c="orange")
plt.scatter(lambdas, small_times, marker='o', c="orange", label="small models")
plt.plot(lambdas, times)
plt.scatter(lambdas, times, marker='x', label="all models")
plt.axvline(4, c="red")
plt.axvline(24, c="red")
plt.text(1, 100, "big")
plt.text(5, 100, "medium")
plt.text(70, 100, "small")
plt.xlim(left=-0.05)
plt.xscale("symlog")
plt.xlabel("$\lambda$")
plt.ylabel("time in seconds")
plt.legend()
plt.tight_layout()
plt.savefig("plots/both_times.png", dpi=DPI)
