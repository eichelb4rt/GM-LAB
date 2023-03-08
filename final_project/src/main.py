import numpy as np
import pandas as pd

import graphs


def main():
    dataset = pd.read_csv("trainset.csv").to_numpy()
    n = dataset.shape[1]

    adjacency_matrix = np.full((n, n), False)
    for i in range(n):
        for j in range(i + 1, n):
            adjacency_matrix[i, j:] = True
    graphs.save(adjacency_matrix, name="full")

    adjacency_matrix = np.full((n, n), False)
    graphs.save(adjacency_matrix, name="empty")


if __name__ == "__main__":
    main()
