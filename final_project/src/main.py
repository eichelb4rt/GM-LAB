import numpy as np
import pandas as pd

import tests

N_ROTATIONS = 8


def main():
    dataset = pd.read_csv("trainset.csv").to_numpy()
    n = dataset.shape[1]
    adjacency_matrix = np.full((n, n), False)
    for i in range(n):
        for j in range(i + 1, n):
            adjacency_matrix[i, j:] = True
    print(adjacency_matrix)
    # test the adjacency matrix
    cross_log_likelihood = tests.cross_validate(adjacency_matrix, dataset, N_ROTATIONS)
    train_log_likelihood = tests.train_log_likelihood(adjacency_matrix, dataset)
    print(f"total log likelihood of tests sets is {cross_log_likelihood}")
    print(f"total log likelihood of train set is {train_log_likelihood}")


if __name__ == "__main__":
    main()
