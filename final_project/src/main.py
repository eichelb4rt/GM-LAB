import numpy as np
import pandas as pd

import tests

N_ROTATIONS = 8


def main():
    dataset = pd.read_csv("trainset.csv").to_numpy()
    n = dataset.shape[1]
    adjacency_matrix = np.full((n, n), False)
    log_likelihood = tests.cross_validate(adjacency_matrix, dataset, N_ROTATIONS)
    print(f"total log likelihood is {log_likelihood}")


if __name__ == "__main__":
    main()
