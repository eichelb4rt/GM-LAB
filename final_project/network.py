import numpy as np
import pandas as pd
from typing import Self


class GaussianBayesNet:
    def __init__(self, adjacency_matrix: np.ndarray, parameters: dict[int, tuple[float, np.ndarray, float]] = None):
        """Bayesian network with Gaussians as conditional probabilities.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            Matrix defining parents of each node. A[i,j] == 1 <=> there is an edge from i to j (i is a parent of j).
        parameters : dict[int, tuple[float, np.ndarray, float]]
            Gaussian parameters for the conditional probabilities of the nodes given their parents. Stored as {node: (beta_0, beta, sigma)}.
        """

        self.adjacency_matrix = adjacency_matrix
        self.network_parameters = parameters
        self.n = adjacency_matrix.shape[0]
        for i in range(self.n):
            assert adjacency_matrix[i, i] != 1, f"A node cannot be the parent of itself (node: {i})."
        # TODO: guarantee DAG (no circles)

    def fit(self, dataset: np.ndarray, lambda_reg: float = 0) -> Self:
        """Fits the parameters to the data, given the adjacency matrix.

        Parameters
        ----------
        dataset : np.ndarray
            m x n matrix, where m is the number of samples and n is the number of features.
        lambda_reg : float
            Regularization constant for the linear regression that is done in order to estimate beta and beta_0.
        """

        self.network_parameters = {}
        for i in range(self.n):
            # get the parents
            parents = self.parents(i)
            # slice the data
            x = dataset[:, parents]
            y = dataset[:, i]
            # do a linear regression and save the parameters
            beta_0, beta, sigma = linear_regression(x, y, lambda_reg)
            self.network_parameters[i] = (beta_0, beta, sigma)
        return self

    def parents(self, node: int) -> list[int]:
        return list(np.argwhere(self.adjacency_matrix[:, node] == 1)[:, 0])

    def log_likelihood(self, dataset: np.ndarray) -> float:
        """Log likelihood of observing the given data.

        Parameters
        ----------
        dataset : np.ndarray
            m x n matrix, where m is the number of samples and n is the number of features.

        Returns
        -------
        float
            Log likelihood of observing the data.
        """

        log_likelihood = 0
        for i in range(self.n):
            # get the parents
            parents = self.parents(i)
            # slice the data
            x = dataset[:, parents]
            y = dataset[:, i]
            # get the network parameters
            beta_0, beta, sigma = self.network_parameters[i]
            mu = beta_0 + x @ beta
            log_likelihood += np.sum(-0.5 * ((x - mu) / sigma)**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi))
        return log_likelihood


def linear_regression(x: np.ndarray, y: np.ndarray, lambda_reg: float = 0) -> tuple[float, np.ndarray, float]:
    """Linear regression with regularization.

    Parameters
    ----------
    x : np.ndarray
        m x n matrix, where m is the number of samples and n is the number of features.
    y : np.ndarray
        m long vector, where m is the number of samples.
    lambda_reg : float, optional
        Regularization constant for the linear regression, by default 0.

    Returns
    -------
    tuple[float, np.ndarray, float]
        Maximum likelihood parameters: offset, weights; and the resulting std deviation.
    """

    m = x.shape[0]
    n = x.shape[1]
    x_padded: np.ndarray = np.c_[np.ones(m), x]
    weights = np.linalg.solve(x_padded.T @ x_padded + lambda_reg * np.eye(n + 1), x_padded.T @ y)
    sigma = np.linalg.norm(x_padded @ weights - y) / np.sqrt(m)
    return weights[0], weights[1:], sigma


def main():
    dataset = pd.read_csv("trainset.csv").to_numpy()
    n = dataset.shape[1]
    adjacency_matrix = np.zeros((n, n))
    gbn = GaussianBayesNet(adjacency_matrix).fit(dataset)
    print(gbn.network_parameters)
    print(gbn.log_likelihood(dataset))


if __name__ == "__main__":
    main()
