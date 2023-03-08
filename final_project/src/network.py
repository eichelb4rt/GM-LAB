import numpy as np
import numpy.typing as npt
from typing import Self

import graphs


class GaussianBayesNet:
    def __init__(self, adjacency_matrix: npt.NDArray[np.bool_], parameters: dict[int, tuple[npt.NDArray[np.float32], float]] = None):
        """Bayesian network with Gaussians as conditional probabilities.

        Parameters
        ----------
        adjacency_matrix : npt.NDArray[np.bool_]
            Matrix defining parents of each node. A[i,j] == True <=> there is an edge from i to j (i is a parent of j).
        parameters : dict[int, tuple[npt.NDArray[np.float32], float]]
            Gaussian parameters for the conditional probabilities of the nodes given their parents. Stored as {node: (beta, sigma)} where beta[0] is the offset and beta[1:] are the linear coefficients.
        """

        self.adjacency_matrix = adjacency_matrix
        self.network_parameters = parameters
        self.n = graphs.n_nodes(adjacency_matrix)
        for i in range(self.n):
            # i know (...) == False can be written as not (...), but i think this is more readable because i'm reading the contents of the adjacency matrix
            assert adjacency_matrix[i, i] == False, f"A node cannot be the parent of itself (node: {i})."
        assert not graphs.has_cycle(adjacency_matrix), "Adjacency matrix has to represent a DAG (a cycle was found)."

    def fit(self, dataset: npt.NDArray[np.float32], lambda_reg: float = 0) -> Self:
        """Fits the parameters to the data, given the adjacency matrix.

        Parameters
        ----------
        dataset : npt.NDArray[np.float32]
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
            beta, sigma = linear_regression(x, y, lambda_reg)
            self.network_parameters[i] = (beta, sigma)
        return self

    def parents(self, node: int) -> list[int]:
        return graphs.neighbours_in(node, self.adjacency_matrix)

    def log_likelihood(self, dataset: npt.NDArray[np.float32]) -> float:
        """Log likelihood of observing the given data.

        Parameters
        ----------
        dataset : npt.NDArray[np.float32]
            m x n matrix, where m is the number of samples and n is the number of features.

        Returns
        -------
        float
            Log likelihood of observing the data.
        """

        m = dataset.shape[0]
        log_likelihood = 0
        for i in range(self.n):
            # get the parents
            parents = self.parents(i)
            # slice the data
            x = dataset[:, parents]
            y = dataset[:, i]
            # get the network parameters
            beta, sigma = self.network_parameters[i]
            # calculate the log likelihood for this node
            x_padded: npt.NDArray[np.float32] = np.c_[np.ones(m), x]
            mu = x_padded @ beta
            log_likelihood += np.sum(-0.5 * ((y - mu) / sigma)**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi))
        return log_likelihood


def linear_regression(x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], lambda_reg: float = 0) -> tuple[npt.NDArray[np.float32], float]:
    """Linear regression with regularization.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        m x n matrix, where m is the number of samples and n is the number of features.
    y : npt.NDArray[np.float32]
        m long vector, where m is the number of samples.
    lambda_reg : float, optional
        Regularization constant for the linear regression, by default 0.

    Returns
    -------
    tuple[npt.NDArray[np.float32], float]
        Maximum likelihood parameters (offset, weights) and the resulting std deviation.
    """

    m = x.shape[0]
    n = x.shape[1]
    x_padded: npt.NDArray[np.float32] = np.c_[np.ones(m), x]
    weights = np.linalg.solve(x_padded.T @ x_padded + lambda_reg * np.eye(n + 1), x_padded.T @ y)
    sigma = np.linalg.norm(x_padded @ weights - y) / np.sqrt(m)
    return weights, sigma
