import numpy as np
import numpy.typing as npt
from typing import Self

import graphs


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


def log_likelihood_linear_regression(x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], beta: npt.NDArray[np.float32], sigma: float) -> float:
    x_padded: npt.NDArray[np.float32] = np.c_[np.ones(x.shape[0]), x]
    mu = x_padded @ beta
    return np.sum(-0.5 * ((y - mu) / sigma)**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi))


class MultivariateGaussian:
    def __init__(self, mu: npt.NDArray[np.float32], sigma: npt.NDArray[np.float32]) -> None:
        self.mu = mu
        self.sigma = sigma
        self.dimension = mu.shape[0]

    def sample(self, n_samples: int) -> npt.NDArray[np.float32]:
        """Returns a n_samples x dimension matrix of samples that follow this multivariate gaussian distribution."""

        L = np.linalg.cholesky(self.sigma)
        # generate a dim x n_samples matrix of samples (n_samples samples that follow a d-dimensional standard gaussian)
        standard_normal_samples = np.random.normal(size=self.dimension * n_samples).reshape(self.dimension, n_samples).astype(np.float32)
        # copy the mean vector along the samples
        offset_matrix = np.repeat(self.mu[:, np.newaxis], n_samples, axis=1)
        # stretch and offset every sample, then transpose the matrix so we have a (n_samples x dim) matrix
        return (L @ standard_normal_samples + offset_matrix).T


class GaussianBayesNet:
    def __init__(self, adjacency_matrix: npt.NDArray[np.bool_], network_parameters: dict[int, tuple[npt.NDArray[np.float32], float]] = None):
        """Bayesian network with Gaussians as conditional probabilities.

        Parameters
        ----------
        adjacency_matrix : npt.NDArray[np.bool_]
            Matrix defining parents of each node. A[i,j] == True <=> there is an edge from i to j (i is a parent of j).
        parameters : dict[int, tuple[npt.NDArray[np.float32], float]]
            Gaussian parameters for the conditional probabilities of the nodes given their parents. Stored as {node: (beta, sigma)} where beta[0] is the offset and beta[1:] are the linear coefficients.
        """

        self.adjacency_matrix = adjacency_matrix
        self.network_parameters = network_parameters
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
            log_likelihood += log_likelihood_linear_regression(x, y, beta, sigma)
        return log_likelihood

    def to_multivariate_gaussian(self) -> MultivariateGaussian:
        """Generates the equivalent multivariate gaussian."""

        # calculate mean vector
        mean_vector = np.empty(self.n, dtype=np.float32)
        mean_done = np.full(self.n, False)
        # start with the root_nodes
        root_nodes = [node for node in range(self.n) if len(self.parents(node)) == 0]
        for root_node in root_nodes:
            beta, sigma = self.network_parameters[root_node]
            mean_vector[root_node] = beta[0]
            mean_done[root_node] = True
        # calculate mu_i when pa(i) are all done
        while not np.all(mean_done):
            for node in range(self.n):
                # if the node is already computed, go on
                if mean_done[node]:
                    continue
                # if the parents are not computed yet, we can't compute the node yet
                if not all([mean_done[parent] for parent in self.parents(node)]):
                    continue
                # compute the mean of this feature
                parent_means = mean_vector[self.parents(node)]
                beta, sigma = self.network_parameters[node]
                # mu_i = beta_0 + beta * mu_pa(i)
                # note that beta_0 is beta[0] and the real linear coefficients are beta[1:]
                parent_means_padded = np.insert(parent_means, 0, 1)
                mean_vector[node] = beta.T @ parent_means_padded
                mean_done[node] = True
        # calculate covariance matrix
        covariance_matrix = np.zeros((self.n, self.n), dtype=np.float32)
        covariance_done = np.full(self.n, False)
        discovered_order: list[int] = []
        # root nodes first
        for root_node in root_nodes:
            beta, sigma = self.network_parameters[root_node]
            covariance_matrix[root_node, root_node] = sigma**2
            covariance_done[root_node] = True
            discovered_order.append(root_node)
        # in the following comments Sigma_ij (upper case S) refers to the covariance matrix, and sigma_i^2 refers to the variance in the conditional distribution p(i | pa(i)) ~ N(beta_0 + beta^T * x, sigma_i^2).
        # calculate Sigma_i,j when Sigma_pa(i),j are all done
        while not np.all(covariance_done):
            for node in range(self.n):
                # if the node is already computed, go on
                if covariance_done[node]:
                    continue
                # if the parents are not computed yet, we can't compute the node yet
                if not all([covariance_done[parent] for parent in self.parents(node)]):
                    continue
                # compute Sigma_ii
                beta, sigma = self.network_parameters[node]
                parents = self.parents(node)
                # slice the covariance matrix to get the covariance matrix of the parents
                parent_covariance_matrix = covariance_matrix[parents, :][:, parents]
                # Sigma_ii = sigma_i^2 + beta^T * Sigma_parents * beta
                # note that beta_0 is beta[0] and the real linear coefficients are beta[1:]
                covariance_matrix[node, node] = sigma**2 + beta[1:].T @ parent_covariance_matrix @ beta[1:]
                # compute Sigma_ij where j in pa(i)
                for parent_index, parent in enumerate(parents):
                    # Sigma_ij = sum_k beta_k * Sigma_jk = beta^T * Sigma_j,:, where Sigma_j,: is the vector (sigma_jk)_{k in pa(i)}
                    # note that beta_0 is beta[0] and the real linear coefficients are beta[1:]
                    covariance_matrix[node, parent] = beta[1:].T @ parent_covariance_matrix[parent_index, :]
                    covariance_matrix[parent, node] = covariance_matrix[node, parent]
                # now compute all the covariance with the other nodes discovered earlier
                for other_node in discovered_order:
                    # don't need to compute the parents again
                    if other_node in parents:
                        continue
                    # compute Sigma_ij where j is not a parent of i
                    # Sigma_ij = sum_k beta_k * Sigma_kj (linear combination of the parents covariance to that other node)
                    # the parents' covariance to that other node have already been computer because all covariances between nodes discovered earlier have been computed
                    covariance_matrix[node, other_node] = beta[1:].T @ covariance_matrix[parents, other_node]
                    covariance_matrix[other_node, node] = covariance_matrix[node, other_node]
                # we're finally done with the covariances for this node
                covariance_done[node] = True
                discovered_order.append(node)
        # mean and covariance matrix have been fully assembled, let's just return the model
        return MultivariateGaussian(mean_vector, covariance_matrix)


def main():
    # test if the conversion to a multivariate gaussian is correct
    # this example is taken from https://helenedk.medium.com/an-introduction-to-gaussian-bayesian-networks-4eeed3d8e6e0
    # p(x_0) ~ N(2, 4)
    # p(x_1 | x_0) ~ N(0.5 * x_0 - 2.5, 4)
    # p(x_2 | x_1) ~ N(-x_1 + 1, 3)
    beta_x_0 = np.array([2], dtype=np.float32)
    beta_x_1 = np.array([-2.5, 0.5], dtype=np.float32)
    beta_x_2 = np.array([1, -1], dtype=np.float32)
    sigma_x_0 = np.sqrt(4)
    sigma_x_1 = np.sqrt(4)
    sigma_x_2 = np.sqrt(3)
    adjacency_matrix: npt.NDArray[np.bool_] = np.array([
        [False, True, False],
        [False, False, True],
        [False, False, False],
    ])
    network_parameters: dict[int, tuple[npt.NDArray[np.float32], float]] = {
        0: (beta_x_0, sigma_x_0),
        1: (beta_x_1, sigma_x_1),
        2: (beta_x_2, sigma_x_2),
    }
    example_gbn = GaussianBayesNet(adjacency_matrix, network_parameters)
    example_multi_gaussian = example_gbn.to_multivariate_gaussian()
    expected_mu = np.array([2, -1.5, 2.5])
    assert np.all(example_multi_gaussian.mu == expected_mu)
    assert example_multi_gaussian.sigma[0, 0] == 4
    assert example_multi_gaussian.sigma[1, 1] == 5
    assert example_multi_gaussian.sigma[2, 2] == 8
    assert example_multi_gaussian.sigma[0, 1] == 2
    assert example_multi_gaussian.sigma[1, 2] == -5
    assert example_multi_gaussian.sigma[0, 2] == -2
    print("network.py: converting to multivariate gaussian correctly.")

    # define a model with adjacency matrix and set the network parameters, sample a lot of data from it, fit another model and compare the two models
    N_DATA_POINTS = 100_000
    RELATIVE_TOLERANCE = 0.02
    synthetic_data = example_multi_gaussian.sample(N_DATA_POINTS)
    trained_gbn = GaussianBayesNet(adjacency_matrix).fit(synthetic_data)
    for node in range(3):
        example_beta, example_sigma = example_gbn.network_parameters[node]
        trained_beta, trained_sigma = trained_gbn.network_parameters[node]
        assert np.all(np.isclose(example_beta, trained_beta, rtol=RELATIVE_TOLERANCE))
        assert np.isclose(example_sigma, trained_sigma, rtol=RELATIVE_TOLERANCE)
    print("network.py: learning the correct parameters.")
    print("network.py: all tests passed.")

    # fun experiment: directions of dependencies don't matter! (watch out: directions don't matter as long as all the conditional independencies are the same. this is not always the case!)
    other_adjacency_matrix: npt.NDArray[np.bool_] = np.array([
        [False, False, False],
        [True, False, True],
        [False, False, False],
    ])
    other_trained_gbn = GaussianBayesNet(other_adjacency_matrix).fit(synthetic_data)

    trained_multi_gaussian = trained_gbn.to_multivariate_gaussian()
    other_trained_multi_gaussian = other_trained_gbn.to_multivariate_gaussian()
    print("\nexample gbn:")
    print(f"likelihood: {example_gbn.log_likelihood(synthetic_data)}")
    print(f"mu:\n{example_multi_gaussian.mu}")
    print(f"sigma:\n{example_multi_gaussian.sigma}")

    print("\nfirst trained gbn (correct adjacency matrix):")
    print(f"likelihood: {trained_gbn.log_likelihood(synthetic_data)}")
    print(f"mu:\n{trained_multi_gaussian.mu}")
    print(f"sigma:\n{trained_multi_gaussian.sigma}")

    print("\nsecond trained gbn (adjacency matrix with different directions):")
    print(f"likelihood: {other_trained_gbn.log_likelihood(synthetic_data)}")
    print(f"mu:\n{other_trained_multi_gaussian.mu}")
    print(f"sigma:\n{other_trained_multi_gaussian.sigma}")


if __name__ == "__main__":
    main()
