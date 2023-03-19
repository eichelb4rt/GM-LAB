import argparse
import numpy as np
import pandas as pd
import numpy.typing as npt

import clock
import graphs
from dynamic_line import Progress
from network import GaussianBayesNet
from structure import GreedySearcher
from test_arg import TestAction


DEFAULT_ROTATIONS = 8


def gen_test_mask(n_samples: int, rotations: int, iteration: int) -> npt.NDArray[np.bool_]:
    """Generates masks for test samples. Generates them such that, if there are 7 samples and 4 rotations, the mask sizes are [2, 2, 2, 1] instead of [1, 1, 1, 4]."""
    samples_per_rotation = n_samples // rotations
    surplus = n_samples - samples_per_rotation * rotations
    # every iteration where we haven't used up the surplus yet, we added 1 more test sample
    surplus_used = min(surplus, iteration)
    start_idx = samples_per_rotation * iteration + surplus_used
    # if we haven't used all the surplus yet, add 1 more test sample
    end_idx = start_idx + samples_per_rotation
    if iteration < surplus:
        end_idx += 1
    # start building the mask
    mask = np.full(n_samples, False)
    mask[start_idx:end_idx] = True
    return mask


def split_train_test(x: npt.NDArray[np.float64], rotations: int, iteration: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n_samples = x.shape[0]
    test_mask = gen_test_mask(n_samples, rotations, iteration)
    x_test = x[test_mask, :]
    x_train = x[~test_mask, :]
    return x_train, x_test


def cross_validate_detective(detective: GreedySearcher, x: npt.NDArray[np.float64], rotations=8) -> float:
    """Calculates the log likelihood with a sort-of cross validation (splits up data into multiple training and test sets and sums up the log likelihood of the test sets). This is done to catch overfitting."""
    
    log_likelihoods = np.empty(rotations)
    progress = Progress(rotations)
    for i in range(rotations):
        x_train, x_test = split_train_test(x, rotations, i)
        clock.start("fitting detective")
        adjacency_matrix = detective.fit(x_train)
        clock.stop("fitting detective")
        gbn = GaussianBayesNet(adjacency_matrix).fit(x_train)
        log_likelihoods[i] = gbn.log_likelihood(x_test)
        progress.update(i + 1)
    # print runtime
    clock.avg("fitting detective")
    clock.clear_all()
    # sum up the log likelihoods (mean is not the right tool here!)
    return np.sum(log_likelihoods, axis=0)


def cross_validate_structure(adjacency_matrix: npt.NDArray[np.bool_], x: npt.NDArray[np.float64], rotations=8, print_time=True) -> float:
    """Calculates the log likelihood with a sort-of cross validation (splits up data into multiple training and test sets and sums up the log likelihood of the test sets). This is done to catch overfitting."""

    log_likelihoods = np.empty(rotations)
    clock.start("total")
    for i in range(rotations):
        x_train, x_test = split_train_test(x, rotations, i)
        clock.start("fitting data")
        gbn = GaussianBayesNet(adjacency_matrix).fit(x_train)
        clock.stop("fitting data")
        clock.start("calculating log likelihood")
        log_likelihoods[i] = gbn.log_likelihood(x_test)
        clock.stop("calculating log likelihood")
    clock.stop("total")
    # print runtime
    if print_time:
        clock.avg("fitting data")
        clock.avg("calculating log likelihood")
        clock.total("total")
    clock.clear_all()
    # sum up the log likelihoods (mean is not the right tool here!)
    return np.sum(log_likelihoods, axis=0)


def train_log_likelihood(adjacency_matrix: npt.NDArray[np.bool_], x: npt.NDArray[np.float64]) -> float:
    """Fits a model to the whole dataset and calculates the likelihood of the trained data."""

    gbn = GaussianBayesNet(adjacency_matrix).fit(x)
    return gbn.log_likelihood(x)


def print_report(adjacency_matrix: npt.NDArray[np.bool_], dataset: npt.NDArray[np.float64], rotations: int = DEFAULT_ROTATIONS):
    print(f"number of parameters: {graphs.n_params(adjacency_matrix)}")
    cross_log_likelihood = cross_validate_structure(adjacency_matrix, dataset, rotations)
    print(f"total log likelihood of test sets is {cross_log_likelihood}")
    train_set_log_likelihood = train_log_likelihood(adjacency_matrix, dataset)
    print(f"total log likelihood of train set is {train_set_log_likelihood}")


def test_tests():
    # test gen_test_mask
    for n_samples in range(50, 100):
        for rotations in range(2, 9):
            test_masks = [gen_test_mask(n_samples, rotations, iteration) for iteration in range(rotations)]
            # assert test mask sizes differ by at most 1
            test_mask_sizes = [np.count_nonzero(test_masks[iteration]) for iteration in range(rotations)]
            assert max(test_mask_sizes) - min(test_mask_sizes) <= 1, f"Test failed: gen_test_mask had a max mask of {max(test_mask_sizes)} and a min mask of {min(test_mask_sizes)}, which is a difference of {max(test_mask_sizes) - min(test_mask_sizes)} (which is >1). n_samples: {n_samples}, rotations: {rotations}."
            # assert test masks are disjoined
            intersection = np.logical_and.reduce(test_masks)
            assert np.count_nonzero(intersection) == 0, f"Test failed: gen_test_mask did not have disjoined test masks. n_samples: {n_samples}, rotations: {rotations}."
            # assert test masks cover all samples
            disjunction = np.logical_or.reduce(test_masks)
            assert np.count_nonzero(disjunction) == n_samples, f"Test failed: gen_test_mask test masks did not cover all samples. n_samples: {n_samples}, rotations: {rotations}."
    print("test_likelihood.py: all tests passed.")


def main():
    parser = argparse.ArgumentParser(description="Reads an adjacency matrix from a .npy file and tests the likelihood of corresponding GBNs.")
    parser.add_argument("filename",
                        help="File with the encoded adjacency matrix.")
    parser.add_argument("-r",
                        "--rotations",
                        type=int,
                        default=DEFAULT_ROTATIONS,
                        help=f"Number of rotations for the cross validation (by default {DEFAULT_ROTATIONS}).")
    parser.add_argument("--test",
                        action=TestAction.build(test_tests),
                        help="Tests the implementation (no other arguments needed).")

    args = parser.parse_args()
    adjacency_matrix: npt.NDArray[np.bool_] = np.load(args.filename)
    dataset = pd.read_csv("trainset.csv").to_numpy()

    print(f"report: {args.filename}")
    print_report(adjacency_matrix, dataset, args.rotations)


if __name__ == "__main__":
    main()
