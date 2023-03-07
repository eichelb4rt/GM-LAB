import numpy as np
import numpy.typing as npt

import clock
from dynamic_line import Progress
from network import GaussianBayesNet


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


def split_train_test(x: npt.NDArray[np.float32], rotations: int, iteration: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    n_samples = x.shape[0]
    test_mask = gen_test_mask(n_samples, rotations, iteration)
    x_test = x[test_mask, :]
    x_train = x[~test_mask, :]
    return x_train, x_test


def main():
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
    print("tests.py: all tests passed.")


def single_log_likelihood(adjacency_matrix: npt.NDArray[np.bool_], x_train: npt.NDArray[np.float32], x_test: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Generates likelihood (for specific x_train, x_test)."""

    clock.start("fitting data")
    gbn = GaussianBayesNet(adjacency_matrix).fit(x_train)
    clock.stop("fitting data")
    clock.start("calculating log likelihood")
    log_likelihood = gbn.log_likelihood(x_test)
    clock.stop("calculating log likelihood")
    return log_likelihood


def cross_validate(adjacency_matrix: npt.NDArray[np.bool_], x: npt.NDArray[np.float32], rotations=8) -> npt.NDArray[np.float32]:
    """Calculates the log likelihood with a sort-of cross validation (splits up data into multiple training and test sets and sums up the log likelihood of the test sets). This is done to catch overfitting."""

    clock.start("total")
    log_likelihoods = np.empty(rotations)
    progress = Progress(rotations)
    for i in range(rotations):
        x_train, x_test = split_train_test(x, rotations, i)
        log_likelihoods[i] = single_log_likelihood(adjacency_matrix, x_train, x_test)
        progress.update(i + 1)
    clock.stop("total")
    # print runtime
    clock.avg("fitting data")
    clock.avg("calculating log likelihood")
    clock.total("total")
    # sum up the log likelihoods (mean is not the right tool here!)
    return np.sum(log_likelihoods, axis=0)


if __name__ == "__main__":
    main()
