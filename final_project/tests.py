import numpy as np
import numpy.typing as npt


def gen_test_mask(n_samples: int, rotations: int, iteration: int) -> npt.NDArray[np.bool_]:
    samples_per_rotation = n_samples // rotations
    start_idx = samples_per_rotation * iteration
    end_idx = samples_per_rotation * (iteration + 1)
    mask = np.full(n_samples, False)

    last_iteration = iteration == rotations - 1
    if not last_iteration:
        mask[start_idx:end_idx] = True
    else:
        mask[start_idx:] = True
    return mask


def split_train_test(x: npt.NDArray[np.float32], rotations: int, iteration: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    n_samples = len(x)
    test_mask = gen_test_mask(n_samples, rotations, iteration)
    x_test = x[test_mask, :]
    x_train = x[~test_mask, :]
    return x_train, x_test
