# Graphical Models LAB final project

- [Adjacency matrices](#adjacency-matrices)
- [Learn Structure](#learn-structure)
- [Learn Parameters](#learn-parameters)
- [Test Likelihood](#test-likelihood)
- [Render Graphs](#render-graphs)
- [Visualization](#visualization)

## Adjacency matrices

Adjacency matrices are stored in `.npy` files in the `graphs` directory.

## Learn Structure

`src/structure.py` does a greedy search with tabu walks and random restarts for the adjacency matrix with the highest objective function $f(x) = \sum_{i \in [n]} S_i(G) - \lambda |E_G|$. For details, see [the report](report/report.pdf), Section 2.4: Choice of Optimized Score.
The `GreedySearcher` has multiple arguments:

- `initial_adjacency_matrix`: the adjacency matrix that the greedy search starts with
- `regularization_constant`: $\lambda$ in the objective function
- `n_tabu_walks`: number of tabu walks in each tabu walk section (total is `n_random_restarts` * `n_tabu_walks` + 1)
- `max_tabu_list_size`: maximum size of the list of tabu-structures
- `tabu_walk_length`: steps walked during each tabu walk
- `n_random_restarts`: number of random restarts
- `random_walk_length`: number of random changes applied to the adjacency matrix during a random walk
- `logging_enabled`: enables logging of the score history and method start times

## Learn Parameters

`src/network.py` learns the parameters of a GBN given structure. In the same file, there is an implementation of a conversion of the GBN to an equivalent multivariate Gaussian. This multivariate Gaussian then has the ability to produce samples. This was used to test if the ML-estimates of the parameters learn the correct parameters.

1. i set up a GBN with preset parameters
2. converted it to a multivariate Gaussian
3. sampled from the multivariate Gaussian
4. learned the parameters from the samples
5. and compared the learned parameters to the preset parameters

You can run the tests yourself with `python src/network.py`.

## Test Likelihood

`src/test_likelihood.py` computes the likelihood of corresponding GBN(s) (plural because of cross-validation) for a given adjacency matrix.
The likelihood is computed based on the data in `trainset.csv`.

### Usage

```text
usage: test_likelihood.py [-h] [-r ROTATIONS] [--test] filename

Reads an adjacency matrix from a .npy file and tests the likelihood of corresponding GBNs.

positional arguments:
  filename              File with the encoded adjacency matrix.

options:
  -h, --help            show this help message and exit
  -r ROTATIONS, --rotations ROTATIONS
                        Number of rotations for the cross validation (by default 8).
  --test                Tests the implementation (no other arguments needed).
```

### Examples

```text
python src/test_likelihood.py graphs/empty.npy
python src/test_likelihood.py graphs/full.npy -r 20
```

## Render Graphs

Adjacency matrices and can be rendered with `./render.sh <filename>`. The results will be stored in the `renders` directory as `.png` files.

### Examples

```text
./render.sh graphs/full.npy
```

## Visualization

`src/visualize.py` visualizes multiple things depending on the directory of your input file:

- `"graphs"`: it assumes the input file is an adjacency matrix stored as a `.npy` file. It the converts the adjacency matrix to a graph encoded in a `.dot` file.
- `"logs"`: it assumes the input file is the score history and method starts saved in a `.npy` file. It then plots the score history and method starts.
- `"results"`: it assumes the input file is `n_params.csv` or `*times*.npy`. It then plots the number of parameters or the runtime over $\lambda$.
