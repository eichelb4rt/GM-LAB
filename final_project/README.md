# Graphical Models LAB final project

- [Adjacency matrices](#adjacency-matrices)
- [Learn Structure](#learn-structure)
- [Test Likelihood](#test-likelihood)
- [Render Graphs](#render-graphs)

## Adjacency matrices

Adjacency matrices are stored in `.npy` files in the `graphs` directory.

## Learn Structure

Learns the structure that maximizes the posterior probability $p(G | D) \propto p(D | G) \cdot p(G)$ where $p(D | G)$ is chosen to be the likelihood of the ML estimate of the parameters given the graph, and $p(G) \propto \frac{1}{|E|^\lambda}$. This results in $\max_G p(G | D) = \max_G \log p(D | G) + \log p(G) = \max_G \log p(D | G) - \lambda |E|$

<!-- NOTES -->

The ML estimate of the 

This results in an update whenever $p(G' | D) > p(G | D) \iff $

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
