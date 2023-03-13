import numpy as np
import numpy.typing as npt


def n_nodes(adjacency_matrix: npt.NDArray[np.bool_]) -> int:
    return adjacency_matrix.shape[0]


def n_edges(adjacency_matrix: npt.NDArray[np.bool_]) -> int:
    return np.count_nonzero(adjacency_matrix)


def n_params(adjacency_matrix: npt.NDArray[np.bool_]) -> int:
    return 2 * n_nodes(adjacency_matrix) + n_edges(adjacency_matrix)


def neighbours_in(node: int, adjacency_matrix: npt.NDArray[np.bool_]) -> list[int]:
    return list(np.where(adjacency_matrix[:, node])[0])


def neighbours_out(node: int, adjacency_matrix: npt.NDArray[np.bool_]) -> list[int]:
    return list(np.where(adjacency_matrix[node])[0])


def all_edges(adjacency_matrix: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    n = n_nodes(adjacency_matrix)
    return [(node_from, node_to) for node_from in range(n) for node_to in range(n) if adjacency_matrix[node_from, node_to]]


def save(adjacency_matrix: npt.NDArray[np.bool_], name: str = "graph"):
    """Saves the adjacency matrix as a .npy file to graphs/{name}.npy"""

    out_file = f"graphs/{name}.npy"
    np.save(out_file, adjacency_matrix)


def has_cycle(adjacency_matrix: npt.NDArray[np.bool_]) -> bool:
    """Determines if the graph has a cycle."""

    # https://en.wikipedia.org/wiki/Kosaraju%27s_algorithm
    # get all the strongly connected components
    n = adjacency_matrix.shape[0]
    work_list: list[int] = []
    visited = np.full(n, False)
    for node in range(n):
        visit(node, adjacency_matrix, visited, work_list)
    components: dict[int, list[int]] = {}
    has_been_assigned = np.full(n, False)
    for node in work_list:
        assign(node, node, adjacency_matrix, components, has_been_assigned)
    # test if the fully connected components are the nodes themselves
    return len(components) != n


def visit(node: int, adjacency_matrix: npt.NDArray[np.bool_], visited: npt.NDArray[np.bool_], work_list: list[int]):
    if visited[node]:
        return
    visited[node] = True
    for out_neighbour in neighbours_out(node, adjacency_matrix):
        if not visited[out_neighbour]:
            visit(out_neighbour, adjacency_matrix, visited, work_list)
    work_list.insert(0, node)


def assign(node: int, root: int, adjacency_matrix: npt.NDArray[np.bool_], components: dict[int, list[int]], has_been_assigned: npt.NDArray[np.bool_]):
    if has_been_assigned[node]:
        return
    if root not in components:
        components[root] = []
    components[root].append(node)
    has_been_assigned[node] = True
    for in_neighbour in neighbours_in(node, adjacency_matrix):
        assign(in_neighbour, root, adjacency_matrix, components, has_been_assigned)


def main():
    n = 5
    # this should have a cycle
    adjacency_matrix = np.full((n, n), False)
    for i in range(n):
        for j in range(i + 1, n):
            adjacency_matrix[i, j:] = True
    adjacency_matrix[1, 0] = True
    assert has_cycle(adjacency_matrix), "Test failed: graphs -> adjacency matrix should have a cycle but none was detected."

    # this should not have a cycle
    adjacency_matrix = np.full((n, n), False)
    for i in range(n):
        for j in range(i + 1, n):
            adjacency_matrix[i, j:] = True
    assert not has_cycle(adjacency_matrix), "Test failed: graphs -> adjacency matrix should not have a cycle but one was detected."

    print("graphs.py: all tests passed.")


if __name__ == "__main__":
    main()
