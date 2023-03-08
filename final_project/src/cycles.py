import numpy as np
import numpy.typing as npt


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
    print(components)
    # test if the fully connected components are the nodes themselves
    return len(components) != n


def visit(node: int, adjacency_matrix: npt.NDArray[np.bool_], visited: npt.NDArray[np.bool_], work_list: list[int]):
    if visited[node]:
        return
    visited[node] = True
    out_neighbours = np.where(adjacency_matrix[node])[0]
    for out_neighbour in out_neighbours:
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
    in_neighbours = np.where(adjacency_matrix[:, node])[0]
    for in_neighbour in in_neighbours:
        assign(in_neighbour, root, adjacency_matrix, components, has_been_assigned)


def main():
    n = 5
    adjacency_matrix = np.full((n, n), False)
    for i in range(n):
        for j in range(i + 1, n):
            adjacency_matrix[i, j:] = True
    adjacency_matrix[1, 0] = True
    print(has_cycle(adjacency_matrix))


if __name__ == "__main__":
    main()
