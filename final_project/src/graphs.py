import numpy as np
from enum import Enum
import numpy.typing as npt


Edge = tuple[int, int]


class ChangeType(Enum):
    Flip = 0
    Addition = 1
    Deletion = 2


def apply_change(adjacency_matrix: npt.NDArray[np.bool_], changed_edge: Edge, change_type: ChangeType):
    """Applies the change to the adjacency matrix. Edge order is relevant."""

    from_node, to_node = changed_edge
    if change_type == ChangeType.Addition:
        adjacency_matrix[from_node, to_node] = True
        return
    if change_type == ChangeType.Deletion:
        adjacency_matrix[from_node, to_node] = False
        return
    if change_type == ChangeType.Flip:
        adjacency_matrix[from_node, to_node] = False
        adjacency_matrix[to_node, from_node] = True
        return
    raise ValueError(f"Unsupported change type: {change_type}")


def revert_change(adjacency_matrix: npt.NDArray[np.bool_], changed_edge: Edge, change_type: ChangeType):
    """Reverts the change to the adjacency matrix, assuming it was applied earlier. Edge order is relevant."""

    from_node, to_node = changed_edge
    if change_type == ChangeType.Addition:
        adjacency_matrix[from_node, to_node] = False
        return
    if change_type == ChangeType.Deletion:
        adjacency_matrix[from_node, to_node] = True
        return
    if change_type == ChangeType.Flip:
        adjacency_matrix[from_node, to_node] = True
        adjacency_matrix[to_node, from_node] = False
        return
    raise ValueError(f"Unsupported change type: {change_type}")


def edge_difference(change_type: ChangeType) -> int:
    """The edge difference of E_G' - E_G when this type of change is applied."""

    if change_type == ChangeType.Addition:
        return 1
    if change_type == ChangeType.Flip:
        return 0
    if change_type == ChangeType.Deletion:
        return -1
    raise ValueError(f"Unsupported change type: {change_type}")


def possible_flips(adjacency_matrix: npt.NDArray[np.bool_]) -> list[Edge]:
    """List of edges that could be flipped without creating a cycle."""

    flips: list[Edge] = []
    # flip some edges
    for edge in all_edges(adjacency_matrix):
        # change the adjacency matrix temporarily
        apply_change(adjacency_matrix, edge, ChangeType.Flip)
        if not has_cycle(adjacency_matrix):
            flips.append(edge)
        # revert the temporal change
        revert_change(adjacency_matrix, edge, ChangeType.Flip)
    return flips


def possible_deletions(adjacency_matrix: npt.NDArray[np.bool_]) -> list[Edge]:
    """List of edges that could be deleted without creating a cycle."""

    return all_edges(adjacency_matrix)


def possible_additions(adjacency_matrix: npt.NDArray[np.bool_]) -> list[Edge]:
    """List of edges that could be added without creating a cycle."""

    n_nodes_ = n_nodes(adjacency_matrix)
    additions: list[Edge] = []
    for from_node in range(n_nodes_):
        for to_node in range(n_nodes_):
            # nodes can't have an edge to themselves
            if from_node == to_node:
                continue
            # don't add edges that already exist
            if adjacency_matrix[from_node, to_node]:
                continue
            # change the adjacency matrix temporarily
            edge = (from_node, to_node)
            apply_change(adjacency_matrix, edge, ChangeType.Addition)
            if not has_cycle(adjacency_matrix):
                additions.append(edge)
            # revert the temporal change
            revert_change(adjacency_matrix, edge, ChangeType.Addition)
    return additions


def possible_changes(adjacency_matrix: npt.NDArray[np.bool_], change_type: ChangeType) -> list[Edge]:
    if change_type == ChangeType.Flip:
        return possible_flips(adjacency_matrix)
    if change_type == ChangeType.Addition:
        return possible_additions(adjacency_matrix)
    if change_type == ChangeType.Deletion:
        return possible_deletions(adjacency_matrix)
    raise ValueError(f"Unsupported change type: {change_type}")


def construct_all_changes(adjacency_matrix: npt.NDArray[np.bool_]) -> list[tuple[ChangeType, Edge]]:
    return [(change_type, changed_edge) for change_type in ChangeType for changed_edge in possible_changes(adjacency_matrix, change_type)]


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


def all_edges(adjacency_matrix: npt.NDArray[np.bool_]) -> list[Edge]:
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
