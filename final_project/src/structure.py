import numpy as np
import pandas as pd
from enum import Enum
from queue import Queue
import numpy.typing as npt

import graphs
from network import linear_regression


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
    for edge in graphs.all_edges(adjacency_matrix):
        # change the adjacency matrix temporarily
        apply_change(adjacency_matrix, edge, ChangeType.Flip)
        if not graphs.has_cycle(adjacency_matrix):
            flips.append(edge)
        # revert the temporal change
        revert_change(adjacency_matrix, edge, ChangeType.Flip)
    return flips


def possible_deletions(adjacency_matrix: npt.NDArray[np.bool_]) -> list[Edge]:
    """List of edges that could be deleted without creating a cycle."""

    return graphs.all_edges(adjacency_matrix)


def possible_additions(adjacency_matrix: npt.NDArray[np.bool_]) -> list[Edge]:
    """List of edges that could be added without creating a cycle."""

    n_nodes = graphs.n_nodes(adjacency_matrix)
    additions: list[Edge] = []
    for from_node in range(n_nodes):
        for to_node in range(n_nodes):
            # nodes can't have an edge to themselves
            if from_node == to_node:
                continue
            # don't add edges that already exist
            if adjacency_matrix[from_node, to_node]:
                continue
            # change the adjacency matrix temporarily
            edge = (from_node, to_node)
            apply_change(adjacency_matrix, edge, ChangeType.Addition)
            if not graphs.has_cycle(adjacency_matrix):
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


def node_score(node: int, parents: list[int], dataset: npt.NDArray[np.float64]) -> float:
    """Calculates the score of a single node. Basically the likelihood of the node having the given parents."""

    n = dataset.shape[0]
    x = dataset[:, parents]
    y = dataset[:, node]
    # do a linear regression
    beta, sigma = linear_regression(x, y)
    # calculate the score (similar to the likelihood)
    x_padded: npt.NDArray[np.float64] = np.c_[np.ones(n), x]
    mu = x_padded @ beta
    return -0.5 * np.sum(((y - mu) / sigma)**2) - n * np.log(sigma)


def construct_all_changes(adjacency_matrix: npt.NDArray[np.bool_]) -> list[tuple[ChangeType, Edge]]:
    return [(change_type, changed_edge) for change_type in ChangeType for changed_edge in possible_changes(adjacency_matrix, change_type)]


def hash_adj(adjacency_matrix: npt.NDArray[np.bool_]) -> bytes:
    """Hashable byte representation of the adjacency matrix."""

    return bytes(np.packbits(adjacency_matrix))


class GreedySearcher:
    def __init__(self, initial_adjacency_matrix: npt.NDArray[np.bool_], regularization_constant: float = 0, n_tabu_walks: int = 0, max_tabu_list_size: int = 0, tabu_walk_length: int = 0, n_random_restarts: int = 0, random_walk_length: int = 0) -> None:
        self.initial_adjacency_matrix = initial_adjacency_matrix
        self.regularization_constant = regularization_constant
        self.n_nodes = graphs.n_nodes(initial_adjacency_matrix)

        self.top_adjacency_matrix: npt.NDArray[np.bool_]
        self.node_scores: npt.NDArray[np.float64]

        self.n_tabu_walks = n_tabu_walks
        self.max_tabu_list_size = max_tabu_list_size
        self.tabu_walk_length = tabu_walk_length
        self.tabu_list: Queue[bytes]
        self.tabu_list_size: int

        self.n_random_restarts = n_random_restarts
        self.random_walk_length = random_walk_length

    def fit(self, dataset: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Greedy search for better adjacency matrices with tabu search and random resets. Returns the top adjacency matrix."""

        # init stuff
        self.top_adjacency_matrix = self.initial_adjacency_matrix.copy()
        self.node_scores = np.empty(self.n_nodes, dtype=np.float64)
        for node in range(self.n_nodes):
            parents = graphs.neighbours_in(node, self.top_adjacency_matrix)
            self.node_scores[node] = node_score(node, parents, dataset)

        self.tabu_list = Queue()
        self.tabu_list_size = 0
        # climb and stuff
        self.hill_climb(dataset)
        for _ in range(self.n_random_restarts):
            for _ in range(self.n_tabu_walks):
                self.tabu_walk(dataset)
                self.hill_climb(dataset)
            self.random_restart(dataset)
            self.hill_climb(dataset)
        return self.top_adjacency_matrix

    def hill_climb(self, dataset: npt.NDArray[np.float64]):
        """Greedy search for better adjacency matrices."""

        # pretend we improved already to get into the loop
        top_improvement = 1
        delta_e = 0
        while top_improvement > self.regularization_constant * delta_e:
            # construct all possible changes
            all_possible_changes = construct_all_changes(self.top_adjacency_matrix)
            # find the top change among them
            top_improvement, top_changed_edge, top_change_type, new_node_scores = self.find_top_change(all_possible_changes, dataset)
            delta_e = edge_difference(top_change_type)
            # we looked through all possible changes and determined the best, now apply it
            if top_improvement > self.regularization_constant * delta_e:
                apply_change(self.top_adjacency_matrix, top_changed_edge, top_change_type)
                # change node scores
                from_node, to_node = top_changed_edge
                new_from_score, new_to_score = new_node_scores
                self.node_scores[from_node] = new_from_score
                self.node_scores[to_node] = new_to_score
                self.update_tabu_list()
                # print(np.sum(self.node_scores))

    def tabu_walk(self, dataset: npt.NDArray[np.float64]):
        for i in range(self.tabu_walk_length):
            # set for fast acces (we don't want to go through the list every time we check if an adjacency matrix is tabu)
            tabu_hashes: set[bytes] = set(self.tabu_list.queue)
            # construct all changes that are not tabu
            all_possible_changes = construct_all_changes(self.top_adjacency_matrix)
            # in the first step, we already visited every possible change (because we came from hill climbing where the last step considered every possible change)
            # so we just get a freebie on this one
            if i == 0:
                non_tabu_changes = all_possible_changes
            else:
                non_tabu_changes = [(change_type, changed_edge) for change_type, changed_edge in all_possible_changes if self.resulting_hash(changed_edge, change_type) not in tabu_hashes]
            # find the top change among them
            top_improvement, top_changed_edge, top_change_type, new_node_scores = self.find_top_change(non_tabu_changes, dataset)
            # we looked through all non-tabu changes and determined the best, now apply it
            apply_change(self.top_adjacency_matrix, top_changed_edge, top_change_type)
            # change node scores
            from_node, to_node = top_changed_edge
            new_from_score, new_to_score = new_node_scores
            self.node_scores[from_node] = new_from_score
            self.node_scores[to_node] = new_to_score
            self.update_tabu_list()

    def random_restart(self, dataset: npt.NDArray[np.float64]):
        for _ in range(self.random_walk_length):
            # get a random change
            all_possible_changes = construct_all_changes(self.top_adjacency_matrix)
            random_change_type, random_changed_edge = all_possible_changes[np.random.choice(len(all_possible_changes))]
            # apply it
            new_from_score, new_to_score = self.changed_scores(random_changed_edge, random_change_type, dataset)
            apply_change(self.top_adjacency_matrix, random_changed_edge, random_change_type)
            from_node, to_node = random_changed_edge
            self.node_scores[from_node] = new_from_score
            self.node_scores[to_node] = new_to_score
            self.update_tabu_list()

    def find_top_change(self, changes: list[tuple[ChangeType, Edge]], dataset: npt.NDArray[np.float64]) -> tuple[float, Edge, ChangeType, tuple[float, float]]:
        """Finds the change among the passed changes that yields the best improvement. Returns:
            - the improvement
            - the changed edge as a tuple
            - the change type
            - the changed node scores of the nodes of the edge"""

        # needed to figure out the best change
        top_improvement = -np.infty
        top_edge_difference = 0
        # keep track of what change was the best
        top_changed_edge: Edge
        top_change_type: ChangeType
        new_node_scores: tuple[float, float]
        for change_type, changed_edge in changes:
            new_from_score, new_to_score = self.changed_scores(changed_edge, change_type, dataset)
            from_node, to_node = changed_edge
            improvement = (new_from_score + new_to_score) - (self.node_scores[from_node] + self.node_scores[to_node])
            current_edge_difference = edge_difference(change_type)
            # if we get a better improvement (that is worth the edges), update
            if improvement - top_improvement > self.regularization_constant * (current_edge_difference - top_edge_difference):
                top_improvement = improvement
                top_changed_edge = changed_edge
                top_change_type = change_type
                new_node_scores = (new_from_score, new_to_score)
                top_edge_difference = current_edge_difference
        return top_improvement, top_changed_edge, top_change_type, new_node_scores

    def changed_scores(self, changed_edge: Edge, change_type: ChangeType, dataset: npt.NDArray[np.float64]) -> tuple[float, float]:
        """Returns the new scores of the affected nodes if the change was applied."""

        from_node, to_node = changed_edge
        # change the adjacency matrix temporarily
        apply_change(self.top_adjacency_matrix, changed_edge, change_type)
        # figure out how much that improved the scores
        if change_type == ChangeType.Addition or change_type == ChangeType.Deletion:
            # regardless whether the edge was added or deleted, the only distribution updated will be the one of the to_node
            parents = graphs.neighbours_in(to_node, self.top_adjacency_matrix)
            new_from_score = self.node_scores[from_node]
            new_to_score = node_score(to_node, parents, dataset)
        elif change_type == ChangeType.Flip:
            # the only distributions updated will be the ones of from_node and to_node
            parents_from = graphs.neighbours_in(from_node, self.top_adjacency_matrix)
            new_from_score = node_score(from_node, parents_from, dataset)
            parents_to = graphs.neighbours_in(to_node, self.top_adjacency_matrix)
            new_to_score = node_score(to_node, parents_to, dataset)
        else:
            raise ValueError(f"Unsupported change type: {change_type}")
        # revert the temporal change
        revert_change(self.top_adjacency_matrix, changed_edge, change_type)
        return new_from_score, new_to_score

    def resulting_hash(self, changed_edge: Edge, change_type: ChangeType) -> bytes:
        apply_change(self.top_adjacency_matrix, changed_edge, change_type)
        changed_hash = hash_adj(self.top_adjacency_matrix)
        revert_change(self.top_adjacency_matrix, changed_edge, change_type)
        return changed_hash

    def update_tabu_list(self):
        adjacency_matrix_hash = hash_adj(self.top_adjacency_matrix)
        self.tabu_list.put(adjacency_matrix_hash)
        if self.tabu_list_size > self.max_tabu_list_size:
            self.tabu_list.get()
        else:
            self.tabu_list_size += 1


def main():
    # n = 12
    # tabu_graphs: Queue[npt.NDArray[np.bool_]] = Queue[npt.NDArray[np.bool_]]()
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         some_adjacency_matrix[i, j:] = True
    # tabu_graphs.put(some_adjacency_matrix)
    # for adjacency_matrix in tabu_graphs.queue:
    #     print(adjacency_matrix)
    # for adjacency_matrix in tabu_graphs.queue:
    #     print(adjacency_matrix)
    # print(len(possible_flips(adjacency_matrix)))
    # for change_type in ChangeType:
    #     print(change_type)
    dataset = pd.read_csv("trainset.csv").to_numpy()
    n = dataset.shape[1]
    empty_adjacency_matrix = np.full((n, n), False)
    # top_adjacency_matrix, node_scores = hill_climb(empty_adjacency_matrix, dataset)


if __name__ == "__main__":
    main()
