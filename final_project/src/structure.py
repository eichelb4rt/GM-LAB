import numpy as np
import pandas as pd
from queue import Queue
import numpy.typing as npt

import graphs
from network import linear_regression


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


def hash_adj(adjacency_matrix: npt.NDArray[np.bool_]) -> bytes:
    """Hashable byte representation of the adjacency matrix."""

    return bytes(np.packbits(adjacency_matrix))


class GreedySearcher:
    def __init__(self, initial_adjacency_matrix: npt.NDArray[np.bool_], regularization_constant: float = 0, n_tabu_walks: int = 0, max_tabu_list_size: int = 0, tabu_walk_length: int = 0, n_random_restarts: int = 0, random_walk_length: int = 0, logging_enabled: bool = False) -> None:
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

        self.logging_enabled = logging_enabled
        self.score_history: list[float]
        self.method_starts: list[tuple[str, int]]

    def fit(self, dataset: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Greedy search for better adjacency matrices with tabu search and random resets. Returns the top adjacency matrix."""

        # init stuff
        self.top_adjacency_matrix = self.initial_adjacency_matrix.copy()
        self.node_scores = np.empty(self.n_nodes, dtype=np.float64)
        for node in range(self.n_nodes):
            parents = graphs.neighbours_in(node, self.top_adjacency_matrix)
            self.node_scores[node] = node_score(node, parents, dataset)

        # init empty tabu list
        self.tabu_list = Queue()
        self.tabu_list_size = 0

        # init score log
        if self.logging_enabled:
            self.score_history = []
            self.score_history.append(self.score())
            self.method_starts = []

        # first hill climb
        self.hill_climb(dataset)
        # then hill climb, tabu, and restart a lot
        for _ in range(self.n_random_restarts):
            for _ in range(self.n_tabu_walks):
                self.tabu_walk(dataset)
                self.hill_climb(dataset)
            self.random_restart(dataset)
            self.hill_climb(dataset)
        # then tabu a bit more
        for _ in range(self.n_tabu_walks):
            self.tabu_walk(dataset)
            self.hill_climb(dataset)
        return self.top_adjacency_matrix

    def hill_climb(self, dataset: npt.NDArray[np.float64]):
        """Searches every possible change until there is none that improves the objective function."""

        # log stuff
        if self.logging_enabled:
            self.method_starts.append(("hill_climb", len(self.score_history)))
        # pretend we improved already to get into the loop
        top_improvement = 1
        delta_e = 0
        while top_improvement > self.regularization_constant * delta_e:
            # construct all possible changes
            all_possible_changes = graphs.construct_all_changes(self.top_adjacency_matrix)
            # find the top change among them
            top_improvement, top_changed_edge, top_change_type, new_node_scores = self.find_top_change(all_possible_changes, dataset)
            delta_e = graphs.edge_difference(top_change_type)
            # we looked through all possible changes and determined the best, now apply it
            if top_improvement > self.regularization_constant * delta_e:
                graphs.apply_change(self.top_adjacency_matrix, top_changed_edge, top_change_type)
                # update node scores
                from_node, to_node = top_changed_edge
                new_from_score, new_to_score = new_node_scores
                self.node_scores[from_node] = new_from_score
                self.node_scores[to_node] = new_to_score
                # one more adjacency matrix to the tabu list :)
                self.update_tabu_list()
                # maybe log the current score
                if self.logging_enabled:
                    self.score_history.append(self.score())

    def tabu_walk(self, dataset: npt.NDArray[np.float64]):
        """Apply the best possible change that hasn't been visited yet. Doesn't matter if it improves the objective function or not. If the result is better than the original, use that."""


        original_adjacency_matrix = self.top_adjacency_matrix.copy()
        original_node_scores = self.node_scores.copy()
        # log stuff
        if self.logging_enabled:
            self.method_starts.append(("tabu_walk", len(self.score_history)))
        # go for a walk
        total_improvement = 0
        for i in range(self.tabu_walk_length):
            # set for fast acces (we don't want to go through the list every time we check if an adjacency matrix is tabu)
            tabu_hashes: set[bytes] = set(self.tabu_list.queue)
            # construct all changes that are not tabu
            all_possible_changes = graphs.construct_all_changes(self.top_adjacency_matrix)
            # in the first step, we already visited every possible change (because we came from hill climbing where the last step considered every possible change)
            # so we just get a freebie on this one
            if i == 0:
                non_tabu_changes = all_possible_changes
            else:
                non_tabu_changes = [(change_type, changed_edge) for change_type, changed_edge in all_possible_changes if self.resulting_hash(changed_edge, change_type) not in tabu_hashes]
            # find the top change among them
            top_node_score_improvement, top_changed_edge, top_change_type, new_node_scores = self.find_top_change(non_tabu_changes, dataset)
            # we looked through all non-tabu changes and determined the best, now apply it
            graphs.apply_change(self.top_adjacency_matrix, top_changed_edge, top_change_type)
            # update node scores
            from_node, to_node = top_changed_edge
            new_from_score, new_to_score = new_node_scores
            self.node_scores[from_node] = new_from_score
            self.node_scores[to_node] = new_to_score
            # update the total improvement of the objective function
            delta_e = graphs.edge_difference(top_change_type)
            total_improvement += top_node_score_improvement - self.regularization_constant * delta_e
            # one more adjacency matrix to the tabu list :)
            self.update_tabu_list()
            # maybe log the current score
            if self.logging_enabled:
                self.score_history.append(self.score())
            # stop the tabu walk if we got better than the original
            if total_improvement > 0:
                return
        # if we're here, it means that we didn't get better than the original -> reset it
        self.top_adjacency_matrix = original_adjacency_matrix
        self.node_scores = original_node_scores
        # maybe log when the reset
        if self.logging_enabled:
            self.score_history.append(self.score())

    def random_restart(self, dataset: npt.NDArray[np.float64]):
        """Just apply some random changes. Doesn't matter if it improves the objective function or not. Picks from a uniform distribution of all possible changes."""

        # log stuff
        if self.logging_enabled:
            self.method_starts.append(("random_restart", len(self.score_history)))
        # go for a walk
        for _ in range(self.random_walk_length):
            # get a random change (uniformly!)
            all_possible_changes = graphs.construct_all_changes(self.top_adjacency_matrix)
            random_change_type, random_changed_edge = all_possible_changes[np.random.choice(len(all_possible_changes))]
            # apply it
            new_from_score, new_to_score = self.changed_scores(random_changed_edge, random_change_type, dataset)
            graphs.apply_change(self.top_adjacency_matrix, random_changed_edge, random_change_type)
            # update node scores
            from_node, to_node = random_changed_edge
            self.node_scores[from_node] = new_from_score
            self.node_scores[to_node] = new_to_score
            # one more adjacency matrix to the tabu list :)
            self.update_tabu_list()
            # maybe log the current score
            if self.logging_enabled:
                self.score_history.append(self.score())

    def find_top_change(self, changes: list[tuple[graphs.ChangeType, graphs.Edge]], dataset: npt.NDArray[np.float64]) -> tuple[float, graphs.Edge, graphs.ChangeType, tuple[float, float]]:
        """Finds the change among the passed changes that yields the best improvement. Returns:
            - the improvement
            - the changed edge as a tuple
            - the change type
            - the changed node scores of the nodes of the edge"""

        # needed to figure out the best change
        top_node_score_improvement = -np.infty
        top_edge_difference = 0
        # keep track of what change was the best
        top_changed_edge: graphs.Edge
        top_change_type: graphs.ChangeType
        new_node_scores: tuple[float, float]
        for change_type, changed_edge in changes:
            # see how this change would impact the scores
            new_from_score, new_to_score = self.changed_scores(changed_edge, change_type, dataset)
            from_node, to_node = changed_edge
            # Delta_S(G_1, G)
            improvement = (new_from_score + new_to_score) - (self.node_scores[from_node] + self.node_scores[to_node])
            current_edge_difference = graphs.edge_difference(change_type)
            # if we get a better improvement (that is worth the edges), update
            if improvement - top_node_score_improvement > self.regularization_constant * (current_edge_difference - top_edge_difference):
                top_node_score_improvement = improvement
                top_changed_edge = changed_edge
                top_change_type = change_type
                new_node_scores = (new_from_score, new_to_score)
                top_edge_difference = current_edge_difference
        return top_node_score_improvement, top_changed_edge, top_change_type, new_node_scores

    def changed_scores(self, changed_edge: graphs.Edge, change_type: graphs.ChangeType, dataset: npt.NDArray[np.float64]) -> tuple[float, float]:
        """Returns the new scores of the affected nodes if the change was applied."""

        from_node, to_node = changed_edge
        # change the adjacency matrix temporarily
        graphs.apply_change(self.top_adjacency_matrix, changed_edge, change_type)
        # figure out how much that improved the scores
        if change_type == graphs.ChangeType.Addition or change_type == graphs.ChangeType.Deletion:
            # regardless whether the edge was added or deleted, the only distribution updated will be the one of the to_node
            parents = graphs.neighbours_in(to_node, self.top_adjacency_matrix)
            new_from_score = self.node_scores[from_node]
            new_to_score = node_score(to_node, parents, dataset)
        elif change_type == graphs.ChangeType.Flip:
            # the only distributions updated will be the ones of from_node and to_node
            parents_from = graphs.neighbours_in(from_node, self.top_adjacency_matrix)
            new_from_score = node_score(from_node, parents_from, dataset)
            parents_to = graphs.neighbours_in(to_node, self.top_adjacency_matrix)
            new_to_score = node_score(to_node, parents_to, dataset)
        else:
            raise ValueError(f"Unsupported change type: {change_type}")
        # revert the temporal change
        graphs.revert_change(self.top_adjacency_matrix, changed_edge, change_type)
        return new_from_score, new_to_score

    def resulting_hash(self, changed_edge: graphs.Edge, change_type: graphs.ChangeType) -> bytes:
        graphs.apply_change(self.top_adjacency_matrix, changed_edge, change_type)
        changed_hash = hash_adj(self.top_adjacency_matrix)
        graphs.revert_change(self.top_adjacency_matrix, changed_edge, change_type)
        return changed_hash

    def update_tabu_list(self):
        adjacency_matrix_hash = hash_adj(self.top_adjacency_matrix)
        self.tabu_list.put(adjacency_matrix_hash)
        if self.tabu_list_size > self.max_tabu_list_size:
            self.tabu_list.get()
        else:
            self.tabu_list_size += 1

    def score(self) -> float:
        return np.sum(self.node_scores) - self.regularization_constant * graphs.n_edges(self.top_adjacency_matrix)


def main():
    dataset = pd.read_csv("trainset.csv").to_numpy()
    n = dataset.shape[1]
    empty_adjacency_matrix = np.full((n, n), False)
    detective = GreedySearcher(empty_adjacency_matrix,
                               regularization_constant=25,

                               n_tabu_walks=3,
                               max_tabu_list_size=10,
                               tabu_walk_length=5,

                               n_random_restarts=5,
                               random_walk_length=5,

                               logging_enabled=False)
    top_adjacency_matrix = detective.fit(dataset)
    graphs.save(top_adjacency_matrix, name=f"top_{graphs.n_params(top_adjacency_matrix)}")


if __name__ == "__main__":
    main()
