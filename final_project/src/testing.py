import numpy as np
import numpy.typing as npt
from queue import Queue


def hash_adj(adjacency_matrix: npt.NDArray[np.bool_]) -> bytes:
    """Hashable byte representation of the adjacency matrix."""

    return bytes(np.packbits(adjacency_matrix))


n = 12
full_adjacency_matrix = np.full((n, n), False)
for i in range(n):
    for j in range(i + 1, n):
        full_adjacency_matrix[i, j:] = True
empty_adjacency_matrix = np.full((n, n), False)


taboo_list: Queue[bytes] = Queue()
taboo_list.put(hash_adj(full_adjacency_matrix))
current_taboo_states: set[bytes] = set(taboo_list.queue)

print(hash_adj(full_adjacency_matrix) in current_taboo_states)
print(hash_adj(empty_adjacency_matrix) in current_taboo_states)
print(hash_adj(full_adjacency_matrix))

print(-3078.8979333084617 + -4827.041815833941)
print(-3431.50537109375 + -4474.4345703125)