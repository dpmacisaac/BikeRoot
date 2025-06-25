
from __future__ import annotations   # allows forward references without quotes
import bisect
from sortedcontainers import SortedKeyList
from typing import Dict, Optional
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

class MultiLabelSet:
    def __init__(self, criteria):
        self.criteria = criteria
        self._by_node = {} # stores [label id] for each label at each node, each nodes list sorted by first cost
        self._store = {} # stores the labels based on id
        self._count = 0
        # label = {id:{node_at: int, cost:dict, prev_id: int}}

        self._queue = SortedKeyList(key=lambda id:tuple(self.get_label(id)['cost'][k] for k in self.criteria)) # holds ids of labels to be processed

    def add_label(self, node_at: int, cost: Dict[str, float], prev_id: Optional[int] = None) -> int:
        label_id = self._count
        self._count += 1
        label = {'node_at': node_at, 'cost': cost, 'prev_id': prev_id}
        self._store[label_id] = label
        if node_at not in self._by_node:
            self._by_node[node_at] = []
        self._by_node[node_at].append(label_id)
        return label_id
    
    def remove_label(self, id: int):
        if id not in self._store:
            return
        label = self._store.pop(id)
        node_at = label['node_at']
        if node_at in self._by_node:
            self._by_node[node_at].remove(id)
            if not self._by_node[node_at]:
                del self._by_node[node_at]

    def remove_from_queue(self, id: int):
        if id in self._queue:
            self._queue.remove(id)

    def add_to_queue(self, id: int):
        if id not in self._store:
            return
        if id not in self._queue:
            # Add the id to the queue if it is not already present
            # The queue is sorted by the cost of the label
            # This ensures that the label with the lowest cost is processed first
            # The key function sorts by the costs in the order of criteria_labels
            self._queue.add(id)

    def get_labels_by_node(self, node:int) -> list[dict]:
        return self._by_node.get(node, [])

    def get_label(self, id):
        if id not in self._store:
            return None
        return self._store[id]
    
    def queue_empty(self):
        return len(self._queue) == 0
    
    def queue_pop(self):
        if self.queue_empty():
            return None
        return self._queue.pop(0)
    
    def backtrack(self, src, dst) -> list[tuple[list[int], list[float]]]:
        if dst not in self._by_node:
            print(f"No labels found for target node {dst}.")
            return []

        paths = []
        for id in self._by_node[dst]:
            path = []
            costs = []
            current_id = id
            cost = self.get_label(current_id)['cost']
            while current_id is not None:
                label = self.get_label(current_id)
                path.append(label['node_at'])
                current_id = label['prev_id']
            path.reverse()
            costs.reverse()
            paths.append((path, cost))
        
        return paths
    
