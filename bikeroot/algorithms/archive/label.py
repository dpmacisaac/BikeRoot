
from __future__ import annotations   # allows forward references without quotes
import bisect
from sortedcontainers import SortedKeyList
from typing import Dict, Optional
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

class Label:
    id_counter = 0
    def __init__(self, cost: Dict[str, float], node_at: int | None, prev: Optional["Label"] = None):
        self.cost = cost
        self.node_at = node_at
        self.prev = prev
        self.id = Label.id_counter
        Label.id_counter += 1


class BiLabelSet():
    def __init__(self, criteria_labels=None):
        if len(criteria_labels) <= 2:
            self._labels = {}
        self.criteria_labels = criteria_labels if criteria_labels else []


    def add_label(self, label: Label):
        print(f"Adding label: {label}")
        node_at = label.node_at
        if label.node_at not in self._labels:
            self._labels[node_at] = [label]
            return True
        else:
            self._labels[node_at].append(label)
            return False
            # criteria1, criteria2 = self.criteria_labels
            # # 1. find insertion point by cost1 (binary search)
            # i = bisect.bisect_left([l.cost[criteria1] for l in self._labels[node_at]], label.cost[criteria1])

            # # 2. check left‑neighbour dominance
            # if i > 0 and self._labels[node_at][i-1].cost[criteria2] <= label.cost[criteria2]:
            #     return False        # dominated

            # # 3. prune any dominated labels to the *right*
            # j = i
            # while j < len(self._labels[node_at]) and self._labels[j].cost[criteria2] >= label.cost[criteria2]:
            #     j += 1
            # self._labels[node_at][i:j] = [label]           # slice‑replace
        return True

    def remove_label(self, label: Label):
        self._labels.remove(label)

    def get_labels(self):
        return list(self.labels)

    def clear(self):
        self.labels.clear()

    def __contains__(self, label: Label):
        return label in self.labels

    def __len__(self):
        return len(self.labels)
    
    def merge(self, i, j, cost_ij):
        # check if j exists in labels. If not, nothing to check against so add all labels from i
        if j not in self._labels:
            for i_label in self._labels[i]:
                cost = {c: i_label.cost[c] + cost_ij[c] for c in self.criteria_labels}
                self.add_label(Label(cost=cost, node_at=j, prev=i_label))
            return True
        
        # create arr for nondominated sorting
        label_arr = np.zeros((len(self._labels[i]) + len(self._labels[j]), len(self.criteria_labels)), dtype=float)
        x = 0
        for i_label in self._labels[i]:
            y = 0
            for c in self.criteria_labels:
                label_arr[x][y] = i_label.cost[c] + cost_ij[c]
                y += 1
            x += 1
        for j_label in self._labels[j]:
            y = 0
            for c in self.criteria_labels:
                label_arr[x][y] = j_label.cost[c]
                y += 1
            x += 1

        # non-dominated sorting
        nds = NonDominatedSorting()
        _, ranks = nds.do(label_arr, return_rank=True)

        # iterate through the labels in reverse order to remove dominated labels and add new labels
        changed = False
        top_index = len(self._labels[i]) + len(self._labels[j]) - 1

        for x in range(top_index, -1, -1):
            if x > len(self._labels[i])-1:
                if ranks[x] != 0:
                    self._labels[j].remove(self._labels[j][len(x - len(self._labels[i]))])

            else:
                if ranks[x] == 0:
                    changed = True if not changed else changed
                    cost = {}
                    for c in self.criteria_labels:
                        cost[c] = label_arr[x][self.criteria_labels.index(c)]
                    
                    new_label = Label(
                        cost = cost,
                        node_at=j,
                        prev = self._labels[i][x])
                    
                    self.add_label(new_label)
        return changed

    def backtrack(self, source, target):
        paths = []
        for label in self._labels[target]:
            path = []
            current = label
            while current.node_at is not source:
                path.append(current.node_at)
                current = current.prev
            path.append(current.node_at)
            path.reverse()
            paths.append((path, label.cost))
        return paths

class MultiLabelSet():
    def __init__(self, criteria_labels=None):
        self.labels = SortedKeyList(key=lambda label: label.cost)
        self.criteria_labels = criteria_labels if criteria_labels else []


    def add_label(self, label: Label):
        self.labels.add(label)

    def remove_label(self, label: Label):
        self.labels.discard(label)

    def get_labels(self):
        return list(self.labels)

    def clear(self):
        self.labels.clear()

    def __contains__(self, label: Label):
        return label in self.labels

    def __len__(self):
        return len(self.labels)
    
    def merge(self):
        pass

    def backtrack(self, source, target):
        paths = []
        for label in self.labels:
            if label.node_from == target:
                path = []
                current = label
                while current.node_from is not source:
                    path.append(current.node_from)
                    current = current.prev
                path.reverse()
                paths.append((path, label.cost))
