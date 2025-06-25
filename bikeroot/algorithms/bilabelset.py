
from __future__ import annotations   # allows forward references without quotes
import bisect
from sortedcontainers import SortedKeyList
from typing import Dict, Optional
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

class BiLabelSet():
    def __init__(self, criteria=None):
        self._by_node = {}
        self._store = {}
        self.criteria = criteria if criteria else []
        self._count = 0


    def add_label(self, node_at: int, cost: Dict[str, float], prev_id = None):
        self._count += 1
        entry = {'node_at': node_at, 'cost': cost, 'prev_id': prev_id}
        self._store[self._count] = entry

        if node_at not in self._by_node:
            self._by_node[node_at] = SortedKeyList(key=lambda t:t[0])
        
        priority = cost[self.criteria[0]]

        self._by_node[node_at].add((priority, self._count))

        return self._count

    def remove_label(self, id):
        entry = self._store.pop(id, None)
        if entry is None:
            return False
        
        node_at = entry['node_at']
        priority = entry['cost'][self.criteria[0]]
        self._by_node[node_at].remove((priority, id))
        if not self._by_node[node_at]:
            del self._by_node[node_at]
        return True

    def print_by_nodes(self,node):
        for x in self._by_node[node]:
            print(f'Node {node} has label {x[1]} with cost {self._store[x[1]]["cost"]}')

    def node_merge(self, i, j, cost_ij):
        # check if j exists in labels. If not, nothing to check against so add all labels from i
        if j not in self._by_node:
            for _,id in self._by_node[i]:
                new_cost = {c: self._store[id]['cost'][c] + cost_ij[c] for c in self.criteria}
                self.add_label(node_at=j, cost = new_cost, prev_id = id)
            return True
        # generate a new list of labels for j based on i + cost_ij
        # temp storage for these labels is quicker than repeadedly accessing and recalculating the cost
        potential_labels = []

        for _, id in self._by_node[i]:
            new_cost = self.get_new_cost(id, cost_ij)
            potential_labels.append({'node_at':i, 'cost':new_cost, 'prev_id':id})

        Mj = []
        Mi = []
        i_idx = 0
        j_idx = 0
        p = len(potential_labels)
        q = len(self._by_node[j])
        first_crit = self.criteria[0]
        second_crit = self.criteria[1]
        while True:
            # if i_idx > p and j_idx > q:
            #     break
            if i_idx >= p:
                # append rest of j to M
                while j_idx < q:
                    Mj.append(self._by_node[j][j_idx][1])
                    j_idx += 1
                break

            if j_idx >= q:
                # append rest of i to M
                while i_idx < p:
                    Mi.append(i_idx)
                    i_idx += 1
                break

            else:
                # if xi1 < yj1 (first val of new label is less than first val of j label) (part a)
                if potential_labels[i_idx]['cost'][first_crit] < self._by_node[j][j_idx][0]:
                    if j_idx < q:
                        j_label_id = self._by_node[j][j_idx][1]
                        while potential_labels[i_idx]['cost'][second_crit] <= self._store[j_label_id]['cost'][second_crit]:
                            j_idx += 1
                            if j_idx < q:
                                j_label_id = self._by_node[j][j_idx][1]
                            else:
                                break
                    # add that i label to M
                    Mi.append(i_idx)
                    i_idx += 1
                    continue
                # if yj1 < xi1 (part b)
                elif self._by_node[j][j_idx][0] < potential_labels[i_idx]['cost'][first_crit]:
                    j_label_id = self._by_node[j][j_idx][1]
                    j_label_cost_second = self._store[j_label_id]['cost'][second_crit]
                    while i_idx < p and j_label_cost_second <= potential_labels[i_idx]['cost'][second_crit]:
                        i_idx += 1

                    Mj.append(j_label_id)
                    j_idx += 1
                    continue
                # if xi2 < yj2 (part c)
                elif potential_labels[i_idx]['cost'][second_crit] < self._store[self._by_node[j][j_idx][1]]['cost'][second_crit]:

                    while True:
                        j_idx += 1

                        if j_idx >= q:
                            break 
                        # until j_2 < i_2

                        j_label_id = self._by_node[j][j_idx][1]
                        j_label_cost_second = self._store[j_label_id]['cost'][second_crit]
                        if j_label_cost_second < potential_labels[i_idx]['cost'][second_crit]:
                            break

                    Mi.append(i_idx)
                    i_idx += 1
                    continue
                # if yj2 < xi2 (part d)
                else:
                    j_label_id = self._by_node[j][j_idx][1]
                    j_label_cost_second = self._store[j_label_id]['cost'][second_crit]
                    while True :
                        i_idx += 1
                        if i_idx >= p:
                            break 

                        if potential_labels[i_idx]['cost'][second_crit] < j_label_cost_second:
                            break

                    Mj.append(j_label_id)
                    j_idx += 1
                    continue
        # remove all labels from j that are not in Mj
        # print(f'labels in Mi: {Mi}')
        # print(f'labels in Mj: {Mj}')

        changed = False
        if len(Mi) > 0 or len(Mj) < q:
            changed = True

        mj_idx = len(Mj)-1 
        
        for j_idx in range(q-1,-1,-1):
            # print(f'checking label {self._by_node[j][j_idx][1]} for node {j}')
            if mj_idx < 0:
                # all labels in j are not in Mj, remove them
                self.remove_label(self._by_node[j][j_idx][1])
                continue
            
            if self._by_node[j][j_idx][1] == Mj[mj_idx]:
                # print(f'keeping label {self._by_node[j][j_idx][1]} for node {j}')
                mj_idx -= 1
                continue
            
            # remove label j_idx
            self.remove_label(self._by_node[j][j_idx][1])

        for id in Mi:
            # print(f'adding label {id} to node {j}')
            self.add_label(node_at=j, cost=potential_labels[id]['cost'], prev_id=potential_labels[id]['prev_id'])
        return changed

    def label_merge(self, node_at, new_cost, id, j):
        if j not in self._by_node:
            # if j does not exist, add label
            id = self.add_label(node_at=j, cost=new_cost, prev_id=id)
            return ([id],[])

        # if j exists, check if new label is dominated by existing labels
        deleted = []
        for _, existing_id in self._by_node[j]:
            existing_cost = self._store[existing_id]['cost']
            if all(new_cost[c] >= existing_cost[c] for c in self.criteria):
                # new label is dominated by existing label. If its dominated by one label, its dominated by all so we can stop checking
                return ([],[])
            elif all(new_cost[c] <= existing_cost[c] for c in self.criteria):
                # existing label is dominated by new label. Remove existing label
                deleted.append(existing_id)
        
        id = self.add_label(node_at=j, cost=new_cost, prev_id=id)
        return ([id],deleted)



            
        return (added,deleted)

    def get_new_cost(self, id, cost_ij):
        entry = self._store[id]
        new_cost = {c: entry['cost'][c] + cost_ij[c] for c in self.criteria}
        return new_cost

    def get_label(self, id):
        if id not in self._store:
            return None
        return self._store[id]

    def get_label_it(self, node, i):
        if node not in self._by_node:
            return None
        if i < 0 or i >= len(self._by_node[node]):
            return None
        id = self._by_node[node][i][1]
        return self._store[id]


    def backtrack(self, source, target):
        paths = []
        for label in self._by_node[target]:
            path = []
            current_id = label[1]
            while self._store[current_id]['node_at'] is not source:
                path.append(self._store[current_id]['node_at'])
                current_id = self._store[current_id]['prev_id']
            path.append(self._store[current_id]['node_at'])
            path.reverse()
            paths.append((path, self._store[label[1]]['cost']))
        return paths


class AllLabels():
    def __init__(self, criteria, bilabels:BiLabelSet):
        
        self.criteria = criteria
        self.bilabels = bilabels
        self._labels = SortedKeyList(key=lambda id:tuple(self.bilabels.get_label(id)['cost'][k] for k in self.criteria))

    def add_label(self, id = int):
        # entry = {'cost': cost, 'node_at': node_at, 'id': id}
        if id in self._labels:
            return
        self._labels.add(id)

    def get_next(self, node):
        return self._labels[0] if self._labels else None
    
    def pop(self):
        if not self._labels:
            return None
        entry = self._labels.pop(0)
        return entry

    def remove_label(self, id):
        if id in self._labels:
            self._labels.remove(id)
        self.bilabels.remove_label(id)

    def empty(self):
        return len(self._labels) == 0