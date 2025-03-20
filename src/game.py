
from itertools import compress, product
from pebble import ProcessPool
import cloudpickle
from concurrent.futures import as_completed
import random

import networkx as nx
import numpy as np

# Note: A "Layer" is a set of two-qubit operations that can happen simultaneously
# i.e. there are no dependencies between any pair of qubits within a layer


class Game:
    def __init__(self, topology: nx.Graph, 
                 data_qubits: list[tuple],
                 workload: list[list[tuple]],
                 lambda_: float = 0,
                 ):
        self.topology_: nx.Graph = topology
        self.workload_: list[list[tuple[int]]] = workload                           # A list (layers) of list (ops per layer) of 2 qubit ids on graph

        self.total_qubits_: set[tuple] = set(topology.nodes)                        # Total qubits
        self.total_qubits_list_ = list(self.total_qubits_)
        self.data_qubits_: set[tuple] = set(data_qubits)                            # Logical qubits used for computation
        self.defective_nodes_: set[tuple] = set()

        self.lambda_: float = lambda_                                               # Per-layer data defect rate

        self.network_: nx.Graph = topology.copy()                                   # Network qubits, made up of ancillas
        for q in data_qubits:
            self.__graph_remove_qubit(self.network_, q)

        self.qubit_to_data_ = {}                                                    # Mapping between qubit index to data qubits (that represent the single logical qubit)
        self.data_to_qubit_ = {}
        for i, q in enumerate(data_qubits):
            self.qubit_to_data_[i] = (1, [q]) # distance 1, single logical qubit
            self.data_to_qubit_[q] = i


    
    def get_data_qubits(self):
        return self.data_qubits_
    
    def get_ancilla_qubits(self):
        return self.total_qubits_ - self.data_qubits_


    # Runs a single trial. Iterate through all layers, applying
    # defects to each layer, then routing available CX's. If 
    # there are no available paths, then cx gate is pushed to 
    # the next layer.
    def run(self) -> tuple[bool, int]:
        layers = 0
        workload = self.workload_.copy()
        progress = True
        st = []

        while progress: # For each layer...
            layers += 1
            progress = False

            # Sample errors
            defects = np.random.poisson(self.lambda_, size=len(self.total_qubits_list_))


            # Apply error and growth
            for i, count in enumerate(defects):
                if count == 0:
                    continue

                node = self.total_qubits_list_[i]
                
                # if it is not already defective
                if node not in self.defective_nodes_:
                    # if it is a data qubit, apply growth
                    if node in self.data_qubits_:
                        qubit = self.data_to_qubit_[node]
                        success = self.__expand_qubit(qubit)
                        
                        if not success:
                            return (False, layers)

                    # Otherwise disable the ancilla
                    else:
                        self.__graph_remove_qubit(self.network_, node)

                    self.defective_nodes_.add(node)


            # Obtain new ops
            if len(workload) != 0:
                st += workload.pop(0)


            # Iterate over new ops, finding and occupying path, update progress
            used_paths = []
            new_st = []
            for c_qbit, t_qubit in st:
                res, path = self.__find_path(c_qbit, t_qubit)
                if res:
                    progress = True
                    used_paths.append(path)
                    for node in path:
                        self.__graph_remove_qubit(self.network_, node)
                else:
                    new_st.append((c_qbit, t_qubit))

            st = new_st
                    

            # Reset occupied paths
            for path in used_paths[::-1]: # We have to inesrt in reverse order 
                for node in path:
                    self.__graph_add_qubit(self.topology_, self.network_, node)
            

            # Check if we've finished the simulation
            if len(st) == 0 and len(workload) == 0: 
                return (True, layers)
            

        # We can only get here if there's no progress
        return (False, layers)


        
    def __graph_remove_qubit(self, graph: nx.Graph, node: tuple):
        if graph.has_node(node):
            graph.remove_node(node)

    def __graph_add_qubit(self, ref_graph: nx.Graph, graph: nx.Graph, node: tuple):
        if not graph.has_node(node):
            graph.add_node(node)
            for u,v in ref_graph.edges(node):
                if (graph.has_node(v)):
                    graph.add_edge(u,v)
            
    def __expand_qubit(self, qubit: int) -> bool: 
        dist, curr_nodes = self.qubit_to_data_[qubit]
        
        new_dist = dist + 1
        new_count = dist + new_dist

        expansion_candidates = []
        curr_nodes_set = set(curr_nodes)
        # Get all immediate neighbors that are ancilla and not defective
        for node in curr_nodes:
            for neighbor in self.topology_.neighbors(node):
                if (neighbor not in curr_nodes_set) and (neighbor not in self.defective_nodes_) and (neighbor in self.network_.nodes()):
                    expansion_candidates.append(neighbor)

        if len(expansion_candidates) < new_count:
            return False
        
        # Otherwise update network_, data_qubits_, qubit_to_data_, data_to_qubit_ 
        for candidate in expansion_candidates[:new_count]:
            self.__graph_remove_qubit(self.network_, candidate)
            self.data_qubits_.add(candidate)
            self.qubit_to_data_[qubit] = (new_dist, self.qubit_to_data_[qubit][1] + [candidate])
            self.data_to_qubit_[candidate] = qubit
        
        return True

    def __find_path(self, c_qubit: int, t_qubit: int) -> tuple[bool, list[tuple]]:
        _, c_nodes = self.qubit_to_data_[c_qubit]
        _, t_nodes = self.qubit_to_data_[t_qubit]

        c_ancillas = []
        for node in c_nodes:
            for neighbor in self.topology_.neighbors(node):
                if neighbor in self.network_.nodes():
                    c_ancillas.append(neighbor)
                elif neighbor in t_nodes:
                    return (True, [])
                
        t_ancillas = []
        for node in t_nodes:
            for neighbor in self.topology_.neighbors(node):
                if neighbor in self.network_.nodes():
                    t_ancillas.append(neighbor)
                elif neighbor in c_nodes:
                    return (True, [])

        
        # Now get the product of the two paths
        combinations = list(product(c_ancillas, t_ancillas))
        path = []
        shortest_length = float('inf')
        for c, t in combinations:
            try:
                tmp = nx.shortest_path(self.network_, c, t)
                if len(tmp) < shortest_length:
                    shortest_length = len(tmp)
                    path = tmp
            except nx.NetworkXNoPath:
                continue

        
        if len(path) == 0:
            return (False, path)
        return (True, path)


def simulate(topology: nx.Graph, 
             data_qubits: list[tuple],
             workload: list[list[tuple]],
             lambda_: float = 0, 
             n: int = 100) -> tuple[float, float]:
    success_hist = []
    layer_hist = []

    for i in range(n):
        game = Game(topology, data_qubits, workload, lambda_)
        success, layers = game.run()
        success_hist.append(success)
        layer_hist.append(layers)

    # Compute success rate and throughput
    success_rate = sum(success_hist) / n
    if success_rate == 0:
        throughput = 0
    else:
        throughput = (sum(success_hist) * len(workload)) / sum(list(compress(layer_hist, success_hist)))

    return (success_rate, throughput, game)



if (__name__ == "__main__"):
    topo = nx.grid_2d_graph(100, 100)

    data = []
    for i in range(1, 100, 20):
        for j in range(1, 100, 20):
            data.append((i,j))

    l = list(range(len(data)))
    random.seed(0)
    random.shuffle(l)
    size = len(l)
    if size % 2 == 1:
        size -= 1
    
    first_l = l[:size // 2]
    last_l = l[size // 2:]
    layer = list(zip(first_l, last_l))
    
    workload = [layer]

    simulate(topo, data, workload, lambda_=0.01, n=100)
    success_rate, throughput = simulate(topo, data, workload, lambda_=0.01, n=100)
    print(f"success rate: {success_rate}, throughput: {throughput}")
