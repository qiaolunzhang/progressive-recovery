import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random

from tree_recovery import plot_graph
from prog import iterate_over_failures, root_to_leaves

def read_gml(path):
    G = nx.read_gml(path)
    return G

# generate a random graph structure with len(recovery_cost) nodes
# and some non-zero amount of edges
def sample_graph(recovery_cost):
    G = nx.Graph()

    utils = {}
    demands = {}

    for node in range(len(recovery_cost)):
        G.add_node(node)
        utils.update({node: 1.0})
        demands.update({node: recovery_cost[node]})
    
    # generate random graph with (n-1) <= x <= n(n-1)/2
    # n(n-1) / 2 ensures the final graph will be fully connected (+ 1 is for exclusion in python)
    # n - 1 some arbitrary lower bound
    n = len(recovery_cost)
    num_edges = random.randint(n-1, (n * (n - 1)) / 2 + 1)
    edges = 0

    while edges <= num_edges:
        # try generating a random edge
        try:
            G.add_edge(random.randint(0, len(recovery_cost) - 1), random.randint(0, len(recovery_cost) - 1))
            edges += 1

        # if we couldn't try again
        except:
            continue

    '''
    # original test graph
    G.add_edge(0,1)
    G.add_edge(0,2)
    G.add_edge(0,5)
    G.add_edge(1,2)
    G.add_edge(2,3)
    G.add_edge(2,5)
    G.add_edge(3,4)
    G.add_edge(4,5)
    ''' 

    nx.set_node_attributes(G, name='util', values=utils)
    nx.set_node_attributes(G, name='demand', values=demands)
    
    tree_recovery.plot_graph(G, 1, 'plots/recovery_graphs/1.png')
    return G

# given a graph G and the cost to recover a node in G (costs), 
# we apply a recovery confiugration (config) and return the total system utility
def simulate_recovery(G, config):
    H = G.copy()
    demands = nx.get_node_attributes(H, name='demand')
    util = 0 
 
    for step in config:
        # remove the resources from rcv_amounts
        for index in range(len(step)):
            demands[index] -= step[index]

        # create a new subgraph H, which is G with all nodes where
        # recovery != 0 are removed
        H = G.copy()
        for node in G:
            if demands[node] > 0:
                H.remove_node(node)

        # The largest connected component in H is our utility at t
        util += len(max(nx.connected_components(H), key=len))

    return util

def max_util(G):
    test_costs = [2, 2, 2, 2, 0, 0]
    root = iterate_over_failures(test_costs, 2)
    all_paths = root_to_leaves(root)

    #print(all_paths)
    #print(len(all_paths))
   
    utils = []
    G = sample_graph(test_costs)

    for path in all_paths:
        utils.append(simulate_recovery(G, path))

    print(utils)
    print(max(utils))

def main():
    # G = read_gml('gml/DIGEX.gml')

    # test_costs = [5, 2, 3, 1, 0, 4]
    test_costs = [2, 2, 2, 2, 0, 0]
    root = iterate_over_failures(test_costs, 2)
    all_paths = root_to_leaves(root)

    #print(all_paths)
    #print(len(all_paths))
   
    utils = []
    G = sample_graph(test_costs)

    for path in all_paths:
        utils.append(simulate_recovery(G, path))

    print(utils)
    print(max(utils))
    #nx.draw(sample_graph(test_costs))
    #plt.draw()
    #plt.savefig('test.png')

if __name__ == "__main__":
    main()