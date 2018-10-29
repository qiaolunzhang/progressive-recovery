import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from prog import iterate_over_failures, root_to_leaves

def read_gml(path):
    G = nx.read_gml(path)
    return G

# given a graph G and the cost to recover a node in G (costs), 
# we apply a recovery confiugration (config) and return the total system utility
def simulate_recovery(G, costs, config):
	return

def main():
    G = read_gml('gml/DIGEX.gml')

    test_costs = [2, 2, 2, 2, 0, 0]
    root = iterate_over_failures(test_costs, 1) 
    all_paths = root_to_leaves(root)
    print(all_paths)
    print(len(all_paths))

    nx.draw(G)
    plt.draw()
    plt.savefig('test.png')

if __name__ == "__main__":
    main()
