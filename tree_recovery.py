import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import os, shutil

# Generates a random tree, with random utility and demand for each node
# Note: we don't use random_tree for this function name since that is
# a networkx function call.
def r_tree(nodes, draw):
    G = nx.random_tree(nodes)
    utils = {}
    demand = {}

    # for a given node, value = util - demand
    value = {}

    # random utils and demand for each node
    for node in G.nodes:
        utils.update({node: random.randint(1, 4)})
        demand.update({node: random.randint(1, 2)})
        value.update({node: utils[node] - demand[node]})

    nx.set_node_attributes(G, name='util', values=utils)
    nx.set_node_attributes(G, name='demand', values=demand)
    nx.set_node_attributes(G, name='value', values=value)

    # if user wants, plot graph
    # plot_graph(G, 'test.png')

    return G

# Given a graph (or subgraph) H, determines the total "value" of the graph.
# That is, returns the sum over all nodes of [utilities - demand]
def evaluate_total_value(H):
    values = nx.get_node_attributes(H, 'value')
    total_value = 0
    for node, node_value in values.items():
        total_value += node_value

    return total_value

# merges two nodes in a given graph
def merge_nodes(H, root, v):
    G = H.copy()
    neighbors = G.neighbors(v)
    for node in neighbors:
        if node != root:
            G.add_edge(node, root)

    G.remove_node(v)

    return G

def plot_graph(G, dir):
    value = nx.get_node_attributes(G, 'value')
    nx.draw(G, with_labels=True, labels=value, node_size=1000)
    plt.draw()
    plt.savefig(dir)

    # clear the plt to not retain the graph next time we wanna plot
    plt.clf()

# Simulates recover of a tree, starting at the root (independent) node. Assumes
# that all other nodes are dependent and therefore to optimally recover, we 
# branch out from the root. Our algorithm works by "merging" any recovered nodes
# into the root node, and re-evaluating all adjacent subtrees.
# ===========================
# Assumptions: Each node in the networkx graph G has the attributes:
# value, util, and demand. Resources: amount of resources per recovery iteration.
def simulate_tree_recovery(G, resources):
    # clean image dir
    folder = 'trees'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    # current and total utility are both 0 to begin
    current_utility = 0
    total_utility = 0
    remaining_resources = 0

    demand = nx.get_node_attributes(G, 'demand')
    utils = nx.get_node_attributes(G, 'util')

    # degrees is a list of tuples of (node, deg) sorted by degree, highest first.
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    # choose root as highest degree node (may not be unique)
    root = degrees[0][0]

    print('Root node value: ', utils[root] - demand[root])
    print('Root node index: ', root)

    # must recover root first, no matter how long it takes. Start measuring total
    # utility after applying the first round of resources _after_ recovering root.
    H = G.copy()
    # iteration counter for saving intermediate graphs
    i = 0
    plot_graph(H, folder + '/{}.png'.format(i))
    while H.number_of_nodes() > 1:
        print('Current utility: ', current_utility, 'Total utility: ', total_utility)
        neighbors = H.neighbors(root)
        possible_recovery = {}
        for neighbor in neighbors:
            # first create an unconnected component
            H.remove_edge(root, neighbor)
            # Now get the nodes of the subgraph in the unconnected component we just created
            # (Not including the root)
            subgraph_nodes = nx.node_connected_component(H, neighbor)
            subgraph_value = evaluate_total_value(H.subgraph(subgraph_nodes))

            # update our possible move list with the value of the move if we recover this node
            possible_recovery.update({neighbor: subgraph_value})
            # now restore the edge we removed
            H.add_edge(root, neighbor)

        print(possible_recovery)
        i += 1

        # choose the best move (look how pythonic this is)
        recovery_node = max(possible_recovery.items(), key=operator.itemgetter(1))[0]
        print('Recovering node: ', recovery_node)

        if remaining_resources != 0:
            resources_this_turn = remaining_resources
            remaining_resources = 0
        else:
            resources_this_turn = resources

        # if our demand is greater than the resources at this time step, apply all resources
        # and continue to the next step
        if demand[recovery_node] > resources_this_turn:
            demand[recovery_node] -= resources_this_turn
            total_utility += current_utility
            continue

        # in this case, we don't increment total utility yet because we still have resources leftover
        elif demand[recovery_node] < resources_this_turn:
            remaining_resources = resources_this_turn - demand[recovery_node]
            demand[recovery_node] = 0
            current_utility += utils[recovery_node]
            H = merge_nodes(H, root, recovery_node)
            plot_graph(H, folder + '/{0}.png'.format(i))
            continue

        # otherwise, we have equal resources as demand at that node
        else:
            demand[recovery_node] = 0
            current_utility += utils[recovery_node]

        # now we merge the node we recovered with our root node
        H = merge_nodes(H, root, recovery_node)
        plot_graph(H, folder + '/{0}.png'.format(i))

        # increment total utility
        total_utility += current_utility

    print(total_utility)


def main():
    # Number of nodes in the random tree
    nodes = 7; draw = True
    G = r_tree(nodes, draw)
    
    # debug printing node util and demand values
    utils = nx.get_node_attributes(G, 'util')
    demand = nx.get_node_attributes(G, 'demand')
    for node in G.nodes:
        print(node, utils[node], demand[node])

    simulate_tree_recovery(G, 1)

if __name__ == "__main__":
    main()