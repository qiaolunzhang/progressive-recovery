import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import os, shutil
import itertools
import time
import sys
from progress.bar import Bar
import math

# Generates a random tree, with random utility and demand for each node
def r_tree(nodes):
    G = nx.random_tree(nodes)
    utils = {}
    demand = {}

    # for a given node, income = util - demand
    income = {}

    # random utils and demand for each node
    for node in G.nodes:
        utils.update({node: random.randint(1, 4)})
        demand.update({node: random.randint(1, 2)})
        income.update({node: utils[node] - demand[node]})

    nx.set_node_attributes(G, name='util', values=utils)
    nx.set_node_attributes(G, name='demand', values=demand)
    nx.set_node_attributes(G, name='income', values=income)

    return G

# Given a graph (or subgraph) H, determines the total "income" of the graph.
# That is, returns the sum over all nodes of [utilities - demand]
def evaluate_total_income(H):
    incomes = nx.get_node_attributes(H, 'income')
    total_income = 0
    for node, node_income in incomes.items():
        total_income += node_income

    return total_income

# merges two nodes in a given graph
def merge_nodes(H, root, v):
    G = H.copy()
    neighbors = G.neighbors(v)
    for node in neighbors:
        if node != root:
            G.add_edge(node, root)

    G.remove_node(v)

    return G

# plots a graph with some features, saves it in dir
def plot_graph(G, root, dir, pos=None):
    utils = nx.get_node_attributes(G, 'util')
    demand = nx.get_node_attributes(G, 'demand')

    # Create a [utils, demand] label for each node
    labels = {}
    for (k,v), (k2,v2) in zip(utils.items(), demand.items()):
        labels[k] = ['n{0}'.format(k), v, v2]

    # color the root node red, all other nodes green
    color = []
    for node in G:
        if node != root:
            color.append('green')
        else:
            color.append('red')
    
    # fix the position to be consistent across all graphs
    if pos is None:
        pos = nx.spring_layout(G)
    
    nx.draw(G, with_labels=True, labels=labels, node_size=1500, node_color=color, pos=pos)

    plt.draw()
    plt.savefig(dir)

    # clear the plt to not retain the graph next time we wanna plot
    plt.clf()

    return pos

# Simulates recover of a tree, starting at the root (independent) node. Assumes
# that all other nodes are dependent and therefore to optimally recover, we 
# branch out from the root. Our algorithm works by "merging" any recovered nodes
# into the root node, and re-evaluating all adjacent subtrees.
# ===============================================================================
# Assumptions: Each node in the networkx graph G has the attributes:
# income, util, and demand. Resources: amount of resources per recovery iteration.
# returns the root (starting independent node) of recovery
def simulate_tree_recovery(G, resources, draw=False):
    # clean image dir
    folder = 'plots/trees'
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

    # DEBUG
    # print('Root node income: ', utils[root] - demand[root])
    # print('Root node index: ', root)

    # must recover root first, no matter how long it takes. Start measuring total
    # utility after applying the first round of resources _after_ recovering root.
    H = G.copy()

    # iteration counter for saving intermediate graphs
    i = 0

    # Create the initial graph, noting all the positions of the nodes (pos) so that we may
    # represent it in the same orientation for sequential graphs
    if draw:
        pos = plot_graph(H, root, folder + '/{}.png'.format(i))

    while H.number_of_nodes() > 1:
        print('Current utility: ', current_utility, 'Total utility: ', total_utility)

        # Dict of possible recovery nodes and their associated eval incomes
        possible_recovery = {}

        # find and iterate through all nodes adjacent to the root
        neighbors = H.neighbors(root)
        for neighbor in neighbors:
            # DEBUG
            #print(neighbor, [utils[neighbor], demand[neighbor]])

            # first create an unconnected component
            test_graph = H.copy()
            test_graph.remove_edge(root, neighbor)

            # Now get the nodes of the subgraph in the unconnected component we just created
            # (Not including the root)
            subgraph_nodes = nx.node_connected_component(test_graph, neighbor)
            subgraph_income = evaluate_total_income(test_graph.subgraph(subgraph_nodes))

            # update our possible move list with the income of the move if we recover this node
            possible_recovery.update({neighbor: subgraph_income})

        print(possible_recovery)
        i += 1

        # choose the best move (look how pythonic this is)
        recovery_node = max(possible_recovery.items(), key=operator.itemgetter(1))[0]
        print('Recovering node: ', recovery_node, [utils[recovery_node], demand[recovery_node]])

        # Use all remaining resources if we had some from the previous turn, otherwise
        # the amount of resources we are allocated this turn is our constant resource income
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

        # If demand < supply this turn, we don't increment total utility yet because we still have resources leftover
        # We recover that node, and note our remaining resources for the next turn
        elif demand[recovery_node] < resources_this_turn:
            remaining_resources = resources_this_turn - demand[recovery_node]
            demand[recovery_node] = 0
            current_utility += utils[recovery_node]
            H = merge_nodes(H, root, recovery_node)
            if draw:
                plot_graph(H, root, folder + '/{0}.png'.format(i), pos)
            continue

        # otherwise, we have equal resources and demand, so apply all resources and continue
        else:
            demand[recovery_node] = 0
            current_utility += utils[recovery_node]

        # now we merge the node we recovered with our root node
        H = merge_nodes(H, root, recovery_node)
        if draw:
            plot_graph(H, root, folder + '/{0}.png'.format(i), pos)

        # increment total utility
        total_utility += current_utility

    print(total_utility)
    return root

# calculates the maximum utility possible from recovering a tree
def max_util(G, resources, root):
    number_of_nodes = G.number_of_nodes()
    # create all possible node recovery orders
    recovery_permutations = list(itertools.permutations([x for x in range(G.number_of_nodes())]))
    # prune all recovery configurations that don't begin with root
    recovery_permutations = [config for config in recovery_permutations if config[0] == root]
    # prune configs that aren't neighbors of nodes already recovered
    pruned_configs = []
    for config in recovery_permutations:
        false_config = False
        # iterate through all indices in that particular config
        for node_index in range(1, len(config)):
            # for each index, check if it is a neighbor of at least one node already recovered
            neighbors = [n for n in G.neighbors(config[node_index])]
            # compute the intersection between the recovered nodes and the neighbors of the node
            # at the current index. If this intersection is null, it is a provably suboptimal 
            # recovery configuration, so we drop it.
            intersection = list(set(config[:node_index]).intersection(neighbors))
            if not intersection:
                false_config = True
                break

        # if every such left-node-slice is not sub-optimal, we add it to the set
        if not false_config:   
            pruned_configs.append(config)

    return (len(pruned_configs), len(recovery_permutations))

# Calculate how much pruning reduces the solution space by iterating through random graphs
# and marking mean, std for the amount pruned
def calc_pruning_stats(node_range_x, node_range_y, graphs_per_range):
    average_pruning = []
    toolbar_width = graphs_per_range
    resources = 1

    # generate 1000 random graphs, compare on average how much we can prune
    for size in range(node_range_x, node_range_y):
        bar = Bar('Processing', max=graphs_per_range)
        current_size = []
        for graph_num in range(graphs_per_range):
            G = r_tree(size)
            # degrees is a list of tuples of (node, deg) sorted by degree, highest first.
            degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
            # choose root as highest degree node (may not be unique)
            root = degrees[0][0]

            pruned, not_pruned = max_util(G, resources, root)
            current_size.append(not_pruned - pruned)
            bar.next()
        stats_tuple = ((np.mean(current_size), np.std(current_size)))
        average_pruning.append(stats_tuple)
        bar.finish()
        unpruned = math.factorial(size)
        print('Average reduction percentage:', stats_tuple[0] / unpruned)
        print('unpruned', unpruned, '\naverage node reduction:', stats_tuple[0], '\nwith std', stats_tuple[1])
        print(size, 'node complete\n')

    return average_pruning

def main():
    # Number of nodes in the random tree
    nodes = 8; draw = True; resources = 1;
    G = r_tree(nodes)
    
    # # debug printing node util and demand values
    # utils = nx.get_node_attributes(G, 'util')
    # demand = nx.get_node_attributes(G, 'demand')
    # for node in G.nodes:
    #     print(node, utils[node], demand[node])

    #root = simulate_tree_recovery(G, resources, draw)
    #pruned, not_pruned = max_util(G, resources, root)

    start_size = 5; end_size = 8;
    average_pruning = calc_pruning_stats(start_size, end_size, 1000)

    labels = ['{0} Nodes'.format(x) for x in range(start_size, end_size)]
    average = np.array([d[0] for d in average_pruning])
    std = np.array([d[1] for d in average_pruning])
    iter_ = 0
    for x in range(start_size, end_size):
        average[iter_] = average[iter_] / math.factorial(x)
        std[iter_] = std[iter_] / math.factorial(x)
        iter_ += 1

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(labels)), average, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)

    ax.set_ylabel('% of Recovery Configurations Pruned \n (Start with n! configurations)')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title('Average Pruning for Random Trees of Size N')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()

    plt.savefig('plots/pruning_data.png')


if __name__ == "__main__":
    main()