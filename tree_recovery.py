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

def r_tree(nodes, height=None):
    '''
    Generates a random tree, with random utility and demand for each node

    :param nodes: Number of nodes in the tree
    :param height: (optional) produces tree with given height.
    :return: Random tree with len{V} = nodes
    '''
    G = nx.random_tree(nodes)
    if height is not None:
        while calc_height(G, get_root(G)) is not height:
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

def evaluate_total_income(H):
    '''
    Given a graph (or subgraph) H, determines the total "income" of the graph.

    :param H: networkx graph
    :return: total income of graph: the sum over all nodes of [utilities - demand]
    '''
    incomes = nx.get_node_attributes(H, 'income')
    total_income = 0
    for node, node_income in incomes.items():
        total_income += node_income

    return total_income

def merge_nodes(H, root, v):
    '''
    Merges two nodes in a given graph, returns a new one

    :param H: networkx graph
    :param root: root node to merge
    :param v: node to merge with root
    :return: new graph G, with V_H - 1 vertices and E_H or E_H - 1 edges.
    '''
    G = H.copy()
    neighbors = G.neighbors(v)
    for node in neighbors:
        if node != root:
            G.add_edge(node, root)

    G.remove_node(v)

    return G

def plot_graph(G, root, dir, pos=None):
    '''
    Plots a graph using pyplot, saves it in dir

    :param G: networkx graph
    :param root: Root node of graph. Colored red.
    :param dir: Directory to save image
    :param pos: Used to keep similar graph plot style across multiple plots. Dict of positions for each node.
    '''
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

def get_root(G):
    '''
    Finds root of tree (node with highest degree). Not necessarily unique.

    :param G: networkx Graph
    :return: root of tree
    '''
    # degrees is a list of tuples of (node, deg) sorted by degree, highest first.
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    # choose root as highest degree node (may not be unique)
    root = degrees[0][0]

    return root

def simulate_tree_recovery(G, resources, draw=False):
    '''
    Simulates recovery of a tree, starting at the root (independent) node. Assumes
    that all other nodes are dependent and therefore to optimally recover, we 
    branch out from the root. Our algorithm works by "merging" any recovered nodes
    into the root node, and re-evaluating all adjacent subtrees.
    Assumptions: Each node in the networkx graph G has the attributes:
    income, util, and demand.

    :param G: networkx graph
    :param resources: Number of resources per time step
    :param draw: If true, plot graph at each step.
    :return: root of G
    '''
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

    root = get_root(G)

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

def max_util_configs(G, resources, root):
    '''
    Calculates number of possibly optimal configs necessary to check for a given
    graph G. Prunes starting from |V_G|!.
    
    :param G: networkx graph
    :param resources: Number of resources at each time step
    :param root: root of tree G
    :return: tuple of (num possibly optimal configs, total number of configurations)
    '''
    print('Fetching all possible configurations...')
    number_of_nodes = G.number_of_nodes()
    # create all possible node recovery orders
    all_permutations = list(itertools.permutations([x for x in range(G.number_of_nodes())]))
            
    # prune configs that aren't neighbors of nodes already recovered
    pruned_configs = []

    bar = Bar('Pruning Configurations', max=len(all_permutations))

    for config in all_permutations:
        # prune all recovery configurations that don't begin with root
        bar.next()
        if config[0] != root:
            continue
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

    bar.finish()
    return pruned_configs

def calc_pruning_stats(node_range_x, node_range_y, graphs_per_range):
    '''
    Calculate how much pruning reduces the solution space by iterating through random graphs
    and marking mean, std for the amount pruned
    
    :param node_range_x: lower bound for number of nodes in graph
    :param node_range_y: upper bound for number of nodes in graph
    :param graphs_per_range: number of graphs to test for each node size graph
    :return: list of tuples, where [0] corresponds to the avg stats (avg pruned, std) for node_range_x.
    '''
    # generate 1000 random graphs, compare on average how much we can prune
    for size in range(node_range_x, node_range_y):
        bar = Bar('Processing', max=graphs_per_range)
        current_size = []
        for graph_num in range(graphs_per_range):
            G = r_tree(size)
            root = get_root(G)

            pruned, not_pruned = max_util_configs(G, resources, root)
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

def graph_pruning_stats():
    '''
    Graphs stats given by calc_pruning_stats. Wrapper function.

    :return: null
    '''
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

def calc_height(G, root):
    '''
    Calculate height of tree, longest path from root to leaf

    :param G: networkx graph
    :param root: Root of tree
    :return: height of G assuming tree.
    '''
    # dict of shortest path lengths
    path_lengths = nx.shortest_path_length(G, root)

    return max(path_lengths.values())

def plot_bar_x(data, label, dir):
    '''
    Plot 1d data as histogram with labels along x axis

    :param data: 1-D array of data to graph
    :param label: labels along x axis
    :param dir: directory to save plot
    :return: null
    '''
    index = np.arange(len(label))
    plt.bar(index, data)
    plt.xlabel('Genre', fontsize=5)
    plt.ylabel('Avg. # of Pruned Configs', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('8 Node Tree Height v. Prunable Configurations')

    plt.savefig(dir)

# DEBUG
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
    #pruned, not_pruned = max_util_configs(G, resources, root)

    # calculate height stats/relationships between height and pruning
    height_stats = [[] for x in range(1, 8)]
    for x in range(2500):
        G = r_tree(nodes)
        root = get_root(G)
        height = calc_height(G, root)
        height_stats[height].append(max_util_configs(G, resources=1, root=root))

    avg_pruning = []
    for lst in height_stats:
        if len(lst) == 0:
            avg_pruning.append(0)
            continue

        avg = 0
        for prune_pair in lst:
            avg += prune_pair[0]
        avg = avg / len(lst)
        avg_pruning.append(avg)

    print(avg_pruning)
    plot_bar_x(avg_pruning, ['height {0}'.format(x) for x in range(len(avg_pruning))], 'plots/pruning.png')

if __name__ == "__main__":
    main()