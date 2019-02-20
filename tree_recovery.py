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
import multiprocessing

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

def update(G, demand, utils):
    nx.set_node_attributes(G, name='demand', values=demand)
    nx.set_node_attributes(G, name='util', values=utils)

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

def par_max_util_configs(G, independent_nodes):
    '''
    Parallelized version of max_util_configs. 
    
    :param G: networkx graph
    :param root: list of independent nodes in G (these don't need to be recovered)
    :return: list of possibly maximum util configurations
    '''
    number_of_nodes = G.number_of_nodes()

    # non independent nodes are nodes we need to recover; they are all nodes that are not-indepedent
    non_independent_nodes = set(range(number_of_nodes)) - set(independent_nodes)

    # create all possible node recovery orders
    all_permutations = list(itertools.permutations(non_independent_nodes))
    all_permutations = [[tuple(independent_nodes), config, G] for config in all_permutations]

    return all_permutations

def prune_map(config_graph):
    '''
    Lambda function to apply using parallel pool.map
    
    :param config_graph: [independent_nodes, config, G] where G contains the graph the config is based on, and config is a node recovery order
    :return: [] if config is not valid, or config if it is a valid recovery configuration
    '''
    independent_nodes = config_graph[0]
    config = independent_nodes + config_graph[1]
    G = config_graph[2]

    # start at the first non_independent node
    for node_index in range(1, len(config)):
        # for each index, check if it is a neighbor of at least one node already recovered
        neighbors = [n for n in G.neighbors(config[node_index])]

        # compute the intersection between the recovered nodes and the neighbors of the node
        # at the current index. If this intersection is null, it is a provably suboptimal 
        # recovery configuration, so we drop it.
        intersection = list(set(config[:node_index]).intersection(neighbors))

        # not a valid intersection i.e. does not have at least 1 neighbor to the left
        if not intersection:
            return []

    return config

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
    plt.plot(index, data)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Total utility in Percentage of Optimal')
    plt.xticks(index, label)
    plt.title('U-D Heuristic: % of Optimal (Sampled)')

    plt.savefig(dir)

def DP_optimal(G, independent_nodes, resources):
    '''
    DP algorithm calculating optimal recovery utility

    :param G: networkx graph with attributes "util" and "demand" for each node
    :param independent_nodes: already functional nodes of the problem
    :param resources: resources per turn
    :return: ordering O = [v1, v2, ..., vn] where vn = |V(G)| of nodes to recover for a star
    '''
    class hashable_set:
        def __init__(self, set):
            self.set = set

    util = nx.get_node_attributes(G, 'util')
    demand = nx.get_node_attributes(G, 'demand')
    print
    V = G.number_of_nodes()
    # note: use (V+1) in range since it is not inclusive
    vertex_set = set(range(V))
    C = resources
    Z = {}
    A = [-1 for x in range(V)]
    A[0] = 0

    for s in range(1, V):
        # generate all |s| size subsets
        s_node_subsets = list(itertools.combinations((range(V)), s))
        for X in s_node_subsets:
            q = float('-inf')
            #v_js : a set of functional nodes
            v_js = vertex_set - set(X)
            print(v_js)
            # generate list of nodes adjacent to any functional nodes
            adj_nodes = [v_i for v_i in X for v_j in v_js if G.has_edge(v_i, v_j)]
            print(adj_nodes)
            for v_i in adj_nodes:
                # take the sum of demands of the remaining nodes after the recovery of v_i (V - {v_i})
                sum_demands = sum([demand[int(v_j)] for v_j in v_js if int(v_j) != int(v_i)])
                q_ = math.ceil((util[int(v_i)] * sum_demands) / C) + Z[hashable_set((set(X) - set([v_i]))).__hash__()]

                if q_ > q:
                    q = q_
                    A[s] = X
                #endif
            #endfor
            Z[hashable_set(set(X)).__hash__()] = q
        #endfor
    #endfor
    return A

def simulate_tree_recovery(G, resources, root, include_root=False, draw=True, debug=False):
    '''
    |U - d| -- Heuristic
    Simulates recovery of a tree, starting at the root (independent) node. Assumes
    that all other nodes are dependent and therefore to optimally recover, we 
    branch out from the root. Our algorithm works by "merging" any recovered nodes
    into the root node, and re-evaluating all adjacent subtrees.
    Assumptions: Each node in the networkx graph G has the attributes:
    income, util, and demand.

    :param G: networkx graph
    :param resources: Number of resources per time step
    :param draw: If true, plot graph at each step.
    :param debug: output logs to std.out
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

    demand = nx.get_node_attributes(G, 'demand')
    utils = nx.get_node_attributes(G, 'util')
    
    # remaining resources is just r - d if we have more than we need
    # or 0 if it is a perfect multiple
    # or the first multiple of resources > demand[root] - resources e.g. for d = 5 and r = 3, ceil(5/3)*5 = 6 -> 6-5 = 1 remaining resource
    if include_root:
        current_utility += utils[root]

        if resources > demand[root]:
            remaining_resources = resources - demand[root]
        elif demand[root] % resources == 0:
            remaining_resources = 0
            total_utility += current_utility
        else:
            remaining_resources = math.ceil(demand[root]/resources)*resources - demand[root]

        if debug:
            print('Remaining resources: ', remaining_resources)

    # if we don't want to include root in our total_utility count
    else:
        remaining_resources = 0

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
        H = update(H, demand, utils)

        if debug:
            print('Current utility: ', current_utility, 'Total utility: ', total_utility)

        # Dict of possible recovery nodes and their associated eval incomes
        possible_recovery = {}

        # find and iterate through all nodes adjacent to the root
        neighbors = H.neighbors(root)
        for neighbor in neighbors:
            # print(neighbor, [utils[neighbor], demand[neighbor]])

            # first create an unconnected component
            test_graph = H.copy()
            test_graph.remove_edge(root, neighbor)

            # Now get the nodes of the subgraph in the unconnected component we just created
            # (Not including the root)
            subgraph_nodes = nx.node_connected_component(test_graph, neighbor)
            subgraph_income = evaluate_total_income(test_graph.subgraph(subgraph_nodes))

            # update our possible move list with the income of the move if we recover this node
            possible_recovery.update({neighbor: subgraph_income})

        if debug:
            print(possible_recovery)
        i += 1

        # choose the best move (look how pythonic this is)
        recovery_nodes = [key for key in possible_recovery if possible_recovery[key] == max(possible_recovery.values())]
        
        if debug:
            print(recovery_nodes)

        # if multiple max nodes, choose the one with the best 'recovery ratio'
        max_ratio = 0; recovery_node = recovery_nodes[0]
        for node in recovery_nodes:
            ratio = utils[node] / demand[node]
            if ratio > max_ratio:
                max_ratio = ratio
                recovery_node = node

        if debug:
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

            # unless we're at the last step:
            if H.number_of_nodes() == 1:
                total_utility += current_utility
                break

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

    return total_utility
