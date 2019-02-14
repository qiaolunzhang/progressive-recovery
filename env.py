import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tree_recovery import get_root, merge_nodes, r_tree, plot_graph, calc_height, simulate_tree_recovery, plot_bar_x, par_max_util_configs, prune_map
from progress.bar import Bar
import multiprocessing

# TODO:
# 1. assume multiple independent nodes. Currently assuming independent nods is list of len 1

class RecoveryEnv:
    def __init__(self, G, independent_nodes):
        '''
        __init__
        :param G: networkx graph
        :param independent_nodes: independent_nodes of graph G; e.g. where we start our recovery
        '''
        self.network = G
        self.independent_nodes = independent_nodes
        self.root = self.independent_nodes[0]

    def recover(self, order, resources, include_root, debug=False, draw=False):
        '''
        Recover our network with the order given.
        :param order: |network| len list with order of nodes to recover
        :param resources: resources per time step (assumed constant)
        :param include_root: include recovering root (1st node) in the total_utility count
        :param debug: print step by step recovery order to check if correct
        :param draw: draw graph at each step of recovery
        :return: total utility
        '''
        demand = nx.get_node_attributes(self.network, 'demand')
        utils = nx.get_node_attributes(self.network, 'util')

        # keep track of what node we are currently recovering, starting from the root
        node_recovery_index = 0

        if draw:
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

        if not include_root:
            current_utility = order[0]
            node_recovery_index += 1

        # DEBUG
        # print('Root node income: ', utils[root] - demand[root])
        # print('Root node index: ', root)

        # keep a copy of our graph to plot change over time/ recovery order more intuitively
        H = self.network.copy()

        # iteration counter for saving intermediate graphs
        i = 0

        # Create the initial graph plot, noting all the positions of the nodes (pos) so that we may
        # represent it in the same orientation for sequential graphs
        if draw:
            pos = plot_graph(H, self.root, folder + '/{}.png'.format(i))

        while node_recovery_index is not len(order):

            recovery_node = order[node_recovery_index]
            if debug:
                print('Current utility: ', current_utility, 'Total utility: ', total_utility)
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
                H = merge_nodes(H, self.root, recovery_node)

                # next node to recover
                node_recovery_index += 1

                # unless we've finished the last node, in which case we increment total utility since we're done recovering completely
                if node_recovery_index is len(order):
                    total_utility += current_utility

                if draw:
                    plot_graph(H, self.root, folder + '/{0}.png'.format(i), pos)
                continue

            # otherwise, we have equal resources and demand, so apply all resources and continue
            else:
                demand[recovery_node] = 0
                current_utility += utils[recovery_node]

                # move to the next node to recover
                node_recovery_index += 1

            # now we merge the node we recovered with our root node
            H = merge_nodes(H, self.root, recovery_node)
            if draw:
                plot_graph(H, self.root, folder + '/{0}.png'.format(i), pos)

            # increment total utility
            total_utility += current_utility

        if debug:
            print('Total util for this config:', total_utility)

        return total_utility

    def optimal(self, resources, include_root=False):
        '''
        Returns the optimal total utility for self.network. May not be unique.
        :param resources: resources per time step
        :param include_root: include recovering root (1st node) in the total_utility count
        :return: optimal total utility over _ceiling{sum(demand) / resources} time steps
        '''
        # get the possible maximum utility configs
        configs = par_get_configs(self.network, [self.root])
        print(configs)
        max_total_utility = 0; max_config = None

        print(len(configs), ' maximum utility configs need to be checked.\n')
        print('Finding max util...')

        # progress bar
        bar = Bar('Simulating Configurations', max=len(configs))

        # check for greatest
        for config in configs:
            config_util = self.recover(config, resources, include_root)

            if config_util > max_total_utility:
                max_config = config
                max_total_utility = config_util
            bar.next()

        bar.finish()

        return max_total_utility, max_config


# Helper functions
# ================================================================================================== #
def deviation_from_optimal(nodes, resources, height=None):
    '''
    Construct simple example to find optimal recovery config
    :param nodes: number of nodes in the random tree
    :param resources: number of resources at each time step
    :param height: height of tree to generate (can be None)
    :return: Tuple (optimal, heuristic) of total utility over all time steps
    '''
    tree = r_tree(nodes=nodes, height=height)
    root = get_root(tree)
    plot_graph(tree, root, 'plots/sample.png')
    G = RecoveryEnv(tree, [root])

    return (G.optimal(resources, include_root=True)[0], simulate_tree_recovery(tree, resources, root))

def par_get_configs(G, independent_nodes):
    '''
    Generate and prune configurations in a parallel fashion. Needs to be at high namespace level.

    :param G: networkx graph
    :param independent_nodes: List of independent nodes
    :return: list of pruned configurations
    '''
    all_permutations = par_max_util_configs(G, independent_nodes)
    pool = multiprocessing.Pool()
    pruned = pool.map(prune_map, all_permutations)

    pruned = [list(config) for config in pruned if config != []]
    pool.close()

    return pruned