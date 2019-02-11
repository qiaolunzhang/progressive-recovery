import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tree_recovery import get_root, max_util_configs, merge_nodes, r_tree, plot_graph, calc_height, simulate_tree_recovery
from progress.bar import Bar

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

    def recover(self, order, resources, debug=False, draw=False):
        '''
        Recover our network with the order given.
        :param order: |network| len list with order of nodes to recover
        :param resources: resources per time step (assumed constant)
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

    def optimal(self, resources):
        '''
        Returns the optimal total utility for self.network. May not be unique.
        :param resources: resources per time step
        :return: optimal total utility over _ceiling{sum(demand) / resources} time steps
        '''
        # get the possible maximum utility configs
        configs = max_util_configs(self.network, resources, self.root)
        max_total_utility = 0; max_config = None

        print(len(configs), ' maximum utility configs need to be checked.\n')
        print('Finding max util...')

        # progress bar
        bar = Bar('Simulating Configurations', max=len(configs))

        # check for greatest
        for config in configs:
            config_util = self.recover(config, resources)
            if config_util > max_total_utility:
                max_config = config
                max_total_utility = config_util
            bar.next()

        bar.finish()

        return max_total_utility, max_config

def deviation_from_optimal(nodes, resources, height=None):
    # construct simple example to find optimal recovery config
    tree = r_tree(nodes=nodes, height=height)
    root = get_root(tree)
    plot_graph(tree, root, 'plots/sample.png')
    G = RecoveryEnv(tree, [root])

    return (G.optimal(resources)[0], simulate_tree_recovery(tree, resources, root))

def main():
    stats = []
    nodes = 7; resources = 2
    for x in range(100):
        stats.append(deviation_from_optimal(nodes, resources))

    print(stats)
    avg = [x[1] / x[0] for x in stats]
    print('Average percentage of optimal for {0} node graph:'.format(nodes), sum(avg)/len(avg))

if __name__ == "__main__":
    main()