import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from graph_helper import plot_graph, calc_height, simulate_tree_recovery, plot_bar_x, r_tree, get_root, merge_nodes, r_graph, DP_optimal

# TODO:
# 1. Test multiple independent nodes for optimality (we are only comparing against U-D heuristic
# for now)


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

    def recover(self, order, resources, include_independent_nodes=False, debug=False, draw=False):
        '''
        Recover our network with the order given.
        :param order: |network| len list with order of nodes to recover
        :param resources: resources per time step (assumed constant)
        :param include_independent_nodes: include recovering independent nodes in the total_utility count
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

        if not include_independent_nodes:
            node_recovery_index += len(self.independent_nodes)

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


def ratio_heuristic(G, independent_nodes, resources):
    """
    Given a graph with attributes utility and demand for each node, we calculate the best greedy stepwise
    recovery based on the ratio of utility to demand. We return the total utility of this recovery.

    :param G: networkx graph G, with attributes utility and demand for each node
    :param independent_nodes: independent nodes of graph G
    :param resources: resources per recovery time step
    :return: total utility of the ordered recovery sequence, excluding independent nodes
    """
    # set starting functional nodes to the given independent nodes
    functional_nodes = independent_nodes.copy()
    util = nx.get_node_attributes(G, 'util')
    demand = nx.get_node_attributes(G, 'demand')

    print('Connected components', nx.number_connected_components(G))
    # we always start our recovery order with independent nodes
    ordered = functional_nodes.copy()
    while len(functional_nodes) is not G.number_of_nodes():
        # get a list of lists of neighbors for each functional node
        adj = [list(G.neighbors(func_node)) for func_node in functional_nodes]

        # flatten list
        adj_nodes = []
        for node_list in adj:
            for node in node_list:
                adj_nodes.append(node)

        # remove functional nodes (already recovered) from possible recovery nodes
        adj_nodes = list(set(adj_nodes) - set(functional_nodes))
        
        # choose the node with the best util/demand ratio
        ratios = {node: (util[node] / demand[node]) for node in adj_nodes}

        ordered.append(max(ratios, key=ratios.get))

        # add node we just recovered to functional nodes
        functional_nodes.append(ordered[-1])

    # use recoveryenv to check the total utility of the ordering
    env = RecoveryEnv(G, independent_nodes)
    total_utility = env.recover(ordered, resources)


    return total_utility


def main():
    for x in range(20):
        graph = r_graph(n=12, edge_prob=0.2)
        root = get_root(graph)
        print(ratio_heuristic(graph, [root], 1))
        print(DP_optimal(graph, [root], 1))

if __name__ == '__main__':
    main()
