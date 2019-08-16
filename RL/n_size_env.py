import networkx as nx
import numpy as np
import math
import copy
from graph_helper import r_tree, get_root, DP_optimal, plot_graph
import random


# TODO: Fully test. Have not used this yet.
# This is another version of rl_environment which only has n-actions. Instead of "allocating resources" at each time
# step, we simply choose what node to bring online. Therefore, each episode will only last n steps, given a graph
# G with n nodes.
class n_environment:
    def __init__(self, G, independent_nodes, resources):
        '''
        :param G: networkx graph with utility and demand attribute set for each node
        :param independent_nodes: Initial independent nodes of G
        :param resources: resources per recovery step (used in calculation of maximum rounds)
        '''
        self.G = G
        self.number_of_nodes = G.number_of_nodes()

        # stays constant across episodes so when we reset we do it cleanly
        self.G_constant = copy.deepcopy(G)
        self.independent_nodes = independent_nodes

        self.start_demand = nx.get_node_attributes(self.G_constant, 'demand')

        # state is an indicator matrix for each node in G. 0 -> node is offline
        # initially, every node is except for independent nodes
        self.state = [0 for x in range(self.number_of_nodes)]
        print('independent nodes:', self.independent_nodes)
        for node in self.independent_nodes:
            self.state[node] = 1

        # max rounds is math.ceil(sum(d) / resources)
        self.round = 1
        self.resources = resources

        # excess reward save
        self.excess_resources = 0

        # True when state is vector of 1's
        self.done = False

    def random_action(self):
        """
        Return a random action from our current state.
        :return: random action (scalar)
        """
        not_recovered = [x if self.state[x] is 0 else None for x in range(self.number_of_nodes)]

        return np.random.choice(not_recovered)

    def step(self, action):
        """
        Take a step in our environment.
        :param action: scalar corresponding to node to be recovered
        :return: 3-tuple of (new_state, reward, done)
        """
        demand = nx.get_node_attributes(self.G_constant, 'demand')
        util = nx.get_node_attributes(self.G_constant, 'util')

        # apply excess resources
        demand[action] -= self.excess_resources
        rounds_to_recover = math.ceil(demand[action] / self.resources)
        # now our new excess will be whatever we have left over after recovering our cur node
        self.excess_resources = demand[action] % self.resources
        demand[action] = 0

        # Compute the subgraph before and after bringing the next node online.
        # Before: we count utility of all functional nodes for rounds_to_recover - 1 nodes
        self.G = self.G_constant.subgraph([x if self.state[x] == 1 else None for x in range(len(self.state))])

        # count utility only for nodes which have a path to an independent node
        count_utility = []
        for node in self.G:
            for id_node in self.independent_nodes:
                if nx.has_path(self.G, id_node, node) and id_node != node:
                    count_utility.append(node)

        util_before = (rounds_to_recover - 1) * sum(
            [util[x] if x in count_utility else 0 for x in range(self.number_of_nodes)])

        # update state, calculate the "after" util
        # print(demand)
        self.state = [1 if demand[x] == 0 or self.state[x] == 1 else 0 for x in range(self.number_of_nodes)]
        print(self.state)

        # G becomes subgraph of functional nodes
        self.G = self.G_constant.subgraph([x if self.state[x] == 1 else None for x in range(len(self.state))])

        # count utility only for nodes which have a path to an independent node
        count_utility = []
        for node in self.G:
            for id_node in self.independent_nodes:
                if nx.has_path(self.G, id_node, node) and id_node != node:
                    count_utility.append(node)

        util_after = sum([util[x] if x in count_utility else 0 for x in range(self.number_of_nodes)])
        reward = util_after + util_before

        # check if we are done
        if sum(self.state) == self.number_of_nodes:
            self.done = True

        # check if we have reached round limit, which is:
        # ceil(sum(demands of non-independent nodes) / resources per turn)
        independent_node_demand = [self.start_demand[x] for x in self.independent_nodes]

        if self.round >= (math.ceil((sum(self.start_demand.values()) - sum(independent_node_demand)) / self.resources)):
            self.done = True

        self.round += rounds_to_recover

        # update demand values in our current graph
        nx.set_node_attributes(self.G_constant, name='demand', values=demand)
        new_state = [demand[x] for x in range(self.number_of_nodes)]

        return new_state, reward, self.done

    def reset(self):
        """
        Reset our state to starting state, return the initial observation

        :return: initial state, 'False' done boolean
        """
        self.G = copy.deepcopy(self.G_constant)
        self.state = [0 for x in range(self.number_of_nodes)]
        for node in self.independent_nodes:
            self.state[node] = 1

        # reset demands, we don't modify utils
        nx.set_node_attributes(self.G_constant, name='demand', values=self.start_demand)

        # True when state is vector of 1's
        self.done = False
        self.round = 1

        return self.state, self.done


def main():
    # test
    num_nodes = 8
    G = r_tree(num_nodes)
    plot_graph(G, 0, 'environment_debug_graph.png')
    env = n_environment(G, [0], 1)
    for x in range(1, num_nodes):
        print(env.step(x))

    env.reset()
    print('Reset env =========================')
    while not env.done:
        print(env.step(random.randint(0, num_nodes - 1)))
        print()


if __name__ == "__main__":
    main()
