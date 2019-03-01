import networkx as nx
import copy
from tree_recovery import r_tree, get_root, DP_optimal
import math

class environment:
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
        for node in self.independent_nodes:
            self.state[node] = 1

        # max rounds is math.ceil(sum(d) / resources)
        self.round = 1
        self.resources = resources

        # True when state is vector of 1's
        self.done = False

    def step(self, action):
        '''
        Applies a partition of resources to the graph G

        :param action: |V(G)| len vector, where sum(action) == resources at a time step
        :return: state, reward, done
        '''
        utils = nx.get_node_attributes(self.G_constant, 'util')
        demand = nx.get_node_attributes(self.G_constant, 'demand')

        # apply resources to demand vector
        demand = [max(demand[x] - action[x], 0) for x in range(len(action))]

        # update state
        self.state = [1 if demand[x] == 0 or self.state[x] == 1 else 0 for x in range(len(action))]

        # G becomes subgraph of functional nodes
        self.G = self.G_constant.subgraph([x if self.state[x] == 1 else None for x in range(len(self.state))])

        # count utility only for nodes which have a path to an independent node
        count_utility = []
        for node in self.G:
            for id_node in self.independent_nodes:
                if nx.has_path(self.G, id_node, node) and id_node != node:
                    count_utility.append(node)

        # utility at this time step is reward
        reward = sum([utils[x] if x in count_utility else 0 for x in range(len(action))])

        # convert demand back to dict
        demand = dict((i, demand[i]) for i in range(len(demand)))
        # update demand values in our current graph
        nx.set_node_attributes(self.G_constant, name='demand', values=demand)

        # check if we are finished with this episode
        if self.state == [1 for x in self.state]:
            self.done = True

        # check if we have reached round limit, which is ceil(sum(demands of non-independent nodes) / resources per turn)
        independent_node_demand = [self.start_demand[x] for x in self.independent_nodes]

        if self.round >= (math.ceil((sum(self.start_demand.values()) - sum(independent_node_demand))/ self.resources)):
            self.done = True

        self.round += 1

        return self.state, reward, self.done

    def reset(self):
        '''
        Reset our state to starting state, return an initial observation

        :return: state
        '''
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
    num_nodes = 5
    G = r_tree(num_nodes)
    env = environment(G, [get_root(G)])
    print(env.step([0,0,0,0,4]))
    print(env.step([0,4,0,0,0]))
    print(env.step([0,0,4,0,0]))
    print(env.step([4,0,0,0,0]))
    print(env.step([0,0,0,4,0]))
    print(env.reset())
    print(env.step([0,0,0,4,0]))

if __name__ == "__main__":
    # main()
    None