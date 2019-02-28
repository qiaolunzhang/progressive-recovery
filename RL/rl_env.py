import networkx as nx
import copy

class rl_env:
    def __init__(self, G, independent_nodes, number_of_nodes):
        '''
        :param G: networkx graph with utility and demand attribute set for each node
        :param independent_nodes: Functional nodes of graph
        '''
        self.G = copy.deepcopy(G)
        self.number_of_nodes = number_of_nodes

        # stays constant across episodes so when we reset we do it cleanly
        self.G_constant = copy.deepcopy(G)
        self.independent_nodes = independent_nodes

        # state is an indicator matrix for each node in G. 0 -> node is offline
        # initially, every node is except for independent nodes
        self.state = [0 for x in range(self.number_of_nodes)]
        for node in self.independent_nodes:
            self.state[node] = 1

        # True when state is vector of 1's
        self.done = False

    def step(self, action):
        '''
        Applies a partition of resources to the graph G

        :param action: |V(G)| len vector, where sum(action) == resources at a time step
        :return: state, reward, done
        '''
        utils = nx.get_node_attributes(self.G, 'util')
        demand = nx.get_node_attributes(self.G, 'demand')

        demand = [min(demand[x] - action[x], 0) for x in range(len(action))]
        self.state = [1 if demand[x] == 0 else 0 for x in range(len(action))]

        # update demand values in our current graph
        nx.set_node_attributes(self.G, name='demand', values=demand)

        # utility at this time step is reward
        reward = sum([utils[x] * demand[x] for x in range(len(action))])

        # check if we are finished with this episode
        if self.state == [1 for x in self.state]:
            self.done = True

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

        # True when state is vector of 1's
        self.done = False

        return self.state