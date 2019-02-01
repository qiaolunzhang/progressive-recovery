import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Define an environment graph class for reinforcement learning and algorithm stuff
class Environment:
    def __init__(self, G, root):
        self.network = G
        self.root = root
        self.done = False
        self.total_utility = 0
        self.demand = nx.get_node_attributes(self.network, 'demand')
        self.utils = nx.get_node_attributes(self.network, 'util')

    def apply_resources(self, node, r):
        if self.demand[node] - r <= 0:
            
        self.demand[node]
    def is_active(self, node):
        x = LENGTH
        for i in range(len(nodes)):
            if nodes[i] == node:
                x = i
        return self.network[x]
    
    def reward(self):
        if not self.ended:
            r = self.calc_reward()
            return r
        return LENGTH
    
    def 