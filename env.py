import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Define an environment graph class for reinforcement learning implementations
class Environment:
    def __init__(self, G, independent_nodes, ):
        self.network = G
        self.ended = False
    
    def apply_resources(self, node, amount):

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
    
    def c