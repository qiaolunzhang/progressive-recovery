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
    
    def calc_reward(self):
        if self.network[node_ids["s"]] == True: #for "s"
            G = nx.Graph()
            for node in nodes:
                if self.network[node_ids[node]] == True:
                    G.add_node(node)
            for edge in edges:
                if self.network[node_ids[edge[0]]] == True and self.network[node_ids[edge[1]]] == True:
                    G.add_edge(*edge)
                elif self.network[node_ids[edge[1]]] == True and self.network[node_ids[edge[0]]] == True:
                    G.add_edge(*edge)
            for component in nx.connected_components(G):
                if "s" in component:
                    return float(len(component))
        else:
            return 0


    def get_state(self):
        k = 0
        h = 0
        for i in range(len(nodes)):
            if self.network[i]== True:
                v = 1
            elif self.network[i] == False:
                v = 0
            h += (2**k) * v
            k += 1
        return h
    
    def game_over(self, force_recalculate=False):
    # returns true if game over (a player has won or it's a draw)
    # otherwise returns false
    # also sets 'winner' instance variable and 'ended' instance variable
        if not force_recalculate and self.ended:
            return self.ended
        
        for i in range(len(nodes)):
            if self.network[i] == False:
                self.ended = False
                return False

        self.ended = True
        return True