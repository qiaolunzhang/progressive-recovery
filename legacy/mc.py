from __future__ import print_function, division
from builtins import range, input

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

LENGTH = 6
nodes = ("s", "a", "b", "c", "d", "e")
node_ids = {}
j = 0
for node in nodes:
    node_ids.update({node:j})
    j += 1

edges = (("s","a"), ("a","c"), ("s","b"), ("b","d"), ("d","e"))

class Agent:
  def __init__(self, eps=0.17, alpha=0.5):
    self.eps = eps # probability of choosing random action instead of greedy
    self.alpha = alpha # learning rate
    self.verbose = False
    self.state_history = []

  def setV(self, V):
    self.V = V

  def set_verbose(self, v):
    # if true, will print values for each position on the board
    self.verbose = v

  def reset_history(self):
    self.state_history = []

  def take_action(self, env, display=False):
    # choose an action based on epsilon-greedy strategy
    r = np.random.rand()
    best_state = None
    if r < self.eps:
      # take a random action
      if self.verbose:
        print("Taking a random action")

      possible_moves = []
      for i in range(len(nodes)):
          if not env.is_active(nodes[i]):
              possible_moves.append(i)
      idx = np.random.choice(len(possible_moves))
      next_move = possible_moves[idx]
    else:
      # choose the best action based on current values of states
      # loop through all possible moves, get their values
      # keep track of the best value
      pos2value = {} # for debugging
      next_move = None
      best_value = -1
      for i in range(len(nodes)):
          if not env.is_active(nodes[i]):
            # what is the state if we made this move?
            env.network[i] = True
            state = env.get_state()
            env.network[i] = False # don't forget to change it back!
            pos2value[nodes[i]] = self.V[state]
            if self.V[state] > best_value:
              best_value = self.V[state]
              best_state = state
              next_move = i 

    # make the move
    if display:
        print("next_move:", nodes[next_move])
    env.network[next_move] = True

  def update_state_history(self, s):
    self.state_history.append(s)

  def update(self, env):
    # V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
    # where V(next_state) = reward if it's the most current state
    reward = env.reward()
    target = reward
    for prev in reversed(self.state_history):
      value = self.V[prev] + self.alpha*(target - self.V[prev])
      self.V[prev] = value
      target = value
    self.reset_history()


# this class represents a tic-tac-toe game
# is a CS101-type of project
class Environment:
    def __init__(self):
        self.network = np.full(LENGTH, False)
        self.ended = False
        self.num_states = 2**LENGTH
    
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


# recursive function that will return all
# possible states (as ints) and who the corresponding winner is for those states (if any)
# (i, j) refers to the next cell on the board to permute (we need to try -1, 0, 1)
def get_state_hash(env, i=0):
    results = []
    for condition in (True, False):
        env.network[i] = condition 
        if i == LENGTH-1:
            state = env.get_state()
            print("state ID:", state, " vector:", env.network)
            ended = env.game_over(force_recalculate = True)
            results.append((state, ended))
        else:
            results += get_state_hash(env, i+1)
    return results


def initialV_x(env, state_winner_triples):
    V = np.zeros(env.num_states)
    for state, ended in state_winner_triples:
        v = 0.5*LENGTH
        if ended:
            v = LENGTH
        V[state] = v
    return V


def play_game(player, env, display=False):
  # loops until the game is ove
  while not env.game_over():
      # current player makes a move
      player.take_action(env, display)
      
      # update state histories
      state = env.get_state()
      player.update_state_history(state)

      if False:
        print("state:", state)
        print("state's value:", player.V[state])
      
      # do the value function update
  player.update(env)


if __name__ == '__main__':
  # train the agent
  player = Agent()

  # set initial V for p1 and p2
  env = Environment()
  state_pairs = get_state_hash(env)

  Vx = initialV_x(env, state_pairs)
  player.setV(Vx)

  T = 20001
  print(player.V)
  for t in range(T):
      display=False
      if t % 2000 == 0:
          print("Game", t)
          display = True
          #for state in state_pairs:
              #print(player.V)
      play_game(player, Environment(), display)
  print(player.V)