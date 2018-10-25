import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def read_gml(path):
    G = nx.read_gml(path)
    return G

# returns true if every element in an arr is zero
def all_zeros(arr):
    zeros = True
    for x in arr:
        if x != 0:
            zeros = False
            break

    return zeros

# Recursively generate all the possible partitions with [total] objects
# and [groups] partitions. Returns a list of list of integers.
def possible_allocs(groups, total):
    if (groups == 1):
        return [[total]];
    else:
        nums = range(total + 1);
        # looping through all values for one of the partitions.
        container1 = [];
        for i in nums:
            # recursive step - generate all combinations without the first 
            # partition
            subset = possible_allocs(groups - 1, total - i);
            # append the first partition onto each element of this list
            container2 = [];
            for l in subset:
                container2 += [([i] + l)];
            container1 += [container2];
        # Flatten just takes a list of lists, and extract everything in each
        # list and mesh everything together.
        return [item for sublist in container1 for item in sublist];

# Given a recovery cost vec and configuration, returns true if we don't overapply resources
# ie. when we apply [3, 0, 0] to the recovery cost [2, 3, 5], returns false because 3 > 2 at v1.
def non_neg(recovery_costs, config):
    iteration = recovery_costs
    alloc_resources = [iteration[i] - config[i] for i in range(len(config))]
    for e in alloc_resources:
        if e < 0:
            return False

    return True

def apply_config(recovery_cost, config):
    return [recovery_cost[i] - config[i] for i in range(len(config))]

class Node:
    def __init__(self, vec):
        self.vec = vec
        self.children = []

# given a vector [v1, v2, ..., vn] of vertices, where v1 corresponds to
# the amount of resources needed for v1 to come back online, we return
# every possible recovery configuration at a single time t given resources r.
def iterate_over_failures(recovery_costs, r):
    if all_zeros(recovery_costs):
        leaf = Node(-1)
        leaf.children = []
        return [leaf]

    # all recovery configurations, not all my be possible
    recovery_configurations = possible_allocs(len(recovery_costs), r)
    
    # we prune the configurations to only those that don't over allocate resources
    pruned_configs = [config for config in recovery_configurations if non_neg(recovery_costs, config)]
   
    children = []
    # now descend down the recursive tree for each pruned config
    for config in pruned_configs:
        # apply config, we know it will be a valid application since it is a pruned config
        new_recovery_cost = apply_config(recovery_costs, config)
        config_node = Node(config)
        children.append(config_node)
        config_node.children = iterate_over_failures(new_recovery_cost, r)
        #print(config_node.children)

    return children

# given a root node, return all possible paths to leaves
def root_to_leaves(root):
    # so we don't print out root, we handle it separately
    for child in root.children:
        recurse(child, [], 0)

def recurse(node, path, pathLen):
    if node.vec is -1:
        return

    # python has no implicit pass by value, so we emulate it by creating 
    # a new, identical list
    newPath = path[:]
    newPath.append(node.vec)
    pathLen += 1

    if len(node.children) == 1 and node.children[0].vec == -1:
        for j in range(pathLen):
            print(newPath[j], end=' ')
        print()
        return

    else:
        for child in node.children:
            recurse(child, newPath, pathLen)
        return

def main():
    test_costs = [5, 3, 1]
    all_configs = iterate_over_failures(test_costs, 3)
    root = Node(test_costs)
    root.children = all_configs

    root_to_leaves(root)
    '''
    G = read_gml('gml/DIGEX.gml')
    nx.draw(G)
    plt.draw()
    plt.savefig('test.png')
    '''

if __name__ == "__main__":
    main()
