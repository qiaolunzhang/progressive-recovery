import time

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

    # check for the special case where the resources don't divide the sum of the costs
    # in this case, at the final step there will be a single non-negative value (rest will be zero)
    negative = 0
    zeros = 0

    for e in alloc_resources:
        if e < 0:
            negative += 1
            # return false since we shouldn't over-allocate
        if e == 0:
            zeros += 1

    # if all our resource costs are <= 0, this is the last config in a config sequence
    if negative + zeros == len(alloc_resources):
        return True

    # otherwise we prune this config
    if negative > 0:
        return False

    return True

# Return a new recovery_cost vector, with the config vector applied. Assumes the 
# config vector is valid for the current recovery_cost (i.e. non_neg is true)
def apply_config(recovery_cost, config):
    return [recovery_cost[i] - config[i] for i in range(len(config))]

# Helper class for creating a tree of possible recovery outcomes
class Node:
    def __init__(self, vec):
        self.vec = vec
        self.children = []

# wrapper for root recursion
def iterate_over_failures(recovery_costs, r):
    all_configs = iterate_over_failures_helper(recovery_costs, r)
    root = Node(recovery_costs)
    root.children = all_configs

    return root

# Returns a recursively defined tree of possible, pruned recovery configurations
def iterate_over_failures_helper(recovery_costs, r):
    if all_zeros(recovery_costs):
        leaf = Node(-1)
        leaf.children = []
        return [leaf]

    # all recovery configurations, not all my be possible
    recovery_configurations = possible_allocs(len(recovery_costs), r)
    
    # we prune the configurations to only those that don't over allocate resources
    pruned_configs = [config for config in recovery_configurations if non_neg(recovery_costs, config)]
    #print(pruned_configs)

    children = []
    # now descend down the recursive tree for each pruned config
    for config in pruned_configs:
        # apply config, we know it will be a valid application since it is a pruned config
        new_recovery_cost = apply_config(recovery_costs, config)
        config_node = Node(config)
        children.append(config_node)
        config_node.children = iterate_over_failures_helper(new_recovery_cost, r)
        #print(config_node.children)

    return children

# print out all possible paths from a root to every leaf node
# dfs with backtracking
def root_to_leaves(root):
    start = time.time()
    # so we don't print out root, we handle it separately
    all_paths = []
    for child in root.children:
        recurse(child, [], 0, all_paths)

    end = time.time()
    print(end - start)
    return all_paths

# recursive helper function 
def recurse(node, path, pathLen, all_paths):
    if node.vec is -1:
        return

    # python has no implicit pass by value, so we emulate it by creating 
    # a new, identical list
    newPath = path[:]
    newPath.append(node.vec)
    pathLen += 1

    if len(node.children) == 1 and node.children[0].vec == -1:
        temp = []
        for j in range(pathLen):
            #print(newPath[j], end=' ')
            temp.append(newPath[j])
        #print()
        # take advantage of python 'pass by reference' to add paths to our current
        # list of all paths
        all_paths.append(temp)
        return

    else:
        for child in node.children:
            recurse(child, newPath, pathLen, all_paths)
        return