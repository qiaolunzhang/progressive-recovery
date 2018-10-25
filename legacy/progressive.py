import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from random import randint

import time as tfunc
import numpy as np

#---------------------------
def savegml(G):
    nx.write_graphml(G, "./graph.graphml")

#---------------------------


#generate the sample graph
def generate_graph():
    G = nx.Graph()

    utils = {} 
    caps = {}
    rcv_amts = {}

    for node in range(8):
        G.add_node(node)
        utils.update({node: 1.0})
        caps.update({node: randint(10,10)})
        rcv_amts.update({node : caps.get(node)})

    G.add_edge(0,1)
    G.add_edge(0,2)
    G.add_edge(0,5)
    G.add_edge(0,7)
    G.add_edge(1,2)
    G.add_edge(2,3)
    G.add_edge(2,5)
    G.add_edge(0,5)
    G.add_edge(3,4)
    G.add_edge(3,5)
    G.add_edge(4,5)
    G.add_edge(5,6)
    G.add_edge(6,7)

    #print(G.nodes())
    #print(G.edges())
    #savegml(G)

    # Nodes have utility, capacity and current amount which the node recieve
    # for recovery
    # in this simulation, im not using utility though
    nx.set_node_attributes(G, "util", 1.0)
    nx.set_node_attributes(G, "cap", caps)
    nx.set_node_attributes(G, "rcv_amt", rcv_amts)

    nx.draw(G,with_labels=True)
    plt.show()

    return G

#---------------------------
def check_recov(H, G, accum_rcv):
    for failed_node in accum_rcv.keys():
        #if the recovery amount reaches the capacity,
        #the node becomes functional
        if accum_rcv.get(failed_node) == G.node[failed_node]["cap"]:
            recover(H, G, failed_node)
    return H



#---------------------------

#put back the recovered node using the original topology setting in G
def recover(H, G, node):
    H.add_node(node, 
            util=G.node[node]["util"],
            cap=G.node[node]["cap"],
            rcv_amt=G.node[node]["cap"])

    for edge in G.edges(node):
        H.add_edge(edge[0],edge[1])

    return H    

#---------------------------

#system utility at time t is defined as
#the size of the maximum connected component
#system utility is the sum of the size over time
def evaluate(H):
    giant_comp = max(nx.connected_components(H), key=len)
    return len(giant_comp)

#---------------------------

#random recovery process where we put recovery resources randomly
#to the failed nodes (even after the recovery, nodes can recieve 
# the recovery resouece
def random_recovery(failed_nodes, resources, time):
    recv_at_t = {}
    for failed_node in failed_nodes:
        recv_at_t.update({failed_node: 0.0})

    total = 0

    while total < resources.get(time):
        target = failed_nodes[randint(0, len(failed_nodes)-1)]
        amount = randint(0, resources.get(time) - total)
        total += amount

        so_far = recv_at_t.get(target) + amount
        recv_at_t.update({target: so_far})

    print(time, total)

    return recv_at_t      


#---------------------------
#representation
#showing the graphic representation
def show(data, node_ids, T, total_val):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Oranges, alpha= 0.8)

    plt.colorbar(heatmap)

    ax.set_xlabel("Failed Node")
    ax.set_ylabel("Time")

    name = "./" + str(total_val) + "--" + str(tfunc.time()) + ".png"
    plt.savefig(name, figsize=(5,5), frameon=False, dpi=80, pad_inches=0)
    plt.show()


#---------------------------


#---------------------------

############################################
#MAIN LOGIC
############################################
G = generate_graph()
T = 10
#the size of failure
print("Decide the size of failure:")
f = 4 #input() # now for this experience, we have 4 failed nodes

failed_nodes = []
#count = 0
#while count < int(f):
#    failure = randint(0,len(G.nodes())-1)
#    if failure not in failed_nodes:
#        failed_nodes.append(failure)
#        count += 1

failed_nodes = [1,4,6,3] # fixed failed nodes


rcv_amts = nx.get_node_attributes(G, "rcv_amt")
for failed_node in failed_nodes:
    rcv_amts[failed_node] = 0

#failed graph
H = G.copy()
for failed_node in failed_nodes:
    H.remove_node(failed_node)

# At time t, we have 4 resoueces
resources = {}
for time in range(T):
    resources.update({time: 4.0})

#failed nodes do not have any resouce right after the failure
accum_rcv = {}
for failed_node in failed_nodes:
    accum_rcv.update({failed_node : 0.0})

progressive_rcv = []
total_val = 0;

#over time 
for time in range(T):
    print("nodes in H: ", H.nodes())
    #get the resource distribution at time t
    rcv_at_t = random_recovery(failed_nodes, resources, time)

    for failed_node in failed_nodes:
        accum_rcv[failed_node] = accum_rcv.get(failed_node) + rcv_at_t.get(failed_node)

    #original list order doesn't change
    progressive_rcv_at_t = []
    for fn in sorted(failed_nodes):
        progressive_rcv_at_t.append(rcv_at_t.get(fn))

    #strore the distribution
    progressive_rcv.append(progressive_rcv_at_t)

    #update H based on recovered nodes
    H = check_recov(H, G, accum_rcv)
    val = evaluate(H)
    print("Eval at time ", time, " - ", val)

    total_val += val

print(total_val)
    
#print(progressive_rcv)
show(progressive_rcv, sorted(failed_nodes), T, total_val)


f = open("result.csv", "a")
s = str(total_val) + "\n"
f.write(s)
f.close()
