import networkx as nx

def calc_avg_cut_size(g):
    sum = 0
    for v in g.nodes():
        S = {v}

        for u in g.nodes():
            T = {u}
            sum += nx.cut_size(g, S, T)

    return float(sum/(2*len(g.nodes)))


def show_graph_props(g):
    print("Diameter: ", nx.diameter(g))
    print("Radius: ", nx.radius(g))
    print("Density: ", nx.density(g))
    print("Avg Cut Size: ", calc_avg_cut_size(g))

def retun_props(g):
    return (nx.diameter(g),nx.radius(g),nx.density(g),calc_avg_cut_size(g))
