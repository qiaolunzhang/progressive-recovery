from env import deviation_from_optimal, RecoveryEnv
from tree_recovery import plot_bar_x, r_tree, DP_optimal, get_root, plot_graph
import time

def heuristic_tester(node_range, sample_size, save_dir, resources=1, height=None):
    '''
    Calculate and graph how good our U - D heuristic is for a variety of node_num trees

    :param node_range: (x, y) tuple s.t. we start at a tree of size x and stop at a tree of size y-1.
    :param sample_size: Number of trees to sample for each tree size
    :param save_dir: where to save final line plot
    :param resources: Resources per recovery turn
    :param height: fix height of each randomly generated tree to some number
    :return: None
    '''
    total_stats = [] 
    time_stats = []
    for nodes in range(node_range[0], node_range[1]):
        start = time.time()
        stats = []
        for x in range(sample_size):
            stats.append(deviation_from_optimal(nodes, resources, height))

        #print(stats)
        avg = [x[1] / x[0] for x in stats]
        #print('Average percentage of optimal for {0} node graph:'.format(nodes), sum(avg)/len(avg))
        total_stats.append(100 * sum(avg)/len(avg))
        end = time.time()
        time_stats.append(end - start)

    print(total_stats)

    labels = ['{0}'.format(x) for x in range(node_range[0], node_range[1])]
    plot_bar_x(total_stats, labels, save_dir)
    # plot_bar_x(time_stats, labels, save_dir)

    return time_stats

def main():
    # generate random tree with 4 nodes
    G = r_tree(4)
    R = RecoveryEnv(G, [get_root(G)])

    print("Optimal util by DP:", DP_optimal(G, [get_root(G)], 1))
    print("=============================================")
    print("Optimal util and config by verified method", R.optimal(resources=1))
    print("\nPlotting graph in 'plots/DP.png'")
    plot_graph(G, get_root(G), 'plots/DP.png')
    
    # times = heuristic_tester(node_range=(4,11), sample_size=50, save_dir='plots/heuristic_optimality.png')    
    # print(times)
    # G = r_tree(8)
    # print(deviation_from_optimal(8, 1))

if __name__ == "__main__":
    main()
