from env import deviation_from_optimal
from tree_recovery import plot_bar_x

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
    for nodes in range(node_range[0], node_range[1]):
        stats = []
        for x in range(sample_size):
            stats.append(deviation_from_optimal(nodes, resources, height))

        #print(stats)
        avg = [x[1] / x[0] for x in stats]
        #print('Average percentage of optimal for {0} node graph:'.format(nodes), sum(avg)/len(avg))
        total_stats.append(100 * sum(avg)/len(avg))

    print(total_stats)

    labels = ['{0} Nodes'.format(x) for x in range(node_range[0], node_range[1])]
    plot_bar_x(total_stats, labels, save_dir)


def main():
    heuristic_tester(node_range=(4,5), sample_size=1, save_dir='plots/heuristic.png')    

if __name__ == "__main__":
    main()