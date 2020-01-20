## DeepPR: Incremental Recovery for Interdependent VNFs with Deep Reinforcement Learning
Abstract: 

The increasing reliance upon cloud services entails more flexible networks that are realized by virtualized network equipment and functions. When such advanced network systems face a massive failure by natural disasters or attacks, the recovery of the entire systems may be conducted in a progressive way due to limited repair resources. The prioritization of network equipment in the recovery phase influences the interim computation and communication capability of systems, since the systems are operated under partial functionality. Hence, finding the best recovery order is a critical problem, which is further complicated by virtualization due to dependency among network nodes and layers. This paper deals with a progressive recovery problem under limited resources in networks with VNFs, where some dependent network layers exist. We prove the NP-hardness of the progressive recovery problem and approach the optimum solution by introducing DeepPR, a progressive recovery technique based on deep reinforcement learning. Our simulation results indicate that DeepPR can obtain 98.4% of the theoretical optimum in certain networks. The results suggest the applicability of Deep RL to more general progressive recovery problems and similar intractable resource allocation problems.

[arXiv](https://arxiv.org/abs/1904.11533)

## Dependencies
* tensorflow 1.11.0 (anything before 2.0 should work)
* numpy 1.15.2
* networkx 2.2
* matplotlib 3.0

## Instructions
To run our reinforcement learning implementation, simply run:

```` bash
$ python RL/q_progressive_recovery.py
````

from the root directory. Within the file `q_progressive_recovery.py` you can find
many graph generators to test our algorithm on, and compare against the described
ratio heuristic with. 

* `random_tree`: Generates random trees with random utilities/demand to run our algorithm on.
* `random_graph`: Generates random graphs by specifying the number of nodes and adding edges with 
a given `p=0.2`. 
* `grid`: Generate two dimensional grid graph.
* `adversarial`: Construct a simple adversarial example with n nodes, "blocking"
the best node from being discovered by the ratio heuristic.
* `gml`: Read in a `.gml` file, like any found [here](http://www.topology-zoo.org/dataset.html).
* `gml_adversarial`: Read in a `.gml` file, but guarantee the existence of an adversarial
example. This means the ratio heuristic will sometimes be outperformed by even a random heuristic.

For each graph generating heuristic, we default to ``util_range=[1,2]``
and ``demand_range=[1,4]`` for simplicity and training time, but these values may be set
by the user.

## Citation
If you found our code to be useful, please consider citing our work:

_To appear at IEEE Globecom 2019_
