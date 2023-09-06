import networkx as nx
import igraph as ig
import argparse
import csv
import community.community_louvain as cl
import numpy as np
import ast
from networkx.algorithms.community import modularity

def get_membership_list_from_file(membership_path):
    membership = dict()
    with open(membership_path) as f:
        for line in f:
            i, m = line.strip().split()
            membership[int(i)] = m
    return membership


def group_to_partition(partition):
    part_dict = {}
    for index, value in partition.items():
        if value in part_dict:
            part_dict[value].append(index)
        else:
            part_dict[value] = [index]
    return part_dict.values()


def partition_statistics(G, partition, show_cluster_size_dist=True):
    cluster_num = len(partition)
    cluster_sizes = [len(c) for c in partition]
    min_size, max_size, mean_size, median_size = np.min(cluster_sizes), np.max(cluster_sizes), np.mean(
        cluster_sizes), np.median(cluster_sizes)
    singletons = [c for c in partition if len(c) == 1]
    singletons_num = len(singletons)
    non_singleton_num = len(partition) - len(singletons)
    modularity_score = modularity(G, partition)
    coverage = (G.number_of_nodes() - len(singletons)) / G.number_of_nodes()

    print('#clusters in partition:', cluster_num)
    if show_cluster_size_dist:
        print('Cluster sizes:')
        print(sorted(cluster_sizes, reverse=True))
    print('min, max, mean, median cluster sizes:', min_size, max_size, mean_size, median_size)
    print('number of singletons:', singletons_num)
    print('number of non-singleton clusters:', non_singleton_num)
    print('modularity:', modularity_score)
    print('coverage:', coverage)
    return cluster_num, min_size, max_size, mean_size, median_size, singletons_num, non_singleton_num, modularity_score, coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold Consensus")
    parser.add_argument("-n", "--edgelist", type=str,  required=True,
                        help="Network edge-list file")
    parser.add_argument("-m", "--membership", type=str, required=True,
                        help="Partition membership")
    #parser.add_argument("-g", "--groundtruth", type=str, required=False,
    #                    help="Ground-truth membership")
    args = parser.parse_args()
    net = nx.read_edgelist(args.edgelist, nodetype=int)
    partition = get_membership_list_from_file(args.membership)
    partition = group_to_partition(partition)
    partition_statistics(net, partition)



