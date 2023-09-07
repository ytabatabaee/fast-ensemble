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


def network_statistics(graph, ground_truth_membership=None, show_connected_components=False):
    print('** Network statistics **')
    node_count, edge_count, isolate_count = graph.number_of_nodes(), graph.number_of_edges(), len(
        list(nx.isolates(graph)))
    connected_components_sizes = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
    connected_component_num = nx.number_connected_components(graph)
    max_connected_component = max(connected_components_sizes)
    degrees = [d for n, d in graph.degree()]
    min_degree, max_degree, mean_degree, median_degree = np.min(degrees), np.max(degrees), np.mean(degrees), np.median(
        degrees)
    print('#nodes, #edges, #singletons:', node_count, edge_count, isolate_count)
    print('num connected comp:', connected_component_num)
    print('max connected comp size:', max_connected_component)
    if show_connected_components:
        print(connected_components_sizes)
    print('min, max, mean, median degree:', min_degree, max_degree, mean_degree, median_degree)
    if ground_truth_membership:
        print('ground truth partition statistics')
        partition_statistics(graph, group_to_partition(membership_list_to_dict(ground_truth_membership)))
    return node_count, edge_count, isolate_count, connected_component_num, max_connected_component, min_degree, max_degree, mean_degree, median_degree


def partition_statistics(G, partition, show_cluster_size_dist=True):
    print('\n** Partition statistics **')
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
        print('cluster sizes:')
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
    network_statistics(net)
    partition = get_membership_list_from_file(args.membership)
    partition = group_to_partition(partition)
    partition_statistics(net, partition)



