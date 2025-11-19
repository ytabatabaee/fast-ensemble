import leidenalg
import networkx as nx
import igraph as ig
import community.community_louvain as cl
import numpy as np
import pandas as pd
import argparse
import seaborn as sns
from networkx.algorithms.community import modularity
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
import csv


def group_to_partition(partition):
    part_dict = {}
    for index, value in partition.items():
        if value in part_dict:
            part_dict[value].append(index)
        else:
            part_dict[value] = [index]
    return part_dict.values()


def membership_list_to_dict(membership_list):
    membership_dict = {}
    for i in range(len(membership_list)):
        membership_dict[i] = membership_list[i]
    return membership_dict


def get_membership_list_from_dict(membership_dict, n):
    memberships = [0]*n
    for i in range(len(membership_dict)):
        for x in membership_dict[i]:
            memberships[x] = i
    return memberships


def communities_to_dict(communities):
    result = {}
    community_index = 0
    for c in communities:
        community_mapping = ({node: community_index for index, node in enumerate(c)})
        result = {**result, **community_mapping}
        community_index += 1
    return result


def initialize(graph, value):
    for u, v in graph.edges():
        graph[u][v]['weight'] = value
    return graph


def get_communities(graph, algorithm, seed, res_val=0.01):
    if algorithm == 'louvain':
        return cl.best_partition(graph, random_state=seed, weight='weight')
    if algorithm == 'leiden-cpm':
        relabelled_graph = ig.Graph.from_networkx(graph)
        networkx_node_id_dict = {}
        igraph_node_id_dict = leidenalg.find_partition(relabelled_graph, leidenalg.CPMVertexPartition,
                                                       resolution_parameter=res_val, n_iterations=1, seed=seed).membership
        print(igraph_node_id_dict)
        for igraph_index, vertex in enumerate(relabelled_graph.vs):
            vertex_attributes = vertex.attributes()
            original_id = int(vertex_attributes["_nx_name"])
            relabelled_id = int(igraph_index)
            networkx_node_id_dict[original_id] = igraph_node_id_dict[relabelled_id]
        return networkx_node_id_dict
    elif algorithm == 'leiden-mod':
        relabelled_graph = ig.Graph.from_networkx(graph)
        networkx_node_id_dict = {}
        igraph_node_id_dict = leidenalg.find_partition(relabelled_graph, leidenalg.ModularityVertexPartition,
                                                       weights='weight', n_iterations=-1, seed=seed).membership
        for igraph_index, vertex in enumerate(relabelled_graph.vs):
            vertex_attributes = vertex.attributes()
            original_id = int(vertex_attributes["_nx_name"])
            relabelled_id = int(igraph_index)
            networkx_node_id_dict[original_id] = igraph_node_id_dict[relabelled_id]
        return networkx_node_id_dict


def check_convergence(G, n_p, delta):
    count = 0
    for wt in nx.get_edge_attributes(G, 'weight').values():
        if wt != 0 and wt != n_p:
            count += 1
    if count > delta * G.number_of_edges():
        return False
    return True


def thresholding(graph, n_p, thresh):
    remove_edges = []
    for u, v in graph.edges():
        if graph[u][v]['weight'] < thresh * n_p:
            remove_edges.append((u, v))
    graph.remove_edges_from(remove_edges)
    return graph


def simple_consensus(G, algorithm='leiden-mod', n_p=10, thresh=0.9, delta=0.02, max_iter=10, res_value=0.01):
    graph = G.copy()
    graph = initialize(graph, 1.0)
    iter_count = 0

    while True:
        iter_count += 1
        if iter_count > max_iter:
            iter_count -= 1
            break
        nextgraph = graph.copy()
        nextgraph = initialize(nextgraph, 0.0)
        partitions = [get_communities(graph, algorithm, i, res_val=res_value) for i in range(n_p)]

        # print('edges', len(graph.edges()))
        for i in range(n_p):
            # print('np: ', i)
            c = partitions[i]
            for node, nbr in graph.edges():
                if graph[node][nbr]['weight'] not in (0, n_p):
                    if c[node] == c[nbr]:
                        nextgraph[node][nbr]['weight'] += 1
                else:
                    nextgraph[node][nbr]['weight'] = graph[node][nbr]['weight']
                # print(node, nbr, nextgraph[node][nbr]['weight'])

        nextgraph = thresholding(nextgraph, n_p, thresh)
        if check_convergence(nextgraph, n_p, delta=delta):
            break
        graph = nextgraph.copy()

    print('number of iterations:', iter_count)
    final_comm = get_communities(graph, algorithm, 0, res_val=res_value)
    #print(final_comm)
    #final_comm = group_to_partition(final_comm)
    #print(final_comm)
    return final_comm

def normal_clustering(graph, algorithm, res_val=0.01):
    if algorithm == 'leiden-cpm':
        return leidenalg.find_partition(ig.Graph.from_networkx(graph),
                                              leidenalg.CPMVertexPartition,
                                              resolution_parameter=res_val,
                                              n_iterations=2).as_cover()
    elif algorithm == 'leiden-mod':
        return leidenalg.find_partition(ig.Graph.from_networkx(graph),
                                         leidenalg.ModularityVertexPartition,
                                         seed=1234).as_cover()
    elif algorithm == 'louvain':
        return group_to_partition(cl.best_partition(graph))


def fast_ensemble(G, algorithm='leiden-cpm', n_p=20, tr=0.2, res_value=0.01):
    graph = G.copy()
    graph = initialize(graph, 1)
    partitions = [get_communities(graph, algorithm, i, res_val=res_value) for i in range(n_p)]

    remove_edges = []
    for node, nbr in graph.edges():
        for i in range(n_p):
            if partitions[i][node] != partitions[i][nbr]:
                graph[node][nbr]['weight'] -= 1/n_p
            if graph[node][nbr]['weight'] < tr:
                remove_edges.append((node, nbr))
                break
    graph.remove_edges_from(remove_edges)

    #return group_to_partition(get_communities(graph, algorithm, 0, res_val=res_value))
    return get_communities(graph, algorithm, 0, res_val=res_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normal clustering")
    parser.add_argument("-n", "--edgelist", type=str,  required=True,
                        help="Network edge-list file")
    parser.add_argument("-alg", "--algorithm", type=str, required=False,
                        help="Clustering algorithm (louvain, leiden-mod or leiden-cpm)", default='leiden-cpm')
    parser.add_argument("-r", "--resolution", type=float, required=False,
                        help="Resolution value for leiden-cpm", default=0.01)
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output community file")
    parser.add_argument("-rl", "--relabel", required=False, action='store_true',
                        help="Relabel network nodes from 0 to #nodes-1.", default=False)

    args = parser.parse_args()
    net = nx.read_edgelist(args.edgelist, nodetype=int)

    # relabeling nodes from 0 to n-1
    if args.relabel:
        mapping = dict(zip(sorted(net), range(0, net.number_of_nodes())))
        net = nx.relabel_nodes(net, mapping)
        reverse_mapping = {y: x for x, y in mapping.items()}

    sc = get_communities(net, args.algorithm, 0, res_val=args.resolution)
    keys = list(sc.keys())
    keys.sort()
    membership_dict = {i: sc[i] for i in keys}
    membership = list(membership_dict.values())

    with open(args.output, 'w') as out_file:
        writer = csv.writer(out_file, delimiter=' ')
        for i in range(len(membership)):
            if args.relabel:
                writer.writerow([reverse_mapping[i]] + [membership[i]])
            else:
                writer.writerow([i] + [membership[i]])
