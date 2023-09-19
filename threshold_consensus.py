import leidenalg
import networkx as nx
import igraph as ig
import argparse
import csv
import community.community_louvain as cl
import numpy as np


def communities_to_dict(communities):
    result = {}
    community_index = 0
    for c in communities:
        community_mapping = ({node: community_index for index, node in enumerate(c)})
        result = {**result, **community_mapping}
        community_index += 1
    return result

def get_communities(graph, algorithm, seed, r=0.001):
    if algorithm == 'louvain':
        return cl.best_partition(graph, random_state=seed, weight='weight')
    elif algorithm == 'leiden-cpm':
        return communities_to_dict(leidenalg.find_partition(ig.Graph.from_networkx(graph),
                                                  leidenalg.CPMVertexPartition,
                                                  resolution_parameter=r,
                                                  n_iterations=2).as_cover())
    elif algorithm == 'leiden-mod':
        return communities_to_dict(leidenalg.find_partition(ig.Graph.from_networkx(graph),
                                        leidenalg.ModularityVertexPartition,
                                        weights='weight',
                                        seed=seed).as_cover())

def initialize(graph, value):
    for u, v in graph.edges():
        graph[u][v]['weight'] = value
    return graph


def thresholding(graph, thresh):
    remove_edges = []
    bound = thresh
    for u, v in graph.edges():
        if graph[u][v]['weight'] < bound:
            remove_edges.append((u, v))
    graph.remove_edges_from(remove_edges)
    return graph

# strict consensus can be achieved by running threshold consensus with tr=1
def threshold_consensus(G, algorithm='leiden-cpm', n_p=20, tr=1, r=0.001):
    graph = G.copy()
    graph = initialize(graph, 1)
    iter_count = 0

    partitions = [get_communities(graph, algorithm, i, r) for i in range(n_p)]

    for i in range(n_p):
        c = partitions[i]
        for node, nbr in graph.edges():
            if c[node] != c[nbr]:
                graph[node][nbr]['weight'] -= 1/n_p

    graph = thresholding(graph, tr)
    return get_communities(graph, algorithm, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold Consensus")
    parser.add_argument("-n", "--edgelist", type=str,  required=True,
                        help="Network edge-list file")
    parser.add_argument("-t", "--threshold", type=float, required=False,
                        help="Threshold value", default=1.0)
    parser.add_argument("-a", "--algorithm", type=str, required=False,
                        help="Clustering algorithm (leiden-cpm, leiden-mod, louvain)", default='leiden-cpm')
    parser.add_argument("-r", "--resolution", type=float, required=False,
                        help="Resolution value for leiden-cpm", default=0.01)
    parser.add_argument("-p", "--partitions", type=int, required=False,
                        help="Number of partitions in consensus clustering", default=10)
    parser.add_argument("-rl", "--relabel", required=False, action='store_true',
                        help="Relabel network nodes from 0 to #nodes-1.", default=False)
    
    args = parser.parse_args()
    net = nx.read_edgelist(args.edgelist, nodetype=int)

    # relabeling nodes from 0 to n-1
    if args.relabel:
        mapping = dict(zip(net, range(0, net.number_of_nodes())))
        net = nx.relabel_nodes(net, mapping)

    tc = threshold_consensus(net, args.algorithm.lower(), args.partitions, args.threshold, args.resolution)
    with open('tc_'+str(args.threshold)+'_'+args.edgelist.split('/')[-1], 'w') as out_file:
        writer = csv.writer(out_file, delimiter=' ')
        for node, mem in tc.items():
            writer.writerow([node]+[mem])

