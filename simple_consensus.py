import igraph as ig
import leidenalg
import csv
import networkx as nx
import numpy as np
import argparse
import community.community_louvain as cl
from networkx.algorithms.community import modularity


def check_convergence(G, n_p, delta):
    count = 0
    for wt in nx.get_edge_attributes(G, 'weight').values():
        if wt != 0 and wt != n_p:
            count += 1
    if count > delta * G.number_of_edges():
        return False
    return True

def communities_to_dict(communities):
    result = {}
    community_index = 0
    for c in communities:
        community_mapping = ({node: community_index for index, node in enumerate(c)})
        result = {**result, **community_mapping}
        community_index += 1
    return result


def thresholding(graph, n_p, thresh):
    remove_edges = []
    for u, v in graph.edges():
        if graph[u][v]['weight'] < thresh * n_p:
            remove_edges.append((u, v))
    graph.remove_edges_from(remove_edges)
    return graph


def initialize(graph, value):
    for u, v in graph.edges():
        graph[u][v]['weight'] = value
    return graph


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


def simple_consensus(G, alg_list, param_list, weight_list, thresh=0.2, delta=0.02, max_iter=10):
    graph = G.copy()
    graph = initialize(graph, 1.0)
    iter_count = 0
    n_p = len(alg_list)
    n_p_max = sum(weight_list)

    while True:
        iter_count += 1
        if iter_count > max_iter:
            iter_count -= 1
            break
        nextgraph = graph.copy()
        nextgraph = initialize(nextgraph, 0.0)

        communities = [get_communities(graph, alg_list[i], i, param_list[i]) for i in range(len(alg_list))]
        for i in range(n_p):
            c = communities[i]
            for node, nbr in graph.edges():
                if graph[node][nbr]['weight'] not in (0, n_p_max):
                    if c[node] == c[nbr]:
                        nextgraph[node][nbr]['weight'] += 1 * weight_list[i]
                else:
                    nextgraph[node][nbr]['weight'] = graph[node][nbr]['weight']

        nextgraph = thresholding(nextgraph, n_p_max, thresh)
        if check_convergence(nextgraph, n_p=n_p_max, delta=delta):
            break
        graph = nextgraph.copy()

    print('number of iterations:', iter_count)
    return get_communities(graph, alg_list[0], 0, param_list[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold Consensus")
    parser.add_argument("-n", "--edgelist", type=str,  required=True,
                        help="Network edge-list file")
    parser.add_argument("-t", "--threshold", type=float, required=False,
                        help="Threshold value", default=1.0)
    parser.add_argument("-r", "--resolution", type=float, required=False,
                        help="Resolution value for leiden-cpm", default=0.01)
    parser.add_argument("-p", "--partitions", type=int, required=False,
                        help="Number of partitions in consensus clustering", default=10)
    parser.add_argument("-d", "--delta", type=float, required=False,
                        help="Convergence parameter", default=0.02)
    parser.add_argument("-m", "--maxiter", type=int, required=False,
                        help="Maximum number of iterations in simple consensus", default=10)
    parser.add_argument("-rl", "--relabel", required=False, action='store_true',
                        help="Relabel network nodes from 0 to #nodes-1.", default=False)

    args = parser.parse_args()
    net = nx.read_edgelist(args.edgelist, nodetype=int)

    # relabeling nodes from 0 to n-1
    if args.relabel:
        mapping = dict(zip(net, range(0, net.number_of_nodes())))
        net = nx.relabel_nodes(net, mapping)

    n_p = args.partitions
    #leiden_alg_list = ['leiden-cpm'] * n_p
    lou_lei_alg_list = ['leiden-mod'] * int(n_p / 2) + ['leiden-cpm'] * (n_p - int(n_p / 2))
    sc = simple_consensus(net, lou_lei_alg_list, [args.resolution] * n_p, [1] * n_p, args.threshold)

    with open('sc_'+str(args.threshold)+'_'+args.edgelist.split('/')[-1], 'w') as out_file:
        writer = csv.writer(out_file, delimiter=' ')
        for node, mem in sc.items():
            writer.writerow([node]+[mem])
