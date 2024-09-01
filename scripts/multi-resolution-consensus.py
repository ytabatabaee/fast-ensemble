import igraph as ig
import leidenalg
import csv
import networkx as nx
import numpy as np
import argparse
import matplotlib.pyplot as plt
import community.community_louvain as cl
from networkx.algorithms.community import modularity


def check_convergence(G, NG, n_p, delta):
    #print (delta)
    #count = 0
    #for wt in nx.get_edge_attributes(G, 'weight').values():
    #    if wt != 0 and wt != n_p:
    #        count += 1
    
    #print (count, delta * G.number_of_edges())
    if nx.utils.graphs_equal(NG,G):
    #if count > delta * G.number_of_edges():
        return True
    return False


def thresholding(graph, n_p, thresh):

    remove_edges = []
    for u, v in graph.edges():
        if graph[u][v]['weight'] < thresh * n_p: 
        # an edge weight represents the proportion of partitions the two nodes are clustered together
            # .3*3=.9->remove edges with weight 0 ie in no partitions together
            #          -> only keep if in at least 1 partition                   
            # .6*3=1.8->remove edges with weight 0 or 1 ie in 0 or 1 partition together 
            #          -> only keep if in at least 2 partitions            
            # .9*3=2.7->remove edges with weight 0 1 or 2 ie in 0 1 or 2 partitions togehter 
            #          -> only keep if in all partitions
            remove_edges.append((u, v))
    graph.remove_edges_from(remove_edges)
    return graph


def initialize(graph, value):
    for u, v in graph.edges():
        graph[u][v]['weight'] = value
    return graph


def get_communities(graph, algorithm, seed, r=0.001):
    print (r)
    if algorithm == 'louvain':
        return cl.best_partition(graph, random_state=seed, weight='weight')
    elif algorithm == 'leiden-cpm':
        return dict(enumerate(leidenalg.find_partition(ig.Graph.from_networkx(graph),
                                                  leidenalg.CPMVertexPartition,
                                                  resolution_parameter=r,
                                                  n_iterations=2).membership))
    elif algorithm == 'leiden-mod':
        return dict(enumerate(leidenalg.find_partition(ig.Graph.from_networkx(graph),
                                        leidenalg.ModularityVertexPartition,
                                        weights='weight',
                                        seed=seed).membership))


def get_node_to_component_dict(graph):
    node_to_component_dict = dict()
    for i, node_set in enumerate(nx.connected_components(graph)):
        node_to_component_dict.update({node: i for node in node_set})
    print (nx.number_connected_components(graph))
    return node_to_component_dict
        

def simple_consensus(G, alg_list, param_list, weight_list, thresh=0.2, delta=0.02, max_iter=10):
    graph = G.copy()
    graph = initialize(graph, 1.0)
    iter_count = 0
    n_p = len(alg_list)
    n_p_max = sum(weight_list)
    print (max_iter)

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
        print (nx.number_connected_components(nextgraph))

        if check_convergence(graph, nextgraph, n_p=n_p_max, delta=delta):
            graph = nextgraph.copy()
            break
        
        graph = nextgraph.copy()

    #nx.draw(graph)
    #plt.show()
    print('number of iterations:', iter_count)
    return get_node_to_component_dict(graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold Consensus")
    parser.add_argument("-n", "--edgelist", type=str,  required=True,
                        help="Network edge-list file")
    parser.add_argument("-t", "--threshold", type=float, required=False,
                        help="Threshold value", default=0.2)
    parser.add_argument("-r", "--resolution", type=float, required=False,
                        help="Resolution value for leiden-cpm", default=0.01)
    parser.add_argument("-p", "--partitions", type=int, required=False,
                        help="Number of partitions in consensus clustering", default=3)
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
        reverse_mapping = {y: x for x, y in mapping.items()}
        
    n_p = args.partitions
    leiden_alg_list = ['leiden-cpm'] * n_p
    resolution_list = [.01,.001, .0001]
    #lou_lei_alg_list = ['leiden-mod'] * int(n_p / 2) + ['leiden-cpm'] * (n_p - int(n_p / 2))
    #print ([args.resolution] * n_p, n_p, args.resolution)
    sc = simple_consensus(net, leiden_alg_list, resolution_list, [1] * n_p, args.threshold,args.delta, args.maxiter)
    #sc = simple_consensus(net, leiden_alg_list, [args.resolution] * n_p, [1] * n_p, args.threshold, args.maxiter)
    with open('sc_'+str(args.threshold)+'_'+args.edgelist.split('/')[-1], 'w') as out_file:
        writer = csv.writer(out_file, delimiter=' ')
        for node, mem in sc.items():
            writer.writerow([reverse_mapping[node]]+[mem])
