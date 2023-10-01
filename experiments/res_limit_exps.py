import leidenalg
import networkx as nx
import igraph as ig
import community.community_louvain as cl
import numpy as np
import pandas as pd
import seaborn as sns
from networkx.algorithms.community import modularity
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt


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
        print(sorted(cluster_sizes, reverse=True))
    print('min, max, mean, median cluster sizes:', min_size, max_size, mean_size, median_size)
    print('number of singletons:', singletons_num)
    print('number of non-singleton clusters:', non_singleton_num)
    print('modularity:', modularity_score)
    print('coverage:', coverage)
    return cluster_sizes

    return cluster_num, min_size, max_size, mean_size, median_size, singletons_num, non_singleton_num, modularity_score, coverage


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

def get_membership_list_from_dict(membership_dict):
    memberships = []
    for i in range(len(membership_dict)):
        memberships.append(membership_dict[i])
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
    elif algorithm == 'leiden-cpm':
        return communities_to_dict(leidenalg.find_partition(ig.Graph.from_networkx(graph),
                                                  leidenalg.CPMVertexPartition,
                                                  resolution_parameter=res_val,
                                                  n_iterations=2).as_cover())
    elif algorithm == 'leiden-mod':
        return communities_to_dict(leidenalg.find_partition(ig.Graph.from_networkx(graph),
                                        leidenalg.ModularityVertexPartition,
                                        weights='weight',
                                        seed=seed).as_cover())

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


def simple_consensus(G, algorithm='leiden', n_p=10, thresh=0.9, delta=0.02, max_iter=10):
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
        partitions = [get_communities(graph, algorithm, i) for i in range(n_p)]

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
    return group_to_partition(get_communities(graph, algorithm, 0)), iter_count

def false_negative():
    return

def false_positive():
    return

def strict_consensus(G, algorithm='leiden-cpm', n_p=20, res_val=0.01):
    graph = G.copy()
    graph = initialize(graph, 1.0)
    iter_count = 0

    partitions = [get_communities(graph, algorithm, i, res_val) for i in range(n_p)]

    for i in range(n_p):
        c = partitions[i]
        for node, nbr in graph.edges():
            if c[node] != c[nbr]:
                graph[node][nbr]['weight'] = 0

    graph = thresholding(graph, 1, 1)

    return group_to_partition(get_communities(graph, algorithm, 0, res_val))

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

def gen_tree_of_cliques(k, n):
  '''
  k: size of clique
  n: number of cliques
  '''
  cliques = [nx.complete_graph(k) for _ in range(n)]
  tree = nx.random_tree(n)
  tree_of_cliques = nx.disjoint_union_all(cliques)
  for s, d in tree.edges():
    tree_of_cliques.add_edge(s*k+4, d*k)
  return tree_of_cliques

if __name__ == "__main__":
    # method is Leiden, Louvain, Leiden-mod
    # partition is SC, original, ground-truth
    df = pd.DataFrame(columns=['k', "n", "method", "partition", "cluster_id", "cluster_size"])
    #for n in [90, 100, 500, 1000, 5000, 10000]:
    #n = 1000
    #for res in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]:
    for n in [90, 100, 500, 1000, 5000, 10000]:
        for k in [10]:
            #ring = nx.ring_of_cliques(num_cliques=n, clique_size=k)
            graph = gen_tree_of_cliques(k, n)
            for method in ['leiden-cpm']: # , 'leiden-mod', 'louvain'
                partition = normal_clustering(graph, method, res_val=0.0001)
                cluster_sizes = partition_statistics(graph, partition)
                for i in range(len(cluster_sizes)):
                    df.loc[len(df.index)] = [k, n, method, 'Leiden-CPM(r=0.0001)', i, cluster_sizes[i]]

                partition = strict_consensus(graph, method, n_p=10, res_val=0.0001)
                cluster_sizes = partition_statistics(graph, partition)
                for i in range(len(cluster_sizes)):
                    df.loc[len(df.index)] = [k, n, method, 'SC(np=10)+Leiden-CPM(r=0.0001)', i, cluster_sizes[i]]

                partition = strict_consensus(graph, method, n_p=50, res_val=0.0001)
                cluster_sizes = partition_statistics(graph, partition)
                for i in range(len(cluster_sizes)):
                    df.loc[len(df.index)] = [k, n, method, 'SC(np=50)+Leiden-CPM(r=0.0001)', i, cluster_sizes[i]]

                #partition = strict_consensus(ring, method, n_p=100, res_val=0.01)
                #cluster_sizes = partition_statistics(ring, partition)
                #for i in range(len(cluster_sizes)):
                #    df.loc[len(df.index)] = [k, n, method, 'SC(np=100)+Leiden-CPM(r=0.01)', i, cluster_sizes[i]]
    df.to_csv('res_limit_exps_leiden_cpm_tree.csv')




