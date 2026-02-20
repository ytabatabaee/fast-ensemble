import leidenalg
import networkx as nx
import igraph as ig
import community.community_louvain as cl
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv


def _partition_worker(args):
    graph, algorithm, seed, res_value, weighted = args
    ig_graph = None if algorithm == 'louvain' else ig.Graph.from_networkx(graph)
    return get_communities(graph, algorithm, seed, res_val=res_value, weighted=weighted, ig_graph=ig_graph)


def group_to_partition(partition):
    part_dict = {}
    for index, value in partition.items():
        if value in part_dict:
            part_dict[value].append(index)
        else:
            part_dict[value] = [index]
    return part_dict.values()


def initialize(graph, value):
    for u, v in graph.edges():
        graph[u][v]['weight'] = value
    return graph


def get_communities(graph, algorithm, seed, res_val=0.01, weighted='weight', ig_graph=None):
    if algorithm == 'louvain':
        return cl.best_partition(graph, random_state=seed, weight=weighted)

    if ig_graph is None:
        ig_graph = ig.Graph.from_networkx(graph)

    if algorithm == 'leiden-cpm':
        membership = leidenalg.find_partition(
            ig_graph,
            leidenalg.CPMVertexPartition,
            resolution_parameter=res_val,
            weights=weighted,
            n_iterations=1,
            seed=seed
        ).membership

    elif algorithm == 'leiden-mod':
        membership = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            weights=weighted,
            n_iterations=-1,
            seed=seed
        ).membership

    networkx_node_id_dict = {}
    for igraph_index, vertex in enumerate(ig_graph.vs):
        original_id = int(vertex["_nx_name"])
        networkx_node_id_dict[original_id] = membership[igraph_index]

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

    return get_communities(graph, algorithm, 0, res_val)


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


def build_partition_specs(algorithm=None, res_value=None, n_p=None,
                          alg_list=None, param_list=None, weight_list=None):

    if alg_list is not None:
        # weighted heterogeneous mode
        total = weight_list.sum()
        if total == 0:
            raise ValueError("Weights must sum to > 0")
        weight_list = weight_list / total
        return [(alg_list[i], param_list[i], weight_list[i])
                for i in range(len(alg_list))]

    # homogeneous mode
    alpha = 1.0 / n_p
    return [(algorithm, res_value, alpha) for _ in range(n_p)]


def _partition_worker_spec(args):
    graph, alg, res, seed, weighted = args
    ig_graph = None if alg == 'louvain' else ig.Graph.from_networkx(graph)
    return get_communities(graph, alg, seed, res_val=res,
                           weighted=weighted,
                           ig_graph=ig_graph)


def fast_ensemble(G, partition_specs=None, algorithm=None,
                  res_value=0.01, n_p=10, tr=0.8, final_alg='leiden-cpm', final_param=0.01,
                  weighted='weight', use_parallel=False):
    graph = G.copy()
    graph = initialize(graph, 1.0)

    if partition_specs is None:
        partition_specs = build_partition_specs(
            algorithm=algorithm,
            res_value=res_value,
            n_p=n_p
        )

    # ---- Generate partitions ----
    if use_parallel:
        with ProcessPoolExecutor() as executor:
            partitions = list(executor.map(
                _partition_worker_spec,
                [(graph, alg, res, i, weighted)
                 for i, (alg, res, _) in enumerate(partition_specs)]
            ))
    else:
        ig_cache = {}
        partitions = []
        for i, (alg, res, _) in enumerate(partition_specs):
            if alg != 'louvain':
                if alg not in ig_cache:
                    ig_cache[alg] = ig.Graph.from_networkx(graph)
                ig_graph = ig_cache[alg]
            else:
                ig_graph = None

            partitions.append(
                get_communities(graph, alg, i,
                                res_val=res,
                                weighted=weighted,
                                ig_graph=ig_graph)
            )

    remove_edges = []

    for u, v in graph.edges():
        weight = 1.0

        for part, (_, _, alpha) in zip(partitions, partition_specs):
            if part[u] != part[v]:
                weight -= alpha
                if weight < tr:
                    remove_edges.append((u, v))
                    break

        graph[u][v]['weight'] = weight

    graph.remove_edges_from(remove_edges)

    final_ig = None if final_alg == 'louvain' else ig.Graph.from_networkx(graph)

    return get_communities(graph,
                           final_alg,
                           0,
                           res_val=final_param,
                           weighted=weighted,
                           ig_graph=final_ig)


def read_alg_list(list_path):
    alg_list, param_list, weight_list = [], [], []
    with open(list_path) as fgt:
        for line in fgt:
            try:
                alg, param, weight = line.strip().split()
                alg_list.append(alg)
                param_list.append(float(param))
                weight_list.append(float(weight))
            except:
                continue
    return alg_list, np.asarray(param_list), np.asarray(weight_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastEnsemble Clustering")
    parser.add_argument("-n", "--edgelist", type=str,  required=True,
                        help="Network edge-list file")
    parser.add_argument("-alg", "--algorithm", type=str, required=False,
                        help="Clustering algorithm (leiden-mod, leiden-cpm or louvain)", default='leiden-cpm')
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output community file")
    parser.add_argument("-t", "--threshold", type=float, required=False,
                        help="Threshold value", default=0.8)
    parser.add_argument("-r", "--resolution", type=float, required=False,
                        help="Resolution value for leiden-cpm", default=0.01)
    parser.add_argument("-falg", "--finalalgorithm", type=str, required=False,
                        help="", default=None)
    parser.add_argument("-fr", "--finalparam", type=float, required=False,
                        help="Parameter (e.g. resolution value) for the final algorithm", default=None)
    parser.add_argument("-p", "--partitions", type=int, required=False,
                        help="Number of partitions in consensus clustering", default=10)
    parser.add_argument("-rl", "--relabel", required=False, action='store_true',
                        help="Relabel network nodes from 0 to #nodes-1.", default=False)
    parser.add_argument("-alglist", "--algorithmlist", type=str, required=False,
                        help="A list of clustering algorithms, with parameters and weights")
    parser.add_argument("-nw", "--noweight", required=False, action='store_true',
                        help="Specify that clustering methods should NOT take the edge weights into account", default=False)
    parser.add_argument("-mp", "--multiprocessing",
                        required=False, action='store_true',
                        help="Enable multiprocessing for partition generation",
                        default=False)

    args = parser.parse_args()
    net = nx.read_edgelist(args.edgelist, nodetype=int)

    # relabeling nodes from 0 to n-1
    if args.relabel:
        mapping = dict(zip(sorted(net), range(0, net.number_of_nodes())))
        net = nx.relabel_nodes(net, mapping)
        reverse_mapping = {y: x for x, y in mapping.items()}

    n_p = args.partitions
    if not args.finalalgorithm:
        args.finalalgorithm = args.algorithm
    if not args.finalparam:
        args.finalparam = args.resolution

    if args.algorithmlist:
        alg_list, param_list, weight_list = read_alg_list(args.algorithmlist)
        partition_specs = build_partition_specs(
            alg_list=alg_list,
            param_list=param_list,
            weight_list=weight_list
        )
    else:
        partition_specs = None

    if partition_specs is not None:
        algorithm = None

    fe = fast_ensemble(net,
                       partition_specs=partition_specs,
                       algorithm=args.algorithm,
                       res_value=args.resolution,
                       n_p=args.partitions,
                       tr=args.threshold,
                       final_alg=args.finalalgorithm,
                       final_param=args.finalparam,
                       weighted=None if args.noweight else 'weight',
                       use_parallel=args.multiprocessing)

    keys = list(fe.keys())
    keys.sort()
    membership_dict = {i: fe[i] for i in keys}
    membership = list(membership_dict.values())

    with open(args.output, 'w') as out_file:
        writer = csv.writer(out_file, delimiter=' ')
        for i in range(len(membership)):
            if args.relabel:
                writer.writerow([reverse_mapping[i]] + [membership[i]])
            else:
                writer.writerow([i] + [membership[i]])
