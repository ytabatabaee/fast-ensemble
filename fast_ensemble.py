import leidenalg
import networkx as nx
import igraph as ig
import community.community_louvain as cl
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

    return get_communities(graph, algorithm, 0, res_val) # group_to_partition(


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


def fast_ensemble(G, algorithm='leiden-cpm', n_p=10, tr=0.8, res_value=0.01,
                  final_alg='leiden-cpm', final_param=0.01,
                  weighted='weight', use_parallel=False):
    graph = G.copy()
    graph = initialize(graph, 1)

    if use_parallel:
        with ProcessPoolExecutor() as executor:
            partitions = list(executor.map(
                _partition_worker,
                [(graph, algorithm, i, res_value, weighted) for i in range(n_p)]
            ))
    else:
        ig_graph = None if algorithm == 'louvain' else ig.Graph.from_networkx(graph)
        partitions = [
            get_communities(graph, algorithm, i,
                            res_val=res_value,
                            weighted=weighted,
                            ig_graph=ig_graph)
            for i in range(n_p)
        ]

    remove_edges = []

    for node, nbr in graph.edges():
        weight = 1.0
        for part in partitions:
            if part[node] != part[nbr]:
                weight -= 1 / n_p
                if weight < tr:
                    remove_edges.append((node, nbr))
                    break
        else:
            graph[node][nbr]['weight'] = weight
            continue
        # only executed if break happens
        graph[node][nbr]['weight'] = weight

    graph.remove_edges_from(remove_edges)

    final_ig = None if final_alg == 'louvain' else ig.Graph.from_networkx(graph)
    return get_communities(graph, final_alg, 0,
                           res_val=final_param,
                           weighted=weighted,
                           ig_graph=final_ig)

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

    fe = fast_ensemble(net,
                       args.algorithm,
                       n_p=args.partitions,
                       tr=args.threshold,
                       res_value=args.resolution,
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
