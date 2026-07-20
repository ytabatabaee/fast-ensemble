import leidenalg
import networkx as nx
import igraph as ig
import community.community_louvain as cl
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import logging
import os
import statistics
import time


LOGGER_NAME = "fast_ensemble"
ALLOWED_ALGORITHMS = {"leiden-cpm", "leiden-mod", "louvain"}
INPUT_WEIGHT_ATTR = "input_weight"


def _elapsed(start_time):
    return f"{time.perf_counter() - start_time:.2f}s"


def _log_step(logger, message, start_time=None):
    if logger is None:
        return
    if start_time is None:
        logger.info(message)
    else:
        logger.info("%s finished in %s", message, _elapsed(start_time))


def _validate_algorithm(algorithm, label="algorithm"):
    if algorithm not in ALLOWED_ALGORITHMS:
        allowed = ", ".join(sorted(ALLOWED_ALGORITHMS))
        raise ValueError(f"Invalid {label} '{algorithm}'. Allowed values: {allowed}")


def _validate_partition_specs(partition_specs):
    if not partition_specs:
        raise ValueError("At least one partition specification is required")

    total_weight = 0
    for index, (algorithm, res_value, weight) in enumerate(partition_specs, start=1):
        _validate_algorithm(algorithm, f"algorithm in partition spec {index}")
        if not np.isfinite(res_value):
            raise ValueError(f"Parameter in partition spec {index} must be finite")
        if weight < 0 or not np.isfinite(weight):
            raise ValueError(f"Weight in partition spec {index} must be finite and non-negative")
        total_weight += weight

    if total_weight <= 0:
        raise ValueError("Partition specification weights must sum to > 0")


def _validate_fast_ensemble_args(partition_specs, algorithm, n_p, tr, final_alg, final_param):
    if not 0 <= tr <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    _validate_algorithm(final_alg, "final algorithm")
    if not np.isfinite(final_param):
        raise ValueError("Final parameter must be finite")

    if partition_specs is None:
        _validate_algorithm(algorithm)
        if n_p <= 0:
            raise ValueError("Number of partitions must be > 0")
    else:
        _validate_partition_specs(partition_specs)


def summarize_partition(membership):
    cluster_sizes = {}
    for cluster_id in membership.values():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    sizes = list(cluster_sizes.values())
    if not sizes:
        return {
            "clusters": 0,
            "singletons": 0,
            "min_size": 0,
            "max_size": 0,
            "mean_size": 0,
            "median_size": 0,
        }

    return {
        "clusters": len(sizes),
        "singletons": sum(1 for size in sizes if size == 1),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "mean_size": statistics.mean(sizes),
        "median_size": statistics.median(sizes),
    }


def log_partition_summary(logger, membership):
    summary = summarize_partition(membership)
    if logger is None:
        return summary

    logger.info(
        "Final partition summary: clusters=%s, singletons=%s, "
        "min_size=%s, max_size=%s, mean_size=%.2f, median_size=%.2f",
        summary["clusters"],
        summary["singletons"],
        summary["min_size"],
        summary["max_size"],
        summary["mean_size"],
        summary["median_size"],
    )
    return summary


def read_network(edge_list_path, weighted_input=False):
    if weighted_input:
        graph = nx.read_weighted_edgelist(edge_list_path, nodetype=int)
        for _, _, data in graph.edges(data=True):
            data[INPUT_WEIGHT_ATTR] = data["weight"]
        return graph

    return nx.read_edgelist(edge_list_path, nodetype=int)


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
        partition_specs = [
            (alg_list[i], param_list[i], weight_list[i])
            for i in range(len(alg_list))
        ]
        _validate_partition_specs(partition_specs)
        return partition_specs

    # homogeneous mode
    if n_p <= 0:
        raise ValueError("Number of partitions must be > 0")
    alpha = 1.0 / n_p
    partition_specs = [(algorithm, res_value, alpha) for _ in range(n_p)]
    _validate_partition_specs(partition_specs)
    return partition_specs


def _partition_worker_spec(args):
    graph, alg, res, seed, weighted = args
    ig_graph = None if alg == 'louvain' else ig.Graph.from_networkx(graph)
    return get_communities(graph, alg, seed, res_val=res,
                           weighted=weighted,
                           ig_graph=ig_graph)


def fast_ensemble(G, partition_specs=None, algorithm=None,
                  res_value=0.01, n_p=10, tr=0.8, final_alg='leiden-cpm', final_param=0.01,
                  weighted='weight', use_parallel=False, logger=None, seed=0):
    total_start = time.perf_counter()
    _validate_fast_ensemble_args(partition_specs, algorithm, n_p, tr, final_alg, final_param)
    graph = G.copy()
    graph = initialize(graph, 1.0)
    _log_step(logger, "Initialized working graph with unit edge weights")

    if partition_specs is None:
        partition_specs = build_partition_specs(
            algorithm=algorithm,
            res_value=res_value,
            n_p=n_p
        )
    _log_step(logger, f"Prepared {len(partition_specs)} partition specification(s)")

    # ---- Generate partitions ----
    partition_start = time.perf_counter()
    if use_parallel:
        _log_step(logger, "Generating ensemble partitions with multiprocessing")
        with ProcessPoolExecutor() as executor:
            partitions = list(executor.map(
                _partition_worker_spec,
                [(graph, alg, res, seed + i, weighted)
                 for i, (alg, res, _) in enumerate(partition_specs)]
            ))
    else:
        _log_step(logger, "Generating ensemble partitions")
        ig_cache = {}
        partitions = []
        for i, (alg, res, _) in enumerate(partition_specs):
            part_start = time.perf_counter()
            if alg != 'louvain':
                if alg not in ig_cache:
                    ig_cache[alg] = ig.Graph.from_networkx(graph)
                ig_graph = ig_cache[alg]
            else:
                ig_graph = None

            partitions.append(
                get_communities(graph, alg, seed + i,
                                res_val=res,
                                weighted=weighted,
                                ig_graph=ig_graph)
            )
            _log_step(
                logger,
                f"Partition {i + 1}/{len(partition_specs)} ({alg}, parameter={res})",
                part_start
            )
    _log_step(logger, f"Generated {len(partitions)} ensemble partition(s)", partition_start)

    remove_edges = []

    threshold_start = time.perf_counter()
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
    _log_step(
        logger,
        f"Applied threshold {tr}; removed {len(remove_edges)} edge(s), "
        f"kept {graph.number_of_edges()} edge(s)",
        threshold_start
    )

    final_start = time.perf_counter()
    _log_step(logger, f"Running final clustering with {final_alg}, parameter={final_param}")
    final_ig = None if final_alg == 'louvain' else ig.Graph.from_networkx(graph)

    final_partition = get_communities(graph,
                                      final_alg,
                                      seed,
                                      res_val=final_param,
                                      weighted=weighted,
                                      ig_graph=final_ig)
    _log_step(
        logger,
        f"Final clustering produced {len(set(final_partition.values()))} cluster(s)",
        final_start
    )
    log_partition_summary(logger, final_partition)
    _log_step(logger, "FastEnsemble run", total_start)
    return final_partition


def read_alg_list(list_path):
    alg_list, param_list, weight_list = [], [], []
    with open(list_path) as fgt:
        for line_number, line in enumerate(fgt, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid algorithm-list line {line_number}: expected "
                    "'<algorithm> <parameter> <weight>'"
                )

            alg, param, weight = parts
            _validate_algorithm(alg, f"algorithm on line {line_number}")

            try:
                param = float(param)
                weight = float(weight)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value on algorithm-list line {line_number}"
                ) from exc

            if not np.isfinite(param):
                raise ValueError(f"Parameter on algorithm-list line {line_number} must be finite")
            if weight < 0 or not np.isfinite(weight):
                raise ValueError(
                    f"Weight on algorithm-list line {line_number} must be finite and non-negative"
                )

            alg_list.append(alg)
            param_list.append(param)
            weight_list.append(weight)

    if not alg_list:
        raise ValueError("Algorithm list is empty")

    return alg_list, np.asarray(param_list), np.asarray(weight_list)


def main():
    parser = argparse.ArgumentParser(description="FastEnsemble Clustering")
    parser.add_argument("-n", "--edgelist", type=str,  required=True,
                        help="Network edge-list file")
    parser.add_argument("-alg", "--algorithm", type=str, required=False,
                        choices=sorted(ALLOWED_ALGORITHMS),
                        help="Clustering algorithm (leiden-mod, leiden-cpm or louvain)", default='leiden-cpm')
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output community file")
    parser.add_argument("-t", "--threshold", type=float, required=False,
                        help="Threshold value", default=0.8)
    parser.add_argument("-r", "--resolution", type=float, required=False,
                        help="Resolution value for leiden-cpm", default=0.01)
    parser.add_argument("-falg", "--finalalgorithm", type=str, required=False,
                        choices=sorted(ALLOWED_ALGORITHMS),
                        help="Clustering algorithm for the final step", default=None)
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
    parser.add_argument("-wi", "--weighted-input", required=False, action='store_true',
                        help="Read the third edge-list column as an input edge weight",
                        default=False)
    parser.add_argument("-mp", "--multiprocessing",
                        required=False, action='store_true',
                        help="Enable multiprocessing for partition generation",
                        default=False)
    parser.add_argument("-s", "--seed", type=int, required=False,
                        help="Base random seed for partition generation and final clustering",
                        default=0)
    parser.add_argument("-q", "--quiet",
                        required=False, action='store_true',
                        help="Suppress progress logging",
                        default=False)

    args = parser.parse_args()
    if not os.path.isfile(args.edgelist):
        parser.error(f"Edge-list file does not exist: {args.edgelist}")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.isdir(output_dir):
        parser.error(f"Output directory does not exist: {output_dir}")

    if args.algorithmlist and not os.path.isfile(args.algorithmlist):
        parser.error(f"Algorithm-list file does not exist: {args.algorithmlist}")

    if not 0 <= args.threshold <= 1:
        parser.error("Threshold must be between 0 and 1")

    if args.partitions <= 0:
        parser.error("Number of partitions must be > 0")

    if not np.isfinite(args.resolution):
        parser.error("Resolution value must be finite")

    if args.finalparam is not None and not np.isfinite(args.finalparam):
        parser.error("Final parameter must be finite")

    logger = logging.getLogger(LOGGER_NAME)
    if not args.quiet:
        logger.setLevel(logging.NOTSET)
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)

    total_start = time.perf_counter()
    read_start = time.perf_counter()
    logger.info("Reading edge list from %s", args.edgelist)
    try:
        net = read_network(args.edgelist, weighted_input=args.weighted_input)
    except Exception as exc:
        parser.error(f"Could not read edge-list file: {exc}")
    logger.info(
        "Loaded network with %s node(s) and %s edge(s) in %s",
        net.number_of_nodes(),
        net.number_of_edges(),
        _elapsed(read_start)
    )
    if args.weighted_input:
        logger.info("Using third edge-list column as input edge weights")

    # relabeling nodes from 0 to n-1
    if args.relabel:
        relabel_start = time.perf_counter()
        mapping = dict(zip(sorted(net), range(0, net.number_of_nodes())))
        net = nx.relabel_nodes(net, mapping)
        reverse_mapping = {y: x for x, y in mapping.items()}
        _log_step(logger, "Relabeled network nodes", relabel_start)

    n_p = args.partitions
    if not args.finalalgorithm:
        args.finalalgorithm = args.algorithm
    if args.finalparam is None:
        args.finalparam = args.resolution

    if args.algorithmlist:
        alg_list_start = time.perf_counter()
        try:
            alg_list, param_list, weight_list = read_alg_list(args.algorithmlist)
            partition_specs = build_partition_specs(
                alg_list=alg_list,
                param_list=param_list,
                weight_list=weight_list
            )
        except ValueError as exc:
            parser.error(str(exc))
        logger.info(
            "Loaded %s algorithm-list entry/entries from %s in %s",
            len(partition_specs),
            args.algorithmlist,
            _elapsed(alg_list_start)
        )
    else:
        partition_specs = None

    if partition_specs is not None:
        algorithm = None

    weighted_attr = None if args.noweight else "weight"
    if args.weighted_input and not args.noweight:
        weighted_attr = INPUT_WEIGHT_ATTR

    logger.info("Using base random seed %s", args.seed)

    fe = fast_ensemble(net,
                       partition_specs=partition_specs,
                       algorithm=args.algorithm,
                       res_value=args.resolution,
                       n_p=args.partitions,
                       tr=args.threshold,
                       final_alg=args.finalalgorithm,
                       final_param=args.finalparam,
                       weighted=weighted_attr,
                       use_parallel=args.multiprocessing,
                       logger=logger,
                       seed=args.seed)

    output_start = time.perf_counter()
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
    logger.info("Wrote %s node membership(s) to %s in %s",
                len(membership), args.output, _elapsed(output_start))
    logger.info("Completed FastEnsemble CLI run in %s", _elapsed(total_start))


if __name__ == "__main__":
    main()
