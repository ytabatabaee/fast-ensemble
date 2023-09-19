import leidenalg
import networkx as nx
import igraph as ig
import argparse
import csv
import community.community_louvain as cl
import numpy as np
import subprocess
import multiprocessing
from pathlib import Path
from multiprocessing import Pool
import shlex
import click


def communities_to_dict(communities):
    result = {}
    community_index = 0
    for c in communities:
        community_mapping = ({node: community_index for index, node in enumerate(c)})
        result = {**result, **community_mapping}
        community_index += 1
    return result

def get_communities_wrapper(args):
    return get_communities(*args)

def get_communities(edgelist, algorithm, seed, r=0.001, graph=None):
    if not graph:
        graph = nx.read_edgelist(edgelist, nodetype=int)
        nx.set_edge_attributes(graph, values=1, name="weight")
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
    raise ValueError(f"{algorithm} not implemented in get_communities()")


def thresholding(graph, thresh):
    remove_edges = []
    bound = thresh
    for u, v in graph.edges():
        if graph[u][v]['weight'] < bound:
            remove_edges.append((u, v))
    graph.remove_edges_from(remove_edges)
    return graph

# strict consensus can be achieved by running threshold consensus with tr=1
@click.command()
@click.option("--edgelist", "-n", type=click.Path(exists=True), required=True, help="Network edge-list file")
@click.option("--num-processors", type=int, default=1, help="Number of parallel workers for the partitions")
@click.option("--threshold", "-t", type=float, default=1.0, required=True, help="Threshold value")
@click.option("--algorithm", "-a", type=click.Choice(["leiden-cpm", "leiden-mod", "louvain"]), default="leiden-cpm", required=True, help="Clustering algorithm")
@click.option("--resolution", "-r", type=float, default=0.01, help="Resolution value for ledien-cpm")
@click.option("--num-partitions", "-p", type=int, default=10, help="Number of partitions in consensus clustering")
@click.option("--output-file", type=click.Path(), required=True, help="Output clustering file")
def threshold_consensus(edgelist, num_processors, threshold, algorithm, resolution, num_partitions, output_file):
    pool = Pool(num_processors)
    args_arr = []

    for i in range(num_partitions):
        current_args = (edgelist, algorithm, i, resolution)
        args_arr.append(current_args)
    results = pool.map(get_communities_wrapper, args_arr)

    graph = nx.read_edgelist(edgelist, nodetype=int)
    nx.set_edge_attributes(graph, values=1, name="weight")

    for c in results:
        for node, nbr in graph.edges():
            if c[node] != c[nbr]:
                graph[node][nbr]['weight'] -= 1/num_partitions

    graph = thresholding(graph, threshold)
    seed = 0
    tc = get_communities("", algorithm, seed, graph=graph)
    with open(f"{output_file}", "w") as f:
        for node, mem in tc.items():
            f.write(f"{node} {mem}\n")

if __name__ == "__main__":
    threshold_consensus()
