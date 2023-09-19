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
import multiprocessing as mp
import shlex
import click
import time
import datetime
from enum import Enum


START_TIME = time.monotonic()
GLOBAL_LOG_FILE = None
GLOBAL_LOG_LOCK = mp.Lock()

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
    write_to_log_file("get_communities() called with {algorithm} and seed={seed}\n")
    if not graph:
        graph = nx.read_edgelist(edgelist, nodetype=int)
        nx.set_edge_attributes(graph, values=1, name="weight")
    if algorithm == 'louvain':
        partition = cl.best_partition(graph, random_state=seed, weight='weight')
        write_to_log_file("get_communities() finished with {algorithm} and seed={seed}\n")
        return partition
    elif algorithm == 'leiden-cpm':
        partition = communities_to_dict(leidenalg.find_partition(ig.Graph.from_networkx(graph),
                                                  leidenalg.CPMVertexPartition,
                                                  resolution_parameter=r,
                                                  n_iterations=2).as_cover())
        write_to_log_file("get_communities() finished with {algorithm} and seed={seed}\n")
        return partition
    elif algorithm == 'leiden-mod':
        partition = communities_to_dict(leidenalg.find_partition(ig.Graph.from_networkx(graph),
                                        leidenalg.ModularityVertexPartition,
                                        weights='weight',
                                        seed=seed).as_cover())
        write_to_log_file("get_communities() finished with {algorithm} and seed={seed}\n")
        return partition
    raise ValueError(f"{algorithm} not implemented in get_communities()")


def thresholding(graph, thresh):
    remove_edges = []
    bound = thresh
    for u, v in graph.edges():
        if graph[u][v]['weight'] < bound:
            remove_edges.append((u, v))
    graph.remove_edges_from(remove_edges)
    return graph

class LogLevel(Enum):
    INFO = 0
    ERROR = 1

def write_to_log_file(message, level=LogLevel.INFO):
    global START_TIME
    global GLOBAL_LOG_FILE
    global GLOBAL_LOG_LOCK

    seconds_elapsed = time.monotonic() - START_TIME
    human_readable_time_elapsed = datetime.timedelta(seconds=seconds_elapsed)

    if(level == LogLevel.INFO):
        GLOBAL_LOG_FILE.write(f"[INFO] t={seconds_elapsed} ({human_readable_time_elapsed}): {message}\n")
    elif(level == LogLevel.ERROR):
        GLOBAL_LOG_FILE.write(f"[ERROR] t={seconds_elapsed} ({human_readable_time_elapsed}): {message}\n")

# strict consensus can be achieved by running threshold consensus with tr=1
@click.command()
@click.option("--edgelist", "-n", type=click.Path(exists=True), required=True, help="Network edge-list file")
@click.option("--num-processors", type=int, default=1, help="Number of parallel workers for the partitions")
@click.option("--threshold", "-t", type=float, default=1.0, required=True, help="Threshold value")
@click.option("--algorithm", "-a", type=click.Choice(["leiden-cpm", "leiden-mod", "louvain"]), default="leiden-cpm", required=True, help="Clustering algorithm")
@click.option("--resolution", "-r", type=float, default=0.01, help="Resolution value for ledien-cpm")
@click.option("--num-partitions", "-p", type=int, default=10, help="Number of partitions in consensus clustering")
@click.option("--output-file", type=click.Path(), required=True, help="Output clustering file")
@click.option("--log-file", type=click.Path(), required=True, help="Output log file")
def threshold_consensus(edgelist, num_processors, threshold, algorithm, resolution, num_partitions, output_file, log_file):
    global START_TIME
    global GLOBAL_LOG_FILE
    global GLOBAL_LOG_LOCK

    with GLOBAL_LOG_LOCK:
        GLOBAL_LOG_FILE = open(f"{log_file}", "w")
        GLOBAL_LOG_FILE.write(f"# This log file starts at time t = 0\n")
        GLOBAL_LOG_FILE.write(f"# format is [INFO/ERROR] t=<seconds elapsed> (<human readable time since start>): <message>\n")


    ## worker setup
    # creating pool
    pool = mp.Pool(num_processors)
    args_arr = []
    write_to_log_file("Worker pool with {num_processors} created")
    for i in range(num_partitions):
        current_args = (edgelist, algorithm, i, resolution)
        args_arr.append(current_args)
    # assigning workers
    write_to_log_file("Starting workers")
    results = pool.map(get_communities_wrapper, args_arr)
    write_to_log_file("All workers finised")


    ## final clustering
    # final graph setup
    write_to_log_file("Started reading graph for the final clustering")
    graph = nx.read_edgelist(edgelist, nodetype=int)
    write_to_log_file("Finished reading graph for the final clustering")
    write_to_log_file("Started setting edge weights for the final graph")
    nx.set_edge_attributes(graph, values=1, name="weight")
    write_to_log_file("Finished setting edge weights for the final graph")

    # final graph edgeweight manipulation
    write_to_log_file("Started subtracting edge weights for the final graph")
    for c in results:
        for node, nbr in graph.edges():
            if c[node] != c[nbr]:
                graph[node][nbr]['weight'] -= 1/num_partitions
    write_to_log_file("Finished subtracting edge weights for the final graph")

    # final graph edgeweight manipulation part 2
    write_to_log_file("Started thresholding for the final graph")
    graph = thresholding(graph, threshold)
    write_to_log_file("Finished thresholding for the final graph")

    # computing final clustering
    seed = 0
    tc = get_communities("", algorithm, seed, graph=graph)

    # writing final clustering
    write_to_log_file("Started writing the final output clustering")
    with open(f"{output_file}", "w") as f:
        for node, mem in tc.items():
            f.write(f"{node} {mem}\n")
    write_to_log_file("Finished writing the final output clustering")
    GLOBAL_LOG_FILE.close()

if __name__ == "__main__":
    threshold_consensus()
