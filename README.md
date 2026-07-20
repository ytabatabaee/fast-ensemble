# Fast Ensemble Clustering
[![PyPI version](https://img.shields.io/pypi/v/fast-ensemble-clustering.svg?cacheSeconds=3600)](https://pypi.org/project/fast-ensemble-clustering/)
[![PyPI license](https://img.shields.io/pypi/l/fast-ensemble-clustering.svg?cacheSeconds=3600)](https://pypi.org/project/fast-ensemble-clustering/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/fast-ensemble-clustering.svg?cacheSeconds=3600)](https://pypi.org/project/fast-ensemble-clustering/)
[![DOI](https://img.shields.io/badge/DOI-10.1371%2Fjournal.pcsy.0000069-blue.svg)](https://doi.org/10.1371/journal.pcsy.0000069)

**FastEnsemble** is a scalable ensemble clustering method that can be used with one or a combination of clustering algorithms. It is currently implemented for use with **Leiden** optimizing **CPM** or **modularity** and the **Louvain** algorithm. 

FastEnsemble supports
- repeated runs of a single clustering algorithm
- combinations of multiple clustering algorithms
- **multi-resolution** ensemble clustering
- **weighted** ensembles
-  multiprocessing

## Installation
FastEnsemble is implemented in Python 3 and can be installed from PyPI:
```
$ python3 -m pip install fast-ensemble-clustering
```

To install the development version from this repository:
```
$ git clone https://github.com/ytabatabaee/fast-ensemble.git
$ cd fast-ensemble
$ python3 -m pip install .
```

To verify successful installation and view command-line options:
```
$ fast-ensemble-clustering --help
```

## Usage
In its simplest form, FastEnsemble combines multiple runs of a *single* clustering algorithm, and can be used with the following command:
```
$ fast-ensemble-clustering -n <edge-list> -o <output-membership> [-t <threshold>] [-alg <algorithm>] [-r <resolution-value>] [-p <number-of-partitions>] [-falg <final-algorithm>] [-fr <final-param>] [-s <seed>] [-wi]
```
The output clustering membership is in the format `<node_id> <community_id>`. 

**Arguments**
- **Required**
```
 -n,  --edgelist               input network edge-list
 -o,  --output                 output community membership
```
- **Optional**
```
 -t,  --threshold              threshold value
 -alg,  --algorithm            clustering algorithm (leiden-cpm, leiden-mod, louvain)
 -r,  --resolution             resolution value for leiden-cpm
 -falg, --finalalgorithm       clustering algorithm for the final step (leiden-cpm, leiden-mod, louvain) - same as -alg if not specified
 -fr, --finalparam             parameter (e.g. resolution value) for the final algorithm - same as -r if not specified
 -p,  --partitions             number of partitions used in consensus clustering
 -alglist,  --algorithmlist    list of clustering algorithms, with parameters and weights
 -rl, --relabel                relabel network nodes from 0 to #nodes-1
 -nw, --noweight               ignore edge weights when clustering
 -wi, --weighted-input         read the third edge-list column as an input edge weight
 -mp, --multiprocessing        enable multiprocessing
 -s, --seed                    base random seed for partition generation and final clustering
 -q, --quiet                   suppress progress logging
```
Use `--seed` for reproducible runs. Partition seeds are derived from the base seed, and the final clustering also uses the base seed.

For weighted edge lists, use `--weighted-input`. The input file should contain three columns: `<node_id> <node_id> <edge_weight>`. Use `--noweight` to ignore edge weights during clustering.

To create a heterogeneous ensemble that allows for an arbitrary combination of clustering algorithms with different parameters (e.g. resolution values) and weights, use the `-alglist` parameter:
```
$ fast-ensemble-clustering -n <edge-list> -o <output-membership> -alglist <algorithm-list> [-falg <final-algorithm> -fr <final-param> -t <threshold>]
```
Each line in the algorithm list should be in the format `<algorithm> <resolution> <weight>`, for example
```
leiden-cpm	0.01	1
leiden-cpm	0.001	2
leiden-mod	1	1
leiden-mod	1	1
leiden-mod	1	1
```
where:
- `<algorithm>` is the clustering algorithm (currently `louvain`, `leiden-mod` and `leiden-cpm` are supported)
- `<resolution>` is the resolution parameter (or other relevant parameters) for `<algorithm>`
- `<weight>` is a weight that specifies the algorithm's influence over the edge weights in the final clustering.

## Example

We demonstrate the use of FastEnsemble on the [Youtube social network](https://snap.stanford.edu/data/com-Youtube.html) and the [Amazon product co-purchasing network](https://snap.stanford.edu/data/com-Amazon.html) from the [SNAP collection](https://snap.stanford.edu/index.html). The [/data](https://github.com/ytabatabaee/fast-ensemble/tree/main/data) directory includes example inputs and outputs.

#### Homogeneous Ensemble

In the simplest setting, FastEnsemble combines multiple runs of a single clustering algorithm:
```
$ fast-ensemble-clustering -n data/youtube-network.dat -t 0.8 -alg leiden-cpm --output data/fe_youtube.dat 
```
#### Heterogeneous Ensemble

FastEnsemble also supports combining different algorithms and resolution values in a single ensemble through an algorithm list:
```
$ fast-ensemble-clustering -n data/amazon-network.dat -alglist data/inputs/ensemble_mod_0.01.txt -o data/fe_weighted_amazon.dat
```
In this example, the file `ensemble_mod_0.01.txt` specifies a mixture of Leiden-modularity and Leiden-CPM runs with associated parameters and weights.


## Publication

If you use FastEnsemble, please cite the following paper:

> Y. Tabatabaee, E. Wedell, M. Park, and T. Warnow. 2025. “FastEnsemble: Scalable Ensemble Clustering on Large Networks.” *PLOS Complex Systems* 2(10): e0000069. https://doi.org/10.1371/journal.pcsy.0000069

A preliminary version of this work appeared at the 2024 International Conference on Complex Networks and Their Applications: https://doi.org/10.1007/978-3-031-82435-7_5

### Data Availability

Datasets and scripts from these papers are available at [ensemble-clustering-data](https://github.com/ytabatabaee/ensemble-clustering-data) repository.

## Calculating accuracy and clustering statistics

### Calculating mixing parameters and clustering statistics
The script [scripts/evaluate_partition.py](https://github.com/ytabatabaee/fast-ensemble/tree/main/scripts/evaluate_partition.py) can be used to evaluate the output partition in terms of cluster statistics, mixing parameter, and modularity, with the following command:
```
$ python3 evaluate_partition.py -n <edge-list> -m <partition-membership>
```

### Calculating accuracy
The script [scripts/clustering_accuracy.py](https://github.com/ytabatabaee/fast-ensemble/tree/main/scripts/clustering_accuracy.py) can be used for computing multiple accuracy measures (NMI, AMI, ARI, false positive rate, false negative rate, precision, recall and F1-score) for a clustering with respect to a ground-truth community membership.
```
$ python3 clustering_accuracy.py -gt <ground-truth-membership> -p <estimated-partition>
```
