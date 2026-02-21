# Fast Ensemble Clustering

**FastEnsemble** is a scalable ensemble clustering method that can be used with one or a combination of clustering algorithms. It is currently implemented for use with **Leiden** optimizing **CPM** or **modularity** and the **Louvain** algorithm. 

FastEnsemble supports
- repeated runs of a single clustering algorithm
- combinations of multiple clustering algorithms
- **multi-resolution** ensemble clustering
- **weighted** ensembles
-  multiprocessing

The algorithm is described in the following paper:

Y. Tabatabaee, E. Wedell, M. Park, T. Warnow (2025). *FastEnsemble: Scalable ensemble clustering on large networks*. PLOS Complex Systems 2(10): e0000069 [preliminary version appeared at International Conference on Complex Networks and their Applications (CNA) 2024] DOI: [10.1371/journal.pcsy.0000069](https://journals.plos.org/complexsystems/article?id=10.1371/journal.pcsy.0000069)

Datasets and scripts from this study are available at [ensemble-clustering-data](https://github.com/ytabatabaee/ensemble-clustering-data) repository.

## Dependencies
FastEnsemble is implemented in Python 3 and have the following dependencies:
- [Python 3.x](https://www.python.org)
- [NetworkX](https://networkx.org)
- [igraph](https://igraph.org/)
- [leidenalg](https://leidenalg.readthedocs.io/en/stable/intro.html)
- [Numpy](https://numpy.org)

If you have Python 3 and pip, you can use `pip install -r requirements.txt` to install the other dependencies.

## Usage Instructions
In its simplest form, FastEnsemble combines multiple runs of a *single* clustering algorithm, and can be used with the following command:
```
$ python3 fast_ensemble.py -n <edge-list> [-t <threshold>] [-alg <algorithm>] [-r <resolution-value>] [-p <number-of-partitions>] [-falg <final-algorithm>] [-fr <final-param>]
```
The output clustering membership is in the format `<node_id> <community_id>`. 

**Arguments**
```
 -n,  --edgelist               input network edge-list
 -t,  --thresh                 threshold value
 -alg,  --algorithm            clustering algorithm (leiden-cpm, leiden-mod, louvain)
 -r,  --resolution             resolution value for leiden-cpm
 -falg, --finalalgorithm       clustering algorithm for the final step (leiden-cpm, leiden-mod, louvain) - same as -alg if not specified
 -fr, --finalparam             parameter (e.g. resolution value) for the final algorithm - same as -r if not specified
 -p,  --partitions             number of partitions used in consensus clustering
 -alglist,  --algorithmlist    list of clustering algorithms, with parameters and weights
 -rl, --relabel                relabel network nodes from 0 to #nodes-1
 -nw, --noweight               ignore edge weights when clustering
 -o, --output                  output community membership
 -mp, --multiprocessing        enable multiprocessing
```
To create a heterogeneous ensemble that allows for an arbitrary combination of clustering algorithms with different parameters (e.g. resolution values) and weights, use the `-alglist` parameter:
```
$ python3 fast_ensemble_weighted.py -n <edge-list> -alglist <algorithm-list> [-falg <final-algorithm> -fr <final-param> -t <threshold>]
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

We demonstrate the use of FastEnsemble on the [Youtube social network](https://snap.stanford.edu/data/com-Youtube.html) from the [SNAP collection](https://snap.stanford.edu/index.html). The [/data](https://github.com/ytabatabaee/fast-ensemble/tree/main/data) directory includes example inputs and outputs.

#### Homogeneous Ensemble

In the simplest setting, FastEnsemble combines multiple runs of a single clustering algorithm:
```
$ python3 fast_ensemble.py -n data/youtube-network.dat -t 0.8 -alg leiden-cpm --output data/fe_youtube.dat 
```
#### Heterogeneous Ensemble

FastEnsemble also supports combining different algorithms and resolution values in a single ensemble through an algorithm list:
```
$ python3 fast_ensemble.py -n data/youtube-network.dat -alglist data/inputs/ensemble_mod_0.01.txt -o data/fe_weighted_youtube_new_cpm.dat
```
In this example, the file `ensemble_mod_0.01.txt` specifies a mixture of Leiden-Modularity and Leiden-CPM runs with associated parameters and weights.

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
