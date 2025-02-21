# Fast Ensemble Clustering

**FastEnsemble** is an ensemble clustering method that can be used with one or a combination of clustering algorithms. It is currently implemented for use with **Leiden** optimizing CPM or modularity and the **Louvain** algorithm. Each clustering algorithm can potentially be used with different resolution values, enabling **multi-resolution** ensemble clustering. This implementation also allows the constituent clustering methods to be **weighted**, so that some clustering methods have more influence than the others on the final output.

The FastEnsemble algorithm is described in the following paper:

Y. Tabatabaee, E. Wedell, M. Park, T. Warnow. FastEnsemble: A new scalable ensemble clustering method. International Conference on Complex Networks and their Applications (CNA) 2024. Preprint available at https://arxiv.org/abs/2409.02077.

Datasets and scripts from this study are available at [ensemble-clustering-data](https://github.com/ytabatabaee/ensemble-clustering-data) repository.

## Dependencies
FastEnsemble is implemented in Python 3 and have the following dependencies:
- [Python 3.x](https://www.python.org)
- [NetworkX](https://networkx.org)
- [Numpy](https://numpy.org)

If you have Python 3 and pip, you can use `pip install -r requirements.txt` to install the other dependencies.

## Usage Instructions
In its simplest form, FastEnsemble combines multiple runs of a *single* clustering algorithm, and can be run with the following command:
```
$ python3 fast_ensemble.py -n <edge-list> -t <threshold> -alg <algorithm> [-r <resolution-value>] -p <number-of-partitions>
```
**Arguments**
```
 -n,  --edgelist           input network edge-list
 -t,  --thresh             threshold value
 -alg,  --algorithm        clustering algorithm (leiden-cpm, leiden-mod, louvain)
 -r,  --resolution         resolution value for leiden-cpm
 -p,  --partitions         number of partitions used in consensus clustering
 -rl, --relabel            relabel network nodes from 0 to #nodes-1
 -nw, --noweight           ignore edge weights when clustering
```
A more advanced version, that allows for an arbitrary combination of different clustering algorithms with different parameters (e.g. resolution values) and weights can be run with the following command:
```
$ python3 fast_ensemble_weighted.py -n <edge-list> -alg <algorithm-list> -falg <final-algorithm> -fr <final-param> -t <threshold>
```
**Arguments**
```
 -n,  --edgelist           input network edge-list
 -t,  --thresh             threshold value
 -alg,  --algorithmlist    list of clustering algorithms, with parameters and weights
 -falg, --finalalgorithm   clustering algorithm (leiden-cpm, leiden-mod, louvain)
 -fr, --finalparam         parameter (e.g. resolution value) for the final algorithm    
 -o, --output              output community file
 -rl, --relabel            relabel network nodes from 0 to #nodes-1
```
The algorithm list should be in the format [algorithm-name   resolution-value   weight], for example
```
leiden-cpm	0.01	1
leiden-cpm	0.001	2
leiden-mod	1	1
leiden-mod	1	1
leiden-mod	1	1
```
where each line has three elements: name of the clustering algorithm (currently **louvain**, **leiden-mod** and **leiden-cpm** are supported), the resolution parameter for that algorithm, and its weight that specifies its influence over the edge weights in the final clustering.

## Useful scripts

### Calculating mixing parameters and clustering statistics
The script `evaluate_partition.py` can be used to evaluate the output partition in terms of cluster statistics, mixing parameter, and modularity, with the following command:
```
$ python3 evaluate_partition.py -n <edge-list> -m <partition-membership>
```

### Calculating accuracy
The script `clustering_accuracy.py` can be used for computing multiple accuracy measures (NMI, AMI, ARI, false positive rate, false negative rate, precision, recall and F1-score) for a clustering with respect to a ground-truth community membership.
```
$ python3 clustering_accuracy.py -gt <ground-truth-membership> -p <estimated-partition>
```
