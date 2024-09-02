# Fast Ensemble Clustering

This respository includes a python implementation of *FastEnsemble* clustering. 


In its simplest form, *FastEnsemble* uses three main parameters: the clustering method, the number of partitions $np$, and the threshold $t$. Given an input matrix $N$, *FastEnsemble* uses the specified clustering method to generate $n_p$ partitions of $N$, and then builds a new network on the same node and edge set but with the edges weighted by the number of partitions the endpoints are in the same cluster. If a given edge has weight less than $t \times np$ (indicating that its endpoints are co-clustered in fewer than that many partitions), then the edge is removed from the network; hence the new network can have fewer edges than the original network.   The new weighted network is then clustered just once more using the selected clustering method.

*FastEnsemble* can be used with one or a combination of clustering paradigms, and we have implemented it for use with Leiden optimizing CPM, Leiden optimizing modularity, Louvain, and other methods. This implementation also allows the constituent clustering methods to be weighted, so that some clustering methods have more influence than the others.

## Dependencies
*FastEnsemble* is implemented in Python 3 and have the following dependencies:
- [Python 3.x](https://www.python.org)
- [NetworkX](https://networkx.org)
- [Numpy](https://numpy.org)

If you have Python 3 and pip, you can use `pip install -r requirements.txt` to install the other dependencies. 

## Usage Instructions
*FastEnsemble* in its simplest form (using multiple runs of a *single* clustering algorithm), can be run with the following command:
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
```
A more advanced version, that allows for an arbitary combination of different clustering algorithms with different parameters (e.g. resolution values) and weights can be run with the following command:
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
An example algorithm list can be as follows:
```
leiden-cpm	0.01	1
leiden-cpm	0.001	2
leiden-mod	1	1
leiden-mod	1	1
leiden-mod	1	1
```
where each line has three elements: name of the clustering algorithm (currently louvain, leiden-mod and leiden-cpm are supported), the resolution parameter for that algorithm, and its weight that specifies its influence over the edge weights in the final clustering.

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

