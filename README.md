# Consensus clustering

This respository includes Simple Consensus (SC) and Threshold Consensus (TC). The algorithms and the codes use many ideas from the Fast Consensus Clustering software available at https://github.com/kaiser-dan/fastconsensus and the paper [Fast consensus clustering in complex networks](https://arxiv.org/pdf/1902.04014.pdf), Phys. Rev. E., 2019.

### Threshold consensus
This implementation of the Threshold Consensus runs a clustering algorithm $n_p$ times with different random seeds in a single iteration and only keeps the edges that appear in at least $\tau$ proportion of the partitions. When $\tau=1$, this is equivalent to *strict* consensus.
```
$ python3 threshold_consensus.py -n <edge-list> -t <threshold> -a <algorithm> -r <resolution-value> -p <number-partitions>
```
**Arguments**
```
 -n,  --edgelist           input network edge-list
 -t,  --threshold          threshold value
 -a,  --algorithm          clustering algorithm (leiden-cpm, leiden-mod, louvain)
 -r,  --resolution         resolution value for leiden-cpm
 -p,  --partitions         number of partitions used in consensus clustering
 -rl, --relabel            relabel network nodes from 0 to #nodes-1
```
### Simple consensus
See description in the report. This does not yet support IKC and strict consensus in the final step and the algorithm list should be set manually in the code. The best-scoring selection in the final step is also not included yet, and the first algorithm in the list will be used to create the final partition.
```
$ python3 simple_consensus.py -n <edge-list> -t <threshold> -r <resolution-value> -p <number-partitions>
```
**Arguments**
```
 -n,  --edgelist           input network edge-list
 -t,  --threshold          threshold value
 -a,  --algorithm          clustering algorithm (leiden-cpm, leiden-mod, louvain)
 -r,  --resolution         resolution value for leiden-cpm
 -p,  --partitions         number of partitions used in consensus clustering
 -d,  --delta              convergence parameter
 -p,  --maxiter            maximum number of iterations
 -rl, --relabel            relabel network nodes from 0 to #nodes-1
```
The script `evaluate_partition.py` can be used to evaluate the output partition in terms of cluster statistics and modularity:
```
$ python3 evaluate_partition.py -n <edge-list> -m <partition-membership>
```

### Calculating accuracy
The script `clustering_accuracy.py` can be used for computing multiple accuracy measures (NMI, AMI, ARI, false positive rate, false negative rate, precision, recall and F1-score) for a clustering with respect to a ground-truth community membership. 
```
$ python3 clustering_accuracy.py -gt <ground-truth-membership> -p <estimated-partition>
```

