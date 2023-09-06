# Consensus clustering

This respository includes two general consensus clustering frameworks: Simple Consensus (SC) and Threshold Consensus (TC). The algorithms and the codes use many ideas from the Fast Consensus Clustering software available at https://github.com/kaiser-dan/fastconsensus and the paper [Fast consensus clustering in complex networks](https://arxiv.org/pdf/1902.04014.pdf), Phys. Rev. E., 2019.

## Threshold consensus
This implementation of the Threshold Consensus runs a clustering algorithm $n_p$ times with different random seeds in a single iteration and only keeps the edges that appear in at least $\tau$ proportion of the partitions. When $\tau=1$, this is equivalent to strict consensus clustering. 
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
```
