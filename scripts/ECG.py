import igraph as ig
import numpy as np
import networkx as nx
import sys
import csv

## needs 2 arguments
if len(sys.argv) != 3:
    print('Usage: ',sys.argv[0],' input_graph_file_path output_partition_filepath')
    exit(-1)

## arguments
graph_filepath = sys.argv[1] 
output_filepath = sys.argv[2] 

## add ECG to the choice of community algorithms
def community_ecg(self, weights=None, ens_size=16, min_weight=0.05):
    W = [0]*self.ecount()
    ## Ensemble of level-1 Louvain
    for i in range(ens_size):
        p = np.random.permutation(self.vcount()).tolist()
        g = self.permute_vertices(p)
        l = g.community_multilevel(weights=weights, return_levels=True)[0].membership
        b = [l[p[x.tuple[0]]]==l[p[x.tuple[1]]] for x in self.es]
        W = [W[i]+b[i] for i in range(len(W))]
    W = [min_weight + (1-min_weight)*W[i]/ens_size for i in range(len(W))]
    part = self.community_multilevel(weights=W)
    part._modularity_params['weights'] = weights
    part.recalculate_modularity()
    ## Force min_weight outside 2-core
    core = self.shell_index()
    ecore = [min(core[x.tuple[0]],core[x.tuple[1]]) for x in self.es]
    part.W = [W[i] if ecore[i]>1 else min_weight for i in range(len(ecore))]
    part.CSI = 1-2*np.sum([min(1-i,i) for i in part.W])/len(part.W)
    return part
ig.Graph.community_ecg = community_ecg

net = nx.read_edgelist(graph_filepath, nodetype=int)
mapping = dict(zip(net, range(0, net.number_of_nodes())))
net = nx.relabel_nodes(net, mapping)
reverse_mapping = {y: x for x, y in mapping.items()}

g = ig.Graph.from_networkx(net)
#Graph.Read_Edgelist(graph_filepath, directed=False)

ec = g.community_ecg(ens_size=10)
#print(ec)
with open(output_filepath, 'w') as out_file:
    writer = csv.writer(out_file, delimiter=' ')
    for i in range(len(ec.membership)):
        writer.writerow([reverse_mapping[i]] + [ec.membership[i]])
