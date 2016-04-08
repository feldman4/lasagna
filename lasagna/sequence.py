import networkx as nx
import Levenshtein
import numpy as np




def edit_distance(sequences):
    """Calculate pairwise distances, return in lower triangular matrix.
    """
    distances = np.zeros(2 * [len(sequences)])
    for i, a in enumerate(sequences):
        for j, b in enumerate(sequences[:i]):
            distances[i,j] = Levenshtein.distance(a, b)
    return distances

def edit_distance_sparse(sequences, k=8):
    # use rolling kmer as local similarity measure
    from collections import defaultdict
    from scipy.sparse import coo_matrix
    kmers = defaultdict(list)

    for i, s in enumerate(sequences):
        s2 = s*(20+k)
        for j in range(20):
            kmers[s2[j:j+k]] += [i]

    # pairs to calculate
    pairs = []
    for v in kmers.values():
        pairs += [(a,b) for i, a in enumerate(v) for b in v[:i]]
    pairs = set(pairs)

    n = len(sequences)
    I,J,V = [], [], []
    for i,j in pairs:
        s1, s2 = sequences[i], sequences[j]
        d = Levenshtein.distance(s1,s2)
        V += [d]
        I += [i]
        J += [j]
    return coo_matrix((V, (I,J)), shape=(n,n))


def make_adjacency(distances, counts, threshold=1):
    """
    Input: pairwise distance (doesn't need to be symmetric?)
    Output: networkx DiGraph
    """
    counts = np.array(counts)
    distances = np.array(distances)
    # find local maxima in adjacency graph
    G = (distances <= threshold) & (distances > 0)

    # scale columns by total reads
    # symmetrize
    G = ((1*G + 1*G.T) > 0).astype(float)
    G = G * counts - counts[:,None]
    G[G < 0] = 0
    return G

def make_adjacency_sparse(distances, counts):
    """
    Input: sparse pairwise distance
    Ouptut: networkx DiGraph
    """
    counts = np.array(counts)
    # find local maxima in adjacency graph
    G = (distances == 1)

    # scale columns by total reads
    # symmetrize
    G = ((1*G + 1*G.T) > 0).astype(float)
    # sparse equivalent of
    # G = G * counts - counts[:,None]
    # G[G < 0] = 0
    G = G.tocsr()
    G.data *= counts[G.indices]
    G = G.tocsc()
    G.data -= counts[G.indices]
    G[G<0] = 0
    return G



def edit_str(a,b):
    arr = []
    op_dict = {'replace': 'R', 'delete': 'D', 'insert': 'I'}
    for op, i, j in Levenshtein.editops(a, b):
        arr += ['%s%d:%s->%s' % (op_dict[op], i, a[i], b[j])]
        # need to track actual insertions/deletions from here on...
        if op in ('insert', 'delete'):
            arr += ['***']
            break
    return '\n'.join(arr)


class UMIGraph(nx.DiGraph):

    def __init__(self, sequences, counts, threshold=1, kmer=None):
        self.sequences = sequences
        self.counts = counts
        if kmer:
            # distance threshold to include edge in adjacency matrix set to 1
            self.distances = edit_distance_sparse(sequences, k=kmer)
            self.G = make_adjacency_sparse(self.distances, counts)
        else:
            self.distances = edit_distance(sequences)
            self.G = make_adjacency(self.distances, counts, threshold=threshold)

        super(UMIGraph, self).__init__(self.G)

        self.components = list(self.find_components())

    def find_components(self):
        components = nx.weakly_connected_component_subgraphs(self)
        return sorted(list(components), key=lambda x: -1 * len(x.nodes()))

    def _default_label(self, base, node):
        edit = edit_str(self.sequences[node], self.sequences[base])
        return '%s\n%d' % (edit, self.counts[node]) 

    def _default_base_label(self, base, node):
        return '%s\n%d' % (self.sequences[base], self.counts[base])
    
    def label_nodes(self, subgraph, label_fcn=None, base_label_fcn=None):
        
        label_fcn = label_fcn or self._default_label
        base_label_fcn = base_label_fcn or self._default_base_label

        base = nx.topological_sort(subgraph.reverse())[0]
        labels = {}
        for node in subgraph.nodes():
            if node == base:
                labels[node] = base_label_fcn(base, node)
            else:
                labels[node] = label_fcn(base, node)

        return labels

    def __class__(self):
        """Needed for self.copy() to work.
        """
        return nx.DiGraph().__class__()

    def draw(self, subgraph, ax=None, labels=None, layout=nx.circular_layout):
        import matplotlib.pyplot as plt



        ax = ax or plt.gca()
        labels = labels or self.label_nodes(subgraph)

        node_size = np.array(self.counts)[subgraph.nodes()]
        node_size = (node_size / max(node_size))*2000 + 200
        print node_size

        nx.draw_networkx(subgraph, labels=self.label_nodes(subgraph), 
                         color='w', node_size=node_size, pos=layout(subgraph), 
                         ax=ax)







