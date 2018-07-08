import networkx as nx
import Levenshtein
import numpy as np

# TODO: move to tests
# sequences  = ['ABCD', 'ABCE', 'GFAP', 'GFF', 'GFAP2']
# counts = [10, 3, 12, 1, 4]
# print lasagna.sequence.edit_distance(sequences)
# print lasagna.sequence.edit_distance_sparse(sequences, k=2).toarray()

def load_fastq(f):
    with open(f, 'r') as fh:
        return fh.read().splitlines()[1::4]

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
        repeat = int(1 + np.ceil((len(s) + k) / len(s)))
        s2 = s*repeat
        for j in range(len(s)):
            kmers[s2[j:j+k]].append(i)

    # extract pairs to calculate from bins
    pairs = []
    for v in kmers.values():
        pairs.extend((a,b) for i, a in enumerate(v) for b in v[:i])

    print len(pairs)
    
    # memory issues...
    # pairs = set(pairs)
    import pandas as pd
    pairs = pd.DataFrame(pairs).drop_duplicates().values()
    

    n = len(sequences)
    I,J,V = [], [], []
    for i,j in pairs:
        s1, s2 = sequences[i], sequences[j]
        d = Levenshtein.distance(s1,s2)
        V.append(d)
        I.append(i)
        J.append(j)
    return coo_matrix((V, (I,J)), shape=(n,n)), kmers


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
    # G[i,j] is the weight for edge from i to j
    # incoming - outgoing
    G = G * counts - counts[:,None]
    # only keep downhill edges, graph rooted at local maxima
    G[G < 0] = 0
    return G

def make_adjacency_sparse(distances, counts, threshold=1):
    """
    Input: sparse pairwise distance
    Ouptut: networkx DiGraph
    """
    counts = np.array(counts)
    # find local maxima in adjacency graph
    G = distances.copy()
    G.data = G.data <= threshold

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
    G.eliminate_zeros()
    return G

def match_patterns(seqs, patterns):
    """Match elements in N sequences to the corresponding pattern.
    Return successful matches for all patterns in dataframe. If any match
    fails, the sequences are returned in unmatched.
    E.g.,
    df, unmatched = match_patterns((read1, read2), (pat1, pat2))
    """
    import regex as re
    import pandas as pd
    arr, unmatched = [], []
    for row in zip(*seqs):
        matches = [re.findall(p, s) for p,s in zip(patterns, row)]
        if all(matches):
            arr += [sum([m[0] for m in matches], tuple())]
        else:
            unmatched += [row]
    return pd.DataFrame(arr), unmatched


def edit_str(a,b):
    arr = []
    op_dict = {'replace': 'R', 'delete': 'D', 'insert': 'I'}
    for op, i, j in Levenshtein.editops(a, b):
        if op in ('insert', 'delete'):
            arr += ['***']
            break
        arr += ['%s%d:%s->%s' % (op_dict[op], i, a[i], b[j])]
        # need to track actual insertions/deletions from here on...
        
    return '\n'.join(arr)


class UMIGraph(object):

    def __init__(self, sequences, counts=None, threshold=1, kmer=None):
        """Construct a networkx.DiGraph representing a set of sequences
        related by edit distance. Sequences are connected if edit distance
        is less than or equal to threshold. Specify kmer size to enable 
        sparse approximation of distance matrix.
        """
        if counts is None:
            sequences, counts = np.unique(sequences, return_counts=True)
        self.counts, self.sequences = zip(*sorted(zip(counts, sequences))[::-1])
        self.kmer = kmer
        self.threshold = threshold
        if kmer:
            # distance threshold to include edge in adjacency matrix set to 1
            self.distances = edit_distance_sparse(self.sequences, k=kmer)
            self.M = make_adjacency_sparse(self.distances, self.counts, threshold=threshold)
        else:
            self.distances = edit_distance(self.sequences)
            self.M = make_adjacency(self.distances, self.counts, threshold=threshold)

        self.G = nx.DiGraph(self.M)

        self.find_components()
        get_peak = lambda c: [k for k,v in c.adj.items() if not v][0]
        self.peaks = [get_peak(c) for c in self.components]
        get_counts = lambda c: sum(self.counts[i] for i in c.nodes())
        self.component_counts = [get_counts(c) for c in self.components]
        self.component_dict = {self.sequences[n]: self.sequences[p] 
                    for c,p in zip(self.components, self.peaks) 
                        for n in c}
        self.sequence_dict = {self.sequences[k]: self.sequences[v] for k,v in self.peak_map.items()}
        self.sequence_dict.update({self.sequences[k]: np.nan for k in self.ambiguous})

    def find_components(self):
        peaks = list(nx.attracting_components(self.G))
        # only local maxima
        assert all(len(p)==1 for p in peaks)
        # return set now?
        peaks = [list(p)[0] for p in peaks]
        components = []
        for peak in peaks:
            nodes = nx.shortest_path(self.G, target=peak).keys()
            components += [nodes]
        
        self.ambiguous = np.where(np.bincount([n for c in components for n in c]) > 1)
        self.ambiguous = set(list(self.ambiguous[0]))
        
        component_graphs = []
        self.peak_map = {}
        for peak, nodes in zip(peaks, components):
            nodes = set(nodes) - self.ambiguous
            self.peak_map.update({n:peak for n in nodes})
            component_graphs += [self.G.subgraph(nodes)]

        count_reads = lambda x: sum(self.counts[i] for i in x.nodes())
        self.components = sorted(list(component_graphs), key=count_reads, reverse=True)

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

    def draw(self, subgraph, ax=None, labels=None, layout=nx.circular_layout):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        labels = labels or self.label_nodes(subgraph)

        node_size = np.array(self.counts)[subgraph.nodes()]
        node_size = (node_size / max(node_size))*2000 + 200

        nx.draw_networkx(subgraph, labels=labels, 
                         color='w', node_size=node_size, pos=layout(subgraph), 
                         ax=ax)



def base_distribution(sequences):
    """Takes raw (uncounted) list of sequences and returns base distribution
    for most common length.
    """
    import pandas as pd
    df = pd.DataFrame(zip(*np.unique(sequences, return_counts=True)),
                      columns=['sequence', 'counts'])
    df['len'] = df['sequence'].apply(len)
    length = np.bincount(df['len'], df['counts']).argmax()
    df.query('len==@length')
    y = [list(s) for s in df.query('len==@length')['sequence'].values()]
    z = [np.unique(x, return_counts=True)[1] for x in np.array(y).T]
    z = np.array(z).astype(float)
    z = z/z.sum(axis=1)[:,None]
    
    bd = pd.DataFrame(z, columns=list('ACGT'))
    bd['bits'] = -1*np.sum(z*np.log2(z), axis=1)
    return bd



