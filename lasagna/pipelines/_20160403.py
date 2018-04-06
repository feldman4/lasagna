import pickle
from lasagna.design import reverse_complement as rc
from lasagna.sequence import UMIGraph
import pandas as pd
import numpy as np

set1 = '/broad/blainey_lab/David/lasagna/probes/stellaris/set1/set1.pkl'
with open(set1, 'r') as fh:
    bs = pickle.load(fh)
refs = bs.refs.set_index('name')
tiles = bs.tiles.set_index('name')

def make_reference_00():
    reference = ''
    ref00_primer = 'ccggtgcctgaaCGCGaCCGGT'
    refs_ = rc(ref00_primer), 'N'*20, refs.ix[0,'sequence']
    for _, tile in tiles.ix[:24:6].iterrows():
        seq = bs.make_barcode([tile['sequence']], refs=refs_, code='rrrst')
        reference += '>ref00_UMI_%s\n%s\n' % (tile.name, seq)
    return reference

def make_reference(i, r1, r2, N=10, rev_comp=True):
    reference = ''
    refs_ = refs.loc[r1,'sequence'], 'N'*N, 'N'*N, refs.loc[r2,'sequence'] 
    for _, tile1 in tiles.ix[i:24:6].iterrows():
        for _, tile2 in tiles.ix[i+1:24:6].iterrows():
            t = tile1['sequence'], tile2['sequence']
            seq = bs.make_barcode(t, refs=refs_, overhangs=bs.overhangs[i:i+1],
                            code='tsrrorrst')
            seq = rc(seq) if rev_comp else seq
            reference += '>%s_%s_UMI_%s_%s\n%s\n' % (tile2.name, r2, r1, tile1.name, seq)
    return reference

def unique_counts(df, col, cutoff=0):
        s, c = np.unique(df[col], return_inverse=True)
        c = np.bincount(c, weights=df['counts'])
        s, c = s[c>=cutoff], c[c>=cutoff]
        return s, c
        
def collapse_UMIs(csv, cutoff=2):
    df = pd.read_csv(csv)
    if 'UMI' in df.columns:
        s, c = unique_counts(df, 'UMI', cutoff=cutoff)
        G_UMI = UMIGraph(s,c, kmer=8, threshold=2)
        with open(csv.replace('.csv', '_UMI.pkl'), 'w') as fh:
            pickle.dump(G_UMI, fh)
            
    if 'UMI_04' in df.columns and 'UMI_05' in df.columns:
        umis = [a+b for _,(a,b) in df[['UMI_04', 'UMI_05']].iterrows()]
        s, c = np.unique(umis, return_inverse=True)
        c = np.bincount(c, weights=df['counts'])
        s, c = s[c>=cutoff], c[c>=cutoff]
        
        G_UMI45 = UMIGraph(s,c, kmer=8, threshold=2)
        with open(csv.replace('.csv', '_UMI45.pkl'), 'w') as fh:
            pickle.dump(G_UMI45, fh)      
            
    if 'UMI_04_rc' in df.columns and 'UMI_05_rc' in df.columns:
        umis = [a+b for _,(a,b) in df[['UMI_05_rc', 'UMI_04_rc']].iterrows()]
        s, c = np.unique(umis, return_inverse=True)
        c = np.bincount(c, weights=df['counts'])
        s, c = s[c>=cutoff], c[c>=cutoff]
        
        G_UMI45 = UMIGraph(s,c, kmer=8, threshold=2)
        G_UMI45.sequences = [rc(s) for s in G_UMI45.sequences]
        with open(csv.replace('.csv', '_UMI45.pkl'), 'w') as fh:
            pickle.dump(G_UMI45, fh)
            
    if 'UMI_08_rc' in df.columns:
        s, c = unique_counts(df, 'UMI_08_rc', cutoff=cutoff)
        G_UMI8 = UMIGraph(s,c, kmer=8, threshold=2)
        G_UMI8.sequences = [rc(s) for s in G_UMI8.sequences]
        with open(csv.replace('.csv', '_UMI8.pkl'), 'w') as fh:
            pickle.dump(G_UMI8, fh)
            
def collapse_csv(csv, regex='.', kmer=None, cutoff=5, threshold=2):
    df = pd.read_csv(csv)
    for col in df.filter(regex=regex):
        s, c = unique_counts(df, col, cutoff=cutoff)
        G_UMI = UMIGraph(s, c, kmer=kmer, threshold=threshold)
        with open(csv.replace('.csv', '_%s.pkl' % col), 'w') as fh:
            pickle.dump(G_UMI, fh)
            
def collapse_table(df, col, graph):
    if isinstance(graph, str):
        with open(graph, 'r') as fh:
            graph = pickle.load(fh)
    peak_map = {graph.sequences[k]: graph.sequences[v] for k,v in graph.peak_map.items()}
    peak_map.update({graph.sequences[k]: np.nan for k in graph.ambiguous})
    a = df[col][df[col].isin(peak_map)]
    # faster than replace...
    df = df.copy()
    df[col] = pd.Series([peak_map[s] for s in a],index=a.index)
    
    return df.dropna()   
    
