from collections import Counter
import os
import pandas as pd
import lasagna.designs.pool1
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np

oligo_arrays = {'pool1': 'Feldman_12K_Array_pool1_table.csv', 
                'pool2': '20171211_Feldman_90K_Array_pool2.csv'}

def histogram_lines(filename):
    """Equivalent of bash hist command.
    """
    with open(filename, 'r') as fh:
        lines = fh.read().strip().split('\n')
    
    counter = Counter(lines)
    it = sorted(counter.items(), key=lambda x: -x[1])
    output = '\n'.join(['%d %s' % (c, s) for s,c in it])
    with open(filename + '.hist', 'w') as fh:
        fh.write(output)
        
    return (sum(counter.values()), len(counter))

def load_database(home='/Users/feldman/lasagna/libraries/'):
    arr = []
    for pool, filename in sorted(oligo_arrays.items()):

        f = os.path.join(home, filename)
        df_design = (pd.read_csv(f)
                       .assign(pool=pool)
                       .rename(columns={'sgRNA_design': 'design'})
                       .drop_duplicates(subset=['sgRNA', 'barcode']))
        arr += [df_design]
    df_sgRNAs, _ = lasagna.designs.pool1.load_tables(path=home)
    return df_design, df_sgRNAs

def analyze_sgRNAs(df, df_design, df_sgRNAs, nearest=0):
    """
    df_sg2 is mapped only. If nearest is 0, throw out mismatches. Otherwise 
    include within edit distance < nearest. 
    """
    # df_sg = df.query('pattern == "sg"').copy()
    df_sg = df.copy().rename(columns={'seq': 'sgRNA'})

    cols = ['source', 'tag', 'gene_symbol', 'gene_id']
    s = df_sgRNAs.set_index('sgRNA')[cols]

    cols = ['well', 'count', 'sgRNA', 'source', 'tag', 'gene_symbol']

    sgRNA_designs = df_design[['sgRNA', 'design']].drop_duplicates().set_index('sgRNA')
    sgRNAs = set(sgRNA_designs.index)

    if nearest:        
        find_nearest = lambda x: sorted((Levenshtein.distance(x, s), s) for s in sgRNAs)[0]
        arr = [find_nearest(seq) for seq in df_sg['sgRNA']]
        dists, sgRNAs = zip(*arr)
        df_sg['nearest_distance'] = dists
        df_sg['sgRNA'] = sgRNAs
        filt = df_sg['nearest_distance'] < nearest
    else:
        filt = df_sg['sgRNA'].isin(sgRNAs)

    df_sg2 = df_sg[filt].join(sgRNA_designs, on='sgRNA')

    df_sg2['fraction'] = df_sg2.groupby('well')['count'].transform(lambda x: x/x.sum())

    return df_sg, df_sg2

def analyze_barcodes(df, df_design, df_sgRNAs):
    df_bc = df.copy().rename(columns={'seq': 'barcode'})

    cols = ['well', 'library', 'gate', 'barcode', 'count']
    # df_bc = df_bc[cols]

    df_bc2 = (df_design.set_index('barcode')
                       .drop('gene_id', axis=1, errors='ignore')
                       .pipe(df_bc.join, on='barcode')
                       )
 

    df_bc2['fraction'] = df_bc2.groupby('well')['count'].transform(lambda x: x/x.sum())

    return df_bc, df_bc2


### plotting

def plot_cdf(values, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    
    y = np.cumsum(values)
    y = y / y.max()
    x = np.linspace(0, 1, len(y))
    ax.plot(x, y, **kwargs)