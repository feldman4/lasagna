# from lasagna.imports import *
import numpy as np
import pandas as pd
import functools
from lasagna.schema import *
from natsort import natsorted

IMAGING_ORDER = 'GTAC'

def do_median_call(df_raw, cycles=12, channels=4):
  X = dataframe_to_values(df_raw, channels=channels)
  Y, W = transform_medians(X.reshape(-1, channels))

  df_reads = call_barcodes(df_raw, Y, cycles=cycles, channels=channels)
  return df_reads.drop([CHANNEL, INTENSITY], axis=1)

def clean_up_raw(df_raw):
    """Categorize, sort. Pre-processing for `dataframe_to_values`.
    """
    # exclude_subset = ['well', 'tile', 'cell', 'intensity', 'blob'] # causes issues with later joins, maybe a pandas bug
    import lasagna.utils
    df_raw = lasagna.utils.categorize(df_raw, subset=[CYCLE])
    order = natsorted(df_raw[CYCLE].cat.categories)
    df_raw[CYCLE] = (df_raw[CYCLE]
                   .cat.as_ordered()
                   .cat.reorder_categories(order))

    df_raw = df_raw.sort_values([WELL, TILE, CELL, BLOB, CYCLE, CHANNEL])
    return df_raw

def call_cells(df_reads):
    """Determine count of top barcodes 
    """
    cols = [WELL, TILE, CELL]
    s = (df_reads
       .drop_duplicates([WELL, TILE, BLOB])
       .groupby(cols)[BARCODE]
       .value_counts()
       .rename('count')
       .sort_values(ascending=False)
       .reset_index()
       .groupby(cols)
        )

    return (df_reads
      .join(s.nth(0)[BARCODE].rename(BARCODE_0),                 on=cols)
      .join(s.nth(0)['count'].rename(BARCODE_COUNT_0).fillna(0), on=cols)
      .join(s.nth(1)[BARCODE].rename(BARCODE_1),                 on=cols)
      .join(s.nth(1)['count'].rename(BARCODE_COUNT_1).fillna(0), on=cols)
      .join(s['count'].sum() .rename(BARCODE_COUNT),             on=cols)
      .drop_duplicates(cols)
      .drop([BARCODE], axis=1) # drop the read barcode
    )

def dataframe_to_values(df, value='intensity', channels=4):
    """Dataframe must be sorted on [cycles, channels]. 
    Returns N x cycles x channels.
    """
    cycles = df['cycle'].value_counts()
    assert len(set(cycles)) == 1
    n_cycles = len(cycles)
    x = np.array(df[value]).reshape(-1, n_cycles, channels)
    return x

def transform_medians(X):
    """For each dimension, find points where that dimension is max. Use median of those points to define new axes. 
    Describe with linear transformation W so that W * X = Y.
    """

    def get_medians(X):
        arr = []
        for i in range(X.shape[1]):
            arr += [np.median(X[X.argmax(axis=1) == i], axis=0)]
        M = np.array(arr)
        return M

    M = get_medians(X).T
    M = M / M.sum(axis=0)
    W = np.linalg.inv(M)
    Y = W.dot(X.T).T.astype(int)
    return Y, W

def call_barcodes(df_raw, Y, cycles=12, channels=4):
    bases = sorted(IMAGING_ORDER[:channels])
    df_reads = df_raw.drop_duplicates([WELL, TILE, BLOB]).copy()
    df_reads[CYCLES_IN_SITU] = call_bases_fast(Y.reshape(-1, cycles, channels), bases)
    Q = quality(Y.reshape(-1, cycles, channels))
    # needed for performance later
    for i in range(len(Q[0])):
        df_reads['Q_%d' % i] = Q[:,i]
 
    # cycles converted straight to barcodes
    df_reads = df_reads.rename(columns={CYCLES_IN_SITU: BARCODE})
    df_reads['Q_min'] = df_reads.filter(regex='Q_\d+').min(axis=1)
    return df_reads

def call_bases_fast(values, bases):
    """4-color: bases='ACGT'
    """
    assert values.ndim == 3
    assert values.shape[2] == len(bases)
    calls = values.argmax(axis=2)
    calls = np.array(list(bases))[calls]
    return [''.join(x) for x in calls]

def quality(X, boost=2):
    """X_2 = (X_1)^a, a in [0..1]
    """
    X = np.abs(np.sort(X, axis=-1).astype(float))
    Q = 1 - np.log(2 + X[..., -2]) / np.log(2 + X[..., -1])
    Q = (Q * 2).clip(0, 1)
    return Q

def reads_to_fastq(df, dataset):
    a = '@MN2157:%s:FCFU' % dataset 
    b = ':{well}:{well_tile}:{cell}:{blob}'
    c = ':{position_i}:{position_j}'
    d = ' 1:N:0:NN'
    e = '\n{barcode}\n+\n{phred}'
    fmt = a + b + c + e
    
    wells = list(lasagna.plates.microwells()['96_well'])
    tile_spacing = 1000
    fields = [WELL, TILE, CELL, BLOB,
              'position_i', 'position_j', 
              BARCODE]
    
    Q = df.filter(like='Q_').values()
    
    reads = []
    for i, row in enumerate(df[fields].values()):
        d = dict(zip(fields, row))
        d['phred'] = ''.join(phred(q) for q in Q[i])
        d['well_tile'] = wells.index(d['well']) * tile_spacing + int(d['tile'])
        reads.append(fmt.format(**d))
    
    return reads
    
def dataframe_to_fastq(df, file, dataset):
    s = '\n'.join(reads_to_fastq(df, dataset))
    with open(file, 'w') as fh:
        fh.write(s)
        fh.write('\n')

def phred(q):
    """Convert 0...1 to 0...30
    No ":".
    No "@".
    No "+".
    """
    n = int(q * 30 + 33)
    if n == 43:
        n += 1
    if n == 58:
        n += 1
    return chr(n)

def unphred_string(phred):
    """Convert 0...30 to 0...1
    """
    arr = [(ord(c) - 33) / 30. for c in phred]
    return arr
        
def unphred_array(phred):
    return (phred - 33) #/ 30.

def table_to_dataframe(file):
    """ no good ideas for out-of-core operation
    would like to be able to 
    - filter tiles
    - subsample cells

    cat run.fastq  | sed 'N;N;N;s/\n/ /g' | rg "@" --replace "" | rg "[:\+]" --replace " " | rg "\s+" --replace " " > run.table
    """
    columns = ['instrument', 'dataset', 'flowcell', 'well', 
           'well_tile', 'cell', 'blob', 'position_i', 'position_j',
           'read', 'quality']

    columns_drop = ['instrument', 'flowcell', 'dataset', 'well_tile']

    df = pd.read_csv(file, sep='\s+', header=None, quoting=3)
    df.columns = columns
    df['tile'] = df['well_tile'] % 1000
    df = df.drop(columns_drop, axis=1)
    return df

def convert_quality(quality_series):
    for x in quality_series:
        n = len(x)
        break

    s = ''.join(quality_series)
    t = np.fromstring(s, dtype=np.uint8)
    q_shape = len(quality_series), n
    columns = ['Q_%02d' % i for i in range(n)]
    x = pd.DataFrame(t.view(np.uint8).reshape(q_shape), 
        columns=columns, index=quality_series.index)
    x = unphred_array(x)
    return x[natsorted(x.columns)]

def locate(well, tile):
    i, j = lasagna.plates.plate_coordinate(well, tile, grid_shape=(7,7), mag='10X')
    return i,j

def make_reference_fasta(df_design):
    """needs work
    """ 
    import random
    random.seed(0)
    def shuffle(s):
        s = list(s)
        random.shuffle(s)
        return ''.join(s)
    
    df_pool = df_design.drop_duplicates('barcode').copy()
    df_pool['scrambled'] = df_pool['barcode'].apply(shuffle)
    fmt = '>{subpool}_{barcode}\n{barcode}'

    arr = []
    for source in ('barcode', 'scrambled'):
        it =  zip(df_pool['subpool'], df_pool[source])
        reference = [fmt.format(subpool=s, barcode=b) for s,b in it]
        arr += ['\n'.join(reference)]

    return arr

def add_6_well(df_reads):
    from lasagna.plates import microwells
    s = microwells().set_index('96_well')['6_well']
    s.index.name = 'well'
    s.name = 'well_6'
    return df_reads.join(s, on='well')

def add_quality(df):
    """Convert string quality column to numeric per-cycle, 
    including Q_min and Q_mean
    """
    df = pd.concat([df, convert_quality(df['quality'])], 
                axis=1)

    df['Q_min']  = df.filter(regex='Q_\d+', axis=1).min(axis=1)
    df['Q_mean'] = df.filter(regex='Q_\d+', axis=1).mean(axis=1)
    return df

def load_NGS_hist(f):
    return (pd.read_csv(f, header=None, sep='\s+')
     .rename(columns={0: 'count', 1: 'barcode_full'})
     .assign(file=f)
     .assign(length=lambda x: x['barcode_full'].map(len))
     .query('length == 12')
     .assign(fraction=lambda x: np.log10(x['count']/x['count'].sum()))
     )

def add_clusters(df_cells, neighbor_dist=50):
    """Assigns -1 to clusters with only one cell.
    """
    from scipy.spatial.kdtree import KDTree
    import networkx as nx

    x = df_cells[GLOBAL_X] + df_cells['j_SBS']
    y = df_cells[GLOBAL_Y] + df_cells['i_SBS']
    barcodes = df_cells[BARCODE_0]
    barcodes = np.array(barcodes)

    kdt = KDTree(zip(x, y))
    num_cells = len(df_cells)
    print('searching for clusters among %d cells' % num_cells)
    pairs = kdt.query_pairs(neighbor_dist)
    pairs = np.array(list(pairs))

    x = barcodes[pairs]
    y = x[:, 0] == x[:, 1]

    G = nx.Graph()
    G.add_edges_from(pairs[y])

    clusters = list(nx.connected_components(G))

    cluster_index = np.zeros(num_cells, dtype=int) - 1
    for i, c in enumerate(clusters):
        cluster_index[list(c)] = i

    df_cells[CLUSTER] = cluster_index
    return df_cells

def add_design(df_cells_all, df_design, 
    design_barcode_col='barcode', 
    cell_barcode_col='cell_barcode_0'):
    
    bc1 = df_cells_all[cell_barcode_col].dropna().iloc[0]
    bc2 = df_design[design_barcode_col].iloc[0]
    assert len(bc1) == len(bc2)

    s = (df_design.drop_duplicates(design_barcode_col)
        .set_index(design_barcode_col)
        [[SUBPOOL, SGRNA_NAME, SGRNA_DESIGN]])

    cols = [WELL, TILE, CELL]
    df_cells = df_cells_all.join(s, on=cell_barcode_col)
    return df_cells

def join_by_cell_location(df_cells, df_ph, max_distance=4):
    from scipy.spatial.kdtree import KDTree
    # df_cells = df_cells.sort_values(['well', 'tile', 'cell'])
    # df_ph = df_ph.sort_values(['well', 'tile', 'cell'])
    i_tree = df_ph['i_ph'] + df_ph['global_y']
    j_tree = df_ph['j_ph'] + df_ph['global_x']
    i_query = df_cells['i_SBS'] + df_cells['global_y']
    j_query = df_cells['j_SBS'] + df_cells['global_x']
    
    kdt = KDTree(zip(i_tree, j_tree))
    distance, index = kdt.query(zip(i_query, j_query))
    cell_ph = df_ph.iloc[index]['cell'].pipe(list)
    cols_left = ['well', 'tile', 'cell_ph']
    cols_right = ['well', 'tile', 'cell']
    cols_ph = [c for c in df_ph.columns if c not in df_cells.columns]
    return (df_cells
                .assign(cell_ph=cell_ph, distance=distance)
                .query('distance < @max_distance')
                .join(df_ph.set_index(cols_right)[cols_ph], on=cols_left)
                # .drop(['cell_ph'], axis=1)
               )
    
def summarize_sg_stats(df_cells, thresholds=None, value='dapi_gfp_nuclear_corr'):
    """A bit slow.
    """
    def describe_range(series, thresholds, index_name='threshold'):
        s = pd.Series({t: (series < t).sum() for t in thresholds})
        s.index.name = index_name
        s.name = 'count'
        return s

    df_cells = df_cells.dropna(subset=[value])
    if thresholds is None:
        thresholds = np.linspace(-1, 1, 21)
    cols = ['stimulant', 'gene', 'sgRNA_name', 'cell_barcode_0']
    f = functools.partial(describe_range, thresholds=thresholds)
    total = df_cells.groupby(cols).size().rename('total')
    df_sg_stats = (df_cells
         .groupby(cols)
         [value].apply(f)
         .reset_index()
         .rename(columns={value: 'count'})
         .join(total, on=cols)
         .assign(ratio=lambda x: x.eval('count / total'))
         .assign(threshold_100=lambda x: (x['threshold'] * 100).astype(int))
        )
    
    return df_sg_stats

