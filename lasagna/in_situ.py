from lasagna.imports import *

# df_reads <=> fastq

# fields

WELL = 'well'
TILE = 'tile'
CELL = 'cell'
BLOB = 'blob'
CYCLES_IN_SITU = 'cycles_in_situ'
QUALITY = 'quality'
BARCODE = 'barcode'
BARCODE_COUNT = 'barcode_count'
BARCODE_0 = 'cell_barcode_0'
BARCODE_1 = 'cell_barcode_1'
BARCODE_COUNT_0 = 'cell_barcode_count_0'
BARCODE_COUNT_1 = 'cell_barcode_count_1'

def do_median_call(df_raw, cycles=12):
  X = dataframe_to_values(df_raw)
  Y, W = transform_medians(X.reshape(-1, 4))

  df_reads = call_barcodes(df_raw, Y, cycles=cycles)
  return df_reads

def clean_up_raw(df_raw):
    """Categorize, sort. Pre-processing for `dataframe_to_values`.
    """
    df_raw = categorize(df_raw)
    order = natsorted(df_raw['cycle'].cat.categories)
    df_raw['cycle'] = (df_raw['cycle']
                   .cat.as_ordered()
                   .cat.reorder_categories(order))

    df_raw = df_raw.sort_values(['well', 'tile', 'cell', 'blob', 'cycle', 'channel'])
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
       .reset_index(BARCODE)
       .groupby(cols)
        )

    return (df_reads
      .join(s.nth(0)[BARCODE].rename(BARCODE_0),                 on=cols)
      .join(s.nth(0)['count'].rename(BARCODE_COUNT_0).fillna(0), on=cols)
      .join(s.nth(1)[BARCODE].rename(BARCODE_1),                 on=cols)
      .join(s.nth(1)['count'].rename(BARCODE_COUNT_1).fillna(0), on=cols)
      .join(s['count'].sum() .rename(BARCODE_COUNT),             on=cols)
    )

def dataframe_to_values(df, value='intensity'):
    """Dataframe must be sorted on [cycles, channels]. 
    Returns N x cycles x channels.
    """
    cycles = df['cycle'].value_counts()
    assert len(set(cycles)) == 1
    n_cycles = len(cycles)
    x = np.array(df[value]).reshape(-1, n_cycles, 4)
    return x

def transform_medians(X):
    """Find median of max along each dimension to define new axes. 
    Describe with linear transformation W so that W * X = Y.
    """

    def get_medians(X):
        arr = []
        for i in range(X.shape[1]):
            arr += [np.median(X[X.argmax(axis=1) == i], axis=0)]
        M = np.array(arr)
        return M

    W = np.eye(4)
    M = get_medians(X).T
    M = M / M.sum(axis=0)
    W = np.linalg.inv(M)
    Y = W.dot(X.T).T.astype(int)
    return Y, W

def call_barcodes(df_raw, Y, cycles=12):
    """Transform the sequencing data into array Y first. This just uses argmax to call bases.
    """
    df_reads = df_raw.drop_duplicates([WELL, TILE, BLOB]).copy()
    df_reads[CYCLES_IN_SITU] = call_bases_fast(Y.reshape(-1, cycles, 4))
    Q = quality(Y.reshape(-1, cycles, 4))
    # might want rank 0 or 1 instead
    df_reads[QUALITY] = Q.mean(axis=1)
    # needed for performance later
    for i in range(len(Q[0])):
        df_reads['Q_%d' % i] = Q[:,i]
 
    # cycles converted straight to barcodes
    df_reads[BARCODE] = df_reads[CYCLES_IN_SITU]
    return df_reads

def call_bases_fast(values, bases='ACGT'):
    assert values.ndim == 3
    assert values.shape[2] == 4
    calls = values.argmax(axis=2)
    calls = np.array(list(bases))[calls]
    return [''.join(x) for x in calls]

# def quality_linear(X):
#     """
#     """
#     X = np.sort(X, axis=-1).astype(float)
#     P = 15000
#     Q = (X[..., -1] - X[..., -2]) / P
#     Q = Q.clip(min=0.0, max=1.0)
#     return Q

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
    it = zip(df['well'], df['tile'])
    tile_spacing = df['tile'].astype(int).max()
    tile_spacing = ((tile_spacing + 100) // 100)*100
    tile_spacing = 1000
    df['well_tile'] = [wells.index(w) * tile_spacing + int(t) for w, t in it]
    fields = [WELL, 'well_tile', BLOB,
              'position_i', 'position_j', 
              BARCODE, CELL]
    
    Q = df.filter(like='Q_').as_matrix()
    
    reads = []
    for i, row in enumerate(df[fields].as_matrix()):
        d = dict(zip(fields, row))
        d['phred'] = ''.join(phred(q) for q in Q[i])
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

def add_sg_names(df_design):
    d = {}
    for design_name, df_ in df_design.query('subpool < 5').groupby(['design']):
        for i, sgRNA in enumerate(sorted(set(df_['sgRNA']))):
            name_ = 'sg_{i}_{design}'.format(design=design_name[:3],i=i)
            d[sgRNA] = name_

    df_design['sgRNA_name'] = [d.get(s, 'fuckyou') for s in df_design['sgRNA']]
    return df_design

def add_xy(df, spacing='10X', grid_shape=(7, 7)):
    df = df.copy()
    from lasagna.plates import plate_coordinate
    it = zip(df['well'], df['tile'])
    ij = [lasagna.plates.plate_coordinate(w, t, spacing=spacing, grid_shape=grid_shape) for w,t in it]
    ij = zip(*ij)
    df['global_x'] = ij[1]
    df['global_y'] = ij[0]
    return df