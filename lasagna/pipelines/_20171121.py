from lasagna.imports import *
import lasagna.plates
from sklearn.decomposition import FastICA, NMF, SparsePCA
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing
import lasagna.pipelines._20171029 as pipeline

def do_ICA(X):
    # reflect
    X = np.vstack([X, X * -1])
    X = sklearn.preprocessing.scale(X)
    Y = cleanup(X, FastICA(n_components=4))
    return Y[:len(Y)/2]

def do_barcode_map(df, Y):
    df2 = df.drop_duplicates(['well', 'tile', 'blob']).copy()
    df2['cycles_in_situ'] = pipeline.call_bases_fast(Y.reshape(-1, 12, 4))
    Q = quality(Y.reshape(-1, 12, 4))
    # might want rank 0 or 1 instead
    df2['quality'] = Q.mean(axis=1)
    # needed for performance later
    for i in range(len(Q[0])):
        df2['Q_%d' % i] = Q[:,i]

    f = '/Users/feldman/lasagna/libraries/Feldman_12K_Array_pool1_table.pkl'
    df_pool1 = pd.read_pickle(f).drop_duplicates('barcode')
    df_pool1['cycles_in_situ'] = df_pool1['barcode']

    n = 12
    barcodes = set(df_pool1['barcode'].apply(lambda x: x[:n]))
    df3 = df2.join(df_pool1.set_index('cycles_in_situ'), on='cycles_in_situ')

    # cycles converted straight to barcodes
    df3['barcode_in_situ'] = df3['cycles_in_situ']
    return df3

def do_mean_quality(df3, df2):
    cols = ['well', 'tile', 'cell', 'barcode_in_situ']
    Qs = df2.filter(like='Q_', axis=1).columns
    mean_quality = df2.pipe(uncategorize).groupby(cols)[Qs].mean()
    df3 = df3.drop(Qs, axis=1).join(mean_quality, on=cols)
    return df3

def quality(X):
    """
    """
    X = np.sort(X, axis=-1).astype(float)
    # P = np.percentile()
    P = 5000
    Q = (X[..., -1] - X[..., -2]) / P
    Q = Q.clip(min=0.0, max=1.0)
    return Q

def pairplot(X, labels=None):
    bases = list('ACGT')
    a = pd.DataFrame(X.reshape(-1, 4), columns=list('ACGT'))
    if labels is None:
        a['call'] = np.array(bases)[X.argmax(axis=1)]
    else:
        a['call'] = labels
        bases = None
    return sns.pairplot(a, hue='call', hue_order=bases, 
                        vars=bases, plot_kws={'s': 10})

def cleanup(X, dec, n=1e4):
    """Provide data and an sklearn decomposition object.
    """
    n = int(n)
    X_train = X[np.random.randint(X.shape[0], size=n)]
    
    dec.fit(X_train.reshape(-1, 4))
    Y = dec.transform(X.reshape(-1, 4)).reshape(X.shape)

    return repair(Y, dec.components_)

def repair(Y, components):
    Y = Y.copy()
    C = np.abs(components)
    C = C / C.sum(axis=1)
    ix = np.argmax(C, axis=0)
    Y = Y[..., ix]
    # Y /= Y.std(axis=0)
    Y = np.abs(Y)
    return Y

def cluster(X):
    arr = []
    for cycle in range(X.shape[1]):
        X_ = X[:,cycle]
        clu = KMeans(n_clusters=4)
        Y = clu.fit_predict(X_)
        arr += [Y]
    Y = np.array(arr).transpose([1, 0])
    Y = np.eye(4)[Y]
    
    # match cluster IDs to dominant components
    C = clu.cluster_centers_
    lasagna.C = C
    ix = np.argmax(C, axis=1)
    Y = Y[:, :, ix]
    return Y

def q_to_alpha(Q):
    import string
    A = np.array(list(string.ascii_uppercase))
    Q2 = ((1 - Q) * 26).astype(int)
    return [''.join(x) for x in A[Q2]]

def reads_to_fastq(df, dataset):
    a = '@MN2157:%s:FCFU' % dataset 
    b = ':{well}:{well_tile}:{cell}:{blob}'
    c = ':{position_i}:{position_j}'
    d = ' 1:N:0:NN'
    e = '\n{barcode_in_situ}\n+\n{phred}'
    fmt = a + b + c + e
    

    wells = list(lasagna.plates.microwells()['96_well'])
    it = zip(df['well'], df['tile'])
    tile_spacing = df['tile'].astype(int).max()
    tile_spacing = ((tile_spacing + 100) // 100)*100
    tile_spacing = 1000
    df['well_tile'] = [wells.index(w) * tile_spacing + int(t) for w, t in it]
    fields = ['well', 'well_tile', 'blob',
              'position_i', 'position_j', 
              'barcode_in_situ', 'cell']
    
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

def load_barcode_table(files, n_cells=None):
    """Load tables, optionally subsampling cells.
    """
    arr = []
    for f in files:
        print f
        df = pd.read_pickle(f)
        if n_cells is not None:
            cells = sorted(set(df['cell']))
            cells_keep = list(np.random.choice(cells, size=n_cells, replace=False))
            df = df.query('cell == @cells_keep')
        arr += [df]

    df = pd.concat(arr)
    return df

def clean_barcode_table(df):
    remap = dict(zip('TGCA','GTAC'))
    df['channel'] = df['channel'].apply(lambda x: remap[x])
    f = lambda s: re.sub('\d+', lambda x: '%02d' % int(x.group()), s)
    df['cycle'] = list(df['cycle'].astype('category').apply(f))
    df = categorize(df)
    df = df.sort_values(['well', 'tile', 'blob', 'cycle', 'channel'])
    retype = (('position_i', np.uint16), 
              ('position_j', np.uint16),
              ('blob', np.uint32))
    for c, t in retype:
        df[c] = df[c].astype(t)
    
    df = df.reset_index(drop=True)
    return df

def make_reference_fasta():
    f = '/Users/feldman/lasagna/libraries/Feldman_12K_Array_pool1_table.pkl'
    df_pool1 = pd.read_pickle(f).drop_duplicates('barcode')

    import random
    random.seed(0)
    def shuffle(s):
        s = list(s)
        random.shuffle(s)
        return ''.join(s)

    fmt = '>pool1_{subpool}_{barcode}\n{barcode}'
    reference = [fmt.format(**row) for _, row in df_pool1.iterrows()]
    reference = '\n'.join(reference)
    with open('/Users/feldman/transfer/pool1.fa', 'w') as fh:
        fh.write(reference)

    # scrambled
    fmt = '>pool1_{subpool}_{scrambled}_shuffled\n{scrambled}'
    df_pool1['scrambled'] = df_pool1['barcode'].apply(shuffle)
    reference = [fmt.format(**row) for _, row in df_pool1.iterrows()]
    reference = '\n'.join(reference)
    with open('/Users/feldman/transfer/pool1_shuffled.fa', 'w') as fh:
        fh.write(reference)

def get_medians(X):
    arr = []
    for i in range(X.shape[1]):
        arr += [np.median(X[X.argmax(axis=1) == i], axis=0)]
    M = np.array(arr)
    return M

def transform_medians(X):
    """Y = W * X
    """
    W = np.eye(4)
    M = get_medians(X).T
    M = M / M.sum(axis=0)
    W = np.linalg.inv(M)
    Y = W.dot(X.T).T.astype(int)
    return Y, W

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
    x = pd.DataFrame(t.view(np.uint8).reshape(q_shape), columns=columns)
    x = unphred_array(x)
    return x

def locate(well, tile):
    i, j = lasagna.plates.plate_coordinate(well, tile, grid_shape=(7,7), mag='10X')
    return i,j

def plot_quality(df3, quality_column):
    a = (df3.groupby(['well', 'tile'])[quality_column]
           .describe()
           .reset_index())

    i, j = zip(*[locate(w, t) for w,t in zip(a['well'], a['tile'])])
    a['tile_x'] = j
    a['tile_y'] = -1 * np.array(i)

    ax = a.plot.scatter(x='tile_x', y='tile_y', c='mean', vmin=0, vmax=20)
    ax.axis('equal')
    return ax


def plot_oversampling():
    def sampler(x, dropout=0.05, cutoff=50):
        """Simulate sampling from a skewed pool. Find number of 
        samples required to see each at least `cutoff` times, with a 
        fixed `dropout` rate. Return the number of samples divided by
        the least number of samples possible.
        
        For a uniform distribution with `dropout = 0.05` and 
        `cutoff = 50` the ratio is ~1.2.
        """
        n = len(x)
        step = 100
        labels = range(n)
        p = np.array(x) / float(sum(x))
        samples = []
        while True:
            samples += list(np.random.choice(labels, p=p, size=step))
            a = np.bincount(samples)
            if sum(a > cutoff) > (1 - dropout) * n:
                return len(samples) / float(n * cutoff)


    xs = np.random.randint(50, 100, 100).astype(float)
    arr = []
    for i in np.linspace(0.5, 5, 8):
        y = xs**i
        y /= y.sum()
        a = np.percentile(y, 10)
        b = np.percentile(y, 90)
        print b / a
        for _ in range(3):
            arr += [(i, b/a, sampler(y))]

    df_ = pd.DataFrame(arr, columns=['exp', '90_10', 'oversample'])
    df_ = df_[df_['90_10'] < 13]

    fig, ax = plt.subplots()
    ax.scatter(x=df_['90_10'], y=df_['oversample'])
    ax.set_xlabel('percentile 90 / 10')
    ax.set_ylabel('oversampling (dropout 0.05, cutoff 50)')
    # fig.savefig('figures/oversampling.pdf')
    return ax