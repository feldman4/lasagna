from lasagna.imports_ipython import *
from lasagna.in_situ import *
from lasagna.plates import plate_coordinate
from collections import OrderedDict

FR_gate = 'corr > 0.85 & ha_median > bkgd_median & ha_median > 2500'



def get_OK_wells(df_reads, df_ph):
    OK_wells = set(df_ph.groupby(['well', 'tile'])['ha_median'].mean()
               .loc[lambda x: x < 2000].index)

    OK_wells |= set(df_ph.groupby(['well' ,'tile']).size()
                    .loc[lambda x: (2500 < x) & (x < 5000)].index)

    return OK_wells

def query_OK_wells(OK_wells):
    """usage: df.pipe(query_OK_wells(OK_wells))
    """
    def wrapped(df):
        return df.set_index(['well', 'tile']).loc[OK_wells].reset_index()
    return wrapped

def to_fastq(f_csv, W):
    dataset = '20180207_6W-G143A'
    f_fastq = name(parse(f_csv), ext='fastq')
    
    df_raw = pd.read_csv(f_csv).pipe(clean_up_raw)
    X = dataframe_to_values(df_raw).reshape(-1, 4)
    Y = W.dot(X.T).T.astype(int)
    df_reads = call_barcodes(df_raw, Y, cycles=12)
    
    dataframe_to_fastq(df_reads, f_fastq, dataset)

def retrieve(df_ph, well, tile, cell):
    bounds = (df_ph.set_index(['well', 'tile', 'cell'])
                   .loc[(well, tile, cell), 'bounds'])
    if isinstance(bounds, str):
        bounds = eval(bounds)
    
    d = dict(mag='10X', subdir='process', tile=tile, well=well)
    f_ph = name(d, tag='phenotype_aligned')
    return f_ph, bounds


def retrieve_arr(df_ph, info):
    arr = [retrieve(df_ph, *x) for x in info]
    return zip(*arr)


def bc_to_files_bounds(df_reads3, df_ph, get_bc):
    cols = ['well', 'tile', 'cell']
    info = df_reads3.query('cell_barcode_0 == @get_bc')[cols]
    FR_pos = df_reads3.query('cell_barcode_0 == @get_bc')['FR_pos']
    info = [tuple(x) for x in info.as_matrix()]
    
    df_ph_ = (df_ph.set_index(['well', 'tile', 'cell'])
               .loc[info].reset_index())

    df_ph_['FR_pos'] = list(FR_pos)
    df_ph_['wt'] = map(str, zip(df_ph_['well'], df_ph_['tile']))
    files, bounds = retrieve_arr(df_ph_, info)
    return files, bounds, df_ph_

def create_median_transform(files):
    """Feed in a subset of files (maybe 10) for performance.
    """
    df_raw = (pd.concat(map(pd.read_csv, files))
            .pipe(clean_up_raw))

    X = dataframe_to_values(df_raw)
    _, W = transform_medians(X.reshape(-1, 4))

    return W


def add_quality(df):
    df = pd.concat([df, convert_quality(df['quality'])], 
                axis=1)

    df['Q_min']  = df.filter(regex='Q_\d+', axis=1).min(axis=1)
    df['Q_mean'] = df.filter(regex='Q_\d+', axis=1).mean(axis=1)
    return df

def add_global_xy(df):
    df = df.copy()
    ij = [plate_coordinate(w, t) for w,t in zip(df['well'], df['tile'])]

    if 'x' in df:
        df['global_x'] = zip(*ij)[1] + df['x']
        df['global_y'] = zip(*ij)[0] + df['y']
    elif 'position_i' in df:
        df['global_x'] = zip(*ij)[1] + df['position_j']
        df['global_y'] = zip(*ij)[0] + df['position_i']
    else:
        df['global_x'] = zip(*ij)[1]
        df['global_y'] = zip(*ij)[0]

    return df

def microwells_96_6():
    columns = {'96_well': 'well', 
               '96_col': 'col', 
               '96_row': 'row'}
    return (microwells()
          .rename(columns=columns)
          .set_index('well'))


def get_finfo(files):
    statinfo = map(os.stat, files)

    cols = ['dataset', 'mag', '6_well', 'tag_ext']

    get_tag_ext = lambda x: ['%s.%s' % y for y in zip(x['tag'], x['ext'])]

    df_finfo = (pd.DataFrame([parse(os.path.abspath(f)) for f in files])
                 .assign(tag_ext=get_tag_ext)
                 .assign(MB=map(lambda x: x.st_size / 1e6, statinfo))
                 .join(microwells_96_6()['6_well'], on='well')         
                 .groupby(cols)['MB']
                 .describe()
                 .assign(count=lambda x: x['count'].astype(int))
                 .assign(GB=lambda x: (0.001 * x['count'] * x['mean']))
                 [[ 'count', 'GB']] 
                 .reset_index()   
               )

    return df_finfo


def cell_mapping_stats(df_cells):

    get_single = lambda x: x.eval('cell_barcode_count_0 == 1')
    get_gt1 = lambda x: x.eval('cell_barcode_count_0 > 1')


    z = (df_cells
         .drop_duplicates(['well', 'tile', 'cell'])
         .assign(mapped=lambda x: ~x['subpool'].isnull())
         .assign(single=get_single)
         .assign(gt1=get_gt1)
    )

    num_cells = len(z)    
    a = z.query('mapped')[['gt1', 'single']].sum()
    b = z.query('~mapped')[['gt1', 'single']].sum()
    a = a.rename(lambda x: x + '_mapped')
    b = b.rename(lambda x: x + '_not_mapped')
    
    df_cell_stats = pd.concat([a,b]).rename(lambda x: 'cells_' + x)
    df_cell_stats['cells'] = num_cells
    
    return df_cell_stats


def read_mapping_stats(df_reads, df_cells):
    reads             = len(df_reads)
    reads_PF          = df_cells.shape[0]
    reads_PF_mapped   = df_cells.query('subpool == subpool').shape[0]
    reads_PF_unmapped = df_cells.query('subpool != subpool').shape[0]

    return  pd.Series(OrderedDict([('reads', reads)
                       ,('reads_PF', reads_PF / float(reads))
                       ,('reads_PF_mapped', reads_PF_mapped / float(reads_PF))
                       ,('reads_PF_unmapped', reads_PF_unmapped / float(reads_PF))
                      ]))

### PLOTTING

def plot_bc(df_reads3, df_ph, barcode, colors=None, jitter=500, **plot_kws):
    
    files, bounds, df_ph_ = bc_to_files_bounds(df_reads3, df_ph, barcode)

    ij = [plate_coordinate(w, t) for w,t in zip(df_ph_['well'], df_ph_['tile'])]
    ij = ij + np.random.rand(*np.array(ij).shape) * jitter
    df_ph_['global_x'] = zip(*ij)[1] + df_ph_['x']
    df_ph_['global_y'] = zip(*ij)[0] + df_ph_['y']

    if colors is None:
        colors = df_ph_['FR_pos'].apply(lambda x: ['blue', 'green'][int(x)])

    ax = df_ph_.plot.scatter('global_x', 'global_y', c=colors, 
                    s=10, **plot_kws)

    ax.axis('equal')
    return ax

def plot_quality_per_cycle(df_reads):

    df = df_reads.pipe(add_global_xy).pipe(add_quality)
    cols = df.filter(axis=1, regex='Q_|gl').columns

    df_ = df.groupby(['well', 'tile'])[cols].mean()

    vmin = 10
    vmax = 30
    
    fig, axs = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
    it = list(df_.filter(axis=1, regex='Q_\d+').columns)
    for ax, col_Q in zip(axs.flatten(), it):
        df_.plot.scatter(x='global_x', y='global_y', c=col_Q, 
                         cmap='viridis', ax=ax, colorbar=False,
                         vmin=vmin, vmax=vmax)
        ax.set_title(col_Q)

    for ax in fig.axes:
        ax.axis('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        y0, y1 = ax.get_ylim()
        ax.set_ylim([y1, y0])

    fig.suptitle('quality range: [%d, %d]' % (vmin, vmax), y=1)
    fig.tight_layout()
    
    return fig

