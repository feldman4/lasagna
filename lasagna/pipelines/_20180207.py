from lasagna.imports_ipython import *
from lasagna import in_situ
import lasagna.plates
from collections import OrderedDict

FR_gate = 'corr > 0.85 & ha_median > bkgd_median & ha_median > 2500'

def analyze_cells(df_reads, Q_min_threshold=4):
    f = '/Users/feldman/lasagna/libraries/20171211_Feldman_90K_Array_pool2.csv'
    df_design = pd.read_csv(f)

    design = (df_design.drop_duplicates('barcode').set_index('barcode')
             [['sgRNA', 'subpool', 'sgRNA_design', 'sgRNA_name']])

    df_cells = (df_reads
        .query('Q_min > @Q_min_threshold')
        .pipe(in_situ.call_cells)
        .pipe(in_situ.add_6_well)
        # .join(design, on='barcode')
    )

    return df_cells

def add_phenotype(df_cells, df_ph, gate=FR_gate):
    df_ph['FR_pos'] = df_ph.eval(gate)

    cols = ['well', 'tile', 'cell']
    phenotype = (df_ph.set_index(cols)
                [['corr', 'bkgd_median', 'dapi_median', 'ha_median', 
                  'area', 'x', 'y', 'FR_pos']])

    return df_cells.join(phenotype, on=cols)

def filter_analyzed_cells(df_cells):
    cols = ['well', 'tile', 'cell']
    return (df_cells
             .query('barcode == cell_barcode_0')
             .drop_duplicates(cols)
             .query('cell_barcode_count_0 > 1 & subpool == "pool2_1"'))

def summarize(df_reads, df_cells, filename=None):
    df_read_stats = read_mapping_stats(df_reads, df_cells).pipe(pd.DataFrame).T
    df_read_stats.index = 'A3',
    df_read_stats.index.name = '6_well'

    cols = ['reads', 'reads_PF', 'reads_PF_mapped', 'reads_PF_unmapped', 'cells',
               'cells_gt1_mapped', 'cells_single_mapped', 'cells_gt1_not_mapped', 
               'cells_single_not_mapped']

    df_stats = (df_cells
        .join(microwells_96_6()['6_well'], on='well')
        .groupby('6_well')
        .apply(cell_mapping_stats)
        .join(df_read_stats)
        [cols].reset_index())

    files = glob('process/*.*')
    df_finfo = get_finfo(files)

    if filename:
        d = {'files': df_finfo, 'read stats': df_stats}
        lasagna.utils.write_excel(filename, d.items())
    
    return df_finfo, df_stats

def get_OK_wells(df_reads, df_ph):
    OK_wells = set(df_ph.groupby(['well', 'tile'])['ha_median'].mean()
               .loc[lambda x: x < 2000].index)

    OK_wells |= set(df_ph.groupby(['well' ,'tile']).size()
                    .loc[lambda x: (2500 < x) & (x < 5000)].index)

    return list(OK_wells)

def filter_OK_wells(df, OK_wells):
    zipper = lambda x: zip(x['well'], x['tile'])
    filt = [(w,t) in OK_wells for w,t in zipper(df)]
    return df[filt]

def query_OK_wells(OK_wells):
    """usage: df.pipe(query_OK_wells(OK_wells))
    """
    def wrapped(df):
        return df.set_index(['well', 'tile']).loc[OK_wells].reset_index()
    return wrapped

def to_fastq(f_csv, W):
    dataset = '20180207_6W-G143A'
    f_fastq = name(parse(f_csv), ext='fastq')
    
    df_raw = pd.read_csv(f_csv).pipe(in_situ.clean_up_raw)
    df_reads = analyze_raw(df_raw, W)

    dataframe_to_fastq(df_reads, f_fastq, dataset)

def analyze_raw(df_raw, W):
    X = in_situ.dataframe_to_values(df_raw).reshape(-1, 4)
    Y = W.dot(X.T).T.astype(int)
    df_reads = (in_situ.call_barcodes(df_raw, Y, cycles=12)
        .drop(['cycle', 'channel'], axis=1))
    return df_reads

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
    info = [tuple(x) for x in info.values()]
    
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
            .pipe(in_situ.clean_up_raw))

    X = in_situ.dataframe_to_values(df_raw)
    _, W = in_situ.transform_medians(X.reshape(-1, 4))

    return W

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
    
    return df_cell_stats.astype(int)

def read_mapping_stats(df_reads):
    reads          = float(len(df_reads))
    reads_mapped   = df_reads.query('subpool == subpool').shape[0]
    reads_unmapped = df_reads.query('subpool != subpool').shape[0]

    return  pd.Series(OrderedDict([('reads', reads)
                       # ,('reads_PF', reads_PF / float(reads))
                       ,('reads_mapped', reads_mapped )
                       ,('reads_unmapped', reads_unmapped )
                      ])).astype(int)

def get_good_well_tiles(cluster_labels):
    good_cluster = (cluster_labels
                    .query('well == "B10" & tile == 4')
                    ['cluster'].iloc[0])
    good_well_tiles = (cluster_labels
                       .query('cluster == @good_cluster')
                       .set_index(['well', 'tile']).index)
    return list(good_well_tiles)

def load_raw_subset(cluster_labels, num_files_median=10):
    good_well_tiles = get_good_well_tiles(cluster_labels)
    
    f = glob('process/*barcodes*csv')[0]
    files = [name(parse(f), tile=t, well=w) for w, t in good_well_tiles]
    
    df_raw_subset = (pd.concat(map(pd.read_csv, tqdm(files)))
                     .pipe(in_situ.clean_up_raw))
    
    W = create_median_transform(files[:num_files_median])
    
    return df_raw_subset, W

### PLOTTING

def plot_bc(df_reads3, df_ph, barcode, colors=None, jitter=500, **plot_kws):
    
    files, bounds, df_ph_ = bc_to_files_bounds(df_reads3, df_ph, barcode)

    ij = [lasagna.plates.plate_coordinate(w, t) for w,t in zip(df_ph_['well'], df_ph_['tile'])]
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

    cols = df_reads.filter(axis=1, regex='Q_|gl').columns

    df_reads_ = df_reads.groupby(['well', 'tile'])[cols].mean()

    vmin = 10
    vmax = 28
    
    fig, axs = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
    it = list(df_reads_.filter(axis=1, regex='Q_\d+').columns)
    for ax, col_Q in zip(axs.flatten(), it):
        df_reads_.plot.scatter(x='global_x', y='global_y', c=col_Q, 
                         cmap='viridis', ax=ax, colorbar=False,
                         linewidths=0, figsize=(8,6),
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

def calc_quality_vs_mapping_rate(df_reads):
    arr = []
    num_reads = len(df_reads)
    for th in range(31):
        q = df_reads.query('Q_min >= @th')
        mapped = (q['subpool'] != 'unmapped')
        arr += [[th, mapped.mean(), mapped.sum(), 
                 len(q) / float(num_reads) ]]

    return np.array(arr)
    
def plot_quality_vs_mapping_rate(df_reads, downsample=1e5):
    """
    """
    num_reads = int(min(downsample, len(df_reads)))
    df_reads_ = df_reads.sample(num_reads)
    arr = calc_quality_vs_mapping_rate(df_reads_)
    
    fig, ax0 = plt.subplots(figsize=(6,4))
    ax1 = ax0.twinx()
    max_q = np.where(arr[:, 2] < 1000)[0][0]
    labels = 'mapping rate', 'reads > threshold'
    line0 = ax0.plot(arr[:max_q,0], arr[:max_q, 1], label=labels[0])
    line1 = ax1.plot(arr[:,0], arr[:, 3], 'g', label=labels[1])
    ax0.legend(line0 + line1, labels, loc='center right')
    
    ax0.set_xlabel('minimum quality threshold')
    ax0.set_ylabel(labels[0])
    ax0.set_ylim([arr[:max_q, 1].min(), 1])
    ax0.set_title('max mapping rate: %.2f%%' % (100 * np.max(arr[:max_q, 1])))
    
    ax1.set_ylabel(labels[1])
    ax1.set_ylim([0, 1])
    # ax1.yaxis.set_visible(False)

    fig.tight_layout()

    return fig



def plot_tile_stats(df_reads, df_ph, df_cells):
    a = calc_tile_num_cells(df_ph)
    b = calc_tile_num_reads(df_reads)
    c = calc_fraction_cells_analyzed(df_ph, df_cells)
    d = calc_tile_consensus_reads_per_cell(df_cells)

    df_tile_stats = (pd.concat([a, b, c, d], axis=1))

    return _plot_tile_stats(df_tile_stats)

def _plot_tile_stats(df_tile_stats):

    def plot_dataframe(c, **kwargs):
        data = kwargs.pop('data')
        ax = plt.gca()
        data.plot.scatter(x='global_x', y='global_y', 
                          c='value', ax=ax, s=120,
                         cmap='viridis', linewidth=0)
        ax.yaxis.set_visible(False)
        ax.axis('equal')

    df_tile_stats.columns.name='statistic'
    fg = (df_tile_stats
        .stack()
        .rename('value').reset_index()
        .pipe(in_situ.add_global_xy)
        .pipe(sns.FacetGrid, col='statistic', col_wrap=2, size=4, aspect=1.4)
        .map_dataframe(plot_dataframe, 'statistic')
        )

    for ax in fg.axes.flatten():
        t = ax.get_title()
        t = (t.replace('statistic = ', '')
              .replace('_', ' ')
              .replace('num', '# of'))
        ax.set_title(t)

        cb = ax.collections[0].colorbar
        cb.set_label('')

    fg.fig.tight_layout()

    return fg

def calc_tile_num_reads(df_reads):
    return df_reads.groupby(['well', 'tile']).size().rename('num_reads')

def calc_tile_num_cells(df_ph):
    return (df_ph
             .drop_duplicates(['well', 'tile', 'cell'])
             .reset_index().groupby(['well', 'tile'])['cell']
             .max().rename('num_cells'))

def calc_tile_consensus_reads_per_cell(df_cells):
    return (df_cells
        .drop_duplicates(['well', 'tile', 'cell'])
        .query('cell_barcode_count_0 > 0')
        .groupby(['well', 'tile'])
        ['cell_barcode_count_0'].mean().rename('consensus_reads_per_cell'))

def calc_fraction_cells_analyzed(df_ph, df_cells):
    cells = calc_tile_num_cells(df_ph)
    called = (df_cells
        .drop_duplicates(['well', 'tile', 'cell'])
        .pipe(filter_analyzed_cells)
        .groupby(['well', 'tile']).size())

    return (called / cells).rename('fraction_cells_analyzed')