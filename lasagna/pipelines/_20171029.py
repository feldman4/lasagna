from lasagna.imports import *
from lasagna.imports_ipython import *

adapters = 'TGCA'

# from snakefile...
cycles_seq = ('c0-DO', 'c1-5B1', 'c2-5B4', 'c3-3B4', 'c4-5B3', 'c5-3B3', 'c6-5B2', 'c7-3B2', 'c8-3B1')

def load_barcode_tables(home):
    # fields provided by parse_filename, not needed
    drop = ['dataset', 'date', 'tag', 'subdir', 'mag', 'home']

    search = os.path.join(home, 'process/*barcodes.pkl')
    files = sorted(glob(search))
    arr = []
    for f in files:
        print os.path.basename(f)
        df = read_pickle(f).drop(drop, axis=1)
        arr.append(df)
    df = pd.concat(arr)

    well_96_to_6 = lasagna.io.microwells().set_index('96_well')['6_well']
    well_96_to_6.index.name = 'well'
    df = df.join(well_96_to_6, on='well')

    return df    

def call(df, DO_threshold=[0, 5000, 0, 0]):
    df = call_bases(df)
    
    df = df[mask]
    df = call_cells(df)

    return df

def call_bases(df):
    """Makes cycles_in_situ. Includes DO.
    """
    cols = ['well', 'tile', 'blob', 'cycle']
    df2 = df.pivot_table(index=cols, columns='channel', values='intensity')

    channels = sorted(set(df['channel'])) # in alphabetical order
    call = np.argmax(np.array(df2), axis=1)
    call = np.array(channels)[call]
    s = pd.Series(call, index=df2.index, name='call')
    df = df.join(s, on=cols)
    
    cols = ['well', 'tile', 'blob']
    df2 = df.pivot_table(index=cols, columns='cycle', values='call', aggfunc='first')

    name = 'cycles_in_situ'
    barcodes = [''.join(x) for x in np.array(df2)]
    s = pd.Series(barcodes, index=df2.index, name=name)
    df = df.join(s, on=cols)

    return df

def call_cells(df):
    cols = ['well', 'tile', 'cell']
    s = (df.drop_duplicates(['well', 'tile', 'blob'])
       .groupby(cols)['barcode_in_situ']
       .value_counts()
       .rename('count')
       .sort_values(ascending=False)
       .reset_index('barcode_in_situ')
       .groupby(cols)
        )

    df2 = (df
      .join(s.nth(0)['barcode_in_situ'].rename('barcode_in_situ_0'), on=cols)
      .join(s.nth(0)['count']          .rename('barcode_count_0'), on=cols)
      .join(s.nth(1)['barcode_in_situ'].rename('barcode_in_situ_1'), on=cols)
      .join(s.nth(1)['count']          .rename('barcode_count_1'), on=cols)
    )
    return df2

def dataframe_to_values(df, value='intensity', adapters='TCGA'):
    """Dataframe must be sorted on [cycles, channels]. 
    Returns N x cycles x channels.
    """
    cycles = df['cycle'].value_counts()
    assert len(set(cycles)) == 1
    n_cycles = len(cycles)
    x = np.array(df[value]).reshape(-1, n_cycles, 4)
    base_order = np.argsort(np.argsort(list(adapters)))
    x = x[:, :, base_order]
    return x


def filter_intensity(df, filt_func):
    """Filter function acts on array of shape N x cycles x channels.
    """
    values = dataframe_to_values(df)
    mask = np.zeros(values.shape, dtype=bool)
    mask[...] = filt_func(values)[:, None, None]
    return mask.flatten()


def load_phenotype(home):
    search = os.path.join(home, 'process/*phenotype.pkl')
    files = sorted(glob(search))
    return pd.concat([read_pickle(f) for f in files])

def add_phenotype(home, df):
    df_p = load_phenotype(home)

    cols = ['well', 'tile', 'cell']
    df_p = df.join(df_p.set_index(cols)[['corr', 'myc_median']], on=cols)
    
    return df_p

def load_design(cycles, subpools=(1, 3)):
    """Mark ambiguous barcodes based on design. Would be better to include 
    both and filter later.
    """
    from lasagna.pipelines import _20171026_NGS
    from lasagna.pipelines._20171018 import narrow_design, cycle_to_index, barcode_to_in_situ
    df_design_, df_sgRNAs = _20171026_NGS.load_database()

    subpools = list(subpools)
    df_design = narrow_design(df_design_.query('subpool==@subpools'), cycles)
    return df_design

def add_design(df, df_design):
    cols = ['barcode', 'subpool', 'design', 'sgRNA']

    x = (df_design
         .query('~ambiguous')
         .set_index('barcode_in_situ')[cols]
         )

    df = df.join(x, on='barcode_in_situ')

    return df

def describe_DO(df):
    bins = [0, 500, 5000, 10000, 30000, 65000, 65536]
    arr = []
    for name, channel in (('DO_barcode', 'G'), ('DO_actb', 'A')):

        s = df.query('cycle == "c0-DO" & channel == @channel')['intensity']
        count, division = np.histogram(s, bins=bins)
        counts = pd.Series(count, name=name + '_count', index=division[1:])
        counts.index.name = 'intensity_right_edge'
        arr.append(counts)
    return pd.concat(arr, axis=1)

def read_pickle(filename):
    tags = lasagna.io.parse_filename(filename)
    df = pd.read_pickle(filename)
    for tag, value in tags.items():
        if value is not None:
            df[tag] = value # could cast to categorical
    return df

def name(description, ext='tif', **kwargs):
    d = dict(description)
    d.update(kwargs)
    if 'cycle' in d:
        basename = '%s_%s_%s-Tile_%s.%s.%s' % (d['mag'], d['cycle'], d['well'], d['tile'], d['tag'], ext)
    else:
        basename = '%s_%s-Tile_%s.%s.%s' % (d['mag'], d['well'], d['tile'], d['tag'], ext)
    
    subdir = d['subdir'] if d['subdir'] else ''
    return os.path.join(d['home'], d['dataset'], subdir, basename)

###PLOTTING###

def plot_cycle_intensities(df, value='log_intensity', row='well', **kwargs):
    """
    """
    col, hue = 'cycle', 'channel'
    distplot_kwargs = dict(kde=True, bins=np.linspace(1, 5.5, 20), norm_hist=False, hist_kws=dict(histtype='step', lw=1))
    distplot_kwargs.update(kwargs)
    
    
    fg = sns.FacetGrid(df, row=row, hue=hue, hue_order=list('TGCA'), col=col)
    fg.map(sns.distplot, 'log_intensity', **distplot_kwargs);
    [ax.set_xlim([1, 5.5]) for ax in fg.axes.flat[:]]
    return fg