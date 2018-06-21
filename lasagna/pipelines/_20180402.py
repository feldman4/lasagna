from lasagna.imports import *
from lasagna.schema import *

gate_cells = and_join(['3000 < dapi_nuclear_max < 12000',
    '60 < area_nuclear < 140', 
    '2000 < gfp_cell_median < 8000',
    'duplicated == False'])

gate_NT = 'dapi_gfp_nuclear_corr < -0.5'

stimulant = {'A': 'TNFa', 'B': 'IL1b', 
	'A1': 'TNFa', 'A2': 'TNFa', 'A3': 'TNFa',
	'B1': 'IL1b', 'B2': 'IL1b', 'B3': 'IL1b'}
positive_genes = {'IL1b': ['MAP3K7', 'NFKBIA', 'IKBKG',
                    'IRAK1', 'MYD88', 'IRAK4', 'TRAF6'],
                'TNFa': ['TNFRSF1A', 'MAP3K7', 'NFKBIA', 'IKBKG', 'TRADD', 
                    'TRAF2', 'IKBKB', 'CHUK', 'RIPK1']}

positive_gene_list = ['MAP3K7', 'NFKBIA', 'IKBKG', 'TRADD', 'TNFRSF1A', 
                    'IRAK1', 'MYD88', 'IRAK4', 'TRAF2', 'TRAF6', 'IKBKB', 
                    'CHUK', 'RIPK1']

tilemap_features = ['gfp_cell_median', 'gfp_nuclear_median', 
                    'dapi_gfp_cell_corr', 'dapi_gfp_nuclear_corr']

def filter_good_tiles(df):
    index = (pd.read_csv('well_tiles_filtered_by_cluster.csv')
        .pipe(lambda x: map(tuple, x.as_matrix())))
    return df.set_index(['well', 'tile']).loc[index].reset_index()

def combine_phenotypes(df_ph_full, df_ph_perimeter):
    """Combine phenotype data from `Snake._extract_phenotype_translocation` and 
    `Snake._extract_phenotype_translocation_ring`.
    """
    key_cols = ['well', 'tile', 'cell']

    val_cols = [
        "dapi_gfp_nuclear_corr",
        "dapi_nuclear_int",
        "dapi_nuclear_max",
        "dapi_nuclear_median",
        "gfp_nuclear_int",
        "gfp_nuclear_max",
        "gfp_nuclear_mean",
        "gfp_nuclear_median",
        "x",
        "y",
        "dapi_gfp_cell_corr",
        "gfp_cell_mean",
        "gfp_cell_median",
        "gfp_cell_int"
    ]
    
    df_ph_perimeter = (df_ph_perimeter
                       .set_index(key_cols)[val_cols]
                       .rename(columns=lambda x: x + '_perimeter'))
    
    return df_ph_full.join(df_ph_perimeter, on=key_cols)

def add_phenotype_cols(df_ph):
    return (df_ph
        .assign(gcm=lambda x: x.eval('gfp_cell_median - gfp_nuclear_median')))

def annotate_cells(df_cells):
    def get_gene_symbol(sgRNA_name):
        try:
          if sgRNA_name.startswith('LG'):
              return 'LG'
          pat = 'sg_(.*?)_'
          return re.findall(pat, sgRNA_name)[0]
        except AttributeError:
          return None

    def get_targeting(sgRNA_name):
        try:
            return 'LG_sg' not in sgRNA_name
        except TypeError:
            return False

    def get_stimulant(well):
        return stimulant[well[0]]

    def categorize_stimulant(s):
        return pd.Categorical(s, categories=['TNFa', 'IL1b'], ordered=True)

    return (df_cells
        # .assign(gate_NT=lambda x: x.eval(gate_NT))
        .assign(gene_symbol=lambda x: x[SGRNA_NAME].apply(get_gene_symbol))
        .assign(targeting=lambda x: x[SGRNA_NAME].apply(get_targeting))
        .assign(stimulant=lambda x: x[WELL].apply(get_stimulant).pipe(categorize_stimulant))
        .assign(positive=get_positives)
        )

def get_positives(df_cells):
    TNFa_pos = positive_genes['TNFa']
    IL1b_pos = positive_genes['IL1b']
    gate_pos = or_join(['stimulant == "TNFa" & gene == @TNFa_pos',
                        'stimulant == "IL1b" & gene == @IL1b_pos'])
    return df_cells.eval(gate_pos)

def annotate_hits(df_cells):
    def classify(positive, gene):
      if positive:
          return 'positive'
      if gene == 'LG':
          return 'nontargeting'
      if pd.isnull(gene) or pd.isnull(positive):
          return None
      else:
        return 'negative'
    return (df_cells
        .assign(sgRNA_class=lambda x: 
              [classify(p,g) for p,g in zip(x['positive'], x['gene'])]))

def annotate_clusters(df_cells):
    features = \
    (df_cells
     .groupby('cluster')['gate_NT']
     .pipe(groupby_reduce_concat,  
            cluster_size='size', cluster_NT='sum_int')
    )
    cols = ['cluster_size', 'cluster_NT']
    return (df_cells.drop(cols, axis=1, errors='ignore').join(features, on='cluster'))

def get_edge_dist(x, y, center):
    x = center - np.abs(center - x)
    y = center - np.abs(center - y)
    return np.min(np.vstack((x, y)), axis=0)

def dump_examples(df_cells, df_ph, num_cells=500):
    # selection criteria
    tile_min_cell_count = 500
    wells = ['B2']
    stimulants = ['IL1b']
    min_edge_dist = 50
    sgRNA_classes = ['positive', 'nontargeting']
    
    # output formatting
    padding = 25
    file_template = 'process/10X_c0-RELA-mNeon/10X_c0-RELA-mNeon_A1_Tile-103.aligned_phenotype.tif'
    description = parse(file_template)
    
    def to_bounds(xy):
        """1 pixel bounds, loaded with padding later. 
        Doesn't use segmented region.
        """
        x, y = xy
        i, j = int(np.floor(y)), int(np.floor(x))
        return i, j, i + 1, j + 1
    
    # tiles with enough cells
    good_wt = (df_ph
               .groupby(['well', 'tile']).size()
               .pipe(lambda x: list(x[x>tile_min_cell_count].index)))
                   
    to_tuples = lambda x: map(tuple, x.as_matrix()) 
    def select_multi(df, cols, values):
      index = to_tuples(df[cols])
      filt = [ix in values for ix in index]
      return df[filt]

    df_cells = (df_cells
     .query('stimulant == @stimulants')
     # good wells
     .query('well == @wells')
     # good tiles
     .pipe(select_multi, ['well', 'tile'], good_wt)
     # pandas .loc with tuple list screws up index and column dtypes...
     # set_index(['well', 'tile'])
     # .loc[good_wt].reset_index()
     # .rename(columns={'level_0': 'well', 'level_1': 'tile'})
     # OK cells
     .query(gate_cells)
     # label sgRNA classes
     .pipe(annotate_hits)
     .query('sgRNA_class == @sgRNA_classes')
     .query('gate_NT == [True, False]')
    )
    
    # no edge cells
    df_ph = (df_ph
         .assign(edge_dist=lambda x: get_edge_dist(x.x, x.y, 500))
         .query('edge_dist > @min_edge_dist')
    )
    
    
    df_ph_wtc = (df_ph[['well', 'tile', 'cell']]
                 .pipe(to_tuples))
    
    # for (class_, NT_status), df_cells_ in df_cells.groupby(['sgRNA_class', 'gate_NT']):
    for class_, df_cells_ in df_cells.groupby('sgRNA_class'):
        # NT_label = 'pos' if NT_status else 'neg'
        NT_label = 'all'
        # select cells
        wtc = (df_cells_
               [['well', 'tile', 'cell']].pipe(to_tuples)
              )
        # only cells with phenotype data passing edge criterion
        wtc = sorted(set(wtc) & set(df_ph_wtc))
        wtc = pd.Series(wtc).sample(min(num_cells, len(wtc)), replace=False, 
          random_state=0).pipe(list)

        # retrieve x,y coordinates
        xy = (df_ph
              .set_index(['well', 'tile', 'cell'])
              .loc[wtc, ['x', 'y']].as_matrix())
        bounds = map(to_bounds, xy)

        # files containing data
        files_data, files_cells = [], []
        for w,t,_ in wtc:
            files_data .append(name(description, well=w, tile=t))
            files_cells.append(name(description, well=w, tile=t, 
                    tag='cells', subdir='process', cycle=None))
                 
        # load data, faster with memmap enabled
        data  = grid_view(files_data,  bounds, padding=padding)
        cells = grid_view(files_cells, bounds, padding=padding)

        cycle = '{class_}-sgRNA-{NT_label}'.format(class_=class_, NT_label=NT_label)
        file_data  = name(description, cycle=cycle, 
                          tag='phenotype_aligned', well='B2', 
                          tile=None, subdir='examples')
        file_cells = name(parse(file_data), tag='cells')
                 
        save(file_data,  data)
        print 'wrote %d examples to %s' % (len(data), file_data)
        # save(file_cells, cells[:, None, :, :])
    
def apply_NN_model(model):

  files = glob('process/10X_c0-RELA-mNeon/*stack-16.tif')
  arr = []
  for f in tqdm(files):
      data = read(f)
      well, tile = parse(f)['well'], parse(f)['tile']
      cells = df_ph.query('tile == @tile & well == @well')['cell'].pipe(list)
      lasagna.io.read_stack._reset()
      
      X = lasagna.learn.pad_zero(data)
      y = model.predict(X/15000.)[:, 0]
      df = pd.DataFrame({'NN_prob': y, 'cell': cells}).assign(well=well, tile=tile)
      arr += [df]
      
  return pd.concat(df)

def hit_table(df_cells, col='gate_NT'):
    return (df_cells.query(gate_cells)
      .groupby(['stimulant', 'gene', 'cell_barcode_0'])[col]
      .pipe(groupby_reduce_concat, 
            fraction='mean', 
            cell_count='size')
      .rename(columns={'fraction': col + '_mean'})
      .reset_index())

def rescale_20X_to_10X(stack_20X, dapi_10X, scale=0.5025):

    rescale = lambda x: (skimage.transform.rescale(x, (scale, scale), preserve_range=True)
                         .astype(np.uint16))
    rescaled_ = np.array(map(rescale, stack_20X))

    num_channels = len(stack_20X)
    
    dapi_10X_ = np.array([dapi_10X] * num_channels)
    rescaled = np.zeros_like(dapi_10X_)
    # pad with zeros, assuming rescaled stack_20X is smaller than dapi_10X
    _, h, w = rescaled_.shape
    rescaled[:, :h, :w] = rescaled_

    images  = [dapi_10X_,  rescaled]
    images_ = [dapi_10X, rescaled[0]]
    result = lasagna.process.register_and_offset(images, registration_images=images_)

    return result[1]

### PLOTTING

def plot_correlation_features(df_cells):
    import seaborn as sns
    corr_vars = df_cells.filter(regex='corr').columns
    pg = \
    (df_cells
     .query(gate_cells)
     .dropna(subset=['subpool'])
     .dropna(subset=corr_vars)
     .sample(5000)
     .pipe(sns.pairplot, vars=corr_vars, hue='positive', size=6, diag_kind='kde')
    )

    return pg

def plot_sgRNAs_by_gene(df_cells, genes):
    import seaborn as sns

    df = (df_cells
      .query(gate_cells)
      .query('gene == @genes'))
    
    df['gene'] = df['gene'].astype('category').cat.reorder_categories(genes)

    df.groupby(['gene'])['gate_NT'].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax = (df
          .groupby(['gene', 'stimulant', 'sgRNA_name'])['gate_NT']
          .pipe(groupby_reduce_concat, fraction='mean', cell_count='size')
          .reset_index()
          .sort_values('gene')
          .pipe(lambda x: (sns.swarmplot(data=x, x='gene', y='fraction',
                                         hue='stimulant', dodge=True, ax=ax)))
        )
    ax.set_xlabel('')
    ax.set_title('fraction of cells scored positive per sgRNA')
    ax.legend(loc='upper left')
    plt.xticks(rotation=30);
    
    return fig

def plot_NT_fraction_vs_count(df_cells, min_count=30):
    import seaborn as sns



    cols = ['stimulant', 'barcode', 'positive', 'gene']
    df = \
    (df_cells
      .query(gate_cells)
      .groupby(cols)['gate_NT']
      .pipe(groupby_reduce_concat, fraction='mean', count='size')
      .reset_index()
      .pipe(annotate_hits)
      .assign(log10_count=lambda x: np.log10(1 + x['count']))
    )

    fg = \
    (df
     .query('count > @min_count')
     .pipe(sns.pairplot, size=5, vars=['fraction', 'log10_count'], hue='sgRNA_class', 
           diag_kind='hist', diag_kws=dict(histtype='step', lw=2, log=False, bins=20, normed=True))
     .add_legend()
    )
    return fg

def filter_clusters(df_cells):
    return (df_cells
            .dropna(subset=['gene'])
            .query(gate_cells)
            .query('cluster > -1')
            .pipe(annotate_clusters)  
            .pipe(annotate_hits))

def plot_cluster_heatmaps(df_cells, x_range=(None,5), y_range=(1, 5)):
    import seaborn as sns
    def plot_heatmap(data, **kwargs):
        ax = plt.gca()
        (data.pivot_table(index='cluster_NT', columns='cluster_size', 
                        values='cluster', aggfunc=len)
             # ensure heatmap covers full range
            .reindex(range(0, y_range[1] + 1), axis=0)
            .reindex(range(0, x_range[1] + 1), axis=1)
            .iloc[slice(*y_range), slice(*x_range)]
            .pipe(lambda x: x / x.sum().sum())
            .pipe(lambda x: x * 100)
            .pipe(sns.heatmap, annot=True, ax=ax)
        )
        ax.invert_yaxis()

    fg = (df_cells
        .pipe(filter_clusters)
        .query('gate_NT')
        .pipe(sns.FacetGrid, row='sgRNA_class',
            size=3, aspect=1.65)
        .map_dataframe(plot_heatmap)
        .set_xlabels('cluster size')
        .set_ylabels('NT positive cells')
    )
    
    return fg
    
def plot_NT_positive_histogram(df_cells):
    import seaborn as sns
    import matplotlib.pyplot as plt
    return (df_cells
        .pipe(filter_clusters)
        .query('gate_NT')
        .pipe(sns.FacetGrid, size=5, hue='sgRNA_class')
        .map(plt.hist, 'cluster_NT', bins=np.arange(0.5, 8), 
             histtype='step', lw=2, normed=True)
        .add_legend()
        .set_xlabels('# pos. cells in cluster')
        .set_ylabels('fraction of pos. cells')
        )

def plot_hit_swarm(df_hits, col, sort_genes=False):
    import seaborn as sns
    if sort_genes:
        df_hits = (df_hits
            .assign(key=lambda x: x.groupby(['stimulant', 'gene'])
                 [col].transform('mean').pipe(list))
            .sort_values('key', ascending=False)
        )

    fig, ax = plt.subplots(figsize=(30, 5))
    sns.swarmplot(data=df_hits, ax=ax,
                  x='gene', y=col, 
                  hue='stimulant', dodge=True)
    plt.xticks(rotation=90);
    fig.tight_layout()

    return fig

