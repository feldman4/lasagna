from lasagna.imports import *

gate_cells = and_join(['3000 < dapi_nuclear_max < 12000',
                       '60 < area_nuclear < 140', 
                       '2000 < gfp_cell_median < 8000',
                       'duplicated == False'])

gate_NT = 'dapi_gfp_nuclear_corr < -0.5'

stimulant = {'A': 'TNFa', 'B': 'IL1b'}
positive_genes = {'IL1b': ['MAP3K7', 'NFKBIA', 'IKBKG', 
                     'IRAK1', 'MYD88', 'IRAK4', 'TRAF6'],
            'TNFa': ['MAP3K7', 'NFKBIA', 'IKBKG', 'TRADD', 
                     'TNFRSF1A', 'TRAF2', 'IKBKB', 'CHUK', 'RIPK1']}

positive_gene_list = ['MAP3K7', 'NFKBIA', 'IKBKG', 'TRADD', 'TNFRSF1A', 
                      'IRAK1', 'MYD88', 'IRAK4', 'TRAF2', 'TRAF6', 'IKBKB', 
                      'CHUK', 'RIPK1', 'CRKL',]

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
    def get_gene(sgRNA_name):
        if sgRNA_name is np.nan:
            return sgRNA_name
        if sgRNA_name.startswith('LG'):
            return 'LG'
        pat = 'sg_(.*?)_'
        return re.findall(pat, sgRNA_name)[0]

    def get_targeting(sgRNA_name):
        if sgRNA_name is np.nan:
            return False
        else:
            return 'LG_sg' not in sgRNA_name

    def get_stimulant(well):
        return stimulant[well[0]]

    def categorize_stimulant(s):
        return s.astype('category').cat.reorder_categories(['TNFa', 'IL1b'])

    def get_positive(df_cells):
        TNFa_pos = positive_genes['TNFa']
        IL1b_pos = positive_genes['IL1b']
        gate_pos = or_join(['stimulant == "TNFa" & gene == @TNFa_pos',
                            'stimulant == "IL1b" & gene == @IL1b_pos'])
        return df_cells.eval(gate_pos)

    return (df_cells
        .assign(gate_NT=lambda x: x.eval(gate_NT))
        .assign(gene=lambda x: x['sgRNA_name'].apply(get_gene))
        .assign(targeting=lambda x: x['sgRNA_name'].apply(get_targeting))
        .assign(stimulant=lambda x: x['well'].apply(get_stimulant).pipe(categorize_stimulant))
        .assign(positive=get_positive)
        )

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