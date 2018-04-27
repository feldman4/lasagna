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


def do_median_call_3(df_raw, cycles, constant_C=1):
    n = len(df_raw)
    X = np.zeros((n / (cycles * 3), cycles, 4), dtype=float)
    X_ = in_situ.dataframe_to_values(df_raw, channels=3)
    X_ = X_ / X_.mean(axis=0)[None]
    # now in A,C,G,T order
    X[:, :, [0, 2, 3]] = X_
    X[:, :, 1] = 1
    # Y, W = in_situ.transform_medians(X.reshape(-1, 4))
    Y = X.reshape(-1, 4)
    df_reads = in_situ.call_barcodes(df_raw, Y, cycles=cycles, channels=4)
    return df_reads.drop(['channel', 'intensity'], axis=1)


def add_phenotype(df_cells, df_ph):
    cols = ['well', 'tile', 'cell']
    return (df_cells     
        .join(df_ph.set_index(cols), on=cols)
        .pipe(annotate_cells)   
        )

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
        return pd.Categorical(s, categories=['TNFa', 'IL1b'], ordered=True)

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
        .assign(stimulant=lambda x: 'IL-1b')
        .assign(positive=get_positive)
        )

def filter_crap(df_raw):
    channels = 3
    X = in_situ.dataframe_to_values(df_raw, channels=channels)
    filt = ~((X > 3000).all(axis=2).any(axis=1))
    filt = ((X != np.nan) & filt[:, None, None])
    filt = filt.reshape(-1)
    return df_raw[filt]

def constant_C(df_raw, value):
    blah = df_raw.drop_duplicates(['well', 'tile', 'blob', 'cycle']).copy()
    blah['channel'] = 'C'
    blah['intensity'] = value

    return pd.concat([df_raw, blah])#.sort_values(['well', 'tile', 'cell', 'blob', 'cycle', 'channel'])

def read_csvs(files):
    def read_csv(f):
        try:
            return pd.read_csv(f)
        except:
            return None
    return pd.concat(map(read_csv, files))
    
def grid_view2(df, **kwargs):
    files = []
    bounds = []
    for _, row in df.iterrows():
        files.append(name(row, tag='phenotype_aligned', mag='10X', 
                          cycle='c0-DAPI-RELA-mNeon',
                          ext='tif', subdir='process/10X_c0-DAPI-RELA-mNeon/')
                    )
        i, j = int(row['y']), int(row['x']) # cell coordinates
        bounds.append((i, j, i+1, j+1))
    return grid_view(files, bounds, **kwargs)