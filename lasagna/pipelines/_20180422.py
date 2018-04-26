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


def do_median_call_3(df_raw, cycles, constant_C):
    n = len(df_raw)
    X = np.zeros((n / (cycles * 3), cycles, 4), dtype=int)
    X[:, :, :3] = in_situ.dataframe_to_values(df_raw, channels=3)
    X[:, :, 3] = constant_C
    Y, W = in_situ.transform_medians(X.reshape(-1, 4))

    df_reads = in_situ.call_barcodes(df_raw, Y, cycles=cycles, channels=4)
    return df_reads.drop(['channel', 'intensity'], axis=1)

def subset_cells(df_cells_all, df_design):
    s = (df_design.drop_duplicates('barcode_in_situ')
     .set_index('barcode_in_situ')
     .rename(columns={'ambiguous': 'duplicated'})
     [['subpool', 'sgRNA_name', 'duplicated']])

    cols = ['well', 'tile', 'cell']
    df_cells = (df_cells_all
     .drop_duplicates(cols)
     .join(s, on='cell_barcode_0')
     .join(df_ph.set_index(['well', 'tile', 'cell']), on=cols)
     .pipe(pipeline.annotate_cells)         
    )
    return df_cells

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