from lasagna.pipelines._20171015_NGS import *


def load_samples(home):
    f = os.path.join(home, 'samples.tsv')
    return pd.read_csv(f, sep='\t').set_index('well')

def load_database(home='/Users/feldman/lasagna/libraries/'):
    f = os.path.join(home, 'Feldman_12K_Array_pool1_table.csv')
    df_design = pd.read_csv(f)
    df_design = df_design.drop_duplicates(subset=['sgRNA', 'barcode'])
    df_sgRNAs, _ = pool1.load_tables(path=home)
    return df_design, df_sgRNAs

def load_read_stats(ssv):
    df_stats = pd.read_csv(ssv, sep='\s+', engine='python', header=None)
    df_stats.columns = 'reads', 'sample'
    df_stats['well'] = df_stats['sample'].apply(get_well)
    df_stats['reads'] = (df_stats['reads'] / 4).astype(int)
    return df_stats[['well', 'reads', 'sample']]

def load_grep(files, df_samples):
    """ files = !ls hist_FACS/*hist
    """
    arr = []
    for f in files:
        try:
            df = pd.read_csv(f, header=None, sep='\s+')
            df.columns = 'count', 'seq'
            well = get_well(f)
            if well not in df_samples.index:
                continue
            df['well']    = well
            df['gate']    = df_samples.loc[well, 'gate']
            df['library'] = df_samples.loc[well, 'library']
            df['pattern'] = get_pattern(f)
            df['pattern_primers'] = df_samples.loc[well, 'pattern']
            df['file']    = f
            arr += [df]
        except (pd.errors.EmptyDataError):
            continue
    df = pd.concat(arr)
    df = df.query('pattern == pattern_primers')

    return df

def analyze_sgRNAs(df, df_design, df_sgRNAs):
    df_sg = df.query('pattern == "sg"').copy()
    df_sg['sgRNA'] = df_sg['seq'].apply(lambda x: x[4:-4])

    cols = ['source', 'tag', 'gene_symbol', 'gene_id']
    s = df_sgRNAs.set_index('sgRNA')[cols]

    cols = ['well', 'count', 'sgRNA', 'source', 'tag', 'gene_symbol']

    x = df_design[['sgRNA', 'design']].drop_duplicates().set_index('sgRNA')

    df_sg2 = df_sg.join(x, on='sgRNA')#.dropna()

    # # require match to +/- control
    # filt = df_sg2['design'].isin(['FR_GFP_TM', 'nontargeting controls'])
    # df_sg2 = df_sg2[filt]

    df_sg2['fraction'] = df_sg2.groupby('well')['count'].transform(lambda x: x/x.sum())

    return df_sg, df_sg2

def analyze_barcodes(df, df_design, df_sgRNAs):
    df_bc = df.query('pattern=="pL42"').copy()
    df_bc['barcode'] = df_bc['seq'].apply(lambda x: x[4:-4])

    cols = ['well', 'library', 'gate', 'barcode', 'count']
    df_bc = df_bc[cols]

    df_bc2 = df_bc.join(df_design.set_index('barcode').drop('gene_id', axis=1), on='barcode')
    df_bc2 = df_bc2.dropna()

    # require match to +/- control
    filt = df_bc2['design'].isin(['FR_GFP_TM', 'nontargeting controls'])
    df_bc2 = df_bc2[filt]

    df_bc2['fraction'] = df_bc2.groupby('well')['count'].transform(lambda x: x/x.sum())

    return df_bc, df_bc2

def crappy_hist(df, thresholds=(0, 3, 10, 100, 500)):
    cols = ['library', 'gate', 'pattern', 'well']
    a = df.groupby(cols)['count'].sum()
    arr = [a]
    for threshold in thresholds:
        s = (df.query('count > %d' % threshold)
               .groupby(cols)
               .size())
        s.name = 'unique > %d' % threshold
        s = s.fillna(0).astype(int)  
        arr += [s]
        
    return pd.concat(arr, axis=1).reset_index()

def more_mapping(df, df_stats, df_sg, df_sg2, df_bc, df_bc2):

    s0 = df_sg.groupby('well')['count'].sum()
    s1 = df_bc.groupby('well')['count'].sum()
    s = pd.concat([s0, s1])
    s.name = 'count_mapped'
    df_stats = df_stats.join(s, on='well').fillna(0)

    s0 = df_sg2.groupby('well')['count'].sum()
    s1 = df_bc2.groupby('well')['count'].sum()
    s = pd.concat([s0, s1])
    s.name = 'count_mapped_FR'
    df_stats = df_stats.join(s, on='well').fillna(0)

    df_stats['mapping']    = df_stats['count_mapped']    / df_stats['reads']
    df_stats['mapping_FR'] = df_stats['count_mapped_FR'] / df_stats['reads']

    return df_stats.drop(['count_mapped', 'count_mapped_FR'], axis=1)

