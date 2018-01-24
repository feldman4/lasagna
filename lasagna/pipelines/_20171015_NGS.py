from lasagna.imports import *
from lasagna.designs import pool1

get_well = lambda x: re.findall('_([ABCDEFGH]..)_', x)[0]
get_pattern = lambda x: re.findall('grep.(.*).hist', x)[0]

samples = ['A01' ,'A02' ,'A03' ,'A04' ,'A05' ,'A06' ,'A07' ,'A08' ,
           'B01' ,'B02' ,'B03' ,'B04' ,'B05' ,'B06' ,'C01' ,'C02' ,
           'C03' ,'C04' ,'C05' ,'C06' ,'C07' ,'C08' ,'D01' ,'D02' ,
           'D03' ,'D04' ,'D05' ,'D06']

def get_gate(well):
    flag_0 = well[0] not in 'AC'
    col = int(well[1:])
    flag_1 = col <= 6
    if flag_0 and flag_1:
        if col % 2 == 1:
            return 'myc'
        else:
            return 'HA'
    return 'no_FACS'

def get_library(well):
    FR1 = ['A01', 'B01', 'B02', 'C01', 'D01', 'D02']
    FR2 = ['A02', 'B03', 'B04', 'C02', 'D03', 'D04']
    FR3 = ['A03', 'B05', 'B06', 'C03', 'D05', 'D06']
    
    if well in FR1:
        return 'pool1_1'
    if well in FR2:
        return 'pool1_2'
    if well in FR3:
        return 'pool1_3'
    
    return 'no_match'

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

def load_grep(files):
    """ files = !ls hist_FACS/*hist
    """
    arr = []
    for f in files:
        try:
            df = pd.read_csv(f, header=None, sep='\s+')
            df.columns = 'count', 'seq'
            df['well'] = get_well(f)
            df['gate'] = get_gate(get_well(f))
            df['library'] = get_library(get_well(f))
            df['file'] = f
            df['pattern'] = get_pattern(f)
            arr += [df]
        except pd.errors.EmptyDataError:
            continue
    df_ = pd.concat(arr)
    df_['prep'] = ['sg' if w[0] in 'AB' else 'pL42' for w in df_['well']]

    # prep tables for analysis
    df = df_.query('prep == pattern')
    filt = df['well'].isin(samples)
    df = df[filt]

    return df, df_

def analyze_sgRNAs(df, df_design, df_sgRNAs):
    df_sg = df.query('pattern == "sg"').copy()
    df_sg['sgRNA'] = df_sg['seq'].apply(lambda x: x[4:-4])

    cols = ['source', 'tag', 'gene_symbol', 'gene_id']
    s = df_sgRNAs.set_index('sgRNA')[cols]

    cols = ['well', 'count', 'sgRNA', 'source', 'tag', 'gene_symbol']
    df_sg.join(s, on='sgRNA').groupby('well').head(2)[cols]

    x = df_design[['sgRNA', 'design']].drop_duplicates().set_index('sgRNA')

    df_sg2 = df_sg.join(x, on='sgRNA').dropna()

    # require match to +/- control
    filt = df_sg2['design'].isin(['FR_GFP_TM', 'LG_TM', 'nontargeting controls'])
    df_sg2 = df_sg2[filt]

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

