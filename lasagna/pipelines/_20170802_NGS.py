from lasagna.imports import *
from Levenshtein import distance

# A01-D12
samples = ['%s%02d' % (r,c) for r,c in product('ABCD', range(1,13))]

get_well = lambda h: re.findall('T1_(...)_S', h)[0]
get_polymerase = lambda x: 'jumpstart' if x[0] in 'AB' else 'herculase'
get_subpool = lambda well: (samples.index(well) % 24) + 1


def find_sgRNA_errors(df):
    """Group sgRNAs by barcode, compute error from each sgRNA to the most
    frequent sgRNA for that barcode. Return errors in like-indexed pd.Series.
    ~3s / 1e5 rows
    """
    def label_errors(df_):
        # assume sorted
        # top_sgRNA = df_.sort_values('count', ascending=False)['sgRNA'].iloc[0]
        top_sgRNA = df_['sgRNA'].iloc[0]
        return df_['sgRNA'].apply(lambda x: distance(x, top_sgRNA))

    sgRNA_error = (df.sort_values(['barcode', 'count'], ascending=[True, False])
                     .groupby(['barcode'])
                     .apply(label_errors))
    sgRNA_error.index = sgRNA_error.index.droplevel(0)
    return sgRNA_error

def label_condition(df_):
    df_['polymerase'] = df_['well'].apply(get_polymerase)
    df_['subpool']    = df_['well'].apply(get_subpool)
    return df_

def load_counts(files):
    """Load regex matches for 20170802_GL_DF.
    """
    columns = 'count', 'sgRNA', 'spacer', 'barcode'
    col_order = ['well', 'count', 'barcode', 'sgRNA', 'spacer']
    
    arr = []
    for f in files:
        try:
            df = pd.read_csv(f, header=None, sep='\t')
        except pd.errors.EmptyDataError as e:
            continue
        df.columns = columns
        df['well'] = get_well(f)
        arr += [df[col_order]]

    df = pd.concat(arr).reset_index(drop=True)
    return df


def load_pool0(csv_path):
    df_pool0 = pd.read_csv(csv_path).rename(columns={'dialout':'subpool'})
    df_pool0['subpool'] += 1
    return df_pool0

def counts_per_barcode(df):
    """Counts per barcode.
    """
    counts = Counter()
    for b,c in zip(df['barcode'], df['count']):
        counts[b] += c
    return [counts[b] for b in df['barcode']]
