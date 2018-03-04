from lasagna.imports import *
import random
from lasagna.designs import pool0
import lasagna.designs.parts

home = '/Users/feldman/lasagna/libraries/'

def load_barcodes():
    f = os.path.join(home, '20171211_12bp_dist3_97304.csv')
    return pd.read_csv(f)

def load_layout():
    f = os.path.join(home, 'Lasagna Oligos - pools.csv')
    df_layout = (pd.read_csv(f, skiprows=4)
        .filter(regex='^(?!Unnamed).*')
        .dropna(axis=0, how='all')
        .dropna(axis=1))
    # return df_layout
    df_layout.columns = [c.replace(' ', '_') for c in df_layout.columns]
    columns_int = ['dialout', 'sgRNAs/gene', '#_of_sgRNAs', 
                   'barcodes/sgRNA', 'spots/oligo', '#_of_barcodes', 
                   '#_of_spots', '#_of_genes']
    for c in columns_int:               
        df_layout[c] = df_layout[c].astype(int)
    return df_layout

def load_sgRNAs():
    f = os.path.join(home, 'UTR_sgRNAs.tsv')
    a = pd.read_csv(f, sep='\s+', header=None)
    a.columns = 'name', 'sgRNA'
    a['sgRNA_design'] = 'UTR_sgRNAs'

    f = os.path.join(home, 'LG_sgRNAs.tsv')
    b = pd.read_csv(f, sep='\s+', header=None)
    b.columns = 'name', 'sgRNA'
    b['sgRNA_design'] = 'FR_LG'
    filt  = [not ('target' in x) and ('mut' in x) for x in b['name']]
    b = b[filt]

    f = os.path.join(home, 'FR_GFP_TM_sgRNAs.tsv')
    c = pd.read_csv(f, sep='\s+', header=None)
    c.columns = 'name', 'sgRNA'
    c['sgRNA_design'] = 'FR_GFP_TM'

    df_sgRNAs = pd.concat([a,b,c])
    return df_sgRNAs

def run():
    # scramble the barcode order
    df_barcodes = (load_barcodes()
        .sample(frac=1, random_state=1)
        .query('hp < 5 & gc < 0.11 & div2 > 2')
        .reset_index(drop=True)
        )

    df_layout = (load_layout()
          .query('pool == "pool2"'))

    df_sgRNAs = load_sgRNAs()

    df_design = design_pool(df_layout, df_sgRNAs, df_barcodes)

    return df_design
    
def design_pool(df_layout, df_sgRNAs, df_barcodes):
    columns = ['sgRNA', 'barcode', 'sgRNA_name', 'gene_id',
               'oligo_design', 'sgRNA_design', 
               'subpool', 'dialout']

    arr = []
    barcodes = iter(df_barcodes['barcode'])
    df_sgRNAs['gene_id'] = df_sgRNAs['gene_id'].fillna(-1)
    for _, row in df_layout.iterrows():
        filt = df_sgRNAs['sgRNA_design'] == row['sgRNA_design']

        df_sgRNAs_ = (df_sgRNAs[filt]
                        .groupby('gene_id')
                        .head(row['sgRNAs/gene']))

        it = zip(df_sgRNAs_['name'], df_sgRNAs_['sgRNA'], df_sgRNAs_['gene_id'])

        for name, sgRNA, gene_id in it:

            for _ in range(int(row['barcodes/sgRNA'])):


                x = {'sgRNA': sgRNA, 
                     'sgRNA_name': name, 
                     'gene_id': gene_id,
                     'barcode': barcodes.next(),
                     'dialout': row['dialout'],
                     'subpool': row['subpool'],
                     'sgRNA_design': row['sgRNA_design'],
                     'oligo_design': row['oligo_design']
                     }
                s = build_degenerate_oligo(sgRNA, x['barcode'], x['oligo_design'], x['dialout'])
                s = degenerate2(s)
                
                arr_ = []
                i = 0
                while len(arr_) < row['spots/oligo']:
                    oligo = s.next()
                    if pool0.count_typeIIS_sites(oligo) == 4:
                        x['oligo'] = oligo
                        arr_.append(x.copy())
                    else:
                        i += 1
                        assert(i < 1e2)

                arr.extend(arr_)

        print len(arr), 'rows done'

    df_design = pd.DataFrame(arr)
    return df_design

def degenerate2(s):
    """returns a generator of degenerated sequences in scrambled order
    """
    from lasagna.utils import base_repr

    n = s.count('N')
    seed = hash(s) % (2**32 - 1)
    rng = random.Random(seed)
    random_base_ix = lambda: base_repr(rng.randint(0, 4**(n + 1) - 1), 4, n + 1)[::-1]
    while True:
        bases = ['ACTG'[int(j)] for j in random_base_ix()]
        s2 = s
        for b in bases:
            s2 = s2.replace('N', b, 1)
        yield s2

def build_degenerate_oligo(sgRNA, barcode, oligo_design, dialout):
    # 1-indexed in layout spreadsheet
    dialout = dialout - 1

    layout = lasagna.designs.parts.default_layouts[oligo_design]
    dialout_5, dialout_3 = lasagna.designs.parts.dialout_primers[dialout]
    parts = {'barcode': barcode
            ,'sgRNA':   sgRNA
            ,'dialout_5': dialout_5
            ,'dialout_3_rc': lasagna.design.rc(dialout_3)
            }

    s = pool0.build_oligo(layout, parts)
    return s

def run_checks(df_design):
    df_design = df_design.copy()
    df_design['typeIIs_site_count'] = df_design['oligo'].apply(pool0.count_typeIIS_sites)

    def count_oligos(df_design, key):
        col = 'oligos/%s' % key
        col_ = 'num_%s' % key
        return (df_design
                     .groupby(['subpool', 'dialout', 'sgRNA_design', key])
                     .size().rename(col)
                     .reset_index().groupby(['subpool', 'dialout', 'sgRNA_design'])
                     [col].value_counts()
                     .rename(col_).reset_index()
                     [['subpool', 'dialout', 'sgRNA_design', col_, col]])

    checks = [
          ('type IIS sites', df_design.groupby(['subpool', 'sgRNA_design'])['typeIIs_site_count'].value_counts()),
          ('unique oligos', len(set(df_design['oligo']))),
          ('unique barcodes', len(set(df_design['barcode']))),
          ('unique sgRNAs', len(set(df_design['sgRNA']))),
          ('oligo length', Counter(map(len, df_design['oligo']))),
          ('oligos per design', df_design.groupby(['subpool', 'dialout', 'sgRNA_design']).size()),
          ('oligos per sgRNA', count_oligos(df_design, 'sgRNA')),
          ('oligos per gene ID', count_oligos(df_design, 'gene_id')),
          ('barcodes per sgRNA', df_design.drop_duplicates(['barcode', 'sgRNA']).groupby(['subpool', 'dialout', 'sgRNA_design', 'sgRNA']).size()),
         ]

    double_bar = '='*20
    sep1 = '\n' + '-'*20 + '\n'
    sep2 = '\n\n' + double_bar + '\n'

    s = sep2.join(sep1.join([str(c) for c in check]) for check in checks)
    s = double_bar + '\n' + s
    return s

  