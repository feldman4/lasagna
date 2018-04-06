from lasagna.imports import *
import random
import lasagna.designs.pool0
import lasagna.designs.pool2
import lasagna.designs.parts

home = '/Users/feldman/lasagna/libraries/'

def load_ncbi_ensg_uniprot_hgnc():
    f = os.path.join(home, 'biomart/ENSG_NCBI_UniProt_HGNC.txt')
    columns = {'NCBI gene ID': 'gene_id', 'Gene name': 'gene_symbol'
              ,'Gene stable ID': 'ENSG'}
    df_biomart = pd.read_csv(f).rename(columns=columns)
    return df_biomart


def load_barcodes():
    f = os.path.join(home, '20180224_12bp_dist3_97406.csv')
    return pd.read_csv(f)


def load_genome_wide_ids():
    f = os.path.join(home, 'pool3/19175_gene_ids.txt')
    gene_ids = list(pd.read_csv(f, header=None)[0])
    return gene_ids


def filter_genome_wide_sgRNAs(df_sgRNAs, n):
    sources = ['Wang 2015', 'brunello', 'CRISPOR', 'ECRISP', 'ensembl', 'benchling']

    order_source = lambda x: (x['source'].astype('category').cat
                    .reorder_categories(sources))

    return (df_sgRNAs.query('source == @sources')
                 .dropna(subset=['gene_id'])
                 .assign(CRISPOR_WangSVM=lambda x: x['CRISPOR_WangSVM'].fillna(-1))
                 .assign(source=order_source)
                 .sort_values(['source', 'CRISPOR_WangSVM'], ascending=[True, False])
                )


def filter_typeIIs_sgRNAs(df_sgRNAs):
    def f(x):
        x = 'CACCG' + x + 'GTTT'
        return lasagna.designs.pool0.count_typeIIS_sites(x)
    filt = df_sgRNAs['sgRNA'].apply(f) == 0
    return df_sgRNAs[filt]



def filter_typeIIs_barcodes(df_barcodes):
    def f(x):
        x = 'TTCC' + x + 'ACTG'
        return lasagna.designs.pool0.count_typeIIS_sites(x)
    filt = df_barcodes['barcode'].apply(f) == 0
    return df_barcodes[filt]



def annotate_sgRNA_designs(df_sgRNAs):
    gene_ids = load_gene_ids()

    arr = []
    for design, ids in gene_ids.items():
        df = df_sgRNAs.query('gene_id == @ids').copy()
        assert(len(set(df['gene_id'])) == len(ids))

        df['sgRNA_design'] = design
        arr += [df.drop_duplicates('sgRNA')]

    return pd.concat(arr)


def split_genome_wide_design(df_sgRNAs):
    it = zip(df_sgRNAs['sgRNA_design'], df_sgRNAs['gene_id'])
    used_gene_ids = set()
    arr = []
    for design, gene_id in it:
        if design == 'genome_wide':
            if gene_id in used_gene_ids:
                arr.append('genome_wide_3')
            else:
                used_gene_ids.add(gene_id)
                arr.append('genome_wide_1')
        else:
            arr.append(design)
    
    return arr

def name_sgRNAs(df_sgRNAs):
    c = Counter()
    it = zip(df_sgRNAs['gene_symbol'], df_sgRNAs['gene_id'])
    fmt = 'sg_{symbol}_{gene_id}_{ix}'

    arr = []
    for symbol, gene_id in it:
        ix = c[(symbol, gene_id)]
        c[(symbol, gene_id)] += 1
        arr += [fmt.format(symbol=symbol, gene_id=gene_id, ix=ix)]

    return arr


def load_gene_ids():
    f = os.path.join(home, 'pool3/Lasagna-Nougat.xlsx')
    
    names = ['JSB_366_ISG', 'JSB_272_UL', 'JSB_296_DUB', 'JSB_87']
    gene_ids = {k: sorted(set(v['gene_id'].dropna().astype(int)))
                for k,v in pd.read_excel(f, sheet_name=names).items()}

    gene_ids['genome_wide'] = load_genome_wide_ids()

    return gene_ids

def fill_gene_ids(df_sgRNAs): 
    return df_sgRNAs['gene_id'].fillna(0).astype(int)


def load_extra_CRISPOR():
    df_biomart = load_ncbi_ensg_uniprot_hgnc()

    f = os.path.join(home, 'pool3/CRISPOR_extra_sgRNAs.tsv')
    clean_symbol = lambda x: x['guideId'].apply(lambda y: y.split('_')[0])

    s = (df_biomart.drop_duplicates('gene_symbol')
     .set_index('gene_symbol')['gene_id'])

    df_extra = (pd.read_csv(f, sep='\t', comment='#')
                  .assign(sgRNA=lambda x: x['guideSeq'])
                  .assign(gene_symbol=clean_symbol)
                [['gene_symbol', 'sgRNA']]
                  .assign(source='CRISPOR')
                .join(s, on='gene_symbol')
               )

    return df_extra

def load_extra_ensembl():
    f = os.path.join(home, 'pool3/extra_sgRNAs_ensembl.csv')
    return pd.read_csv(f)


def load_extra_nontargeting():
    files = ('pool3/extra_nontargeting_Wang2015.csv','pool3/extra_nontargeting_GeCKO.csv')
    return (pd.concat([pd.read_csv(os.path.join(home, f)) for f in files])
              .assign(sgRNA_design=lambda x: x['source'])
              .pipe(filter_typeIIs_sgRNAs))


def parse_biomart_fasta(s, ncols):
    entries = [x for x in s.split('>') if x]
    pat = '\|'.join(['(.*)'] * ncols) + '((?:\n[ACTG]+)+)'
#     print pat
    results = [re.findall(pat, x)[0] for x in entries]
    results = [list(x[:-1]) + [x[-1].replace('\n', '')] for x in results]
    return results

def load_exons():
    """if an exon is marked minus strand, the sequence is antisense (as expressed)
    """
    txt = ''
    for f in ('biomart/exons.fasta', 'biomart/exons_from_symbols.fasta'):
        f2 = os.path.join(home, f)
        with open(f2, 'r') as fh:
            txt += fh.read()
        
    exons = parse_biomart_fasta(txt, 1)
    exons = pd.DataFrame(exons, columns=['ENSE', 'exon_seq'])

    arr = []
    for f in ('biomart/exon_table.csv', 'biomart/exon_table_from_symbols.csv'):
        f2 = os.path.join(home, f)
        arr += [pd.read_csv(f)]

    columns = {'exon_stable_id': 'ENSE', 'gene_stable_id': 'ENSG', 'transcript_stable_id': 'ENST'}
    df_exons = \
    (pd.concat(arr)
     .rename(columns=lambda x: x.lower().replace(' ', '_').replace('_(bp)', ''))
     .rename(columns=columns)
     .join(exons.set_index('ENSE'), on='ENSE')
    )

    return df_exons


def get_coding_sequences(df_exons):

    def get_coordinates(x):
        return x.apply(lambda y: exon_coding_coordinates(**y.to_dict()), axis=1)
        
    coords = get_coordinates(df_exons)
    arr = []
    for (i0, i1), exon_seq in zip(coords, df_exons['exon_seq']):
        arr.append(exon_seq[i0:i1])
        
    return arr

def exon_coding_coordinates(exon_region_start, exon_region_end, 
                            genomic_coding_start, genomic_coding_end, strand, **kwargs):
    """for each ENSE, get the indices that are coding relative to the strand
    """
    if strand == 1:
        start = genomic_coding_start - exon_region_start
        end   = genomic_coding_end - exon_region_start
    elif strand == -1:
        start = exon_region_end - genomic_coding_end
        end   = exon_region_end - genomic_coding_start
    else:
        raise ValueError(strand)
    start, end = int(start), int(end)
#     print start, end, exon_region_end - exon_region_start
    return start, end 


def exons_to_sgRNAs(df_exons):
    d = os.getcwd()
    os.chdir('/Users/feldman/packages/crisporWebsite-master/')
    import crisporEffScores

    arr = []
    df_exons = df_exons.reset_index(drop=True)
    it = zip(df_exons.index, df_exons['exon_seq_coding'])
    for ix, exon in it:
        arr += [(ix, sg) for sg in find_sgRNAs(exon)]
        
    df = (pd.DataFrame(arr, columns=[df_exons.index.name, 'sgRNA'])
          .set_index(df_exons.index.name))

    df['CRISPOR_WangSVM'] = crisporEffScores.calcWangSvmScores(list(df['sgRNA']))

    os.chdir(d)

    return df_exons.join(df)

def find_sgRNAs(seq):
    pat    = '(.{20}).GG'
    sgRNAs = re.findall(pat, seq)
    sgRNAs += map(rc, re.findall(pat, rc(seq)))
    return sgRNAs
    

def process_exons():
    s = (load_ncbi_ensg_uniprot_hgnc().drop_duplicates('ENSG')
        .set_index('ENSG')[['gene_id', 'gene_symbol']])

    df_exons = \
    (load_exons()
     .drop_duplicates('ENSE')
     .join(s, on='ENSG')
     .dropna()
     .sort_values(['gene_id', 'constitutive_exon', 'exon_rank_in_transcript'], 
        ascending=[True, False, True])
     # .query('constitutive_exon == 1')
     .assign(exon_seq_coding=get_coding_sequences)
     .pipe(exons_to_sgRNAs)
    )
    
    return df_exons


def export_nontargeting():
    f = 'pool3/extra_nontargeting_Wang2015.csv'
    df_wang = lasagna.designs.pool0.load_wang()

    df_wang_ = \
    (df_wang[df_wang['gene_symbol'].isnull()]
     .query('CRISPOR_WangSVM > 70')
     .sample(frac=1, random_state=0)[['sgRNA', 'CRISPOR_WangSVM']]
     .assign(source='Wang_nontargeting')
    )

    df_wang_.to_csv(f, index=None)
    print 'exported %d nontargeting guides to %s' % (len(df_wang_), f)

    return df_wang_

def load_extra_ECRISP():

    f = 'pool3/gene_ID_symbol.tsv'
    columns={'Gene ID': 'gene_id', 'Gene Symbol': 'gene_symbol'}
    s = (pd.read_csv(f, sep='\t')
         .rename(columns=columns)
         .set_index('gene_symbol')['gene_id'])

    f = 'pool3/extra_ECRISP.tsv'
    df = (pd.read_csv(f, sep='\t')
             .assign(sgRNA=lambda x: x['Nucleotide sequence'].apply(lambda y: y.split()[0]))
             .assign(ENSG=lambda x: x['Gene Name'].apply(lambda y: y.split('::')[0]))
             .assign(gene_symbol=lambda x: x['Gene Name'].apply(lambda y: y.split('::')[1]))
             [['sgRNA', 'ENSG', 'gene_symbol']]
             .assign(source='ECRISP')
             .join(s, on='gene_symbol')
             .dropna()
            )
    return df


def load_benchling_sgRNAs():
    
    f = 'pool3/Lasagna-Nougat.xlsx'
    return (pd.read_excel(f, sheet_name='benchling_sgRNAs')
              .assign(sgRNA=lambda x: x['sgRNA'].apply(lambda y: y.upper()))
              .pipe(filter_typeIIs_sgRNAs)
              [['gene_symbol', 'gene_id', 'sgRNA']]
              .assign(source='benchling'))


def load_layout():
    f = os.path.join(home, 'pool3/Lasagna Oligos - pools.csv')
    df_layout = (pd.read_csv(f, skiprows=5)
        .filter(regex='^(?!Unnamed).*')
        .dropna(axis=0, how='all')
        .dropna(axis=1))

    df_layout.columns = [c.replace(' ', '_') for c in df_layout.columns]
    columns_int = ['dialout', 'sgRNAs/gene', '#_of_sgRNAs', 
                   'barcodes/sgRNA', 'spots/oligo', '#_of_barcodes', 
                   '#_of_spots', '#_of_genes']
    for c in columns_int:               
        df_layout[c] = df_layout[c].astype(int)
    return df_layout


def get_JSB_13_barcodes():
    f = os.path.join(home, 'pool3/JSB_13.csv')
    design = pd.read_csv(f).dropna()
    design['oligo'] = design['Sequence']

    pat = 'ttcc(.*)actg'
    get_bc = lambda a: re.findall(pat, a)[0]
    design['barcode'] = design['oligo'].apply(get_bc)
    return sorted(set(design['barcode']))

