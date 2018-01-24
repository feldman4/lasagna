import pandas as pd
import numpy as np
from itertools import product, combinations, cycle
from lasagna.design import rc, generate_sequence
from Levenshtein import distance
from collections import Counter
import regex as re
import os
from lasagna.designs.parts import *


GFP_TM_SGRNAS = 'FR_GFP_TM_sgRNAs.tsv'
LG_SGRNAS  = 'LG_sgRNAs.tsv'

def load_tables(path=''):
    brunello = load_brunello(path=path)
    mitosis = load_cheese_sgRNAs(path=path)
    GFP_TM = load_GFP_TM_sgRNAs(path=path)
    LG_TM = load_LG_TM_sgRNAs(path=path)
    benchling = load_benchling(path=path)
    controls = load_nontargeting_sgRNAs(path=path)
    # patch brunello with ECRISP and Wang sgRNAs to get 3 non-typeIIS / gene
    # selection order is brunello > ECRISP > Wang
    ecrisp = load_ecrisp(path=path)
    wang = load_wang(path=path)

    include = [mitosis, GFP_TM, LG_TM, controls, brunello, wang, ecrisp, benchling]
    df_sgRNAs = (pd.concat(include).reset_index(drop=True)).drop_duplicates('sgRNA')

    # add # of GO terms for each gene ID
    df_go = pd.read_csv(path + 'go_entrez.tsv', sep='\t').dropna()
    df_go = df_go.rename(columns={'NCBI gene ID': 'gene_id'})
    df_go['gene_id'] = df_go['gene_id'].astype(int)

    go_count = df_go.groupby('gene_id').count().iloc[:,0]
    go_count.name = 'go_count'
    df_sgRNAs = df_sgRNAs.join(go_count, on='gene_id')
    df_sgRNAs['go_count'] = df_sgRNAs['go_count'].fillna(0).astype(int)

    # add # of endocytosis GO terms for each gene ID
    df_go['endo'] = df_go['GO term name'].apply(lambda x: 'endocytosis' in x)
    endo_go_count = df_go.groupby('gene_id')['endo'].sum().astype(int)
    endo_go_count.name = 'endo_go_count'
    df_sgRNAs = df_sgRNAs.join(endo_go_count, on='gene_id')
    df_sgRNAs['endo_go_count'] = df_sgRNAs['endo_go_count'].fillna(0).astype(int)

    # add shitty ENCODE GAII RNAseq from 2011
    # weak correlation with GO annotations
    enst_ncbi = load_enst_ncbi(path=path)
    s = enst_ncbi.set_index('Transcript stable ID')['gene_id']

    # hela_rnaseq = pd.read_csv(path + 'HeLa_RNAseq/ENCFF000DMU/abundance.tsv', sep='\t')
    # hela_rnaseq = (hela_rnaseq.join(s, on='target_id')
    #             .dropna()
    #             .groupby('gene_id')['tpm'].sum())
    # hela_rnaseq.index = hela_rnaseq.index.astype(int)
    # hela_rnaseq.name = 'hela_tpm'
    # df_sgRNAs = df_sgRNAs.join(hela_rnaseq, on='gene_id')

    return df_sgRNAs, df_go


def load_brunello(path=''):
    f = os.path.join(path, 'brunello_SuppTables/STable 21 Brunello.csv')
    brunello = pd.read_csv(f)

    columns = {'Target Gene ID': 'gene_id'
              ,'Target Transcript': 'transcript'
              ,'Genomic Sequence': 'chromosome'
              ,'Target Gene Symbol': 'gene_symbol'
              ,'sgRNA Target Sequence': 'sgRNA'
              ,'Exon Number': 'exon'
              ,'Rule Set 2 score': 'rule_set_2'
              }
    brunello = brunello[columns.keys()].rename(columns=columns)
    brunello['source'] = 'brunello'
    brunello['tag'] = 'brunello'
    return brunello


def load_mckinley(path=''):
    """from McKinley 2017, sgRNA sequences standardized
    """
    columns = {'Gene': 'gene_symbol', 'Guide sequence': 'sgRNA'}
    df_mckinley = (pd.read_excel(path + 'mitosis/2017 McKinley_2017_STable1.xls')
                     .rename(columns=columns))
    f = lambda x: x[4:].replace('g', '')
    # 5' G from old U6 rule
    weird = {'GGCTTTTACACGAGGAGAACAGG': 'GGCTTTTACACGAGGAGAAC'
            ,'GAAAGAAGCCTCAACTTCGTC'  : 'AAAGAAGCCTCAACTTCGTC'
            ,'GGATAAAGCGGAAGGCTCCCCG' : 'ATAAAGCGGAAGGCTCCCCG'
            ,'GATTCAGGTTCTTCCGGGTGC'  : 'ATTCAGGTTCTTCCGGGTGC'
            ,'GTGAGTCACGAGAACACGTTT'  : 'TGAGTCACGAGAACACGTTT'}
    g = lambda x: weird.get(x, x)
    df_mckinley['sgRNA'] = df_mckinley['sgRNA'].apply(f).apply(g)
    
    df_cheese = load_cheese(path=path)
    df_mckinley['cheese'] = df_mckinley['gene_symbol'].isin(df_cheese['gene_symbol'])
    return df_mckinley


def load_cheese(path=''):
    """76 genes suggested by Iain Cheeseman
    """
    # SGO1 not in brunello (why?)
    # KNL1 in brunello with symbol CASC5
    columns = {'Gene Target': 'gene_symbol'}
    df_cheese = pd.read_excel(path + 'Cheeseman_sgRNAs_20170710.xlsx')
    df_cheese = df_cheese.rename(columns=columns)
    return df_cheese


def load_cheese_sgRNAs(path=''):
    """cut down to 2 sgRNAs (some have 4 sgRNAs)
    """
    df_mckinley = load_mckinley(path=path)
    df_mckinley = (df_mckinley[df_mckinley['cheese']]
                        .groupby('gene_symbol').head(2))

    enst_ncbi = (load_enst_ncbi(path=path)
                  .drop_duplicates(['gene_symbol', 'gene_id']))

    symbols = df_mckinley['gene_symbol'].tolist()
    df_mckinley['gene_id'] = (enst_ncbi.set_index('gene_symbol')
                                      .loc[symbols, 'gene_id']).astype(int).tolist()

    df_mckinley['source'] = 'McKinley 2017'
    df_mckinley['tag'] = 'Cheeseman'
    columns = ['gene_symbol', 'sgRNA', 'gene_id', 'source', 'tag']
    return df_mckinley[columns]


def load_wang(path=''):
    columns = {'sgRNA sequence':'sgRNA'
          ,'Symbol': 'gene_symbol'}
    f = os.path.join(path, 'Wang_2015_supplement/Wang_2015_STable1.csv')
    df_wang = (pd.read_csv(f)
                 .rename(columns=columns)[columns.values()])

    enst_ncbi = (load_enst_ncbi(path=path)
                  .drop_duplicates(['gene_symbol', 'gene_id'])
                  .set_index('gene_symbol'))
    df_wang = df_wang.join(enst_ncbi['gene_id'], on='gene_symbol')
    df_wang['source'] = 'Wang 2015'

    return df_wang


def load_enst_ncbi(path=''):
    path = os.path.join(path, 'ENS_to_NCBI.tsv')
    columns = {'NCBI gene ID': 'gene_id', 'Gene name': 'gene_symbol'}
    enst_ncbi = (pd.read_csv(path, sep='\t')
                   .rename(columns=columns))
    return enst_ncbi


def load_ecrisp(path=''):
    """Not in use.
    """
    enst_ncbi = load_enst_ncbi(path=path)
    enst_ncbi = (enst_ncbi.drop_duplicates('Gene stable ID')
                          .set_index('Gene stable ID')['gene_id']
                          .to_dict())

    columns = {'Nucleotide sequence': 'sgRNA'}
    f = os.path.join(path, 'ECRISPED.tsv')
    df_ecrisp = (pd.read_csv(f, sep='\t')
                   .rename(columns=columns))

    df_ecrisp['sgRNA'] = df_ecrisp['sgRNA'].apply(lambda x: x[:20])
    df_ecrisp['ENSG'] = df_ecrisp['Gene Name'].apply(lambda x: x.split('::')[0])
    # trustworthy?
    
    df_ecrisp['gene_id'] = df_ecrisp['ENSG'].apply(lambda x: enst_ncbi[x])
    scores = ['S-Score', 'A-Score', 'E-Score']
    df_ecrisp['ecrisp_score'] = df_ecrisp[scores].sum(axis=1)

    symbols = (load_enst_ncbi(path=path).drop_duplicates(['gene_symbol', 'gene_id'])
                                 .set_index('gene_id')['gene_symbol'])
    df_ecrisp = df_ecrisp.join(symbols, on='gene_id')
    df_ecrisp['source'] = 'ECRISP'

    columns = ['sgRNA', 'gene_id', 'gene_symbol', 'source']
    return df_ecrisp[columns]


def load_benchling(path=''):
    df_benchling = pd.read_csv(path + 'benchling_sgRNAs.tsv', sep='\t')
    df_benchling['source'] = 'zBenchling'
    return df_benchling


def load_GFP_TM_sgRNAs(path=''):
    columns = {0: 'name', 1: 'sgRNA'}
    f = os.path.join(path, GFP_TM_SGRNAS)
    GFP_TM = (pd.read_csv(f, '\t', header=None)
                .rename(columns=columns))
    GFP_TM['source'] = 'GFP_TM'
    GFP_TM['tag'] = 'GFP_TM'
    columns = ['sgRNA', 'source', 'tag']
    return GFP_TM[columns]

def load_LG_TM_sgRNAs(path=''):
    columns = {0: 'name', 1: 'sgRNA'}
    f = os.path.join(path, LG_SGRNAS)
    LG_TM = (pd.read_csv(f, '\t', header=None)
                .rename(columns=columns))
    LG_TM['source'] = 'LG_TM'
    LG_TM['tag'] = 'LG_TM'
    columns = ['sgRNA', 'source', 'tag']

    filt = LG_TM['name'].str.contains('^LG_sg\d+_mut$')

    return LG_TM.loc[filt, columns]

def load_nontargeting_sgRNAs(path=''):
    controls = pd.read_csv(path + 'nontargeting_sgRNAs.tsv', header=None)
    controls.columns = 'sgRNA', 
    controls['source'] = 'CRISPOR'
    controls['tag'] = 'nontargeting'
    return controls


def contiguous(s):
    longest = 0
    current = 1
    last = ''
    for c in s:
        if c == last:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
        last = c
    return longest


def GC_penalty(s):
    """ in range [0, 1]
    """
    GC = sum(nt in 'GC' for nt in s) / float(len(s))
    penalty = abs((0.5 - GC)**3) / 0.5**3
    return penalty


def homopolymer_penalty(s, n=4):
    """ +1 for each base over n
    """
    homopolymer = contiguous(s)
    penalty = 0 if homopolymer <= n else (homopolymer - n)
    return penalty


def degenerate(s):
    """a generator would be nice
    """
    bases = {'N': list('ACTG')}
    arr = []
    for c in s:
        arr += [bases.get(c, [c])]
    return [''.join(x) for x in product(*arr)]


def score(barcode):
    a = GC_penalty(barcode)
    b = homopolymer_penalty(barcode, n=3)
    penalty = 2 * a + b
    return 1. / (1 + penalty)


def generate_barcodes():
    import rpy2.interactive as r
    import rpy2.interactive.packages
    DNABarcodes = r.packages.importr('DNABarcodes')
    barcodes = DNABarcodes.create_dnabarcodes(8, dist=2, heuristic='conway')
    barcodes = list(barcodes)

    pd.DataFrame(barcodes, columns=['dist2']).to_csv('DNABarcodes_8bp_2dist_conway.csv', index=None)


def load_8bp_barcodes():
    dist2 = pd.read_csv('DNABarcodes_8bp_2dist_conway.csv')['dist2']

    df_barcodes = pd.DataFrame(degenerate('NNNNNNNN'), columns=['8bp'])
    df_barcodes['score'] = map(score, df_barcodes['8bp'])
    df_barcodes['dist2'] = df_barcodes['8bp'].isin(dist2)

    df_barcodes['5bp'] = df_barcodes['8bp'].apply(lambda x: x[:5])
    subset_5bp = df_barcodes[df_barcodes['dist2']].groupby('5bp').head(1)
    df_barcodes['subset_5bp'] = df_barcodes['8bp'].isin(subset_5bp['8bp'])
    return df_barcodes


def prepare_barcodes():
    """Top level
    """
    df_barcodes = load_8bp_barcodes()
    full = degenerate_and_fix()
    df_barcodes['12bp'] = [full[x] for x in df_barcodes['8bp']]
    df_barcodes['12bp_layout'] = df_barcodes['12bp'].apply(layout_12bp_barcode)
    df_barcodes = filter_barcodes_pL42(df_barcodes)
    df_barcodes = color(df_barcodes)

    return df_barcodes


def filter_barcodes_pL42(df_barcodes):
    layout = 'sticky_Pd42_5', 'barcode', 'sticky_Pd42_3'
    f = lambda x: build_oligo(layout, override_parts={'barcode': x})

    filt = ~df_barcodes['12bp_layout'].apply(f).apply(contains_typeIIS)
    return df_barcodes[filt]


def filter_sgRNAs(df_sgRNAs):
    """Remove typeIIS
    """
    layout = 'sticky_U6', 'sgRNA', 'sticky_scaffold'
    f = lambda x: build_oligo(layout, override_parts={'sgRNA': x})

    filt = ~df_sgRNAs['sgRNA'].apply(f).apply(contains_typeIIS)
    return df_sgRNAs[filt]


def color(df_barcodes, red=196, orange=596, green=608, seed=0):
    """Green 5bp not in red. Orange 5bp not in red. 
    """
    df_barcodes = df_barcodes.copy().set_index('8bp')
    df_barcodes['red'] = False
    df_barcodes['orange'] = False
    df_barcodes['green'] = False
    
    red_barcodes = (df_barcodes.query('subset_5bp')
                               .sample(red, random_state=seed))
    df_barcodes.loc[red_barcodes.index, 'red'] = True

    green_barcodes = (df_barcodes.query('subset_5bp')
                                 .query('~red')
                                 .sample(green, random_state=seed + 1))
    df_barcodes.loc[green_barcodes.index, 'green'] = True
    
    # red 5bp sequences are excluded from orange and green
    filt = ~df_barcodes['5bp'].isin(red_barcodes['5bp'])
    # assert False
    orange_barcodes = (df_barcodes[filt]
                                  .query('~green & dist2')
                                  .drop_duplicates('5bp')
                                  .sample(orange, random_state=seed + 2))
    df_barcodes.loc[orange_barcodes.index, 'orange'] = True
    
    return df_barcodes.reset_index()


def contains_typeIIS(s):
    s = s.upper()

    a, b, c = lasagna_enzymes['BsmBI'], lasagna_enzymes['BsaI'], lasagna_enzymes['BbsI']
    sites = a, rc(a), b, rc(b), c, rc(c)
    sites = a, rc(a), c, rc(c)

    return any(x in s for x in sites)


def count_typeIIS_sites(s):
    s = s.upper()

    a, b, c = lasagna_enzymes['BsmBI'], lasagna_enzymes['BsaI'], lasagna_enzymes['BbsI']
    sites = a, rc(a), b, rc(b)
    sites = a, rc(a), c, rc(c)

    return sum(s.count(x) for x in sites)


def neighbors(s):
    arr = []
    for i, c in enumerate(s):
        for nt in 'ACTG':
            if nt == c:
                continue
            arr.append(s[:i] + nt + s[i+1:])
    return arr


def layout_12bp_barcode(s):
    # order for cutting
    return s[:4] + s[8:12] + s[4:8]
    

def degenerate_and_fix():
    fillers = degenerate('N'*4)
    full = {}
    for s in degenerate('N'*8):
        filler = np.random.choice(fillers)
        full[s] = s + filler
    bad = full.keys()
    
    while bad:
        print 'searching', len(bad)
        new_bad = []
        for i, s in enumerate(bad):
            arr = [full[n] for n in neighbors(s)]
            for a,b in combinations(arr, 2):
                if distance(a,b) < 2:
                    new_bad.append(a[:-4])
                    new_bad.append(b[:-4])
            if contains_typeIIS(layout_12bp_barcode(full[s])):
                new_bad.append(s)
        bad = sorted(set(new_bad))
        
        print 'replacing', len(bad)
        fillers = degenerate('N'*4)
        for s in bad:
            filler = np.random.choice(fillers)
            full[s] = s + filler
            
    return full


def build_oligo(layout, override_parts=None):
    sequence = ''
    parts = default_parts.copy()
    if override_parts:
        parts.update(override_parts)
    for i, name in enumerate(layout):
        if i % 2:
            sequence += parts[name].upper()
        else:
            sequence += parts[name].lower()
    return sequence
    

def degenerate_substrings(s):
    """Replace 
    """
    pat = '([nN]+|[^nN]+)'
    output = ''
    for substring in re.findall(pat, s):
        if substring.upper().startswith('N'):
            # degenerate
            n = len(substring)
            if n == 1:
                output += np.random.choice(list('ACTG'))
            else:
                output += generate_sequence(0.5, n)
        else:
            output += substring

    return output


def complete_oligo(s):
    """degenerate without creating new typeIIS sites
    """
    sites = count_typeIIS_sites(s)
    for _ in range(100):
        s_ = degenerate_substrings(s)
        if count_typeIIS_sites(s_) == sites:
            return s_
    raise ValueError('no valid substitutions in %s' % s)


def pair_sgRNAs_barcodes(df_sgRNAs, df_barcodes, df_layout):
    df_barcodes = df_barcodes.sort_values('score', ascending=False)
    tagged_barcodes = {'red':    iter(df_barcodes.query('red')['12bp_layout'])
                      ,'green':  iter(df_barcodes.query('green')['12bp_layout'])
                      ,'orange': iter(df_barcodes.query('orange')['12bp_layout'])
                      ,'white':  iter(df_barcodes.query('~red & ~green & ~orange')['12bp_layout'])
                      ,'gecko': cycle([None])
                      }

    def brunello_3X(df_sgRNAs):
        filt = df_sgRNAs['source'].isin(['brunello', 'Wang 2015', 'ECRISP', 'zBenchling'])
        return df_sgRNAs[filt].sort_values('source').groupby('gene_id').head(3)


    def endo_go_ranked_sgRNAs(df_sgRNAs, n=3):
        return (df_sgRNAs.query('endo_go_count > 0')
                         .sort_values(['endo_go_count', 'source'], ascending=False)
                         .groupby('gene_id').head(n))

    def patched_Cheeseman(df_sgRNAs):
        cheeseman_ids = load_cheese_sgRNAs(path=path)['gene_id']

        filt = df_sgRNAs['tag'].isin(['Cheeseman', 'brunello'])
        filt &= df_sgRNAs['gene_id'].isin(cheeseman_ids)
        return (df_sgRNAs[filt].sort_values('tag', ascending=True)
                               .groupby('gene_id')
                               .head(2)
                               .sort_values('gene_id'))
         

    design_names = {
      'Cheeseman':             lambda x: patched_Cheeseman(x)
    , 'GFP TM':      lambda x: x.query('tag=="GFP_TM"')
    , 'endocytosis GO top 50': lambda x: endo_go_ranked_sgRNAs(x, n=3)[:150]
    , 'endocytosis GO all':    lambda x: endo_go_ranked_sgRNAs(x)
    , 'nontargeting controls': lambda x: x.query('tag=="nontargeting"')
    , 'brunello 3X':           lambda x: brunello_3X(x)
    }

    # multiple subpools draw from this
    brunello_3X_iter = iter(zip(brunello_3X(df_sgRNAs)['sgRNA']
                               ,brunello_3X(df_sgRNAs)['gene_id']))

    withdrawn = Counter()
    oligos = []
    it = zip(df_layout.iterrows(), enumerate(dialout_primers))
    for (_, row), (dialout_ix, (dialout_5, dialout_3)) in it:
        tag = row['tag']
        design = row['sgRNA design']
        layout = row['barcode design']
        barcode_coverage = int(row['barcodes/sgRNA'])
        spot_coverage = int(row['spots/oligo'])
        num_sgRNAs = int(row['# of sgRNAs'])

        if design.startswith('brunello 3X_'):
            df_sgRNAs_subset = pd.DataFrame([brunello_3X_iter.next() for _ in range(num_sgRNAs)])
            df_sgRNAs_subset.columns = 'sgRNA', 'gene_id'
        else:
            df_sgRNAs_subset = design_names[design](df_sgRNAs)

        print '-'*10, '%s | %s | dialout %d' % (layout, design, dialout_ix), '-'*10 
        print 'layout uses %d / %d available sgRNAs' % (num_sgRNAs, len(df_sgRNAs_subset))
        print
        
        it = zip(df_sgRNAs_subset['sgRNA'], df_sgRNAs_subset['gene_id'])
        for sgRNA, gene_id in it[:num_sgRNAs]:
            for _ in range(barcode_coverage):
                barcode = tagged_barcodes[tag].next()
                if barcode:
                    withdrawn[tag] += 1
                parts = {'barcode': barcode
                        ,'sgRNA':   sgRNA
                        ,'dialout_5': dialout_5
                        ,'dialout_3_rc': rc(dialout_3)
                        }
                
                oligo = build_oligo(default_layouts[layout], override_parts=parts)
                oligo = complete_oligo(oligo)
                [oligos.append([gene_id, sgRNA, barcode, dialout_ix, tag, design, layout, oligo]) for _ in range(spot_coverage)]

    columns = 'gene_id', 'sgRNA', 'barcode', 'dialout', 'tag', 'design', 'layout', 'oligo'
    oligos = pd.DataFrame(oligos, columns=columns)

    print 'used %d / %d available barcodes' % (sum(withdrawn.values()), len(df_barcodes))
    print withdrawn
    
    return oligos


def fixed_FWD_primers(n=24):
    bsaI_3_ = 'GGTCTCcCACCG'
    return [adapter + bsaI_3_ for adapter, _ in dialout_primers[:n]]

