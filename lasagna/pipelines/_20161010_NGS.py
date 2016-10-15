from glob import glob
import os, re
from collections import Counter
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(style='white', font_scale=1.5)


sgRNAs = {'sgV-101': 'GAGACCAGCAGAACCGACAA',
 'sgV-106': 'CGGGAGGGCAGGGCACGAAA',
 'sgV-107': 'GCGCAACAGAGAGGGGAGCG',
 'sgV-116': 'AGGGCACAGACCCCAGGGAG',
 'sgV-117': 'CAGGGGGGAGCGAAAGAGAA'}



def read_fastq(filepath):
    with open(filepath, 'r') as fh:
        return fh.read().split('\n')[1::4]

def make_sample_table(prefix=''):
    """Run from the experiment directory, containing sequence data
    in fastq/*.fastq
    """
    R1 = glob(os.path.join(prefix, 'fastq/*R1*'))
    R2 = glob(os.path.join(prefix, 'fastq/*R2*'))

    df = pd.DataFrame([R1, R2]).T
    df.columns = 'R1', 'R2'

    get_well = lambda f: re.findall('fastq/(...)', f)[0]
    df['well'] = [get_well(f) for f in df['R1']]
    df['reads'] = [len(read_fastq(f)) for f in df['R1']]
    df = df[['well', 'reads', 'R1', 'R2']]

    df['row'] = [s[0] for s in df['well']]
    df['col'] = [s[1:] for s in df['well']]

    if prefix:
        df['run'] = prefix

    return df


def match_patterns_samples(samples, patterns):
    """Matches regex patterns to samples given as {sample_name: fastq}.
    """
    arr = []
    for sample, fastq in samples.items():
        reads = read_fastq(fastq)
        df_ = match_patterns(reads, patterns)            
        df_['sample'] = [sample]*len(df_)
        arr += [df_]
    return pd.concat(arr)

def match_patterns(reads, patterns):
    arr = []
    for pattern_name, pattern in patterns.items():        
        matches = [re.findall(pattern, read) for read in reads]
        matches = [m[0] for m in matches if m]
        counted = Counter(matches)
        for match, count in counted.items():
            arr += [[pattern_name, pattern, match, count]]
        columns = 'pattern_name', 'pattern', 'match', 'count'
    return pd.DataFrame(arr, columns=columns)


def jaccard(a,b):
    a = a['match']
    b = b['match']
    return len(set(a) & set(b)) / float(len(set(a) | set(b)))


def calculate_pairwise(df, groupby='sgV', func=jaccard):
    """Group samples by 
    """
    samples = np.unique(df[groupby])
    arr = []
    for i, a in enumerate(samples):
        for b in samples[i:]:
            matches_a = df[df[groupby] == a]
            matches_b = df[df[groupby] == b]
            score = func(matches_a, matches_b)
            arr += [[a, b, score]]
            if a != b:
                arr += [[b, a, score]]
    columns = groupby + '0', groupby + '1', 'score'
    df = pd.DataFrame(arr, columns=columns)
    df = df.pivot(index=columns[0], columns=columns[1], values='score')
    return df


def add_features(df, add_well_info_from_sample=True):
    """ Additional columns for output of match_patterns().
    Expects a column of 
    """
    df = df.copy()
    if add_well_info_from_sample:
        df['run'] = [s[0] for s in df['sample']]
        df['well'] = [s[1] for s in df['sample']]
        df['row'] = [s[0] for s in df['well']]
        df['col'] = [int(s[1:]) for s in df['well']]

    df['length'] = [len(s) for s in df['match']]
    df['frame'] = (df['length'] + 2) % 3
    df['kozak'] = ['GCCACCATG' in s for s in df['match']]
    ORFs = [re.findall('(?=ATG(.*))', m) for m in df['match']]
    df['ATG_0'] = [any((len(x)%3 == 0) for x in hit) for hit in ORFs]
    df['ATG_1'] = [any((len(x)%3 == 1) for x in hit) for hit in ORFs]
    df['ATG_2'] = [any((len(x)%3 == 2) for x in hit) for hit in ORFs]

    # safe for duplicate index
    def f(x):
        x = x.copy()
        x['mapped'] = x['count'].sum()
        return x

    df = df.groupby(['pattern_name', 'sample']).apply(f)
    df['fraction'] = df['count'] / df['mapped']

    return df.reset_index(drop=True)


def mark_controls(df, target_info):
    df = df.copy()
    # should there be signal?
    A = (df['dox'] == 'no dox')
    B = (df['sgRNA'] != target_info.loc[df['target']]['sgRNA'])
    C = df['sgRNA'].apply(lambda s: 'K' in s)

    df['control'] = A | (B & ~C)
    df['library'] = ['K' in s for s in df['sgRNA']]

    return df


def load_conditions(sheet, g_file='Lasagna NGS'):
    """ Load conditions from Lasagna NGS.
    """
    from lasagna.conditions_ import Experiment
    exp = Experiment(sheet, g_file=g_file)
    ivt = exp.ind_vars_table
    ivt['well'] = [s[0] + '0' + s[1] if len(s)==2 else s for s in ivt.index]
    ivt['run'] = sheet
    return ivt


def display_ind_var(df, ind_var, run=None, fix_rows_cols=True):
    """Check the well layout of an independent variable.
    If the dataframe contains multiple runs, specify which to display.
    fix_rows_cols fills in empty rows/columns with None
    """
    if run:
        df = df[df['run'] == run].copy()
    if run is None and 'run' in df.columns and len(np.unique(df['run'])) > 1:
        raise ValueError('which run do you want? I see %s' % np.unique(df['run']))
    df2 = df.drop_duplicates(['run','well', 'sgRNA']).set_index(['run', 'well'])
    df2 = df2.pivot_table(index='row', columns='col', values=ind_var, aggfunc=lambda x: x[0])

    if fix_rows_cols:
        for i in range(min(df2.columns), max(df2.columns)):
            if i not in df2.columns:
                df2.loc[:,i] = None
        df2 = df2.loc[:, sorted(df2.columns)]

        rows = [ord(c) for c in df2.index]
        for i in range(min(rows), max(rows)):
            if i not in rows:
                df2.loc[i] = None

        df2 = df2.sort_index()

    return df2


def load_target_info():
    from lasagna.conditions_ import load_sheet
    x = load_sheet('targets', g_file='inventory for a time machine')
    return pd.DataFrame(x[1:], columns=x[0]).set_index('target')


def msa(template, sequences, match=2, mismatch=-1, open=-1, extend=-0.5):
    """Fake multiple sequence alignment to template. Do pairwise alignment
    to determine gap locations, then repeat alignments. For the second pass, 
    don't allow gaps in the template.
    """
    from Bio import pairwise2
    template = str(template)
    for s in sequences:
        results = pairwise2.align.localms(template.upper(), 
                                          s, match, mismatch, open, extend)
        template = results[0][0]
    aligned = []
    for s in sequences:
        results = pairwise2.align.localmd(template.upper(), 
                                          s, match, mismatch, -100, -100, open, extend)
        aligned += [results[0][1]]
    return [template] + aligned


def format_msa(template, aligned):
    def color(c, shade=None):
        return '<span style="color: %s">%s</span>' % (shade, c)
    
    arr = []
    for s in aligned:
        span = ''
        for c0, c1 in zip(template, s):
            if c1 != c0 and c1 != '-':
                span += color(c1, 'red')
            else:
                span += color(c1)
        arr += ['%s' % span]
    
    return '<div style="font-family: Consolas">%s<br>%s</div>' % (template, '<br>'.join(arr))


def display_table(table, cols=('formatted_alignment', 'sgRNA')):
    from IPython.display import HTML
    cols = list(cols)
    with pd.option_context('display.max_colwidth', 100000):
        html = table[cols].to_html(escape=False,
            float_format=lambda x: "%.3f%%" % (100*x))
    return HTML(html)


def display_target_sgRNAs(target, df, target_info):

    def collapse_sgRNAs(df, n=3):
        """Take the top 3 sgRNAs for 
        """
        a = df.query('pattern_name=="sgRNA"').groupby(['sgRNA', 'match'])['count'].sum().reset_index()
        a = a.sort_values(['sgRNA', 'count'], ascending=[True, False]).groupby('sgRNA').head(n)
        return a.set_index('sgRNA')

    collapsed_sgRNAs = collapse_sgRNAs(df)
    
    sgRNA = target_info.loc[target, 'sgRNA']
    t = target_info.loc[target, 'FWD_seq']
    s = target_info.loc[target, 'sgRNA_seq']

    sequences, counts = collapsed_sgRNAs.loc[sgRNA].as_matrix().T
    aligned = msa(t, sequences)
    template, aligned = aligned[0], aligned[1:]
    
    arr = []
    for a,s,c in zip(aligned, sequences, counts):
        arr += [[target, sgRNA, template, a, s, c]]
    
    df2 = pd.DataFrame(arr)
    df2.columns = 'target', 'sgRNA', 'template', 'aligned', 'sgRNA_sequence', 'count'
    
    formatted = format_msa(template, aligned)    
    formatted = formatted.replace('ATG', '<span style="color: green">ATG</span>')
    formatted = formatted.replace('AGGCCT', '<span style="color: blue">AGG</span>CCT')
    
    df2['formatted_alignment'] = formatted
    
    return df2

def count_sgRNAs_d9(df):
    """Used for normalizing 
    """
    x = df.query('day=="d9"&pattern_name=="sgRNA"&dox=="dox"&row!="A"')
    return x.pivot_table(values='fraction', columns='sgRNA', index='match')



oddballs = {
    '101.fcs' : 'E01',
    '102.fcs' : 'E02',
    '106.fcs' : 'E03',
    '107.fcs' : 'E04',
    '116.fcs' : 'E05',
    '117.fcs' : 'E06',

    '01-96W-101.fcs' : 'E01',
    '01-96W-102.fcs' : 'E02',
    '01-96W-106.fcs' : 'E03',
    '01-96W-107.fcs' : 'E04',
    '01-96W-116.fcs' : 'E05',
    '01-96W-117.fcs' : 'E06',

    'sgV101.fcs' : 'E01',
    'sgV102.fcs' : 'E02',
    'sgV106.fcs' : 'E03',
    'sgV107.fcs' : 'E04',
    'sgV116.fcs' : 'E05',
    'sgV117.fcs' : 'E06',

    'A1-.fcs' : 'C01',
    'A2-.fcs' : 'C02',
    'A3-.fcs' : 'C03',
    'A4-.fcs' : 'C04',
    'A5-.fcs' : 'C05',
    'A6-.fcs' : 'C06',
    'B1-.fcs' : 'D01',
    'B2-.fcs' : 'D02',
    'B3-.fcs' : 'D03',
    'B4-.fcs' : 'D04',
    'B5-.fcs' : 'D05',
    'B6-.fcs' : 'D06',
}
oddballs_d7 = {

    '01-96W-A1.fcs' : 'C01',
    '01-96W-A2.fcs' : 'C02',
    '01-96W-A3.fcs' : 'C03',
    '01-96W-A4.fcs' : 'C04',
    '01-96W-A5.fcs' : 'C05',
    '01-96W-A6.fcs' : 'C06',
    '01-96W-B1.fcs' : 'D01',
    '01-96W-B2.fcs' : 'D02',
    '01-96W-B3.fcs' : 'D03',
    '01-96W-B4.fcs' : 'D04',
    '01-96W-B5.fcs' : 'D05',
    '01-96W-B6.fcs' : 'D06',
   }


def load_fcs_stats(csv):
    columns = {
    'cells | Count': 'cells',
    'cells/Single Cells | Count': 'single cells',
    'Cells_2 | Count': 'cells',
    'Cells_2/Single Cells | Count': 'single cells',
    'cells/Single Cells/GFP+_d3_cytoflex | Count': 'GFP+',
    'cells/Single Cells/GFP+_d5_cytoflex | Count': 'GFP+',
    'cells/Single Cells/GFP+_d7_cytoflex | Count': 'GFP+',
    'cells/Single Cells/GFP+_d9_cytoflex | Count': 'GFP+',
    'Cells_2/Single Cells/GFP+ | Count': 'GFP+',
    'cells/Single Cells/MC+ | Count': 'mCherry+',
    'cells/Single Cells/MC+ | Count': 'mCherry+',
    'Cells_2/Single Cells/MC+ | Count': 'mCherry+',
    'Unnamed: 0': 'fcs'}
    
    df = pd.read_csv(csv)
    columns = {k: columns[k] for k in columns if k in df.columns}
    df = df.rename(columns=columns).loc[:,columns.values()]
    df = df[~df['fcs'].isin(['Mean', 'SD'])]

    wells = []
    for f in df['fcs']:
        sony_names = '3K', '1K', 'Data Source'
        if any(n in f for n in sony_names):
            wells += ['tube']
            continue
        if f in oddballs:
            wells += [oddballs[f]]
            continue

        if 'd7' in csv and f in oddballs_d7:
            wells += [oddballs_d7[f]]
            continue

        well_pattern = '([A-H][0-1][1-9])|([A-H][1-9])'
        matches = re.findall(well_pattern, f)
        if matches:
            fmt0, fmt1 = matches[0]
            if fmt1:
                well = fmt1[0] + '%02d' % int(fmt1[1:])
            else:
                well = fmt0
        else:
            raise ValueError('unrecognized well in %s' % f)

        wells += [well]

    df['well'] = wells

    df['row']  = [w[0] if w is not 'tube' else None for w in df['well']]
    df['col']  = [int(w[1:]) if w is not 'tube' else None for w in df['well']]

    df['negative'] = (df['single cells'] 
                      - df['mCherry+'] 
                      - df['GFP+'])

    df['% negative'] = (df['negative'] / df['single cells']) * 100
    df['% mCherry+'] = (df['mCherry+'] / df['single cells']) * 100
    df['% GFP+']     = (df['GFP+'] / df['single cells']) * 100

    return df


def load_flowstats():
    arr = []
    for csv in glob('FlowStats/*.csv'):
        run = csv.split('/')[1]
        day = run[:2]
        df = load_fcs_stats(csv)
        df['run'] = run
        df['day'] = day

        df['instrument'] = None
        if 'cytoflex' in csv.lower():
            df['instrument'] = 'cytoflex'
        if 'sony' in csv.lower():
            df['instrument'] = 'sony'

        arr += [df]
    return pd.concat(arr)






def load_target_info():
    from lasagna.conditions_ import load_sheet
    x = load_sheet('targets', g_file='inventory for a time machine')
    df = pd.DataFrame(x[1:], columns=x[0])
    return df.set_index('target') 



def add_closest_sgRNA(df, sgRNAs):

    def closest_sgRNAs(sequence, sgRNAs):
        from Levenshtein import distance
        return sorted([[distance(sequence, s), name] for name, s in sgRNAs.items()])

    df = df.copy()
    
    df['closest_sgRNA'] = 'other'
    df['closest_sgRNA_dist'] = -1
    
    filt = df['pattern_name'] == 'sgRNA'
    closest = [closest_sgRNAs(s, sgRNAs) for s in df.loc[filt, 'match']]
    
    df.loc[filt, 'closest_sgRNA'] = [n[0][1] for n in closest]
    df.loc[filt, 'closest_sgRNA_dist'] = [n[0][0] for n in closest]
    
    return df
