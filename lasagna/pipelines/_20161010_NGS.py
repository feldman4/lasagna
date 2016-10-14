from glob import glob
import os, re
from collections import Counter
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(style='white', font_scale=1.5)


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
    df = df[['sample', 'reads', 'R1', 'R2']]

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
        df['col'] = [s[1:] for s in df['well']]

    df['length'] = [len(s) for s in df['match']]
    df['frame'] = df['length'] % 3
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
 


    return df.reset_index(drop=True)


def load_conditions(sheet):
    """ Load conditions from Lasagna NGS.
    """
    from lasagna.conditions_ import Experiment
    exp = Experiment(sheet, g_file='Lasagna NGS')
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
    x = load_sheet('targets', g_file='inventory for a time machine')
    return pd.DataFrame(x[1:], columns=x[0])