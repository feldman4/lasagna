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



def make_sample_table():
    """Run from the experiment directory, containing sequence data
    in fastq/*.fastq
    """
    R1 = glob('fastq/*R1*')
    R2 = glob('fastq/*R2*')

    df = pd.DataFrame([R1, R2]).T
    df.columns = 'R1', 'R2'

    get_sample = lambda f: re.findall('fastq/(...)', f)[0]
    df['sample'] = [get_sample(f) for f in df['R1']]
    df['reads'] = [len(read_fastq(f)) for f in df['R1']]
    df = df[['sample', 'reads', 'R1', 'R2']]
    return df


def match_patterns_samples(samples, patterns):
    """Matches regex patterns to samples given as {sample_name: fastq}.
    """
    arr = []
    for sample, fastq in samples.items():
        reads = read_fastq(fastq)
        df_ = match_patterns(reads, patterns)            
        df_['sample'] = sample
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
