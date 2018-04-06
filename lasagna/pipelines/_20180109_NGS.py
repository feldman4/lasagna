import pandas as pd
import re
from lasagna.pipelines._20171015_NGS import *


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
            df['pattern'] = get_pattern(f)
            df['file']    = f
            arr += [df]
        except (pd.errors.EmptyDataError):
            print('error reading', f)
            continue
    df = pd.concat(arr)

    return df.join(df_samples, on='well')
