import gspread
import json
import regex as re
import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict, OrderedDict
import urllib2
from lasagna.design import rc

import lasagna.models
import lasagna.config


ROW_INDICATOR = 'A'

ROWS = 'ABCDEFGH'
COLS = [str(x) for x in range(1,17)]

doc_ids = {'Lasagna Supreme' : '1Q-4z1NwvncMjhTencZYdua-_E2IUdZzyo4cm12XdZ38',
           'Lasagna Oligos'  : '16Pn5RB0nOj_an_3ad0YUgTmWklQXNvINncs93r2rG9I',
           'inventory for a time machine': '1Ov5aPW-J8s_K_tGQ3kc9zbx17c7HbgfOPo-Jt86oDzc'}

google_csv_url = 'https://docs.google.com/spreadsheets/d/%s/gviz/tq?tqx=out:csv&sheet=%s'

default_translation_table = ''.join([chr(x) for x in range(256)])

default_dyes = 'Cy3', 'Cy5', 'A647', 'Tex615', 'AlexF488', 'AlexF594'


def format_output(columns=None):
    def f(series):
        output = [list(s) for s in series]
        return pd.DataFrame(output, index=series.index, columns=columns)
    return f


def apply_df(f, formatter=format_output(), **kwargs):
        
    def call(series):
        return f(**series.to_dict())

    listargs = []
    keys = sorted(kwargs.keys())
    for k in keys:
        v = kwargs[k]
        if not isinstance(v, list):
            v = [v]
        listargs.append(v)
    
    listargs = list(product(*listargs))
    arguments = pd.DataFrame(listargs, columns=keys)
    output = arguments.apply(call, axis=1)
    output = formatter(output)
    return pd.concat([arguments, output], axis=1)


def oligo_to_nucleotides(s):
    table = default_translation_table
    table = table.replace('U', 'T').upper()
    return s.translate(table, 'm+')


def mod_to_dye(m):
    for dye in default_dyes:
        if dye.lower() in mod.lower():
            return dye
    return 'dye not in' + dyes


def load_google_csv(sheet='Lasagna Oligos', worksheet='in situ'):
    url = google_csv_url % (urllib2.quote(doc_ids[sheet]), urllib2.quote(worksheet))
    response = urllib2.urlopen(url)
    html = response.read()
    return html
    return pd.read_csv(url)


def load_in_situ_sequences():
    # load oligos
    # kinds = 'DO', 'Padlock', 'LNA primer'
    rename = {'Name': 'name', 'Sequence': 'sequence', 'Kind of oligo': 'kind'}
    cols_to_keep = 'name', 'sequence', 'kind', "5' mod.", "3' mod."

    df_oligos = load_google_csv(sheet='Lasagna Oligos', worksheet='in situ')
    df_oligos = df_oligos.rename(columns=rename)
    df_oligos['sequence'] = df_oligos['sequence'].apply(oligo_to_nucleotides)
    # filt = df_oligos['kind'].isin(kinds)
    df_oligos = df_oligos.loc[:, cols_to_keep].dropna(how='all').fillna('')
    
    # load barcoded plasmids
    cols_to_keep = 'name', 'sequence'
    rename = {'oligo FWD': 'sequence'}
    df_barcodes = load_google_csv(sheet='Lasagna Supreme', worksheet='pL43_pL44')
    df_barcodes = df_barcodes.rename(columns=rename)
    df_barcodes = df_barcodes.loc[:, cols_to_keep].dropna(how='all').fillna('')
    df_barcodes['sequence'] = df_barcodes['sequence'].apply(oligo_to_nucleotides)

    # load misc. padlocks
    df_padlocks_misc = load_google_csv(sheet='Lasagna Oligos', worksheet='misc. padlocks')
    df_padlocks_misc = df_padlocks_misc.loc[:, cols_to_keep].dropna(how='all').fillna('')
    df_padlocks_misc['sequence'] = df_padlocks_misc['sequence'].apply(oligo_to_nucleotides)

    df_seqs = pd.concat([df_oligos, df_barcodes, df_padlocks_misc]).set_index('name')
    df_seqs = df_seqs[df_seqs.columns.sort_values(ascending=False)].fillna('')
    return df_seqs


def gapfill_ligate(padlock, transcript, binding_threshold=10):
    """Returns gapfill (may be the empty string) if padlock matches
    transcript, None otherwise. Padlock and transcript are both sense.
    """
    for i in range(1, len(padlock)):
        if padlock[-i:] not in transcript:
            i-=1
            break
    ix = transcript.find(padlock[-i:])
    for j in range(1, len(padlock)):
        if padlock[:j] not in transcript:
            j-=1
            break
    jx = transcript.find(padlock[:j])
    fillin = transcript[ix+i:jx]

    if (i>binding_threshold) & (j>binding_threshold):
        return fillin
    else:
        return None


def show_signal(df_seqs, 
                transcript_name,
                padlock_name,
                anchor_primer_name,
                labeled_oligo_names,
                verbose=False):

    

    if isinstance(labeled_oligo_names, str):
        labeled_oligo_names = [labeled_oligo_names]

    dyes = set()
    for labeled_oligo_name in labeled_oligo_names:

        # get sequences
        mod_names = ["5' mod.", "3' mod."]
        transcript    = df_seqs.loc[transcript_name, 'sequence']
        padlock       = df_seqs.loc[padlock_name, 'sequence']
        anchor_primer = df_seqs.loc[anchor_primer_name, 'sequence']
        anchor_mods   = df_seqs.loc[anchor_primer_name, mod_names]
        labeled_oligo = df_seqs.loc[labeled_oligo_name, 'sequence']
        labeled_mods  = df_seqs.loc[labeled_oligo_name, mod_names]

        gapfill, dye = show_signal_(transcript, 
                padlock, 
                anchor_primer,
                anchor_mods,
                labeled_oligo,
                labeled_mods,
                verbose=verbose)

        dyes |= set(dye)

    return gapfill, sorted(dyes)


def show_signal_(transcript, 
                padlock, 
                anchor_primer,
                anchor_mods,
                labeled_oligo,
                labeled_mods,
                verbose=False):
    """
    Return None if there is no binding.
    """
    def vprint(s):
        if verbose:
            print s

    dye = [m for m in labeled_mods if m and 'Phos' not in m]
    
    # do the gapfill + ligation
    gapfill = gapfill_ligate(padlock, transcript)
    if gapfill is None:
        vprint('no detection')
        return None, ''
    vprint('gapfill added %d bases: %s' % (len(gapfill), gapfill))
    
    # circularize and amplify
    # RCA product is antisense
    RCA = padlock + gapfill + padlock[:-17]
    
    # hybridize anchor primer, ligate
    # annealing combinators would be nice
    ix = RCA.find(anchor_primer)
    if labeled_oligo in RCA:
        vprint('detection oligo bound %s' % labeled_oligo)
        return gapfill, dye
    if ix == -1:
        vprint('anchor primer did not bind')
        return gapfill, []
    
    # sequencing from anchor primer
    # adapter has 5' phosphate
    pattern = labeled_oligo.replace('N', '.')
    if 'phos' in labeled_mods[0].lower():
        if re.findall(anchor_primer + pattern, RCA):
            return gapfill, dye
    # anchor primer has 5' phosphate
    if 'phos' in anchor_mods[0].lower():
        if re.findall(pattern + anchor_primer, RCA):
            return gapfill, dye

    return gapfill, []


def get_adapters(df_seqs, base):
    """
    f = partial(get_adapters, df_seqs)
    f('5B1')
    """
    filt = df_seqs['kind']==base
    return list(df_seqs[filt].index)


def flatten_layout_row_col(x):
    x = pd.DataFrame(x.stack())
    index = [a + str(b) for a, b in x.index.values]
    x.index = pd.Index(index, name='sample')
    return x


def load_sheet(worksheet, g_file='Lasagna FISH'):
    """Load sheet as array of strings (drops .xls style index)
    gspread allows for .xlsx export as well, which can capture style info.
    :param worksheet: provide None to return a dictionary of all sheets
    :param g_file:
    :return:
    """
    # see http://gspread.readthedocs.org/en/latest/oauth2.html

    from oauth2client.service_account import ServiceAccountCredentials


    scope = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(lasagna.config.credentials, scope)

    gc = gspread.authorize(credentials)
    xsheet = gc.open(g_file)

    if type(worksheet) is int:
        wks = xsheet.get_worksheet(worksheet)
    if worksheet is None:
        return {x.title: np.array(x.get_all_values()) for x in xsheet.worksheets()}
    else:
        wks = xsheet.worksheet(worksheet)
    xs_values = np.array(wks.get_all_values())
    return xs_values


class Experiment(object):
    def __init__(self, worksheet=None, g_file='Lasagna FISH', ind_var_transforms=()):
        """Represent independent variables and their values for each sample.
        Load spreadsheet, extract independent variable dict and sample layout.
        Optionally update independent variable dict (layout notation may not match ind. var.
        list).
        Construct DataFrame with samples as index, independent variables as columns.
        Optionally restructure columns to indicate nested independent variables (e.g., probes
        in each round).
        Optionally pivot into binary DataFrame which can be expanded and reshaped into ndarray.
        Optionally extract first and second order comparisons from either DataFrame (edit distance)
        or ndarray (geometric, written earlier).

        To construct manually:

        exp = Experiment()
        exp.sheet = load_sheet(worksheet)
        exp.parse_grids()
        exp.parse_ind_vars()
        # modify exp.ind_vars if keys don't match layout notation
        exp.make_ind_vars_table()

        :return:
        """

        self.grids = {}
        self.ind_vars = {}
        self.sheet = None
        self.ind_vars_table = None
        self.flatten_layout = flatten_layout_row_col

        if worksheet:
            self.load_and_go(worksheet, g_file=g_file, ind_var_transforms=ind_var_transforms)

    def load_and_go(self, worksheet, g_file='Lasagna FISH', ind_var_transforms=()):
        """Shortcut to load sheet, parse sheet, and make independent variable table when
        notation is sheet is sufficient. If provided, transforms are used to update the 
        independent variable dictionary.
        :param worksheet:
        :param g_file:
        :param ind_var_transforms:
        :return:
        """
        self.sheet = load_sheet(worksheet, g_file=g_file)
        self.parse_grids()
        self.parse_ind_vars()
        [t(self.ind_vars) for t in ind_var_transforms]
            
        return self.make_ind_vars_table()

    def parse_grids(self, title_offset=(-1, 0), A_offset=(1, 0)):
        """Look for first row labelled by ROW_INDICATOR, find origin and return boolean mask to values.
        :param numpy.ndarray xs_values:
        :param tuple grid_size:
        :param tuple title_offset:
        :param tuple A_offset:
        :return: (dict, wells): ({title: mask}, {well_name:
        :rtype : dict[str, numpy.nparray]
        """
        # some weird bug with np.pad and string dtype
        xs_values = np.zeros(np.array(self.sheet.shape) + 1, dtype='S123')
        xs_values[:-1, :-1] = self.sheet

        grids = OrderedDict()
        for candidate in zip(*np.where(xs_values == ROW_INDICATOR)):
            origin = np.array(candidate) - A_offset

            columns = xs_values[origin[0], origin[1]+1:]
            columns = columns[:list(columns).index('')]

            rows = xs_values[origin[0] + 1:, origin[1]]
            rows = rows[:list(rows).index('')]

            shape = len(rows), len(columns)

            values = xs_values[origin[0] + 1: origin[0] + 1 + shape[0],
                               origin[1] + 1: origin[1] + 1 + shape[1]]

            title = xs_values[tuple(origin + title_offset)]
            grids[title] = pd.DataFrame(values, index=pd.Index(rows, name='row'),
                                        columns=pd.Index(columns, name='column'))

        self.grids.update(grids)
        return grids

    def parse_ind_vars(self):
        """Define values of independent variables by parsing first example of form:
        [var name]  [value 0]
                    [value 1]
                    ...
                    [value n]
                    [blank]
                *or*
        [text]
        :return:
        """
        selem = np.array([[1, 0],
                         [1, 1]])
        # some weird bug with np.pad and string dtype
        s_type = 'S%d' % (max([len(x) for y in self.sheet for x in y]) + 10)
        xs_values = np.zeros(np.array(self.sheet.shape) + 1, dtype=s_type)
        xs_values[:-1, :-1] = self.sheet

        mask = (xs_values[:, :2] != '').astype(int)
        mask[:, 1] *= 2
        mask_string = ''.join(['ABCD'[i] for i in mask.sum(axis=1)])

        ind_vars = {}
        for x in re.finditer('(DC+)[ABD]', mask_string):
            name = xs_values[x.span()[0], 0]
            values = xs_values[x.span()[0]:x.span()[1] - 1, 1]
            ind_vars[name] = list(values)

        self.ind_vars.update(ind_vars)

    def make_ind_vars_table(self):
        """Create table with independent variables as columns, samples as rows,
        and independent variable values as entries.
        :return:
        """

        def apply_try(y):
            def func(x):
                try:
                    x_ = float(x)
                    if x_ % 1:
                        return y[x_]
                    return y[int(x_)]
                except ValueError:
                    return x  # default
            return func

        arr = []
        for ind_var, grid in self.grids.items():
            values = self.ind_vars[ind_var]
            grid = self.flatten_layout(grid)
            grid.columns = [ind_var]
            # # set default value (no index in grid) based on type of first entry
            # try:
            #     default = type(values.values()[0])()
            # except AttributeError:
            #     default = type(values[0])()
            arr += [grid.applymap(apply_try(values))]

        table = pd.concat(arr, axis=1).fillna('')
        table = table[(table != '').any(axis=1)]
        self.ind_vars_table = table[sorted(table.columns)]


    def melt(self, model):
        # export i.v. table
        # exclude single characters ('M', 'b')
        table = self.ind_vars_table.filter(regex='..')
        # make nice the tuples
        def nice_tuple(z): return [', '.join(y).encode('ascii') for y in z]
        for column, series in self.ind_vars_table.iteritems():
            if any(isinstance(value, tuple) for value in series):
                table[column] = nice_tuple(table[column])

        results = []
        for well, row in self.ind_vars_table.iterrows():
            results += [model.evaluate(row['M'], row['b']).stack()]
        results = pd.concat(results, axis=1).transpose()
        results.index = self.ind_vars_table.index

        # preserve MultiIndex of results when combining
        if isinstance(results.columns, pd.MultiIndex):
            base = ('',) * (results.columns.nlevels - 1)
            table.columns = pd.MultiIndex.from_tuples([base + (c,) for c in table.columns])

        return pd.concat([table, results], axis=1)
        

def get_named_wells(wells, conditions):
    """Convert dict of wells with indexed conditions to dict with named conditions
    """
    return {k: [arr[x] for x, arr in zip(v, conditions.values())] for k, v in wells.items()}


def get_named_comparisons(comparisons, conditions):

    named_comparisons = []
    for coords in comparisons:
        arr = []
        for index, (variable, condition) in zip(coords, conditions.items()):
            arr += [{variable: [condition[i] for i in index]}]
        named_comparisons += [arr]

    return named_comparisons


def find_comparisons(cube):
    all_idx = zip(*np.unravel_index(range(cube.size),
                                cube.shape))

    faces = []
    for dim in range(cube.ndim):
        for idx in all_idx:
            tmp = list(idx)
            tmp[dim] = None
            faces.append(tuple(tmp))
    faces = np.vstack({tuple(row) for row in faces})

    comparisons = defaultdict(list)
    for face in faces:
        slic = tuple(slice(None) if x is None else x for x in face)
        tmp = (cube[slic]>0)
        comparisons[tmp.sum()] += [(face, np.where(tmp)[0])]

    # not useful
    comparisons.pop(0)
    comparisons.pop(1)
    out = []
    for n, arr in comparisons.items():
        for index, subs in arr:
            index2 = []
            for i in index:
                if i is not None:
                    index2.append((i,))
                else:
                    index2.append(tuple(subs))
            out.append(tuple(index2))

    return tuple(out)


def find_comparisons_second_order(cube):
    comparisons = find_comparisons(cube)
    n = len(comparisons)
    comparisons2 = []
    for i in range(n):
        for j in range(i+1, n):
            # make combined index, check if it's OK
            c = []
            for a, b in zip(comparisons[i], comparisons[j]):
                # exclude comparisons that fail these tests
                if len(a) > 1 and len(b) > 1:
                    break
                if len(a) == 1 and len(b) == 1 and a != b:
                    break
                # if one index has multiple values, be sure to keep it
                c += [a] if len(a) > 1 else [b]
            else:
                comparisons2.append(tuple(c))

    # clear out anything that doesn't have at least one well for every condition
    keep = []
    for index in comparisons2:
        ix = np.ix_(*index)
        test = cube[ix].flatten()
        if test.sum() == test.size:
            keep.append(index)

    return {k: 0 for k in keep}.keys()


def set_credentials(path):
    CREDENTIALS_JSON = path


def prepare_linear_model(experiment):
    """Create LinearModel, set probes used in experiment, and generate matrices.
    Requires probes as an independent variable.
    Requires at least one probes grid named "probes round 1" or similar.
    :return:
    """
    model = lasagna.models.LinearModel()
    # split comma-separated probes
    lasagna.config.set_linear_model_defaults(model)
    all_probes = experiment.ind_vars['probes']
    model.indices['l'] = list(set(sum([x.split(', ') for x in all_probes], [])))

    model.matrices_from_tables()

    ivt = experiment.ind_vars_table
    model.indices['j'] = [x for x in ivt.columns
                          if 'round' in x]

    # derive matrix input to LinearModel (M and b) from table
    M = {sample: pd.DataFrame([], index=model.indices['j'],
                              columns=model.indices['l']).fillna(0)
         for sample in ivt.index}
    b = {sample: pd.Series({x: 0 for x in model.indices['m']})
         for sample in ivt.index}

    for sample, row in ivt.iterrows():
        for rnd in model.indices['j']:
            M[sample].loc[rnd, list(ivt.loc[sample, rnd])] = 1
        # empty tuple would index whole pd.Series ...
        if not pd.isnull(row['barcodes']) and row['barcodes']:
            b[sample][row['barcodes']] = 1

    experiment.ind_vars_table['M'] = [M[x] for x in ivt.index]
    experiment.ind_vars_table['b'] = [b[x] for x in ivt.index]

    return model