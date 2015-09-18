import json
from itertools import product
from collections import defaultdict, OrderedDict
import regex as re

import gspread
import numpy as np
from oauth2client.client import SignedJwtAssertionCredentials
import pandas as pd

import lasagna.config


ROW_INDICATOR = 'A'

ROWS = 'ABCDEFGH'
COLS = [str(x) for x in range(1,17)]


def flatten_layout_row_col(x):
    x = pd.DataFrame(x.stack())
    index = [a + str(b) for a, b in x.index.values]
    x.index = pd.Index(index, name='sample')
    return x


class Experiment(object):
    def __init__(self, worksheet=None, g_file='Lasagna FISH'):
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
        exp.load_sheet(worksheet)
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
            self.load_and_go(worksheet, g_file=g_file)

    def load_and_go(self, worksheet, g_file='Lasagna FISH'):
        """Shortcut to load sheet, parse sheet, and make independent variable table when
        notation is sheet is sufficient.
        :param worksheet:
        :param g_file:
        :return:
        """
        self.load_sheet(worksheet, g_file=g_file)
        self.parse_grids()
        self.parse_ind_vars()
        return self.make_ind_vars_table()

    def load_sheet(self, worksheet, g_file='Lasagna FISH'):
        """Load sheet as array of strings (drops .xls style index)
        :param worksheet:
        :param g_file:
        :return:
        """
        # see http://gspread.readthedocs.org/en/latest/oauth2.html
        json_key = json.load(open(lasagna.config.credentials))
        scope = ['https://spreadsheets.google.com/feeds']
        credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'], scope)
        gc = gspread.authorize(credentials)
        xsheet = gc.open(g_file)

        if type(worksheet) is int:
            wks = xsheet.get_worksheet(worksheet)
        else:
            wks = xsheet.worksheet(worksheet)
        xs_values = np.array(wks.get_all_values())
        self.sheet = xs_values

    def parse_grids(self, title_offset=(-1, 0), A_offset=(1, 0)):
        """Look for first row labelled by ROW_INDICATOR, find origin and return boolean mask to values.
        :param numpy.ndarray xs_values:
        :param tuple grid_size:
        :param tuple title_offset:
        :param tuple A_offset:
        :return: (dict, wells): ({title: mask}, {well_name:
        :rtype : dict[str, numpy.nparray]
        """
        xs_values = np.pad(self.sheet, ((0, 1), (0, 1)), mode='constant')

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
        xs_values = np.pad(self.sheet, ((0, 1), (0, 1)), mode='constant')

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
                    return x
            return func

        arr = []
        for ind_var, grid in self.grids.items():
            values = self.ind_vars[ind_var]
            grid = self.flatten_layout(grid)
            grid.columns = [ind_var]
            arr += [grid.applymap(apply_try(values))]

        table = pd.concat(arr, axis=1).fillna('')
        self.ind_vars_table = table[(table != '').any(axis=1)]







#
#
#
#
#
#
# def load_sheet(worksheet, gfile='Lasagna FISH', grid_size=(6, 6),
#               find_conditions=True):
#     """Find conditions demarcated in grid format from local .xls or google sheet.
#     :param str worksheet: name of google sheet, can provide int index as well
#     :param str file: google sheet to search in
#     :param tuple grid_size: dimensions of sample layout (rows, columns)
#     :return (dict[str, tuple], dict[str, list], numpy.ndarray): (wells: dict of tuples, {well: condition},
#      conditions: OrderedDict, {variable_name: conditions},
#      cube: N-d array representing space of conditions, integer entries indicate number of replicates)
#     """
#
#
#     wells = defaultdict(list)
#     for (i, j) in product(range(grid_size[0]), range(grid_size[1])):
#         well = ROWS[i] + COLS[j]
#         for title, grid in grids.items():
#             val = xs_values[grid][i*grid_size[1] + j]
#             wells[well] += [val]
#
#
#     wells_ = {}
#     for well, values in wells.items():
#         # exclude fully empty wells
#         if all(x == '' for x in values):
#             continue
#
#         tmp = []
#         for x in values:
#             try:
#                 tmp += [float(x)]
#             except ValueError:
#                 tmp += [np.nan]
#         wells.update({well: tmp})
#
#     wells = {a: [float(x) for x in b] for a, b in wells.items()
#              if not all(x == '' for x in b)}
#
#     # easy out for non-standard condition indexing
#     if not find_conditions:
#         return wells
#
#     # get the named conditions for each variable
#     conditions = extract_conditions(xs_values, grids.keys())
#
#     cube = np.zeros([max(x) + 1 for x in zip(*wells.values())])
#     for coords in wells.values():
#         cube[tuple(coords)] += 1
#
#     return wells, conditions, cube
#




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