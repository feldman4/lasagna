import json, gspread
import numpy as np
from oauth2client.client import SignedJwtAssertionCredentials
from itertools import product
from collections import defaultdict, OrderedDict

CREDENTIALS_JSON = '/Users/feldman/Downloads/gspread-da2f80418147.json'

ROW_INDICATOR = 'A'

ROWS = 'ABCDEFGH'
COLS = [str(x) for x in range(1,17)]


def load_xsheet(worksheet, gfile='Lasagna FISH'):
    # see http://gspread.readthedocs.org/en/latest/oauth2.html
    json_key = json.load(open(CREDENTIALS_JSON))
    scope = ['https://spreadsheets.google.com/feeds']
    credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'], scope)
    gc = gspread.authorize(credentials)
    xsheet = gc.open(gfile)

    if type(worksheet) is int:
        wks = xsheet.get_worksheet(worksheet)
    else:
        wks = xsheet.worksheet(worksheet)
    xs_values = np.array(wks.get_all_values())
    return xs_values


def load_sheet(worksheet, gfile='Lasagna FISH', grid_size=(6, 6),
              find_conditions=True):
    """Find conditions demarcated in grid format from local .xls or google sheet.
    :param str worksheet: name of google sheet, can provide int index as well
    :param str file: google sheet to search in
    :param tuple grid_size: dimensions of sample layout (rows, columns)
    :return (dict[str, tuple], dict[str, list], numpy.ndarray): (wells: dict of tuples, {well: condition},
     conditions: OrderedDict, {variable_name: conditions},
     cube: N-d array representing space of conditions, integer entries indicate number of replicates)
    """
    xs_values = load_xsheet(worksheet, gfile=gfile)
    # pad edges with empty string
    xs_values = np.pad(xs_values, ((0, 1), (0, 1)), mode='constant')

    # find grids containing variable conditions
    grids = find_grids(xs_values, grid_size)

    wells = defaultdict(list)
    for (i, j) in product(range(grid_size[0]), range(grid_size[1])):
        well = ROWS[i] + COLS[j]
        for title, grid in grids.items():
            val = xs_values[grid][i*grid_size[1] + j]
            wells[well] += [val]
            
    
    wells_ = {}
    for well, values in wells.items():
        # exclude fully empty wells
        if all(x == '' for x in values):
            continue
            
        tmp = []
        for x in values:
            try:
                tmp += [float(x)]
            except ValueError:
                tmp += [np.nan]
        wells.update({well: tmp})
            
    wells = {a: [float(x) for x in b] for a, b in wells.items()
             if not all(x == '' for x in b)}

    # easy out for non-standard condition indexing
    if not find_conditions:
        return wells
    
    # get the named conditions for each variable
    conditions = extract_conditions(xs_values, grids.keys())

    cube = np.zeros([max(x) + 1 for x in zip(*wells.values())])
    for coords in wells.values():
        cube[tuple(coords)] += 1

    return wells, conditions, cube


def find_grids(xs_values, grid_size, title_offset=(-1, 0), A_offset=(1, 0)):
    """Look for first row labelled by ROW_INDICATOR, find origin and return boolean mask to values.
    :param numpy.ndarray xs_values:
    :param tuple grid_size:
    :param tuple title_offset:
    :param tuple A_offset:
    :return: (dict, wells): ({title: mask}, {well_name:
    :rtype : dict[str, numpy.nparray]
    """
    grids = OrderedDict()
    for candidate in zip(*np.where(xs_values == ROW_INDICATOR)):
        origin = np.array(candidate) - A_offset
        grid = np.zeros(xs_values.shape, bool)
        coords = tuple([slice(a + 1, a + b + 1) for a, b in zip(origin, grid_size)])
        grid[coords] = True

        title = xs_values[tuple(origin + title_offset)]
        grids[title] = grid

    return grids


def get_named_wells(wells, conditions):
    """Convert dict of wells with indexed conditions to dict with named conditions
    """
    return {k: [arr[x] for x, arr in zip(v, conditions.values())] for k, v in wells.items()}


def extract_conditions(xs_values, names):
    # assume conditions are listed in blocks
    conditions = OrderedDict()
    [conditions.update({title: []}) for title in names]
    rows = []
    for title in names:
        hits = sorted(zip(*np.where(xs_values == title)), key=lambda x: x[1])
        rows.append(hits[0][0])
        if len(hits) < 2:
            # condition names must exactly match
            raise IndexError
        index = np.array(hits[0]) + (0, 1)
        while xs_values[tuple(index)]:
            conditions[title] += [xs_values[tuple(index)]]
            index += (1, 0)
    # sort by row
    conditions = OrderedDict((key, conditions[key]) for _, key in sorted(zip(rows, names)))

    return conditions

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