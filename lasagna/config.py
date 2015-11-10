import os
import pandas as pd
import lasagna.conditions_

up = os.path.dirname
home = up(up(__file__))
paths = None
calibration = None
calibration_short = None

experiment = None

fonts = os.path.join(home, 'resources', 'fonts')
luts = os.path.join(home, 'resources', 'luts')

visitor_font = os.path.join(fonts, 'visitor1.ttf')

credentials = os.path.join(os.path.dirname(home), 'gspread-da2f80418147.json')

cloning = None


def load_sheets():
    """Load from google spreadsheet. Report duplicates on key "name". Drops empty rows.
    Column names must be in first row.
    :return:
    """
    g_file = 'Lasagna Cloning'
    name = 'name'
    x = lasagna.conditions_.load_sheet(None, g_file=g_file)
    global cloning
    cloning = {}
    for title, values in x.items():
        df = pd.DataFrame(values[1:], columns=values[0])
        # drop empty rows
        df = df[(df != '').any(axis=1)]
        duplicates = df[df.duplicated(subset=name)]
        if duplicates.size:
            print '[%s: %s] dropped duplicates in column [%s]' % (g_file, title, name)
            print list(duplicates['name'])
        # useful for pivoting
        df['dummy'] = 1
        cloning[title] = df.drop_duplicates(subset=name).set_index('name')


def set_linear_model_defaults(model):
    """
    Default is not set for index j, corresponding to round.
    :param model:
    :return:
    """
    pr = cloning['probes']
    pr['oligos'] = pr['oligos'].convert_objects(convert_numeric=True).fillna(0)
    x = pr.reset_index().pivot_table(values='oligos', fill_value=0, index='name', columns='targets')

    tiles = (cloning['barcodes'].loc[:, 'tiles'])
    tiles = {k: v.split(', ') for k,v in dict(tiles).items()}

    B = pd.DataFrame()
    for barcode, tiles in tiles.items():
        B[barcode] = x[tiles].sum(axis=1)

    model.tables['B'] = B
    model.tables['C'] = (cloning['dyes'].drop('dummy', 1)
                         .astype(float))
    model.tables['D'] = (cloning['probes'].reset_index()
                         .pivot_table(values='dummy', index='name',
                                      columns='dye', fill_value=0))

    # default indices
    model.indices['k'] = list(model.tables['C'].index)
    model.indices['l'] = list(model.tables['B'].index)
    model.indices['m'] = list(model.tables['B'].columns)
    model.indices['n'] = list(model.tables['C'].columns)
