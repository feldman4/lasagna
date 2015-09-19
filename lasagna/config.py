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

credentials = '/Users/feldman/Downloads/gspread-da2f80418147.json'

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
        cloning[title] = df.drop_duplicates(subset=name)
