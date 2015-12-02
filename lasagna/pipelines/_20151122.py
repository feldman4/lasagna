import lasagna.io
import lasagna.models
import lasagna.process
import os
import numpy as np
import pandas as pd
import skimage.transform
import skfmm
import skimage
import scipy.stats


dye_colors = {'A594': 'r', 'Atto647': 'm', 'Cy3': 'g'}

GSPREAD_CREDENTIALS = '/broad/blainey_lab/blainey-ipython/DF/gspread-da2f80418147.json'

# name of lasagna folder and sheet in Lasagna FISH
dataset = '20151122_96W-G020'

# relative magnification, empirical
scale_40_100 = 0.393

pipeline = None
tile_configuration = 'calibration/TileConfiguration.registered.txt'
channels = 'DAPI', 'Cy3', 'A594', 'Atto647'
luts = lasagna.io.BLUE, lasagna.io.GREEN, lasagna.io.RED, lasagna.io.MAGENTA

display_ranges = ((500, 80000),
                  (2000, 6000),
                  (800, 3500),
                  (800, 3500))


def setup(lasagna_path='/broad/blainey_lab/David/lasagna/'):
    """Construct Paths and Calibration objects for dataset. Copy a smaller version of Calibration object
    to be synced to remote engines.
    :return:
    """
    lasagna.config.paths = lasagna.io.Paths(dataset, lasagna_path=lasagna_path)
    
    config_path = lasagna.config.paths.full(lasagna.config.paths.calibrations[0])
    # lasagna.config.calibration = lasagna.process.Calibration(config_path,
    #                                                          dead_pixels_file='dead_pixels_empirical.tif')
    # c = copy.deepcopy(lasagna.config.calibration)
    # c.calibration = None
    # lasagna.config.calibration_short = c
    global tile_configuration
    tile_configuration = lasagna.config.paths.full(tile_configuration)
    return lasagna.config.paths





# def load_conditions():
#     """Load Experiment and prepare ind. var. table based on A + 0.01*B notation for probes
#      in sheet layout.
#     :return:
#     """
#     experiment = lasagna.conditions_.Experiment()
#     experiment.sheet = lasagna.conditions_.load_sheet(dataset)
#     experiment.parse_ind_vars()
#     experiment.parse_grids()    

#     experiment.ind_vars['probes round 1'] = \
#             [tuple(x.split(', ')) for x in experiment.ind_vars['probes']]

#     experiment.make_ind_vars_table()


#     cells = experiment.ind_vars_table['cells']
#     virus = lasagna.config.cloning['cell lines'].loc[cells, 'lentivirus']
#     plasmids = lasagna.config.cloning['lentivirus'].loc[virus, 'plasmid']
#     plasmids = plasmids.fillna('')
#     barcodes = lasagna.config.cloning['plasmids'].loc[plasmids, 'barcode']

#     barcodes = barcodes.fillna('')
#     # split comma-separated list of barcodes
#     experiment.ind_vars_table['barcodes'] = [tuple(x.split(', ')) for x in barcodes]

#     lasagna.config.experiment = experiment
#     return experiment


# def prepare_linear_model():
#     """Create LinearModel, set probes used in experiment, and generate matrices.

#     :return:
#     """
#     model = lasagna.models.LinearModel()
#     # split comma-separated probes
#     lasagna.config.set_linear_model_defaults(model)
#     all_probes = lasagna.config.experiment.ind_vars['probes']
#     model.indices['l'] = list(set(sum([x.split(', ') for x in all_probes], [])))

#     model.matrices_from_tables()

#     ivt = lasagna.config.experiment.ind_vars_table

#     model.indices['j'] = [x for x in ivt.columns
#                           if 'round' in x]

#     # reformat entries in independent vars table as matrix input to LinearModel
#     M = {sample: pd.DataFrame([], index=model.indices['j'],
#                               columns=model.indices['l']).fillna(0)
#          for sample in ivt.index}
#     b = {sample: pd.Series({x: 0 for x in model.indices['m']})
#          for sample in ivt.index}

#     for sample, row in ivt.iterrows():
#         for rnd in model.indices['j']:
#             M[sample].loc[rnd, list(ivt.loc[sample, rnd])] = 1
#         if not pd.isnull(row['barcodes']):
#             b[sample][row['barcodes']] = 1

#     lasagna.config.experiment.ind_vars_table['M'] = [M[x] for x in ivt.index]
#     lasagna.config.experiment.ind_vars_table['b'] = [b[x] for x in ivt.index]

#     return model





