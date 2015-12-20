import lasagna.io
import lasagna.models
from lasagna.utils import cells_to_barcodes, probes_to_rounds
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


def load_conditions():
    """Load Experiment and prepare ind. var. table based on A + 0.01*B notation for probes
     in sheet layout.
    :return:
    """
    transforms = (probes_to_rounds(rounds=4),)
    experiment = lasagna.conditions_.Experiment(worksheet=dataset,
                                                ind_var_transforms=transforms)
    cells_to_barcodes(experiment.ind_vars_table, cloning=lasagna.config.cloning)
    lasagna.config.experiment = experiment
    return experiment

