import lasagna.io
import lasagna.models
import lasagna.process
import os
import numpy as np
import pandas as pd
import skimage.transform
import skimage
import scipy.stats


# name of lasagna folder and sheet in Lasagna FISH
dataset = '20160414_96W-G030'
dead_pixels_file = 'calibration/dead_pixels_20151219_MAX.tif'
channels = 'DAPI', 'Cy3', 'A594', 'Atto647'
luts = lasagna.io.BLUE, lasagna.io.GREEN, lasagna.io.RED, lasagna.io.MAGENTA
tile_configuration = None
display_ranges = ((500, 45000),
                    (1400, 5000),
                    (800, 5000),
                    (800, 5000))

#  glueviz
name_map = {('all','bounds',1.): 'bounds',
            ('all', 'file', 1.): 'file',
            ('all', 'contour',1.):'contour',
            ('well', '', ''): 'well',
            ('row', '', ''): 'row',
            ('column', '', ''): 'column',
            ('all', 'x', 1.): 'x',
            ('all', 'y', 1.): 'y'}


def setup(lasagna_path='/broad/blainey_lab/David/lasagna/'):
    """Construct Paths and Calibration objects for dataset. Copy a smaller version of Calibration object
    to be synced to remote engines.
    :return:
    """
    lasagna.config.paths = lasagna.io.Paths(dataset, lasagna_path=lasagna_path)
    
    config_path = lasagna.config.paths.calibrations[0]
    lasagna.config.calibration = lasagna.process.Calibration(config_path, 
                                    dead_pixels_file=dead_pixels_file, illumination_correction=False)
    # c = copy.deepcopy(lasagna.config.calibration)
    # c.calibration = None
    # lasagna.config.calibration_short = c
    return lasagna.config.paths


def load_conditions():
    """Load Experiment and prepare ind. var. table based on A + 0.01*B notation for probes
     in sheet layout.
    :return:
    """
    experiment = lasagna.conditions_.Experiment()
    experiment.sheet = lasagna.conditions_.load_sheet(dataset)
    experiment.parse_ind_vars()
    experiment.parse_grids()    

    # make each round of probes its own variable, described by tuple of probes 
    for rnd in '123':
        experiment.ind_vars['probes round %s' % rnd] = \
                [tuple(x.split(', ')) for x in experiment.ind_vars['probes']]

    experiment.ind_vars['cells'] = \
                [tuple(x.split(', ')) for x in experiment.ind_vars['cells']]

    experiment.make_ind_vars_table()

    not_found = {}
    for ix, cells in experiment.ind_vars_table['cells'].iteritems():
    
        virus = lasagna.config.cloning['cell lines'].loc[cells, 'lentivirus']
        # TODO: map comma-separated cell lines to tuple
        try:
            virus = virus.fillna('')
            plasmids = lasagna.config.cloning['lentivirus'].loc[virus, 'plasmid']
            plasmids = plasmids.fillna('')
            barcodes = lasagna.config.cloning['plasmids'].loc[plasmids, 'barcode']
            barcodes = barcodes.fillna('')
            # split comma-separated list of barcodes
            experiment.ind_vars_table.loc[ix, 'barcodes'] = tuple(barcodes)
        except KeyError:
            not_found[cells] = None
            experiment.ind_vars_table.loc[ix, 'barcodes'] = tuple()

    print 'no barcodes found for cells: %s' % '\n'.join(','.join(x) for x in not_found)

    lasagna.config.experiment = experiment
    return experiment
