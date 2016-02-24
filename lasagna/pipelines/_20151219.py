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
dataset = '20151219_96W-G024'
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


def align(files, save_name, n=500, trim=150, take_max=True):
    """Align data using first channel (DAPI). Register corners using FFT, build similarity transform,
    warp, and trim edges.
    :param files: files to align
    :param save_name:
    :param n: width of corner alignment window
    :param trim: # of pixels to trim in from each size
    :return:
    """
    import os  # mysteriously not imported by dview.execute
    if not os.path.isabs(save_name):
        save_name = lasagna.config.paths.full(save_name)

    data = [lasagna.io.read_stack(f) for f in files]
    data = lasagna.io.compose_stacks(data)
    if take_max:
        data = data.max(axis=1)

    offsets, transforms = lasagna.process.get_corner_offsets(data[:, 0], n=n)
    for i, transform in zip(range(data.shape[0]), transforms):
        for j in range(data.shape[1]):
            data[i, j] = skimage.transform.warp(data[i, j], transform.inverse, preserve_range=True)

    data = data[:, :, trim:-trim, trim:-trim]
    lasagna.io.save_hyperstack(save_name, data, luts=luts, display_ranges=display_ranges)
    return data


def align_scaled(x, y, scale, **kwargs):
    """Align two images taken at different magnification. The input dimensions 
    are assumed to be [channel, height, width], and the alignment is based on 
    the first channel. The first image should contain the second image.
    Additional kwargs are passed to lasagna.process.register_images.
    """
    # TODO: MODIFY TO RETURN OVERLAPPING WINDOW ONLY
    x = x.transpose([1, 2, 0])
    y = y.transpose([1, 2, 0])

    # downsample 100X image and align to get offset
    if float(scale) != 1.:
        y_ds = skimage.transform.rescale(y, scale, preserve_range=True)
    else:
        y_ds = y
    _, offset = lasagna.process.register_images([x[..., 0], y_ds[..., 0]],
                                                **kwargs)
    ST = skimage.transform.SimilarityTransform(translation=offset[::-1])

    # warp 40X image and resize to match 100X image
    x_win = skimage.transform.warp(x, inverse_map=ST, 
                                      output_shape=y_ds.shape[:2], 
                                      preserve_range=True)
    x_win = skimage.transform.resize(x_win, y.shape)

    # combine images along new leading dimension
    return np.r_['0,4', x_win, y].transpose([0,3,1,2])


def clean_signal(im, bg_radius=160, clahe_clip=0.08, clahe_ntiles=4):

    selem_bg = skimage.morphology.square(bg_radius)

    # background subtraction
    im_bs = skimage.morphology.white_tophat(im, selem=selem_bg)

    # CLAHE
    # ImageJ: blocksize=250, histogram bins=256, maximum slope=3.00
    clahe = skimage.exposure.equalize_adapthist
    return clahe(im_bs, ntiles_x=clahe_ntiles, 
                            ntiles_y=clahe_ntiles, clip_limit=clahe_clip)



def segment_cells(nuclei, mask, small_holes=100, remove_boundary_cells=True):
    import skfmm
    selem_3 = skimage.morphology.square(3)

    # voronoi
    phi = (nuclei>0) - 0.5
    speed = np.ones(phi.shape)
    time = skfmm.travel_time(phi, speed).astype(np.uint16)
    time[nuclei>0] = 0

    w = skimage.morphology.watershed(time, nuclei)

    if remove_boundary_cells:
        cut = w[0,:], w[-1,:], w[:,0], w[:,-1]
        w.flat[np.in1d(w, np.unique(cut))] = 0
        w = skimage.measure.label(w)

    # apply mask
    w[mask==0] = 0
    w = skimage.morphology.closing(w)

    # only take biggest component for each cell
    relabeled = skimage.measure.label(w, background=0) + 1
    relabeled[w==0] = 0
    regions = skimage.measure.regionprops(relabeled, 
                                          intensity_image=nuclei)
    cut = [reg.label for reg in regions if reg.intensity_image.max() == 0]
    relabeled.flat[np.in1d(relabeled, np.unique(cut))] = 0

    # fill small holes
    holes = skimage.measure.label(relabeled==0, background=0) + 1
    regions = skimage.measure.regionprops(holes,
                intensity_image=skimage.morphology.dilation(relabeled))

    for reg in regions:
        if reg.area < small_holes:
            vals = reg.intensity_image[reg.intensity_image>0]
            relabeled[holes == reg.label] = scipy.stats.mode(vals)[0][0]

    return relabeled




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


def prepare_linear_model():
    """Create LinearModel, set probes used in experiment, and generate matrices.

    :return:
    """
    model = lasagna.models.LinearModel()
    # split comma-separated probes
    lasagna.config.set_linear_model_defaults(model)
    all_probes = lasagna.config.experiment.ind_vars['probes']
    model.indices['l'] = sorted(set(sum([x.split(', ') for x in all_probes], [])))

    model.matrices_from_tables()

    ivt = lasagna.config.experiment.ind_vars_table

    model.indices['j'] = [x for x in ivt.columns
                          if 'round' in x]

    # reformat entries in independent vars table as matrix input to LinearModel
    M = {sample: pd.DataFrame([], index=model.indices['j'],
                              columns=model.indices['l']).fillna(0)
         for sample in ivt.index}
    b = {sample: pd.Series({x: 0 for x in model.indices['m']})
         for sample in ivt.index}

    for sample, row in ivt.iterrows():
        for rnd in model.indices['j']:
            M[sample].loc[rnd, list(ivt.loc[sample, rnd])] = 1
        if not pd.isnull(row['barcodes']):
            b[sample][list(row['barcodes'])] = 1

    lasagna.config.experiment.ind_vars_table['M'] = [M[x] for x in ivt.index]
    lasagna.config.experiment.ind_vars_table['b'] = [b[x] for x in ivt.index]

    return model



def stitch(df, translations=None, clip=True):
    """Stitches images with alpha blending, provided Paths DataFrame with tile filenames in column
    'calibrated' and TileConfiguration.registered.txt file (from GridCollection stitching) with location
    stored in pipeline.tile_configuration. Alternately, provide list of translations (xy).
    :param df: has 'calibrated' and 'stitched' columns
    :param translations: list of [x,y] offsets of tiles
    :return:
    """

    if translations is None:
        translations = lasagna.io.load_tile_configuration(tile_configuration)
    if isinstance(translations, str):
        translations = lasagna.io.load_tile_configuration(translations)

    # kluge for 5x5 vs 3x3 grids present in some data
    grid_size = int(np.sqrt(len(df['calibrated']))) * np.array([1, 1])

    save_name = lasagna.config.paths.full(df['stitched'][0])
    print save_name
    files = np.array([f for f in df['calibrated']]).reshape(grid_size)
    data = np.array([[lasagna.io.read_stack(lasagna.config.paths.full(x)) for x in y] for y in files])
    print data.shape
    arr = []
    for channel in range(data.shape[2]):
        arr += [lasagna.process.alpha_blend(data[:, :, channel].reshape(-1, *data.shape[-2:]),
                                            translations, edge=0.48,
                                            edge_width=0.01,
                                            clip=clip)]

    lasagna.io.save_hyperstack(save_name, np.array(arr), display_ranges=display_ranges, luts=luts)


