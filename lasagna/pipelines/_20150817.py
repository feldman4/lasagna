from itertools import product
import scipy.ndimage.filters
import lasagna.io
import lasagna.config
import lasagna.process
import lasagna.utils
import lasagna.conditions_
import lasagna.models
import copy
import numpy as np
import skimage.transform
import skimage.morphology
import skimage.feature
import os
import pandas as pd
pdx = pd.IndexSlice

display_ranges = ((500, 20000),
                  (500, 3500),
                  (500, 3500),
                  (500, 3500))

# for matplotlib
dye_colors = {'A594': 'r', 'Atto647': 'm', 'Cy3': 'g'}

GSPREAD_CREDENTIALS = '/broad/blainey_lab/blainey-ipython/DF/gspread-da2f80418147.json'

worksheet = '20150816 96W-G008'

dataset = '20150817 6 round'

pipeline = None
tile_configuration = 'calibration/TileConfiguration.registered.txt'
channels = 'DAPI', 'Cy3', 'A594', 'Atto647'
luts = lasagna.io.BLUE, lasagna.io.GREEN, lasagna.io.RED, lasagna.io.MAGENTA

filters = lasagna.utils.Filter2DReal(lasagna.process.double_gaussian(10, 1)),


def setup(lasagna_path):
    """Construct Paths and Calibration objects for dataset. Copy a smaller version of Calibration object
    to be synced to remote engines.
    :return:
    """
    lasagna.config.paths = lasagna.io.Paths(dataset, lasagna_path=lasagna_path)
    for condition in ('strip', 'hyb'):
        lasagna.config.paths.table[condition] = [
            condition in name for name in lasagna.config.paths.table.index.get_level_values('set')]
        lasagna.config.paths.table.set_index(condition, append=True, inplace=True)

    config_path = lasagna.config.paths.full(lasagna.config.paths.calibrations[0])
    lasagna.config.calibration = lasagna.process.Calibration(config_path,
                                                             dead_pixels_file='dead_pixels_empirical.tif')
    c = copy.deepcopy(lasagna.config.calibration)
    c.calibration = None
    lasagna.config.calibration_short = c
    global tile_configuration
    tile_configuration = lasagna.config.paths.full(tile_configuration)


def initialize_engines(client):
    """Import modules and define Paths and Calibration objects on remote engines. Use short version
    of Calibration object to save on time.
    :param client:
    :return:
    """
    dview = client[:]
    dview.execute('import numpy as np')
    dview.execute('import pandas as pd')
    dview.execute('import lasagna.io')
    dview.execute('import lasagna.process')
    dview.execute('import lasagna.config')
    dview.execute('import lasagna.pipelines._20150817')
    dview.execute('import os')
    dview.execute('import lasagna.pipelines._20150817 as pipeline')

    dview['lasagna.config.paths'] = lasagna.config.paths
    dview.execute('paths = lasagna.config.paths')
    dview['lasagna.config.calibration'] = lasagna.config.calibration_short
    print len(client.ids), 'engines initialized'


def calibrate(row):
    """Calibrate row of Paths DataFrame. Call with map_async to farm out.
    :param row:
    :return:
    """
    channels = [None, 'Cy3', 'A594', 'Atto647']

    raw, calibrated = row['raw'], row['calibrated']
    raw_data = lasagna.io.read_stack(lasagna.config.paths.full(raw))
    raw_data = np.array([lasagna.config.calibration.fix_dead_pixels(frame) for frame in raw_data])
    fixed_data = np.array([lasagna.config.calibration.fix_illumination(frame, channel=channel)
                           for frame, channel in zip(raw_data, channels)])
    lasagna.io.save_hyperstack(lasagna.config.paths.full(calibrated), fixed_data,
                               display_ranges=display_ranges, luts=luts)


def stitch(files_in, file_out, translations=None, clip=True):
    """Stitches images with alpha blending, provided Paths DataFrame with tile filenames in column
    'calibrated' and TileConfiguration.registered.txt file (from GridCollection stitching) with location
    stored in pipeline.tile_configuration. Alternately, provide list of translations (xy).
    :param files_in: list of calibrated files, relative to dataset
    :param file_out: output file name, relative to dataset
    :param translations: list of [x,y] offsets of tiles
    :return:
    """

    if translations is None:
        translations = lasagna.io.load_tile_configuration(tile_configuration)
    if isinstance(translations, str):
        translations = lasagna.io.load_tile_configuration(translations)

    # kluge for 5x5 vs 3x3 grids present in some data
    grid_size = int(np.sqrt(len(files_in))) * np.array([1, 1])
    if grid_size[0] == 3:
        index = [0, 1, 2,
                 5, 6, 7,
                 10, 11, 12]
        translations = [translations[i] for i in index]

    save_name = lasagna.config.paths.full(file_out)
    print save_name
    files = np.array([f for f in files_in]).reshape(grid_size)
    data = np.array([[lasagna.io.read_stack(lasagna.config.paths.full(x)) for x in y] for y in files])
    print data.shape
    arr = []
    for channel in range(data.shape[2]):
        arr += [lasagna.process.alpha_blend(data[:, :, channel].reshape(-1, *data.shape[-2:]),
                                            translations, edge=0.48,
                                            edge_width=0.01,
                                            clip=clip)]

    lasagna.io.save_hyperstack(save_name, np.array(arr), display_ranges=display_ranges, luts=luts)


def align(files, save_name, n=500, trim=150):
    """Align data using first channel (DAPI). Register corners using FFT, build similarity transform,
    warp, and trim edges.
    :param files: files to align
    :param save_name:
    :param n: width of corner alignment window
    :param trim: # of pixels to trim in from each size
    :return:
    """
    import os  # mysteriously not imported by dview.execute
    if not os.path.isfile(save_name):
        save_name = lasagna.config.paths.full(save_name)

    data = [lasagna.io.read_stack(lasagna.config.paths.full(f)) for f in files]
    data = lasagna.io.compose_stacks(data)

    offsets, transforms = lasagna.process.get_corner_offsets(data[:, 0], n=n)
    for i, transform in zip(range(data.shape[0]), transforms):
        for j in range(data.shape[1]):
            data[i, j] = skimage.transform.warp(data[i, j], transform.inverse, preserve_range=True)

    data = data[:, :, trim:-trim, trim:-trim]
    lasagna.io.save_hyperstack(save_name, data)


def find_nuclei(row, block_size, source='aligned'):
    """Find nuclei in data from row of Paths DataFrame, using local threshold and watershed separation.
    Automatically processes single frame of trailing two dimensions of data only.
    :param row:
    :param block_size:
    :return:
    """
    data = lasagna.io.read_stack(lasagna.config.paths.full(row[source]))
    s = [0] * (data.ndim - 2) + [slice(None)] * 2
    nuclei = lasagna.process.get_nuclei(data[s], block_size=block_size)
    lasagna.io.save_hyperstack(lasagna.config.paths.full(row['nuclei']), nuclei)


def table_from_nuclei(row, index_names=None, save_name=None, round_=1,
                      nuclei_dilation=None):
    """Build nuclei table from segmented nuclei files. Separate entry for each round, labels not preserved.
    :param row: row from Paths DataFrame
    :param index_names: name of levels in row MultiIndex (row from DataFrame saves level values but not names)
    :param round_: number of round, one-indexed
    :return:
    """
    if index_names is None:
        index_names = lasagna.config.paths.table.index.names

    data = lasagna.io.read_stack(lasagna.config.paths.full(row['aligned']))
    data = data[round_ - 1]

    # update row name to correspond to round being analyzed
    x = list(row.name)
    x[index_names.index('round')] = round_
    row.name = x

    df_ = lasagna.process.table_from_nuclei(row, index_names, data=data,
                                            channels=channels, nuclei_dilation=nuclei_dilation)
    if save_name is None:
        save_name = lasagna.config.paths.export(row['file_well'] + '.' + str(round_) + '.pkl')
    df_.to_pickle(save_name)


def match_nuclei_table(table , area_fraction=0.9):
    """Match nuclei between rounds based on overlap of segmented regions.
    :param area_fraction: minimum required overlap, defined as # shared pixels/max(# of pixels)
    :return:
    """

    pass


def apply_watermark(arr, label, trail=3, **kwargs):
    """Apply label over trailing dimensions of array and append watermark of result.
    If label is a function, it should return string or list of strings.
    If label is a numpy.ndarray, its shape must match arr.shape[:-trail], with a single optional additional
        trailing dimension. The trailing dimension will be used to form a list of str.
    If label is a pandas.DataFrame, it will first be converted to a numpy.ndarray of type str via
        lasagna.utils.to_nd_array.

    Watermark is appended to channel dimension.

    :param numpy.ndarray arr: image data of shape [..., channel, height, width].
    :param label: function, numpy.ndarray, or pandas.DataFrame
    :param int trail: number of trailing dimensions when applying label, >= 3.
    :param kwargs: passed to lasagna.io.watermark
    :return:
    """

    if isinstance(label, pd.DataFrame):
        label = lasagna.utils.to_nd_array(label.astype(str))[0]
    if isinstance(label, np.ndarray):
        label = label.reshape([np.prod(arr.shape[:-trail]), -1])
        it = iter([x for x in label])
        label = lambda _: it.next()

    assert(trail >= 3)
    arr_ = arr.reshape(-1, *arr.shape[-trail:]).copy()
    new_arr = []
    for stack in arr_:
        annotation_shape = list(stack.shape)
        annotation_shape[-3] = 1
        try:
            annotation = lasagna.io.watermark(annotation_shape[-2:], label(stack), **kwargs)
            annotation = np.resize(annotation, annotation_shape)
        except ValueError:
            annotation = np.zeros(annotation_shape)
        new_arr += [np.concatenate([stack, annotation], axis=-3)]

    new_shape = list(arr.shape)
    new_shape[-3] += 1
    return np.array(new_arr).reshape(new_shape)


def blob_max_median(df, detect_round=1, detect_channel=3, neighborhood=(9, 9),
                    window_filter=lambda x: x, pad_width=None, max_sigma=5,
                    starting_threshold=1):
    """Detect blobs in given round, mark strongest blob. For each round and channel, find max signal
    in a window of width `neighborhood` after applying `window_filter` to [round, channel, height, width]
    data. Also find and record median of applying filter and max window over entire nucleus, may be useful
     for normalization.
    :return:
    """

    pad_width = neighborhood if pad_width is None else pad_width

    data = lasagna.io.get_row_stack(df.xs(detect_round, level='round').ix[0], pad=pad_width)

    detect_frame = data[detect_round - 1, detect_channel]

    threshold = starting_threshold
    blobs = []
    while True:
        blobs = skimage.feature.blob_log(detect_frame, max_sigma=max_sigma, threshold=threshold)
        if len(blobs) != 0:
            break
        threshold /= 4.

    max_size = [1, 1] + list(neighborhood)
    data_max = scipy.ndimage.filters.maximum_filter(window_filter(data), size=max_size)

    # find blob with most signal
    blob_values = detect_frame[blobs[:, 0], blobs[:, 1]]
    best = blob_values.argmax()

    max_values = data_max[:, :, blobs[best, 0], blobs[best, 1]]
    max_median_values = np.median(data_max, axis=[2, 3])

    # uses channels from outer scope
    for i, channel in enumerate(channels):
        df[channel, 'blob_max'] = max_values[:, i].astype(float)
        df[channel, 'blob_max_median'] = max_median_values[:, i].astype(float)

    # actual pad width will be smaller if near edge of full data
    bounds = df.ix[0]['all', 'bounds']
    pad_width_adj = [min(pad_width[0], bounds[0]),
                     min(pad_width[1], bounds[1]), 0]
    pad_width_adj = np.array(pad_width_adj).astype(float)
    blob_info_columns = 'blob_i', 'blob_j', 'blob_sigma'
    blob_info = blobs[best, :] - pad_width_adj


    for k, v in zip(blob_info_columns, blob_info):
        df['all', k] = v

    return df.sortlevel(axis=1)


def load_conditions():
    """Load Experiment and prepare ind. var. table based on A + 0.01*B notation for probes
     in sheet layout.
    :return:
    """
    experiment = lasagna.conditions_.Experiment()
    experiment.sheet = lasagna.conditions_.load_sheet(worksheet)
    experiment.parse_ind_vars()
    experiment.parse_grids()

    # convert list to dict that accepts floats as combos
    probes = experiment.ind_vars['probes']
    probes_ = {}
    for i, a in enumerate(probes):
        i += 1
        for j, b in enumerate(probes):
            j += 1
            if i == j:
                probes_.update({i: (a,)})
                continue
            probes_.update({i + 0.01*j: (a, b)})
            probes_.update({j + 0.01*i: (b, a)})

    for ind_var in experiment.grids:
        if 'probe' in ind_var:
            experiment.ind_vars[ind_var] = probes_

    experiment.make_ind_vars_table()

    # remove _d60 and add corresponding barcode
    cells = [x[:-4] for x in experiment.ind_vars_table['cells']]
    virus = lasagna.config.cloning['cell lines'].loc[cells, 'lentivirus']
    plasmids = lasagna.config.cloning['lentivirus'].loc[virus, 'plasmid']
    barcodes = lasagna.config.cloning['plasmids'].loc[plasmids, 'barcode']

    experiment.ind_vars_table['barcodes'] = list(barcodes)

    lasagna.config.experiment = experiment
    return experiment


def prepare_linear_model():
    """Create LinearModel, set probes used in experiment, and generate matrices.
    
    :return:
    """
    model = lasagna.models.LinearModel()
    lasagna.config.set_linear_model_defaults(model)
    model.indices['l'] = lasagna.config.experiment.ind_vars['probes']
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
        b[sample][row['barcodes']] = 1

    print b['A1']
    lasagna.config.experiment.ind_vars_table['M'] = [M[x] for x in ivt.index]
    lasagna.config.experiment.ind_vars_table['b'] = [b[x] for x in ivt.index]

    return model


###################
# NOT IN USE, needs sub-pixel alignment, better blending, individual offsets instead of mean
###################
def _stitch(df, offsets=None, overlap=925. / 1024):
    """Stitch calibrated images into square grid. Calculate offsets using lasagna.process.stitch_grid
    unless provided.
    :param df:
    :param offsets:
    :param overlap:
    :return:
    """

    stitch = df.ix[0, 'calibrated'].replace('analysis/calibrated/raw',
                                            'stitch')
    stitch = lasagna.config.paths.full(stitch)

    data = np.array([lasagna.io.read_stack(lasagna.config.paths.full(f))
                     for f in df['calibrated']])
    grid = np.sqrt(data.size / np.prod(data.shape[-3:]))
    print data.shape
    gridded_data = data.reshape([grid, grid] + list(data.shape[-3:]))

    if offsets is None:
        offsets = lasagna.process.stitch_grid(gridded_data[:, :, 0, :, :], overlap)
    y = lasagna.process.compress_offsets(offsets)

    stitched_data = np.array([lasagna.process.alpha_blend(gridded_data[:, :, i, :, :], y)
                              for i in range(gridded_data.shape[2])])

    lasagna.io.save_hyperstack(lasagna.config.paths.full(stitch), stitched_data)
    return offsets
