import lasagna.io
import lasagna.config
import lasagna.process
import lasagna.utils
import copy
import numpy as np
import skimage.transform
import skimage.morphology
import os
import pandas as pd

display_ranges = ((500, 20000),
                  (500, 3500),
                  (500, 3500),
                  (500, 3500))

dataset = '20150817 6 round'

pipeline = None
tile_configuration = 'calibration/TileConfiguration.registered.txt'
channels = 'DAPI', 'Cy3', 'A594', 'Atto647'

luts = lasagna.io.DEFAULT_LUTS

filters = lasagna.utils.Filter2DReal(lasagna.process.double_gaussian(10, 1)),


def setup():
    """Construct Paths and Calibration objects for dataset. Copy a smaller version of Calibration object
    to be synced to remote engines.
    :return:
    """
    lasagna.config.paths = lasagna.io.Paths(dataset)
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
    dview.execute('import lasagna.io')
    dview.execute('import lasagna.process')
    dview.execute('import lasagna.config')
    dview.execute('import lasagna.pipelines._20150817')
    dview.execute('import os')
    dview.execute('import lasagna.pipelines._20150817 as pipeline')

    dview['lasagna.config.paths'] = lasagna.config.paths
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


def stitch(df, translations=None, clip=True):
    """Stitches images with alpha blending, provided Paths DataFrame with tile filenames in column
    'calibrated' and TileConfiguration.registered.txt file (from GridCollection stitching) with location
    stored in pipeline.tile_configuration. Alternately, provide list of translations (xy).
    :param df:
    :param translations: list of [x,y] offsets of tiles
    :return:
    """

    if translations is None:
        translations = lasagna.io.load_tile_configuration(tile_configuration)

    # kluge for 3x3 vs 5x5 grids
    grid_size = int(np.sqrt(df.shape[0])) * np.array([1, 1])
    if grid_size[0] == 3:
        index = [0, 1, 2,
                 5, 6, 7,
                 10, 11, 12]
        translations = [translations[i] for i in index]

    files = np.array([f for f in df['calibrated']]).reshape(grid_size)
    data = np.array([[lasagna.io.read_stack(lasagna.config.paths.full(x)) for x in y] for y in files])
    arr = []
    for channel in range(data.shape[2]):
        arr += [lasagna.process.alpha_blend(data[:, :, channel].reshape(-1, *data.shape[-2:]),
                                            translations, edge=0.48,
                                            edge_width=0.01,
                                            clip=clip)]
    save_name = lasagna.config.paths.full(df['stitched'][0])
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


def table_from_nuclei(row, index_names, save_name=None, round_=1,
                      nuclei_dilation=None):
    """Build nuclei table from
    :param df:
    :return:
    """

    data = lasagna.io.read_stack(lasagna.config.paths.full(row['aligned']))
    data = data[round_ - 1]

    # update row name to correspond to round being analyzed
    x = list(row.name)
    x[index_names.index('round')] = round_
    row.name = x

    df_ = lasagna.process.table_from_nuclei(row, index_names, data=data,
                                            channels=channels, nuclei_dilation=nuclei_dilation)
    if save_name is None:
        save_name = lasagna.config.paths.export(row['file_well'] + '.pkl')
    df_.to_pickle(save_name)


def apply_watermark(arr, func, trail=3, **kwargs):
    """Apply function over trailing dimensions of array and append watermark of result. Function should
    return string or list of strings.
    :param arr:
    :param func:
    :return:
    """
    n = np.prod(arr.shape[:-trail])
    arr_ = arr.reshape(-1, *arr.shape[-3:]).copy()
    new_arr = []
    for stack in arr_:
        try:
            annotation = lasagna.io.watermark(stack.shape[1:], func(stack), **kwargs)
        except ValueError:
            annotation = np.zeros(stack.shape[1:])
        new_arr += [np.concatenate((stack, annotation[None, :, :]))]

    new_shape = list(arr.shape)
    new_shape[-3] = -1  # expand to fill
    return np.array(new_arr).reshape(new_shape)


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
