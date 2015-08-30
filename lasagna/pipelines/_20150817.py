import lasagna.io
import lasagna.config
import lasagna.process
import lasagna.utils
import copy
import numpy as np
import skimage.transform

display_ranges = ((500, 20000),
                  (500, 3500),
                  (500, 3500),
                  (500, 3500))

dataset = '20150817 6 round'

pipeline = None

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
    lasagna.io.save_hyperstack(lasagna.config.paths.full(calibrated), fixed_data, luts=luts)


def align(files, save_name, n=500, trim=150):
    """Align data using first channel (DAPI). Register corners using FFT, build similarity transform,
    warp, and trim edges.
    :param files: files to align
    :param save_name:
    :param n: width of corner alignment window
    :param trim: # of pixels to trim in from each size
    :return:
    """
    data = [lasagna.io.read_stack(f) for f in files]
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
    s = [0]*(data.ndim - 2) + [slice(None)]*2
    nuclei = lasagna.process.get_nuclei(data[s], block_size=block_size)
    lasagna.io.save_hyperstack(lasagna.config.paths.full(row['nuclei']), nuclei)


def table_from_nuclei(df, *args, **kwargs):
    df_ = lasagna.process.table_from_nuclei(df, *args, **kwargs)
    save_name = df['file'][0] + '.pkl'
    df_.to_pickle(lasagna.config.paths.export(save_name))
    return df_


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
def stitch(df, offsets=None, overlap=925. / 1024):
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
