import lasagna.io
import lasagna.config
import lasagna.process
import copy
import numpy as np
import os

display_ranges = ((500, 20000),
                  (500, 3500),
                  (500, 3500),
                  (500, 3500))

dataset = '20150817 6 round'

pipeline = None

channels = 'DAPI', 'Cy3', 'A594', 'Atto647'


def setup():
    """Construct Paths and Calibration objects for dataset. Copy a smaller version of Calibration object
    to be synced to remote engines.
    :return:
    """
    lasagna.config.paths = lasagna.io.Paths(dataset)
    lasagna.config.paths.add_analysis('raw', 'calibrated')

    config_path = lasagna.config.paths.full(lasagna.config.paths.calibrations[0])
    lasagna.config.calibration = (
    lasagna.process.Calibration(config_path, dead_pixels_file='dead_pixels_empirical.tif'))
    c = copy.deepcopy(lasagna.config.calibration)
    c.calibration = None
    lasagna.config.calibration_short = c


def table_from_nuclei(df, *args, **kwargs):
    df_ = lasagna.process.table_from_nuclei(df, *args, **kwargs)
    save_name = df['file'][0] + '.pkl'
    df_.to_pickle(lasagna.config.paths.export(save_name))
    return df_


# TODO check if single function can be called with lview.map_async(func, [arg1], ..., chunksize=10)
def calibrate(row, paths, calibration):
    channels = ['Cy3', 'Cy3', 'A594', 'Atto647']
    luts = [lasagna.io.BLUE, lasagna.io.GREEN, lasagna.io.RED, lasagna.io.MAGENTA]
    raw, calibrated = row['raw'], row['calibrated']
    raw_data = lasagna.io.read_stack(paths.full(raw))
    raw_data = np.array([calibration.fix_dead_pixels(frame) for frame in raw_data])
    fixed_data = np.array([calibration.fix_illumination(frame, channel=channel)
                           for frame, channel in zip(raw_data, channels)])
    lasagna.io.save_hyperstack(paths.full(calibrated), fixed_data, luts=luts)


def find_nuclei(row, block_size):
    M = lasagna.io.read_stack(lasagna.config.paths.full(row['stitch']))
    N = lasagna.process.get_nuclei(M[0, :, :], block_size=block_size)
    lasagna.io.save_hyperstack(lasagna.config.paths.full(row['nuclei']), N)


def calibrate_sge(dfs, lview):
    """Apply current calibration, with results saved according to current paths. Farm out jobs to
     provided load-balanced view.
    :param dfs:
    :param lview:
    :return:
    """
    async_results = []
    c_ = copy.deepcopy(lasagna.config.calibration)
    c_.calibration = None

    for df in dfs:
        def func(df_, paths=lasagna.config.paths, calibration=c_):
            for ix, row in df_.iterrows():
                pipeline.calibrate(row, paths, calibration)

        async_results += [lview.apply_async(func, df)]

    return async_results


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
                     for f in df['raw']])
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
