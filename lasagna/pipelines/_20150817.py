import lasagna.io
import lasagna.config
import lasagna.process
import copy

display_ranges = ((500, 20000),
                  (500, 3500),
                  (500, 3500),
                  (500, 3500))

dataset = '20150817 6 round'


def setup():
    lasagna.config.paths = lasagna.io.Paths(dataset)
    lasagna.config.paths.add_analysis('raw', 'calibrated')

    config_path = lasagna.config.paths.full(lasagna.config.paths.calibrations[0])
    lasagna.config.calibration = (lasagna.process.Calibration(config_path,
                                                              dead_pixels_file='dead_pixels_empirical.tif'))
    c = copy.deepcopy(lasagna.config.calibration)
    c.calibration = None
    lasagna.config.calibration_short = c


def calibrate(row, paths, calibration):
    import numpy as np
    import lasagna.io

    channels = ['Cy3', 'Cy3', 'A594', 'Atto647']
    luts = [lasagna.io.BLUE, lasagna.io.GREEN, lasagna.io.RED, lasagna.io.MAGENTA]
    raw, calibrated = row['raw'], row['calibrated']
    raw_data = lasagna.io.read_stack(paths.full(raw))
    raw_data = np.array([calibration.fix_dead_pixels(frame) for frame in raw_data])
    fixed_data = np.array([calibration.fix_illumination(frame, channel=channel)
                           for frame, channel in zip(raw_data, channels)])
    lasagna.io.save_hyperstack(paths.full(calibrated), fixed_data, luts=luts)


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
            import lasagna.pipelines._20150817 as pipeline
            for ix, row in df_.iterrows():
                pipeline.calibrate(row, paths, calibration)
        async_results += [lview.apply_async(func, df)]

    return async_results


def initialize_engines(client):
    dview = client[:]
    dview.execute('import numpy as np')
    dview.execute('import lasagna.io')
    dview.execute('import lasagna.process')
    dview.execute('import lasagna.pipelines._20150817')

    dview['paths'] = lasagna.config.paths
    dview['calibration'] = lasagna.config.calibration_short



def my_func(df, y, paths):
    import numpy as np
    import lasagna.io, lasagna.process
    stitch = df.ix[0, 'calibrated'].replace('analysis/calibrated/raw',
                                            'stitch')
    stitch = paths.full(stitch)
    if not os.path.isdir(os.path.dirname(stitch)):
        os.makedirs(os.path.dirname(stitch))
        print 'made dir', os.path.dirname(stitch)

    data = np.array([lasagna.io.read_stack(paths.full(f))
                  for f in df['raw']])
    grid = np.sqrt(data.size / (4*1024*1024))
    data = data.reshape([grid,grid,4,1024,1024])

#     off = lasagna.process.stitch_grid(data[:,:,0,...], 925./1024)
#     y = lasagna.process.compress_offsets(off)


    data2 = np.array([lasagna.process.alpha_blend(data[:,:,i,:,:], y)
                      for i in range(data.shape[2])])

    lasagna.io.save_hyperstack(paths.full(stitch), data2)
    return 'saved', stitch