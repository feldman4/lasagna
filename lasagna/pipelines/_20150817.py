import lasagna.io
import lasagna.config
import lasagna.process

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

