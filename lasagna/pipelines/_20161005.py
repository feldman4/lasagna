from lasagna.imports import *


# name of lasagna folder and sheet in Lasagna FISH
datasets = '20160914_96W-G064', '20160927_96W-G065'
# dead_pixels_file = 'calibration/dead_pixels_20151219_MAX.tif'
# tile_configuration = None

channels = 'DAPI', 'FITC', 'Cy3', 'TexasRed', 'Cy5'
luts = GRAY, CYAN, GREEN, RED, MAGENTA
display_ranges = ((500, 10000),
                  (500, 7000),
                  (500, 7000),
                  (500, 7000),
                  (500, 7000))

#  glueviz
name_map = {('all','bounds',1.): 'bounds',
            ('all', 'file', 1.): 'file',
            ('all', 'contour',1.):'contour',
            ('well', '', ''): 'well',
            ('row', '', ''): 'row',
            ('column', '', ''): 'column',
            ('all', 'x', 1.): 'x',
            ('all', 'y', 1.): 'y'}


def find_peaks(aligned, n=5):
    """At peak, max value in neighborhood. Returns max-min at peak, 0 otherwise.
    Doesn't guarantee peak has 1 pixel.
    """
    from scipy.ndimage import filters
    neighborhood_size = (1,)*(aligned.ndim-2) + (n,n)
    data_max = filters.maximum_filter(aligned, neighborhood_size)
    data_min = filters.minimum_filter(aligned, neighborhood_size)
    peaks = data_max - data_min
    peaks[aligned!=data_max] = 0
    
    # remove peaks close to edge
    mask = np.ones(peaks.shape, dtype=bool)
    mask[...,n:-n, n:-n] = False
    peaks[mask] = 0
    
    return peaks

def analyze_peaks(data, labeled):
    features = {'max': lambda r: r.max_intensity,
                '20p': lambda r: np.percentile( r.intensity_image.flat[:], 20),
                'x'  : lambda r: r.centroid[0],
                'y'  : lambda r: r.centroid[1]}

    index = (('channel', ('pdPuro', 'pdActB', 'T', 'G', 'C', 'A')),)
    arr = []
    for mask, source in zip(labeled, index[0][1]):
        table = lasagna.process.build_feature_table(data, mask, features, index)
        table['source'] = source
        arr += [table]
        print source

    return pd.concat(arr)
    

def analyze_objects(labeled):
    sources = 'pdPuro', 'pdActB', 'T', 'G', 'C', 'A'
    features = lasagna.process.default_object_features
    arr = []
    for mask, source in zip(labeled, sources):
        table = lasagna.process.feature_table(mask, mask, features)
        table['source'] = source
        arr += [table]
        print source
    df_objects = pd.concat(arr)
    
    return df_objects




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



def find_files(start='', include='', exclude=''):
    # or require glob2
    files =  glob(os.path.join(start, '*.tif'))
    files += glob(os.path.join(start, '*/*.tif'))
    files += glob(os.path.join(start, '*/*/*.tif'))
    files += glob(os.path.join(start, '*/*/*/*.tif'))
    files_ = []
    for f in files:
        if include and not re.findall(include, f):
            continue
        if exclude and re.findall(exclude, f):
            continue
        files_ += [f]
    return files_


def paths():
    
    files = find_files(exclude='tmp|gt')

    pat =   '(?P<dataset>.*)\/' + \
            '(?P<mag>[0-9]+X).' + \
            '(?:(?P<round>.*?[^_])(?:.MMStack)?.)?' + \
            '(?P<well>[A-H][0-9]|montage)' + \
            '(?:\.(?P<tag>.*))*\.tif'

    matches = [re.match(pat, f).groupdict() for f in files]

    df = pd.DataFrame(matches)
    df['file'] = files
    
    return df



def do_alignment():
    """Aligns 20160927 data
    """
    df = paths()

    df2 = df.query('dataset=="20160927_96W-G065" & mag=="20X"')

    df2 = df2.groupby('well').filter(lambda x: len(x)==2)

    # alignment
    luts = GRAY, CYAN, GREEN, RED, MAGENTA
    dr = ((400, 7000),) + ((500, 4000),)*4

    for well, df_ in df2.groupby('well'):
        do, b3 = sorted(df_['file'])

        d1 = read(b3)
        d0 = np.zeros(d1.shape, d1.dtype)
        d0[[0, 2, 4]] = read(do)

        offsets = lasagna.process.register_images([d0[0], d1[0]])

        arr = []
        for d, offset in zip((d0, d1), offsets):
            arr += [lasagna.io.offset(d, offset.astype(int))[None]]

        data = np.concatenate(arr)

        f = b3.replace('base3_1_MMStack_', '').replace('stitched', 'aligned')

        save(f, data[...,20:-20, 20:-20], luts=luts, display_ranges=dr)
        print well

def do_peak_detection():
    df = paths()
    for f in df.query('tag=="aligned"')['file']:
        aligned = read(f)
        peaks = pipeline.find_peaks(aligned)
        f2 = f.replace('aligned', 'aligned.peaks')
        if f != f2:
            save(f2, peaks, luts=luts, compress=1)
            print f
        else:
            print 'wtf',


def peak_to_region(peak, data, threshold=2000, n=5):
    """Expand peaks above threshold to nxn square neighborhood.
    Overlapping pixels go to highest peak.
    """
    selem = np.ones((n,n))
    peak = peak.copy()
    peak[peak<threshold] = 0
    
    labeled = skimage.measure.label(peak)
    regions = lasagna.utils.regionprops(labeled, intensity_image=data)
    # hack for rare peak w/ more than 1 pixel
    for r in regions:
        if r.area > 1:
            labeled[labeled==r.label] = np.array([r.label] + [0]*(r.area-1))

    # dilate labels so higher intensity regions win
    fwd = [r.max_intensity for r in regions]
    fwd = np.argsort(np.argsort(fwd)) + 1
    rev = np.argsort(fwd)

    labeled[labeled>0] = fwd

    labeled = skimage.morphology.dilation(labeled, selem)
    labeled[labeled>0] = rev[labeled[labeled>0] - 1]

    return labeled


