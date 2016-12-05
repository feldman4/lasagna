from lasagna.imports import *
from lasagna.process import default_object_features

# name of lasagna folder and sheet in Lasagna FISH
datasets = '20161203_96W-G073',
# dead_pixels_file = 'calibration/dead_pixels_20151219_MAX.tif'
# tile_configuration = None

channels = 'DAPI', 'FITC', 'Cy3', 'TexasRed', 'Cy5'
luts = GRAY, CYAN, GREEN, RED, MAGENTA
lasagna.io.default_luts = luts
display_ranges = ((500, 50000),
                  (500, 3000),
                  (500, 3000),
                  (500, 3000),
                  (500, 3000))

DO_index = [0, 2, 4]
DO_thresholds = 1000, 500

peak_features =  {'max':  lambda r: r.max_intensity,
                  '20p':  lambda r: np.percentile(r.intensity_image.flat[:], 20),
                  'median': lambda r: np.percentile(r.intensity_image.flat[:], 50),
                  'min':  lambda r: r.min_intensity,
                  'mean': lambda r: r.mean_intensity}


DO_index = (('cycle',   ('c0-DO',))
           ,('channel', ('DO_Puro', 'DO_ActB')))

IS_index = (('cycle',   ('c1-5B3', 'c2-5B1'))
           ,('channel', ('T-FITC', 'G-Cy3', 'C-TxRed', 'A-Cy5')))
                  
                  
#  glueviz
name_map = {('all','bounds',1.): 'bounds',
            ('all', 'file', 1.): 'file',
            ('all', 'contour',1.):'contour',
            ('well', '', ''): 'well',
            ('row', '', ''): 'row',
            ('column', '', ''): 'column',
            ('all', 'x', 1.): 'x',
            ('all', 'y', 1.): 'y'}


file_pattern = \
        r'(?P<dataset>.*)[\/\\]' + \
        r'(?P<mag>[0-9]+X).' + \
        r'(?:(?P<cycle>[^_\.]*).*?(?:.*MMStack)?.)?' + \
        r'(?P<well>[A-H][0-9]|montage)' + \
        r'(?:\.(?P<tag>.*))*\.tif'


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



def analyze_peaks(data, labeled, index):
    features = {'max': lambda r: r.max_intensity,
                '20p': lambda r: np.percentile( r.intensity_image.flat[:], 20),
                'x'  : lambda r: r.centroid[0],
                'y'  : lambda r: r.centroid[1]}

    arr = []
    table = lasagna.process.build_feature_table(data, mask, features, index)

    return pd.concat(arr)
    

def analyze_objects(labeled, sources, features=default_object_features):
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
    
    files = find_files(include='20161203.*(((20X_c[0-9]).*stitched.tif)|(20X_.*aligned.*tif))')

    matches = []
    for f in files:
        match = re.match(file_pattern, f)
        if match:
            matches += [match.groupdict()]

    df = pd.DataFrame(matches)
    df['file'] = files
    
    return df

def make_filename(dataset, mag, well, tag, cycle=None):
    if cycle:
        filename = '%s_%s_%s.%s.tif' % (mag, cycle, well, tag)
    else:
        filename = '%s_%s.%s.tif' % (mag, well, tag)
    return os.path.join(dataset, filename)

def crop(data):
    return data[..., 30:-30, 30:-30]
    
def do_alignment(df_files):
    """Aligns 20161203_96W-G073 data. DO and sequencing cycles are aligned using DAPI.
    Within each sequencing cycle, all channels except DAPI are aligned to FITC.
    """

    for well, df_ in df_files.groupby('well'):
        files = df_.sort_values('cycle')['file']
        data = []
        for f in files:
            data_ = read(f)
            if 'DO' in f:
                shape = 5, data_.shape[1], data_.shape[2]
                data2 = np.zeros(shape, np.uint16)
                data2[DO_index] = data_
                data += [data2]
            elif '5B' in f or '3B' in f:
                # register sequencing channels to FITC
                # data_reg = data_.copy()
                # data_reg[data_reg > 2500] = 2500
                offsets = [[0, 0]] + [o.astype(int) for o in register_images(data_[1:])]
                if max([a for b in offsets for a in b])**2 < 100: 
                    print 'weird channel offset', offsets, 'in', f
                data_ = [lasagna.io.offset(d, o) for d, o in zip(data_, offsets)]
                data += [np.array(data_)]
        assert len(set([d.shape for d in data])) == 1 # all data for this well is the same shape
        data = np.array(data)
        
        # register DAPI
        offsets = register_images([d[0] for d in data])
        data = [lasagna.io.offset(d, o.astype(int)) for d, o in zip(data, offsets)]
        data = np.array(data)

        # register channels
        data = data
        dataset, mag, well = df_.iloc[0][['dataset', 'mag', 'well']]
        f = make_filename(dataset, mag, well, 'aligned')
        save(f, crop(data), luts=luts, display_ranges=display_ranges)
        print well, f
        lasagna.io._get_stack._reset()
        

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

