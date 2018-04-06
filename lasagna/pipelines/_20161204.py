from lasagna.imports import *
from lasagna.process import build_feature_table, feature_table

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

DO_slices = [2, 4]
DO_thresholds = ('DO_Puro', 1000), ('DO_ActB', 500)

object_features = lasagna.process.default_object_features
peak_features =  {'max':  lambda r: r.max_intensity,
                  '20p':  lambda r: np.percentile(r.intensity_image.flat[:], 20),
                  'median': lambda r: np.percentile(r.intensity_image.flat[:], 50),
                  'min':  lambda r: r.min_intensity,
                  'mean': lambda r: r.mean_intensity}

 
all_index= (('cycle',   ('c0-DO', 'c1-5B3', 'c2-5B1', 'c3-DO', 'c4-5BX'))
           ,('channel', ('DAPI', 'FITC', 'Cy3', 'TxRed', 'Cy5')))


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
        r'(?P<dataset>(?P<date>[0-9]{8}).*)[\/\\]' + \
        r'(?P<mag>[0-9]+X).' + \
        r'(?:(?P<cycle>[^_\.]*).*?(?:.*MMStack)?.)?' + \
        r'(?P<well>[A-H][0-9]|montage)' + \
        r'(?:\.(?P<tag>.*))*\.tif'


def add_row_col(df):
    df['row'] = [s[0] for s in df['well']]
    df['col'] = [s[1] for s in df['well']]
    return df


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
    
    files = find_files(include='stitched.tif|aligned.*tif')

    matches, files_keep = [], []
    for f in files:
        match = re.match(file_pattern, f)
        if match:
            matches += [match.groupdict()]
            files_keep += [f]

    df = pd.DataFrame(matches)
    df['file'] = files_keep
    df = add_row_col(df)
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


def peak_to_region(peak, threshold, n=5):
    """Expand peaks above threshold to nxn square neighborhood.
    Overlapping pixels go to highest peak.
    """
    selem = np.ones((n,n))
    peak = peak.copy()
    peak[peak<threshold] = 0
    
    labeled = skimage.measure.label(peak)
    regions = lasagna.utils.regionprops(labeled, intensity_image=peak)
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


def get_features(data, peaks, DO_thresholds=DO_thresholds):
    """ Uses peaks from DO only. 
    peaks: C x X x Y
    DO_thresholds: ((name, val),) * C
    """
    DO_masks = peaks_to_DO_masks(peaks, DO_thresholds)

    arr = []
    # loop over peak sources
    for source, mask in DO_masks.items():
        # peak object features
        objects = feature_table(mask, mask, object_features)
        # DO/sequencing features
        table = build_feature_table(data, mask, peak_features, all_index)
        table = objects.join(table)
        table['source'] = source
        arr += [table]
    return pd.concat(arr)


def peaks_to_DO_masks(peaks, DO_thresholds):
    DO_regions = {}
    for p, (s, t) in zip(peaks, DO_thresholds):
        DO_regions[s] = peak_to_region(p, t)
    return DO_regions


def tidy_to_long(df, values, extra_values=None):
    """ For viewing in glue
    """
    # values = ['max', 'min', '20p']
    # extra_values = ['well', 'source', 'label']
    columns = ['channel', 'cycle']
    index = ['file', 'source', 'label']

    df.set_index('file', 'channel')
    df_long = pd.pivot_table(df, columns=columns, 
               values=values, index=index)

    object_values = ['x', 'y', 'bounds', 'contour', 'well']
    if extra_values:
        object_values += extra_values

    df2 = df.drop_duplicates(index).set_index(index)
    for col in object_values:
        df_long[col] = df2[col]


    return df_long.reset_index()

