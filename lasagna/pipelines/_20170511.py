from lasagna.imports import *
from lasagna.process import build_feature_table, feature_table

# name of lasagna folder and sheet in Lasagna FISH
worksheet = 'Mushroom Lasagna'
datasets = '20170511_96W-G105-A',
file_pattern = lasagna.io.default_file_pattern
channels = 'DAPI', 'FITC', 'Cy3', 'TexasRed', 'Cy5'
luts = GRAY, CYAN, GREEN, RED, MAGENTA
lasagna.io.default_luts = luts
display_ranges = ((1000, 65000),
                  (1000, 5000),
                  (1000, 13000),
                  (1000, 12000),
                  (1000, 16000))

celltracker_index = [0, 4]
DO_index = [0, 1, 2, 3, 4]

 
DO_thresholds = ( ('DO_pd_pL34', 5000)
                , ('DO_Puro', 5000)
                , ('DO_pd_pL33', 5000)
                , ('DO_ActB', 10000)
                , )

object_features = lasagna.process.default_object_features
peak_features =  {'max':  lambda r: r.max_intensity,
                  '20p':  lambda r: np.percentile(r.intensity_image.flat[:], 20),
                  'median': lambda r: np.percentile(r.intensity_image.flat[:], 50),
                  'min':  lambda r: r.min_intensity,
                  'mean': lambda r: r.mean_intensity}
 
all_index= (('cycle',   ('c0-DO', 'c1-5B1', 'c2-5B2', 'c3-5B3'))
           ,('channel', ('DAPI', 'FITC', 'Cy3', 'TxRed', 'Cy5')))

adapters = {'FITC': 'T', 'Cy3': 'G', 'TxRed': 'C', 'Cy5': 'A'}

offsets_5B2_AnchP_pL30 = [[0, 0], [0, 0], [5, -10], [0, -3], [8, -8]]
offsets_5B2_1          = [[0, 0], [0, 0], [2, -11], [0,  0], [4, -2]]

offsets_5B2_AnchP_pL30 = [[0, 0], [2, 0], [0, 0], [0, 0], [1, 0]]
offsets_5B2_1          = [[0, 0], [2, 0], [0, 0], [0, 0], [1, 0]]



def paths(file_pattern=file_pattern, depth=2, datasets=datasets):
    """Finds all files in the current directory that match the file pattern.
    Run from the lasagna data directory.
    If data is in ~/lasagna/20170305_96W-G078
    run this from ~/lasagna and filter the resulting pd.DataFrame.
    """
    include = 'stitched.tif|aligned.*tif|stitched.cut.tif'
    files = lasagna.io.find_files(include=include, depth=depth)

    matches, files_keep = [], []
    for f in files:
        match = re.match(file_pattern, f)
        if match:
            matches += [match.groupdict()]
            files_keep += [f]

    df = pd.DataFrame(matches)
    df['file'] = files_keep
    df = lasagna.io.well_to_row_col(df)
    df = df[df['dataset'].isin(datasets)]
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


def make_filename(dataset, mag, well, tag, cycle=None):
    if cycle:
        filename = '%s_%s_%s.%s.tif' % (mag, cycle, well, tag)
    else:
        filename = '%s_%s.%s.tif' % (mag, well, tag)
    return os.path.join(dataset, filename)


def crop(data):
    return data[..., 30:-30, 30:-30]
    

def do_alignment(df_files):
    """Aligns 20170428_96W-G078 data. DO and sequencing cycles are aligned using DAPI.
    Within each sequencing cycle, all channels except DAPI are aligned to FITC.
    """

    for well, df_ in df_files.groupby('well'):
        files = df_.sort_values('cycle')['file']
        data = []
        # if data doesn't have all 5 channels, need to fill in with zeros
        for f in files:
            offsets = 'none'
            data_ = read(f)
            if 'DO' in f:
                shape = 5, data_.shape[1], data_.shape[2]
                data2 = np.zeros(shape, np.uint16)
                data2[DO_index] = data_
                data += [data2]

            elif '5B2_1' in f:
                offsets = offsets_5B2_1
                data_ = [lasagna.io.offset(d, o) for d, o in zip(data_, offsets)]
                data += [np.array(data_)]

            elif '5B' in f or '3B' in f:
                # register sequencing channels to FITC
                offsets = [[0, 0]] + [o.astype(int) for o in register_images(data_[1:])]
                if max([a for b in offsets for a in b])**2 > 100: 
                    print 'weird channel offset', offsets, 'in', f
                data_ = [lasagna.io.offset(d, o) for d, o in zip(data_, offsets)]
                data += [np.array(data_)]
            elif 'celltracker' in f:
                shape = 5, data_.shape[1], data_.shape[2]
                data2 = np.zeros(shape, np.uint16)
                data2[celltracker_index] = data_
                data += [data2]

            print f, 'offsets', offsets

        # all data for this well is the same shape
        assert len(set([d.shape for d in data])) == 1 
        data = np.array(data)
        
        # register DAPI
        offsets = register_images([d[0] for d in data])
        data = [lasagna.io.offset(d, o.astype(int)) for d, o in zip(data, offsets)]
        data = np.array(data)

        # register channels
        # average over sequencing data
        meaned = data[:, 1:].mean(axis=1)
        offsets = np.array(register_images(meaned)).astype(int)
        print 'sequencing cycle offsets', offsets
        # assert False
        data = np.array([lasagna.io.offset(d_, off) for d_, off in zip(data, offsets)])
        
        dataset, mag, well = df_.iloc[0][['dataset', 'mag', 'well']]
        f = make_filename(dataset, mag, well, 'aligned')

        save(f, data, luts=luts, display_ranges=display_ranges)
        print well, f
        lasagna.io._get_stack._reset()


def peak_to_region(peak, threshold, n=5):
    """Expand peaks above threshold to nxn square neighborhood.
    Overlapping pixels go to highest peak.
    """
    selem = np.ones((n,n))
    peak = peak.copy()
    peak[peak<=threshold] = 0
    
    labeled = skimage.measure.label(peak)
    regions = lasagna.utils.regionprops(labeled, intensity_image=peak)
    # hack for rare peak w/ more than 1 pixel
    for r in regions:
        if r.area > 1:
            labeled[labeled==r.label] = np.array([r.label] + [0]*(r.area-1))


    # dilate labels so higher intensity regions win
    ranks = [r.max_intensity for r in regions]
    ranks = np.argsort(np.argsort(ranks)) + 1
    rank_to_label = np.argsort(ranks) + 1

    # switch to ranks for the dilation
    labeled[labeled>0] = ranks
    labeled = skimage.morphology.dilation(labeled, selem)
    # convert pixels labeled by rank back to the original label
    labeled[labeled>0] = rank_to_label[labeled[labeled>0] - 1]

    return labeled


def get_features(data, peaks, DO_thresholds=DO_thresholds):
    """ Uses peaks from DO only. 
    data: N x X x Y
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


def get_nuclear_features(dapi, nuclei):
    features = dict(object_features)
    features.update(peak_features)
    return feature_table(dapi, nuclei, features)


def peaks_to_DO_masks(peaks, DO_thresholds):
    DO_regions = {}
    for p, (s, t) in zip(peaks, DO_thresholds):
        DO_regions[s] = peak_to_region(p, t)
    return DO_regions


def pivot_for_glue(df, values, 
            columns=('channel', 'cycle'), 
            index=('file', 'source', 'label'), 
            extra_values=None):
    """ Pivot a dataframe of observations in long format for viewing in glue.
    """

    columns = list(columns)
    index = list(index)

    object_values = ['x', 'y', 'bounds', 'contour', 'well']
    if extra_values:
        object_values += list(extra_values)

    return lasagna.utils.long_to_wide(df, values, columns, index, object_values)


def show_DO(*args, **kwargs):
    luts = GRAY, GREEN, MAGENTA
    display_ranges=None
    return show(*args, luts=luts, display_ranges=display_ranges, **kwargs)


def show_al(*args, **kwargs):
    global luts
    global display_ranges
    luts = luts
    display_ranges=display_ranges
    return show(*args, luts=luts, display_ranges=display_ranges, **kwargs)


def save_al(*args, **kwargs):
    return save(*args, luts=luts, display_ranges=display_ranges, **kwargs)


def parse_landmarks(f):
    """Reads landmarks from ImageJ TurboReg.
    """
    lines = open(f).read().split('\n')
    titles = 'Refined source landmarks', 'Target landmarks'
    arr = []
    for title in titles:
        ix = lines.index(title)
        points = [[float(x) for x in l.split('\t')] for l in lines[ix+1:ix+4]]
        arr += [points]

    return np.array(arr)


def warp_from_landmarks(f_landmarks, img1, img2):
    from skimage.transform import warp, SimilarityTransform

    landmarks = parse_landmarks(f_landmarks)
    
    st = SimilarityTransform()
    st.estimate(landmarks[1], landmarks[0])

    inverse_map = np.array(landmarks)
    out = warp(img2, st, preserve_range=True)
    return out


def offset2D(data, offsets, ij=True):
    from skimage.transform import warp, SimilarityTransform
    h, w = data.shape[-2:]
    arr = []
    for frame, offset in zip(data.reshape(-1, h, w), offsets):
        # skimage.transform uses xy coordinates
        if ij:
            offset = offset[::-1]
        st = SimilarityTransform(translation=offset)
        arr += [warp(frame, st, preserve_range=True)]
    return np.array(arr).reshape(data.shape)


def remove_overlapped_DO(df, c1='TxRed', c2='Cy3', source='DO_Puro', ratio=0.9):
    """hack"""
    filt = df['source'] == source
    dfx = (pivot_for_glue(df[filt], 'peak')
            .set_index(['source', 'label', 'well'])
            .sortlevel(axis=1))

    get = lambda c: dfx.loc[:,pdx[c,'c0-DO']]
    remove = (get(c1) / get(c2)) > ratio
    df = df.set_index(['source', 'label', 'well'])
    cut = df.index.isin(remove[remove].index)
    return df[~cut].reset_index()


def unmix_cy3_txred(data, angle=(3.14*0.3), threshold=None):
    assert data.ndim == 3
    assert data.shape[0] == 2

    if threshold is None:
        d = dict(DO_thresholds)
        threshold = min([d['DO_Puro'], d['DO_pd_pL33']]) - 1

    data = data.copy()
    unmixed = unmix(data[0], data[1], angle=angle)
    unmixed[unmixed < 0] = 0
    unmixed[0] = denoise(unmixed[0], threshold=threshold)
    unmixed[1] = denoise(unmixed[1], threshold=threshold)
    return unmixed


def unmix(green, red, angle=3.14/4):
    """total hack, converts to float"""
    normalize = lambda x: x / (((x**2).sum())**0.5)
    
    d = np.array([green, red]).astype(float)
    n1 = normalize(np.array([1, 0.1]))
    n2 = [np.cos(angle), np.sin(angle)]

    R = np.vstack([n1, n2]).T
    R = np.linalg.inv(R)
    
    ans = R.dot(d.reshape(2, -1)).reshape(d.shape)
    return ans


def denoise(img, threshold=2500):
    from scipy.ndimage.filters import median_filter
    assert img.ndim == 2
    med = median_filter(img, size=3)
    mask = (img > threshold) & (med < threshold)
    img[mask] = threshold
    return img


def make_barcode_subset(data_collection, barcodes, label, color=None):
    from glue.core.roi import CategoricalROI
    from glue.core.subset import CategoricalROISubsetState

    categorical = CategoricalROI(categories=barcodes)
    subset_state = CategoricalROISubsetState(att='barcode', roi=categorical)
    subset_group = data_collection.new_subset_group( 
                                     subset_state=subset_state,
                                     label=label,
                                     # color=color
                                     )
    return subset_group


def call_barcodes_faster(df):
    x = df.pivot_table(index=['cycle', 'label', 'source'], columns='channel', values='rank_norm')
    signal = x.values()
    am = np.argsort(signal, axis=1)[:, ::-1]
    ranked_signal = np.sort(signal, axis=1)[:, ::-1]
    
    bases = np.array([adapters[c] for c in x.columns])

    x['call'] = bases[am[:,0]]

    x['Q'] = ((ranked_signal[:,0] - ranked_signal[:,1]) 
                                 / 
              (ranked_signal[:,0].astype(float)))


    y = x.reset_index().groupby(['label', 'source']).apply(lambda x: ''.join(x['call']))
    y.name = 'barcode'
    z = x.reset_index().groupby(['label', 'source'])['Q'].apply(lambda x: sorted(x)[1])
    assert all(z >= 0)
    z.name = 'min_quality'
    return df.set_index(['label', 'source']).join(y).join(z).reset_index()


# def launch_glue(df):
#     pass