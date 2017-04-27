from lasagna.imports import *
from lasagna.process import build_feature_table, feature_table

# name of lasagna folder and sheet in Lasagna FISH
datasets = '20170410_96W-G088'
file_pattern = lasagna.io.default_file_pattern
channels = 'DAPI', 'FITC', 'Cy3', 'TexasRed', 'Cy5'
luts = GRAY, CYAN, GREEN, RED, MAGENTA
lasagna.io.default_luts = luts
display_ranges = ((500, 50000),
                  (500, 3000),
                  (500, 3000),
                  (500, 3000),
                  (500, 3000))

celltracker_index = [0, 4]
DO_index = [0, 2, 4]
DO_thresholds = ('DO_Puro', 1000), ('DO_ActB', 500)

object_features = lasagna.process.default_object_features
peak_features =  {'max':  lambda r: r.max_intensity,
                  '20p':  lambda r: np.percentile(r.intensity_image.flat[:], 20),
                  'median': lambda r: np.percentile(r.intensity_image.flat[:], 50),
                  'min':  lambda r: r.min_intensity,
                  'mean': lambda r: r.mean_intensity}
 
all_index= (('cycle',   ('c0-DO', 'c1-5B1'))
           ,('channel', ('DAPI', 'FITC', 'Cy3', 'TxRed', 'Cy5')))


def paths(file_pattern=file_pattern, depth=2):
    """Finds all files in the current directory that match the file pattern.
    Run from the lasagna data directory.
    If data is in ~/lasagna/20170305_96W-G078
    run this from ~/lasagna and filter the resulting pd.DataFrame.
    """
    files = lasagna.io.find_files(include='stitched.tif|aligned.*tif', depth=depth)

    matches, files_keep = [], []
    for f in files:
        match = re.match(file_pattern, f)
        if match:
            matches += [match.groupdict()]
            files_keep += [f]

    df = pd.DataFrame(matches)
    df['file'] = files_keep
    df = lasagna.io.well_to_row_col(df)
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
    """Aligns 20170305_96W-G078 data. DO and sequencing cycles are aligned using DAPI.
    Within each sequencing cycle, all channels except DAPI are aligned to FITC.
    """

    for well, df_ in df_files.groupby('well'):
        files = df_.sort_values('cycle')['file']
        data = []
        # if data doesn't have all 5 channels, need to fill in with zeros
        for f in files:
            data_ = read(f)
            if 'DO' in f:
                shape = 5, data_.shape[1], data_.shape[2]
                data2 = np.zeros(shape, np.uint16)
                data2[DO_index] = data_
                data += [data2]
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

        # all data for this well is the same shape
        assert len(set([d.shape for d in data])) == 1 
        data = np.array(data)
        
        # register DAPI
        offsets = register_images([d[0] for d in data])
        data = [lasagna.io.offset(d, o.astype(int)) for d, o in zip(data, offsets)]
        data = np.array(data)

        # register channels
        data = data
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


def find_nearest(centers, points):
    """Assign points to the nearest center. Returns (distance, assignment).
    Uses euclidean distance.
    """
    distances, labels = scipy.spatial.KDTree(centers).query(points)
    return distances, labels


def to_ij_contour(contours, bounds):
    """Automatically adjusts and packs the contours for viewing
    in ImageJ.
    """
    contours = np.array([x + y[:2] for x,y in zip(contours, bounds)])
    packed = lasagna.utils.pack_contours(contours)
    return packed


def mark_blobs_by_nuclei(df_blobs, df_nuclei):
    """Adds nucleus_distance and nucleus_label columns to df_blobs. Blob to nuclei
    distances are calculated using centroids.
    """
    def get_xy(df, index=('label', 'well', 'source')):
        cols = [c for c in index if c in df]
        df = df.set_index(cols)
        
        df = df[~df.index.duplicated(keep='first')]
        return df[['x', 'y']]

    df = df_blobs
    df_n = df_nuclei
    arr = []
    for well, df_ in df.groupby('well'):

        nucleus_c = get_xy(df_n[df_n['well'] == well])
        blob_c    = get_xy(df_)

        distances, labels = find_nearest(nucleus_c, blob_c)
        xs = np.array(nucleus_c.reset_index()['label'])


        blob_c['nucleus_distance'] = distances
        blob_c['nucleus_label'] = xs[labels]

        df2 = df_.set_index(['label', 'well', 'source'])

        df2['nucleus_distance'] = blob_c['nucleus_distance']
        df2['nucleus_label']    = blob_c['nucleus_label']
        arr += [df2.reset_index()]

    df = pd.concat(arr)
    return df


def check_max(row):
    from lasagna.glueviz import grid_view
    
    grid = grid_view([row['file']], [row['bounds']], padding=0)
    
    d = dict(all_index)
    channel_ix = d['channel'].index(row['channel'])
    cycle_ix   = d['cycle'].index(row['cycle'])
    
    data = grid[0, cycle_ix, channel_ix]
    
    mask = row['mask'].mask
    
    assert data[mask].max() == row['max']


class ImageGrid(object):
    default_title = 'ImageGrid'
    def __init__(self, grid, mask, title=default_title):
        """
        ig = ImageGrid(df['file'][::20], df3['bounds'][::20])
        ig.get_selected()
        """
        self.imp = show_al(montage(grid), title=title)
        self.grid, self.grid_mask = grid, mask

        self.labels = [x - 1 for x in range(mask.max() + 1)]

    @staticmethod
    def from_files_bounds(files, bounds, title=default_title, padding=0):
        grid, mask = grid_view(files, bounds, padding=padding, with_mask=True)
        return ImageGrid(grid, mask, title=title)

    @staticmethod
    def from_dataframe(df, title=default_title, padding=0):
        files = df['file']
        bounds = df['bounds']
        ig = ImageGrid.from_files_bounds(files, bounds, title=title, padding=padding)
        ig.labels = df['label']
        return ig
        
    def get_selected(self):
        """Gets indices corresponding to the current PointROI.
        """
        self.labels = list(self.labels) # auto-convert 
        x, y = self.get_point_xy(self.imp)
        selected = montage(self.grid_mask)[y, x]
        
        return [self.labels[i - 1] if i != 0 else -1 for i in selected ]

    def __iter__(self):
        for i in range(len(self.grid)):
            yield self.__getitem__(i)

    def __getitem__(self, key):
        """Returns image without padding.
        """
        grid, mask = self.grid[key], self.grid_mask[key]
        i, j = [max(x) for x in np.where(mask)]
        return grid[..., :i+1, :j+1]
     
    @staticmethod
    def get_point_xy(imp):
        poly = imp.roi.getPolygon()
        x = list(poly.xpoints)
        y = list(poly.ypoints)
        return x, y


def overlay_df(imp, df, awt_color=None, overlay=None):
    """Add colored contours to overlay. Need to keep stuff around. Would be 
    more convenient to wrap imp in a python class with added functionality.
    """
    assert lasagna.config.j is not None
    j = lasagna.config.j

    if awt_color is None:
        awt_color = j.java.awt.Color.GREEN

    packed = to_ij_contour(df['contour'], df['bounds'])
    
    if overlay:
        n = len(overlay.toArray())
        overlay = j.overlay_contours(packed, overlay=overlay)
        rois = range(n, n + len(df))
        j.set_overlay_contours_color(rois, imp, awt_color)
    else:
        overlay = j.overlay_contours(packed, overlay=overlay)
        # sets color for everything
        overlay.strokeColor = awt_color
    imp.updateAndDraw()
    return overlay


def textures(image):
    """A few off-the-shelf features.
    """
    gcm = skimage.feature.greycomatrix
    gcp = skimage.feature.greycoprops
    image = skimage.img_as_ubyte(image)
    
    # make the co-occurence matrix
    P = gcm(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4]
            , normed=True, symmetric=True)
    
    # derive the chosen features
    features = 'dissimilarity', 'homogeneity', 'contrast'
    x = pd.Series([gcp(P, f)[0, 0] for f in features], index=features)
    x['max'] = image.max()
    x['mean'] = image[image>0].mean()
    x['area'] = (image>0).sum()
    
    return x

def get_masked_dapi(row):
    """Convenience.
    """
    from lasagna.glueviz import grid_view
    grid = grid_view([row['file']], [row['bounds']],padding=0)
    dapi = grid[0, 0, 0]
    dapi[~row['mask'].mask] = 0
    return dapi