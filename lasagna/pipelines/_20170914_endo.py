from lasagna.imports import *

dr = [(14000, 29000), (1000, 10000)
      , (5000, 23000), (2000, 17000)
      , (10000, 18000)]
luts = GRAY, GRAY, GREEN, RED, MAGENTA

def get_tags(f):
    xs = f.split('/')[-1].split('_')
    
    minute = re.findall('_(..min|.min)', f)[0]
    line = re.findall('(lineA|lineB)', f)[0]
    tfn = 'Tfn' in f
    egf = 'EGF' in f
    acid = True
    if 'noacid' in f:
        acid = False
    stains = '(EEA1_LAMP1|CLC)'
    stain = re.findall(stains, f)[0]
    dyna = 'dyna'
    
    tags = {  'tfn': tfn
            , 'egf': egf
            , 'acid': acid
            , 'minute': minute
            , 'line': line
            , 'stain': stain
           }

    tags['channel_order'] = get_channel_order(tags)

    return tags

def get_channel_order(tags):
    candidates = \
        (('CLC', 'EGF', 'Tfn'),
        ('LAMP1', 'EEA1', 'Tfn'),
        ('LAMP1', 'EGF', 'EEA1'),
        ('CLC', 'EGF', 'dummy'),
        ('CLC', 'dummy', 'Tfn'))

    candidates = {tuple(sorted(c)): c for c in candidates}

    channels = []
    if tags['tfn']:
        channels += ['Tfn']
    if tags['egf']:
        channels += ['EGF']
    channels += tags['stain'].split('_')

    channels = candidates[tuple(sorted(channels))]
    return ('BF', 'DAPI') + channels

def get_info(files):
    tags = [get_tags(f) for f in files]
    df_info = pd.DataFrame(tags)
    df_info['file'] = files
    data_all = np.array([read(f) for f in files])
    xs = data_all[:,0,:,:].mean(axis=(1,2))
    df_info['phase'] = xs > 5000

    return df_info

def segment_hela(dapi, brightfield):
    resize_shape = 256, 256
    d = skimage.transform.resize(dapi, resize_shape)
    bf = skimage.transform.resize(brightfield, resize_shape)
    
    n = lasagna.process.find_nuclei(d, smooth=3)
    mask = brightfield_to_mask(bf)
    c = lasagna.process.find_cells(n, mask)
    
    n[c==0] = 0
    n, _, _ = skimage.segmentation.relabel_sequential(n)
    c, _, _ = skimage.segmentation.relabel_sequential(c)
    
    nuclei = skimage.transform.resize(n, dapi.shape, order=0, preserve_range=True)
    cells = skimage.transform.resize(c, dapi.shape, order=0, preserve_range=True)
    return nuclei.astype(int), cells.astype(int)

def brightfield_to_mask(bf):
    l = skimage.filters.laplace(bf.astype(float))
    l = np.abs(l) / l.max()
    l_ = skimage.filters.gaussian(l, sigma=5)
    return l_ > skimage.filters.threshold_li(l_)

def rescale_image(image, low=0, high=23000):
    in_range = low, high
    image = skimage.exposure.rescale_intensity(image, in_range=in_range)
    image = skimage.img_as_ubyte(image)
    return image

def gcm_props(region, prop='contrast'):
    from skimage.feature import greycomatrix
    from skimage.feature import greycoprops

    image = region.intensity_image
    image = rescale_image(image)
    distances = 0, 2, 5
    angles = 0, np.pi/2

    P = greycomatrix(image, distances, angles, 
                     normed=True, symmetric=True)
    # ignore mask pixels
    P[0, :, :, :] = 0
    P[:, 0, :, :] = 0
    return greycoprops(P, prop).flatten()

def make_table(file_data, file_segment, channels):

    index = ('channel', channels),

    data = read(file_data)
    nuclei, cells = read(file_segment)
    stack = data[2:5]
    mask = cells

    features = lasagna.process.default_intensity_features
    features.update(lasagna.process.default_object_features)
    table = lasagna.process.build_feature_table(stack, mask, features, index)

    # stack features
    features = {'label': lambda r: r.label}
    for i,j in ((2, 3), (3, 4)):
        column = 'corr_%s_%s' % (channels[i], channels[j])
        features.update({column: partial(correlated_pixels, i=i, j=j)})

    table_ = feature_table_stack(data, cells, features)
    table_ = expand_columns(table_)
    table = table.join(table_.set_index('label'), on='label')

    table['file'] = file_data

    return table

def correlated_pixels(region, i, j, threshold_corr=15):
    """
    """
    data = region.intensity_image_full
    # only pixels in region
    a = data[i][region.image]
    b = data[j][region.image]

    corr = spearman_corr_bsub(a, b)
    mask_corr = corr > threshold_corr

    # count fraction of EGF pixels above threshold that are also
    # correlated with EEA1
    threshold_a = skimage.filters.threshold_otsu(a)
    threshold_b = skimage.filters.threshold_otsu(b)
    mask_a = a > threshold_a
    mask_b = b > threshold_b
    
    # fraction of pixels
    c_a = (mask_a & mask_corr).sum() / (1. + mask_a.sum())
    c_b = (mask_b & mask_corr).sum() / (1. + mask_a.sum())

    return c_a, c_b

def feature_table_stack(data, labels, features, global_features=None):
    """Apply functions in features to regions in data specified by
    integer labels.
    """
    regions = regionprops_stack(labels, intensity_image=data)
    results = {feature: [] for feature in features}
    for region in regions:
        for feature, func in features.items():
            results[feature] += [func(region)]
    if global_features:
        for feature, func in global_features.items():
            results[feature] = func(data, labels)
    return pd.DataFrame(results)

def expand_columns(df):
    """Expand all columns containing an iterable of fixed length.
    """
    arr = [expand_column(df[c]) for c in df]
    return pd.concat(arr, axis=1)

def expand_column(x):
    """Returns a dataframe with expanded columns.
    """
    try:
        arr = [len(a) for a in x]
        n = arr[0]
        if all(a == n for a in arr):
            columns = [x.name + '_%d' % i for i in range(n) ]
            return pd.DataFrame(zip(*x), index=columns).T
        else:
            # print 'expansion failed, not all elements of length %d' % n
            return [x]
    except TypeError:
        return pd.DataFrame(x)

def spearman_corr(a, b):

    a = a.astype(float)
    b = b.astype(float)

    x = (a - a.mean()) * (b - b.mean())
    x = x / (a.std() * b.std())
    return x 

def bsub(x, diameter=20):
    from scipy.ndimage.filters import minimum_filter
    return x - minimum_filter(x, size=diameter)

def spearman_corr_bsub(a, b, diameter=20):
    return spearman_corr(bsub(a, diameter=diameter), bsub(b, diameter=diameter))

def regionprops_stack(label, intensity_image):
    """Supplement skimage.measure.regionprops with additional field containing full intensity image in
    bounding box (useful for filtering).
    :param args:
    :param kwargs:
    :return:
    """
    import skimage.measure

    regions = skimage.measure.regionprops(label)
    for region in regions:
        b = region.bbox
        region.intensity_image_full = intensity_image[..., b[0]:b[2], b[1]:b[3]]

    return regions

def do_czi_tiff(files):
    """files = glob(home + '*czi')
    """
    for f in files:
        czifile.czi2tif(f)
        data = read(f + '.tif')
        save(f, data, display_ranges=dr, luts=luts)

def do_segment(files):
    """files = glob(home + '*czi.tif')
    """
    for f in files:
        data = read(f)
        bf, dapi = data[:2]
        n, c = pipeline.segment_hela(dapi, bf)
        
        f_ = f.replace('.tif', '.segment.tif')
        save(f_, np.array([n, c]), luts=(GLASBEY, GLASBEY), compress=1)
        print f

def get_5chan_files(files):
    """files = glob(home + '*czi.tif')
    """
    data = [read(f) for f in files]
    weird_files = [f for d,f in zip(data, files) if d.shape[0] == 4]
    return sorted(set(files) - set(weird_files))

def file_segment(f):
    return f.replace('.tif', '.segment.tif')