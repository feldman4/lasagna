from lasagna.imports import *
import skimage.io
from decorator import decorator
from scipy.ndimage.filters import gaussian_laplace

def register_and_offset(images, registration_images=None, verbose=False):
    if registration_images is None:
        registration_images = images
    offsets = register_images(registration_images)
    if verbose:
        print np.array(offsets)
    aligned = [lasagna.io.offset(d, o) for d,o in zip(images, offsets)]
    return np.array(aligned)

def align(data, verbose=False):
    # initial cycle alignment
    if verbose:
        print 'initial cycle alignment'
    data2 = register_and_offset(data, data[:,0], verbose=verbose)
    # second cycle alignment
    means = data2[:,1:].mean(axis=1)
    if verbose: 
        print 'second cycle alignment'
    data3 = register_and_offset(data2, means, verbose=verbose)
    # sequencing channel alignment
    alignment_order = [2,1,3,4]
    data4 = data3.copy()
    for cycle in range(data.shape[0]):
        if verbose: 
            print 'sequencing channel alignment, cycle', cycle
        s = np.s_[cycle, alignment_order]
        data4[s] = register_and_offset(data3[s], data3[s], verbose=verbose)
    
    return data4 

def align_(data, verbose=False):
    cycles, channels, h, w = data.shape
    # initial cycle alignment
    if verbose:
        print 'initial cycle alignment'
    data2 = register_and_offset(data, data[:,0], verbose=verbose)
    
    # sequencing channel alignment to DO
    sequencing = data2[1:, 1:].reshape([-1, h, w])
    reference = data2[0, 1:].mean(axis=0)
    to_align = np.array([reference] + list(sequencing))
    
    if verbose:
        print 'sequencing channel alignment to DO'

    aligned = register_and_offset(to_align, to_align, verbose=verbose)
    data2[1:, 1:] = aligned[1:].reshape([cycles - 1, channels - 1, h, w])
    
    return data2

def concatMap(f, xs):
    return sum(map(f, xs), [])

def tile(arr, n, m, pad=None):
    """Divide a stack of images into tiles of size m x n. If m or n is between 
    0 and 1, it specifies a fraction of the input size. If pad is specified, the
    value is used to fill in edges, otherwise the tiles may not be equally sized.
    Tiles are returned in a list.
    """
    assert arr.ndim > 1
    h, w = arr.shape[-2:]
    # convert to number of tiles
    m = np.ceil(h / m) if m >= 1 else np.round(1 / m)
    n = np.ceil(w / n) if n >= 1 else np.round(1 / n)
    m, n = int(m), int(n)

    if pad is not None:
        pad_width = (arr.ndim - 2) * ((0, 0),) + ((0, -h % m), (0, -w % n))
        arr = np.pad(arr, pad_width, 'constant', constant_values=pad)
        print arr.shape

    tiled = np.array_split(arr, m, axis=-2)
    tiled = concatMap(lambda x: np.array_split(x, n, axis=-1), tiled)
    return tiled

def laplace_only(data, *args, **kwargs):
    from skimage.filters import laplace
    h, w = data.shape[-2:]
    arr = []
    for frame in data.reshape((-1, h, w)):
        arr += [laplace(frame, *args, **kwargs)]
    return np.array(arr).reshape(data.shape)

def log_ndi(data, *args, **kwargs):
    """Apply laplacian of gaussian to each image in a stack of shape
    (..., I, J). 
    Extra arguments are passed to scipy.ndimage.filters.gaussian_laplace.
    """
    from scipy.ndimage.filters import gaussian_laplace
    h, w = data.shape[-2:]
    arr = []
    for frame in data.reshape((-1, h, w)):
        arr += [gaussian_laplace(frame, *args, **kwargs)]
    return np.array(arr).reshape(data.shape)

@decorator
def applyXY(f, arr, *args, **kwargs):   
    """Apply a function that expects 2D input to the trailing two
    dimensions of an array.
    """
    h, w = arr.shape[-2:]
    reshaped = arr.reshape((-1, h, w))

    arr_ = []
    for frame in reshaped:
        arr_ += [f(frame, *args, **kwargs)]
    return np.array(arr_).reshape(arr.shape)

laplace_ndi = applyXY(gaussian_laplace)
    
def log_ndi(arr, sigma=1, **kwargs):
    arr_ = -1 * laplace_ndi(arr.astype(float), sigma, **kwargs)
    arr_[arr_ < 0] = 0
    arr_ /= arr_.max()
    return skimage.img_as_uint(arr_)

CGRM = (np.array([CYAN, GREEN, RED, MAGENTA])
       .reshape(4,3,256)
       [:,:,::4]
       .transpose([1, 0, 2])
       .reshape(3*256))

def map_CGRM(stack):
    assert stack.shape[0] == 4
    assert stack.dtype == np.float64
    arr = np.zeros_like(stack[0])
    for i, frame in enumerate(stack):
        frame_ = frame * 0.25
        frame_[frame_>0] += i*0.25
        arr += frame_
    return arr

def map_CGRM_jpg(stack, invert=True):
    """Provide input in range (0, 1).
    """
    assert stack.shape[0] == 4
    rgbs = (0, 0.5, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1)

    if invert:
        arr = np.ones_like(stack)[:3]
        for frame, rgb in zip(stack, rgbs):
            rgb = 1 - np.array(rgb)
            arr -= frame[None] * rgb[:,None,None]
        arr[:, stack.sum(axis=0) == 0] = 0
        return arr
    else:
        arr = np.zeros_like(stack)[:3]
        for frame, rgb in zip(stack, rgbs):
            rgb = np.array(rgb)
            arr += frame[None] * rgb[:,None,None]
        return arr

def add_overlay(imp, call, overlay=None, x_offset=0, y_offset=0, alpha=1):
    j = lasagna.config.j
    x = map_CGRM_jpg(call).transpose([1, 2, 0])
    png = os.path.join(lasagna.config.fiji_target, 'overlay.png')
    skimage.io.imsave(png, x)
    imp_ = j.ij.ImagePlus(png)
    ip = imp_.getChannelProcessor()
    roi = j.ij.gui.ImageRoi(x_offset, y_offset, ip)
    roi.setZeroTransparent(True) 
    roi.setOpacity(alpha)
    if overlay is None:
        imp.setOverlay(roi, j.java.awt.Color.RED, 0, j.java.awt.Color.RED)
        overlay = imp.getOverlay()
    else:
        overlay.add(roi, 'overlay')
    return overlay

@applyXY
def contrast(arr, top=1, q1=75, q2=99.9, ceiling=0.25, verbose=False):
    """Apply skimage.exposure.rescale_intensity based on quantiles.
    Pixels values are rescaled such that values at or below q1 are set to 0, 
    and other values are set to ((x - q1_) / (q2_ - q1_)) * ceiling, 
    with values
    being clipped to a maximum of 65535.
    """
    from skimage.exposure import rescale_intensity
    q1_ = np.percentile(arr, q1)
    q2_ = np.percentile(arr, q2)
    q1_, q2_, ceiling = float(q1_), float(q2_), float(ceiling)
    if verbose:
        print q1_, q2_
    # scaled in range 0 .. 1 .. (max is greater than 1)
    arr_ = (arr - q1_) / (q2_ - q1_)
    # clip at maximum for dtype / ceiling
    arr_ = np.clip(arr_ * ceiling, 0, 1)
    return skimage.img_as_uint(arr_)

def unmix(arr):
    """kludge
    """
    green, red = 1, 2
    
    M = np.matrix(np.eye(4))
    M[green,  red] = -2
    M[green,green] = 1
    M[red,  green] = -0.2
    M[red,    red] = 1
    
    x = M.dot(arr.reshape(4,-1).astype(float))
    y = (np.array(x)
           .reshape(arr.shape)
           .clip(0, 2**16 - 1)
           .astype(np.uint16))
    return y

def find_blobs(data, threshold=500):
    """Pass in DO
    """
    from lasagna.pipelines._20170511 import find_peaks
    import skimage.morphology

    peaks = find_peaks(data)
    ix0, ix1, ix2 = 1,2,3
    blobs = peaks > threshold
    mask = skimage.morphology.dilation(blobs[ix1], np.ones((3,3)))
    blobs[ix0, mask] = 0 # remove in other DO channels
    blobs[ix2, mask] = 0
    mask = skimage.morphology.dilation(blobs[ix0], np.ones((3,3)))
    blobs[ix2, mask] = 0
    blobs = blobs[[ix0, ix1, ix2]].sum(axis=0)
    return blobs

def call(data,foreground,width=5,do_threshold=10000,worst_threshold=3000):
    """Thresholds are after contrast adjustment
    """
    from scipy.ndimage.filters import maximum_filter
    cycles, channels, _, _ = data.shape
    assert foreground.ndim == 2
    foreground = foreground.astype(bool)

    size = (1,1,width,width)
    fore = foreground

    # call pixels in the foreground 
    # only keep the max value per cycle
    maxd = maximum_filter(data, size)
    # tie-breaker
    tie_breaker = np.arange(channels) / 100.
    maxd_ = maxd + tie_breaker[None, :, None, None]

    call = maxd #.copy()
    call[maxd_ != maxd_.max(axis=1)[:,None]] = 0
    mask = call.max(axis=1)[0] > do_threshold
    # rank called values
    call_ = call.reshape(-1, call.shape[-2], call.shape[-1]).copy()
    call_.sort(axis=0)
    # nix the worst one
    mask &= call_[-call.shape[-3],...] > worst_threshold
    # dilate to compensate for alignment
    mask = skimage.morphology.dilation(mask, np.ones((width, width)))
    call[:,:,~(mask & fore)] = 0

    return call

def make_offsets(n):
    k = int(np.ceil(np.sqrt(n)))
    offsets = []
    for i in range(n):
        offsets += [(int(i / k), i % k)]
    return offsets

def pixelate(called, blobs, offset=1):
    # create a pixelated overlay showing called base in each cycle
    cycles = called.shape[0]
    offsets = make_offsets(cycles)
    call2 = called.copy()
    call2[:,:,blobs==0] = 0

    for k, (i, j) in enumerate(offsets):
        call2[k,:] = np.roll(call2[k,:],i, axis=-2)
        call2[k,:] = np.roll(call2[k,:],j, axis=-1)
    call2 = call2.max(axis=0)
    # a few bad points 
    mask = (call2>0).sum(axis=0) > 1
    print mask.sum(), 'bad points'
    call2[:,mask] = 0
    return call2
  
def contrast_112B(data_, fore, verbose=False):
    data_ = data_.copy()
    slices = (('contrast GFP sequencing',      np.s_[2:, 1, fore],    20)
              ,('contrast non-GFP sequencing', np.s_[1:, 2:, fore],   50)
              ,('contrast non-GFP DO',         np.s_[0:1, 2:4, fore], 50))

    for msg, s, q1 in slices:
        if verbose:
            print msg
        data_[s] = contrast(data_[s][...,None], 
                            q1=20, q2=99., verbose=verbose)[...,0]

    return data_

def number_barcode(barcode):
    bases = [4**i * 'TGCA'.index(x) for i, x in enumerate(barcode)]
    return bases

def load_design(path=None):
    if path is None:
        path = '/Users/feldman/Downloads/Lasagna Supreme.xlsx'
    df_design = pd.read_excel(path, sheetname='pL43_pL44_sg-bc_design', skiprows=1)
    
    get_bases = lambda x: ''.join(x[i] for i in [0, 1, 2, 6, 5])
    
    df_design = df_design.loc[:, 'well':'oligo FWD'].dropna()
    df_design['short'] = ['C' + get_bases(s) for s in df_design['barcode DNA']]
    return df_design

def load_pools(path=None):
    if path is None:
        path = '/Users/feldman/Downloads/Lasagna Supreme.xlsx'
    df_pools = pd.read_excel(path, sheetname='pL43_44_pools')
    df_pools = df_pools.dropna(axis=0, how='all').dropna(axis=1,how='all')

    return remove_unnamed(df_pools)

def remove_unnamed(df):
    """should use df.filter
    """
    cols = [c for c in df.columns if 'Unnamed' not in c]
    return df[cols]

def call_cells(f, order='TGCA'):
    """
    f = '20170529_96W-G112-B/E?_tile-?.aligned.tif'
    """
    f2 = f.replace('.aligned', '.log.aligned')
    f3 = f.replace('.aligned', '.segment.aligned')
    f4 = f.replace('.aligned', '.called.aligned')

    data = read(f2, memmap=True)
    blobs = find_blobs(data[0])
    nuclei, cells = read(f3)
    called = read(f4)
    cycles = data.shape[0]

    img = called.copy()
    img[:,:,blobs==0] = 0
    lasagna.config.wtf = img.copy()
    df = labels_to_barcodes(img, cells, order=order)
    df_top = (df.sort_values('count', ascending=False)
            .groupby('label')
            .head(1)
            .reset_index()
            .query('count > 1'))

    
    tops = [[order.index(c) for c in x] for x in df_top['barcode']]
    labels = np.zeros((nuclei.max() + 1, cycles), dtype=int)
    labels[df_top['label']] = np.array(tops) + 1
    labeled = labels[nuclei, :].transpose([2, 0, 1])

    return labels, df, df_top

def labels_to_barcodes(img, cells, order='TGCA'):
    """...replace with feature_table
    """
    ix = img.any(axis=1).all(axis=0) & (cells > 0)
    z = img[:, :, ix].max(axis=1)

    bases_ix = np.argmax(img[:, :, ix], axis=1)
    bases = np.array(list(order))[bases_ix].T

    codes = [''.join(x) for x in bases]
    lasagna.config.wtf = zip(cells[ix], bases_ix.T, z.T)
    tmp = [[a,b,c] for (a,b), c in Counter(zip(cells[ix], codes)).items()]
    df = pd.DataFrame(tmp, columns=['label', 'barcode', 'count'])
    return df

def yoshis_lasISland(nuclei, labels, six=False):
    """ 
    labels : (nuclei labels) x cycles, range ()
    nuclei : 2D integer mask
    six    : flag for index layout, affects plaid layout
    """
    cycles = labels.shape[1]
    k = 3
    h,w = nuclei.shape[-2:]
    x = np.mgrid[0:h,0:w]
    outline =((x % 12) == 0).any(axis=0)
    if six:
        x = np.floor(x * (1./4)) 
        x[0] = (x[0] % 2) * 3
        x[1] = x[1] % 3
    else:
        x = np.floor(x * (1./4)) % k
        x[0] *= k
    x = x.sum(axis=0).astype(int)

    neutral = x > (cycles-1)
    first = skimage.morphology.erosion(x==0, np.ones((3,3)))
    x[x > (cycles - 1)] = 0

    a = nuclei.flatten()
    b = x.flatten()

    c = labels[a, b].reshape(x.shape) + 1
    mask = (labels.sum(axis=1)[nuclei] == 0)
    c[neutral] = 1
    c[mask] = 0

    neutral_shade = [[0, 0.5, 0, 0.5]]
    cmap = np.vstack((np.zeros((1,4)), neutral_shade, np.eye(4)))
    d = cmap[c, :].transpose([2,0,1])
    d[:, outline] *= 0.5

    return d

def roi_to_mask(roi, h=2533, w=2531):
    mask = np.zeros((h,w), bool)
    mask[roi[:,0], roi[:,1]] = True
    return mask

