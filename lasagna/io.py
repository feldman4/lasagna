from itertools import product
import skimage.morphology
from lasagna import config
import lasagna.utils
import pandas
import struct
import os
import time
import PIL.ImageFont
import PIL.Image
import PIL.ImageDraw
import numpy as np
from numpy.lib.stride_tricks import as_strided
import regex as re
import StringIO, pickle, zlib
from lasagna.external.tifffile_old import TiffFile, imsave, imread

imagej_description = ''.join(['ImageJ=1.49v\nimages=%d\nchannels=%d\nslices=%d',
                              '\nframes=%d\nhyperstack=true\nmode=composite',
                              '\nunit=\\u00B5m\nspacing=8.0\nloop=false\n',
                              'min=764.0\nmax=38220.0\n'])

UM_PER_PX = {'10X' : 0.66,
             '20X' : 0.33,
             '40X' : 0.175,
             '60X' : 0.11,
             '100X': 0.066}

BINNING = 2

RED = tuple(range(256) + [0] * 512)
GREEN = tuple([0] * 256 + range(256) + [0] * 256)
BLUE = tuple([0] * 512 + range(256))
MAGENTA = tuple(range(256) + [0] * 256 + range(256))
GRAY = tuple(range(256) * 3)
CYAN = tuple([0] * 256 + range(256) * 2)

DEFAULT_LUTS = GRAY, CYAN, GREEN, RED, MAGENTA, GRAY, GRAY, GRAY

DIR = {}

VISITOR_FONT = PIL.ImageFont.truetype(config.visitor_font)


def get_file_list(str_or_list):
    """Get a list of files from a single filename or iterable of filenames. Glob expressions accepted.
    :param str_or_list: filename, glob, list of filenames, or list of globs
    :return:
    """
    if type(str_or_list) is str:
        return glob(str_or_list)
    return [y for x in str_or_list for y in glob(x)]


def add_dir(path, dir_to_add):
    x = path.split('/')
    return '/'.join(x[:-1] + [dir_to_add] + [x[-1]])


def find_files(start='', include='', exclude='', depth=2):
    # or require glob2
    files = []
    for d in range(depth):
        path_str = '*/' * d + '*.tif'
        files += glob(os.path.join(start, path_str))

    def keep(f):
        flag1 = include and not re.findall(include, f)
        flag2 = exclude and re.findall(exclude, f)
        return (not flag1 or flag2)

    return filter(keep, files)


def well_to_row_col(df, in_place=False, col_to_int=True):
    if not in_place:
        df = df.copy()
    f = lambda s: int(s) if col_to_int else s

    df['row'] = [s[0] for s in df['well']]
    df['col'] = [f(s[1:]) for s in df['well']]

    return df


def get_row_stack(row, full=False, nuclei=False, apply_offset=False, pad=0):
    """For a given DataFrame row, get image data for full frame or cell extents only.
    :param row:
    :param full:
    :param nuclei:
    :param apply_offset:
    :param pad: constant, included on both sides
    :return:
    """
    filename = row[('all', 'file')]
    if nuclei:
        filename = config.paths.lookup('nuclei', stitch=row[('all', 'file')])
    I = _get_stack(filename)
    if full is False:
        I = subimage(I, row[('all', 'bounds')], pad=pad)
    if apply_offset:
        offsets = np.array(I.shape) * 0
        offsets[-2:] = 1 * row['offset x'], 1 * row['offset y']
        I = offset(I, offsets)

    return I


def compose_rows(df, load_function=lambda x: get_row_stack(x)):
    """Load and concatenate stacks corresponding to rows in a DataFrame.
    Stacks are expanded to maximum in each dimension by padding with zeros.
    :param df: N rows in DataFrame, stacks have shape [m,...,n]
    :param load_function:
    :return: ndarray of shape [N,m,...,n]
    """

    arr_ = []
    for ix, row in df.iterrows():
        z = load_function(row).copy()
        arr_.append(z)

    return pile(arr_)


def pile(arr):
    """Concatenate stacks of same dimensionality along leading dimension. Values are
    filled from top left of matrix. Fills background with zero.
    :param arr:
    :return:
    """
    shape = [max(s) for s in zip(*[x.shape for x in arr])]
    # strange numpy limitations
    arr_out = []
    for x in arr:
        y = np.zeros(shape, x.dtype)
        slicer = [slice(None, s) for s in x.shape]
        y[slicer] = x
        arr_out += [y[None, ...]]

    return np.concatenate(arr_out, axis=0)


def montage(arr, shape=None):
    """tile ND arrays ([..., height, width]) in last two dimensions
    first N-2 dimensions must match, tiles are expanded to max height and width
    pads with zero, no spacing
    if shape=(rows, columns) not provided, defaults to square, clipping last row if empty
    """
    sz = zip(*[img.shape for img in arr])
    h, w, n = max(sz[-2]), max(sz[-1]), len(arr)
    if not shape:
        nr = nc = int(np.ceil(np.sqrt(n)))
        if (nr - 1) * nc >= n:
            nr -= 1
    else:
        nr, nc = shape
    M = np.zeros(arr[0].shape[:-2] + (nr * h, nc * w), dtype=arr[0].dtype)

    for (r, c), img in zip(product(range(nr), range(nc)), arr):
        s = [[None] for _ in img.shape]
        s[-2] = (r * h, r * h + img.shape[-2])
        s[-1] = (c * w, c * w + img.shape[-1])
        M[[slice(*x) for x in s]] = img

    return M


def tile(arr, n, m, pad=None):
    """Divide a stack of images into tiles of size m x n. If m or n is between 
    0 and 1, it specifies a fraction of the input size. If pad is specified, the
    value is used to fill in edges, otherwise the tiles may not be equally sized.
    Tiles are returned in a list.
    """
    assert arr.ndim > 1
    h, w = arr.shape[-2:]
    # convert to number of tiles
    m = m if m >= 1 else np.round(1 / m)
    n = n if n >= 1 else np.round(1 / n)
    m, n = int(m), int(n)

    if pad is not None:
        pad_width = (arr.ndim - 2) * ((0, 0),) + ((0, -h % m), (0, -w % n))
        arr = np.pad(arr, pad_width, 'constant', constant_values=pad)
        print arr.shape

    tiled = np.array_split(arr, m, axis=-2)
    tiled = lasagna.utils.concatMap(lambda x: np.array_split(x, n, axis=-1), tiled)
    return tiled


@lasagna.utils.Memoized
def _get_stack(name):
    data = imread(config.paths.full(name), multifile=False)
    while data.shape[0] == 1:
        data = np.squeeze(data, axis=(0,))
    return data


# TODO fix extension of luts and display_ranges when not provided
def save_stack(name, data, luts=None, display_ranges=None, 
               resolution=1., compress=0):
    """
    :param data: numpy array with 5, 4, 3, or 2 dimensions [TxZxCxYxX]. float64 
        is automatically converted to float32 for ImageJ compatibility
    :type data: array[...xYxX](bool | uint8 | uint16 | float32 | float64)

    :param resolution: resolution in microns per pixel



    input ND array dimensions as ([time], [z slice], channel, y, x)
    leading dimensions beyond 5 could be wrapped into time, not implemented
    if no lut provided, use default and pad extra channels with GRAY
    """
    
    if name.split('.')[-1] != 'tif':
        name += '.tif'
    name = os.path.abspath(name)

    if data.ndim == 2:
        data = data[None]

    if (data.dtype == np.int64):
        if (data>=0).all() and (data<2**16).all():
            data = data.astype(np.uint16)
            print 'Cast int64 to int16 without loss'
        else:
            data = data.astype(np.float32)
            print 'Cast int64 to float32'
    if data.dtype == np.float64:
        data = data.astype(np.float32)
        print 'Cast float64 to float32'

    if data.dtype == np.bool:
        data = 255 * data.astype(np.uint8)

    if not data.dtype in (np.uint8, np.uint16, np.float32):
        raise ValueError('Cannot save data of type %s' % data.dtype)

    nchannels = data.shape[-3]

    if isinstance(resolution, str):
        resolution = UM_PER_PX[resolution] * BINNING
    resolution = (1./resolution,)*2
    resolution[0]

    if luts is None:
        luts = DEFAULT_LUTS + (GRAY,) * nchannels

    if display_ranges is None:
        display_ranges = tuple([(x.min(), x.max())
                                for x in np.rollaxis(data, -3)])

    try:
        luts = luts[:nchannels]
        display_ranges = display_ranges[:nchannels]
    except IndexError:
        raise IndexError('Must provide at least %d luts and display ranges' % nchannels)

    # metadata encoding LUTs and display ranges
    # see http://rsb.info.nih.gov/ij/developer/source/ij/io/TiffEncoder.java.html
    description = ij_description(data.shape)
    tag_50838 = ij_tag_50838(nchannels)
    tag_50839 = ij_tag_50839(luts, display_ranges)

    if not os.path.isdir(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))

    imsave(name, data, photometric='minisblack',
           description=description, resolution=resolution, compress=compress,
           extratags=[(50838, 'I', len(tag_50838), tag_50838, True),
                      (50839, 'B', len(tag_50839), tag_50839, True),
                      ])


def ij_description(shape):
    """Format ImageJ description for hyperstack.
    :param shape:
    :return:
    """
    s = shape[:-2]
    if not s:
        return imagej_description % (1, 1, 1, 1)
    n = np.prod(s)
    if len(s) == 3:
        return imagej_description % (n, s[2], s[1], s[0])
    if len(s) == 2:
        return imagej_description % (n, s[1], s[0], 1)
    if len(s) == 1:
        return imagej_description % (n, s[0], 1, 1)
    # bad shape
    assert False


def ij_tag_50838(nchannels):
    """ImageJ uses tag 50838 to indicate size of metadata elements (e.g., 768 bytes per ROI)
    :param nchannels:
    :return:
    """
    info_block = (20,)  # summary of metadata fields
    display_block = (16 * nchannels,)  # display range block
    luts_block = (256 * 3,) * nchannels  #
    return info_block + display_block + luts_block


def ij_tag_50839(luts, display_ranges):
    """ImageJ uses tag 50839 to store metadata. Only range and luts are implemented here.
    :param tuple luts: tuple of 255*3=768 8-bit ints specifying RGB. Class constants io.RED etc may be used.
    :param tuple display_ranges: tuple of (min, max) pairs for
    :return:
    """
    d = struct.pack('<' + 'd' * len(display_ranges) * 2, *[y for x in display_ranges for y in x])
    # insert display ranges
    tag = ''.join(['JIJI',
                   'gnar\x01\x00\x00\x00',
                   'stul%s\x00\x00\x00' % chr(len(luts)),
                   d])
    tag = struct.unpack('<' + 'B' * len(tag), tag)
    return tag + tuple(sum([list(x) for x in luts], []))


def parse_MM(s):
    """Parses Micro-Manager MDA filename.
    100X_round1_1_MMStack_A1-Site_15.ome.tif => ('100X', 1,  'A1', 15)
    100X_scan_1_MMStack_A1-Site_15.ome.tif => ('100X', None, 'A1', 15)
    """
    pat = '.*?([0-9]*X).*?([a-zA-Z]+([0-9])*).*_MMStack_(.*)-Site_([0-9]*).*'
    m = re.match(pat, s)
    try:
        mag, _, rnd, well, site = m.groups()
        if rnd:
            rnd = int(rnd)
        return mag, rnd, well, int(site)
    except TypeError:
        raise ValueError('Filename %s does not match MM pattern' % s)
    

def subimage(stack, bbox, pad=0):
    """Index rectangular region from [...xYxX] stack with optional constant-width padding.
    Boundary is supplied as (min_row, min_col, max_row, max_col).
    If boundary lies outside stack, raises error.
    If padded rectangle extends outside stack, fills with fill_value.

    bbox can be bbox or iterable of bbox (faster if padding)
    :return:
    """ 
    i0, j0, i1, j1 = bbox + np.array([-pad, -pad, pad, pad])

    sub = np.zeros(stack.shape[:-2]+(i1-i0, j1-j0), dtype=stack.dtype)

    i0_, j0_ = max(i0, 0), max(j0, 0)
    i1_, j1_ = min(i1, stack.shape[-2]), min(j1, stack.shape[-1])
    s = (Ellipsis, 
         slice(i0_-i0, (i0_-i0) + i1_-i0_),
         slice(j0_-j0, (j0_-j0) + j1_-j0_))

    sub[s] = stack[..., i0_:i1_, j0_:j1_]
    return sub


def offset(stack, offsets):
    """Applies offset to stack, fills with zero. Only applies integer offsets.
    :param stack: N-dim array
    :param offsets: list of N offsets
    :return:
    """
    if len(offsets) != stack.ndim:
        if len(offsets) == 2 and stack.ndim > 2:
            offsets = [0] * (stack.ndim - 2) + list(offsets)
        else:
            raise IndexError("number of offsets must equal stack dimensions, or 2 (trailing dimensions)")

    offsets = np.array(offsets).astype(int)

    n = stack.ndim
    ns = (slice(None),)
    for d, offset in enumerate(offsets):
        stack = np.roll(stack, offset, axis=d)
        if offset < 0:
            index = ns * d + (slice(offset, None),) + ns * (n - d - 1)
            stack[index] = 0
        if offset > 0:
            index = ns * d + (slice(None, offset),) + ns * (n - d - 1)
            stack[index] = 0

    return stack


def grid_view(files, bounds, padding=40, with_mask=False):
    """Mask is 1-indexed. Zero values indicate background.
    """
    padding = int(padding)

    arr = []
    for filename, bounds_ in zip(files, bounds):
        I = read_stack(filename, memmap=False, copy=False) # some memory issue right now
        I_cell = subimage(I, bounds_, pad=padding)
        arr.append(I_cell.copy())

    if with_mask:
        arr_m = []
        for i, (i0, j0, i1, j1) in enumerate(bounds):
            shape = i1 - i0 + padding, j1 - j0 + padding
            img = np.zeros(shape, dtype=np.uint16) + i + 1
            arr_m += [img]
        return pile(arr), pile(arr_m)

    return pile(arr)


def watermark(shape, text, spacing=1, corner='top left'):
    """ Add rasterized text to empty 2D numpy array.
    :param shape: (height, width)
    :param text: string or list of strings
    :param spacing: spacing between lines, in pixels
    :return:
    """
    bm = bitmap_text(text, spacing=spacing)
    if any(x > y for x, y in zip(bm.shape, shape)):
        raise ValueError('not enough space, text %s occupies %s pixels' %
                         (text.__repr__(), bm.shape.__repr__()))
    b = np.zeros([shape[0], shape[1]])
    if corner == 'top left':
        b[:bm.shape[0], :bm.shape[1]] += bm
    if corner == 'bottom left':
        b[-bm.shape[0]:, :bm.shape[1]] += bm
    return b


def bitmap_text(text, spacing=1):
    if type(text) in (str, np.string_):
        text = [text]

    def get_text(s):
        img = PIL.Image.new("RGBA", (len(s) * 8, 10), (0, 0, 0))
        draw = PIL.ImageDraw.Draw(img)
        draw.text((0, 0), s, (255, 255, 255), font=VISITOR_FONT)
        draw = PIL.ImageDraw.Draw(img)

        n = np.array(img)[2:7, :, 0]
        if n.sum() == 0:
            return n
        return n[:, :np.where(n.any(axis=0))[0][-1] + 1]

    m = [get_text(s).copy() for s in text]

    full_shape = np.max([t.shape for t in m], axis=0)
    full_text = np.zeros(((full_shape[0] + spacing) * len(text) - spacing, full_shape[1]))

    for i, t in enumerate(m):
        full_text[i * (full_shape[0] + spacing):(i + 1) * full_shape[0] + i * spacing, :t.shape[1]] = t

    return full_text


def mark_text(arr, text, inplace=False, value=255, **kwargs):
    """ arr[...,channels,height,width]
    if inplace=True, add text to last channel; otherwise insert into new channel
    """
    if not inplace:
        frames = np.zeros(arr.shape[:-3] + (1,) + arr.shape[-2:], dtype=arr.dtype)
    for index in np.ndindex(arr.shape[:-3]):
        frame = watermark(arr[index][-1].shape, text[index], **kwargs)
        if inplace:
            arr[index][-1] += frame * value
        else:
            frames[index][-1] = frame * value
    if not inplace:
        return np.concatenate([arr, frames], axis=-3)
    return arr


def mark_disk(shape, features, type='box'):
    """ features contains [i, j, diameter]
    """
    disk_size = 31
    disks = [np.pad(skimage.morphology.square(i, dtype=np.uint16), disk_size / 2 - i / 2, mode='constant')
             for i in range(1, disk_size / 2, 2)]

    disks = [skimage.morphology.dilation(d) - d for d in disks]

    im = np.zeros(shape)
    im2 = np.pad(im, disk_size / 2, mode='constant')
    for i, j, size in features:
        im2[i:i + disk_size, j:j + disk_size] += disks[size]

    return im2[disk_size / 2:-disk_size / 2 + 1, disk_size / 2:-disk_size / 2 + 1]


def mark_blobs(row, n):
    channels = ('DAPI', 'Cy3', 'A594', 'Cy5')
    im = np.zeros(n.shape[1::], dtype='uint16')
    for channel, blobs in row.loc[:, 'blob'].iteritems():
        if channel != 'DAPI':
            i = channels.index(channel)
            im += mark_disk(im.shape, blobs) * 2 ** i
    return im


def compress_obj(obj):
    s = StringIO.StringIO()
    pickle.dump(obj, s)
    out = zlib.compress(s.getvalue())
    s.close()
    return out


def decompress_obj(string):
    return pickle.load(StringIO.StringIO(zlib.decompress(string)))


def read_lut(name):
    file_name = os.path.join(config.luts, name + '.txt')
    with open(file_name, 'r') as fh:
        lines = fh.readlines()
        values = [line[:-1].split() for line in lines]
    return [int(y) for x in zip(*values) for y in x]

# DEPRECATED
def read_registered(path):
    """DEPRECATED

    Reads output of Fiji Grid/Collection stitching (TileConfiguration.registered.txt),
    returns list of tile coordinates.
    :param path:
    :return:
    """
    with open(path, 'r') as fh:
        tile_config = fh.read()
    m = re.findall('\(.*\)', tile_config)
    translations = [[float(x) for x in pos[1:-1].split(',')] for pos in m]
    return translations

GLASBEY = read_lut('glasbey_inverted')

hash_cache = {}
# decorate to store hash_cache in function
def show_IJ(data, title='image', imp=None, check_cache=False, **kwargs):
    """Display image in linked ImageJ instance. If imp is provided,
    replaces contents; otherwise opens new ij.ImagePlus. 

    Images are first exported to .tif, then loaded in ImageJ. Default behavior is to
    check the file cache using a hash of metadata and sparsely sampled pixels, and only 
    export if not found in cache.
    """
    
    if not isinstance(data, np.ndarray):
        data = np.array(data) # for pandas and xarray, could use data.values

    # have we done this before?
    skip = min(100, data.size)
    key = hash(str(kwargs)) + \
          hash(tuple(data.flat[::data.size / skip])) + \
          hash(data.shape)

    if key in hash_cache and check_cache:
        savename = hash_cache[key]
    else:
        # export .tif
        savename = config.fiji_target + title + time.strftime('_%Y%m%d_%s.tif')
        save_stack(savename, data, **kwargs)
        hash_cache[key] = savename
    
    new_imp = config.j.ij.IJ.openImage(savename)

    if imp:
        # rather than convert from grayscale to composite, just replace
        # TODO: remove this sketchy shit
        if new_imp.getDisplayMode() != imp.getDisplayMode():

            # move the old listeners over
            new_imp.show()

            new_canvas = new_imp.getWindow().getCanvas()
            [new_canvas.removeKeyListener(x) for x in new_canvas.getKeyListeners()]

            listeners = imp.getWindow().getCanvas().getKeyListeners()
            [new_canvas.addKeyListener(x) for x in listeners]

            imp.close()
                
        else:
            imp.setImage(new_imp)
            imp.updateAndRepaintWindow() # TODO: sometimes this doesn't redraw. flipping Z frame manually fixes it.
            imp.setTitle(title)
            new_imp.close()
            return imp

    new_imp.setTitle(title)
    new_imp.show()
    # set contrast for single channel images
    if data.ndim == 2:
        new_imp.setDisplayRange(data.min(), data.max())
        new_imp.updateAndDraw()
    return new_imp


def read_stack(filename, memmap=False, copy=True):
    """Read a .tif file into a numpy array, with optional memory mapping.
    """
    if memmap:
        data = _get_mapped_tif(filename)
    else:
        # os.stat detects if file has been updated
        # data = _imread(filename, hash(os.stat(filename)))
        data = _imread(filename, 0, copy=copy)
        while data.shape[0] == 1:
            data = np.squeeze(data, axis=(0,))
    return data

# @lasagna.utils.Memoized
def _imread(filename, dummy, copy=True):
    """Call TiffFile imread. Dummy arg to separately memoize calls.
    """
    return imread(filename, multifile=False)

@lasagna.utils.Memoized
def _get_mapped_tif(filename):
    TF = TiffFile(filename)
    # check the offsets
    offsets = []
    for i in range(len(TF.pages) - 1):
        offset, shape = TF.pages[i].is_contiguous
        offsets += [TF.pages[i+1].is_contiguous[0] - (offset + shape)]
    # doesn't work unless all IFD headers are the same size
    if not all(np.array(offsets)==offsets[0]):
        raise IOError('attempted to memmap unequal offsets in:\n%s' % paths.full(f))
    stride_offset = offsets[0]
    # adjust strides to account for offset
    shape = [x for x in TF.series[0].shape if x != 1]
    strides = np.r_[np.cumprod(shape[::-1])[::-1][1:], [1]] * 2
    # TODO: figure out what the fuck is going on here
    skip_count = np.cumprod(shape[-3:0:-1]) # of IFDs skipped over in leading dimensions
    skip_count = np.r_[[1], skip_count][::-1]
    strides[:-2] += skip_count.astype(int) * stride_offset

    # make the initial memmap
    fh = TF.filehandle
    offset = TF.pages[0].is_contiguous[0]
    
    # RHEL and osx are little-endian, ImageJ defaults to big-endian
    dtype = np.dtype(TF.byteorder + 'u2')

    mm = np.memmap(fh, dtype=dtype, mode='r',
                                    offset=offset,
                                    shape=np.prod(shape), order='C')
    # update the memmap with adjusted strides
    return as_strided(mm, shape=shape, strides=strides)

def grab_image():
    """Return contents of currently selected ImageJ window.
    """
    path = os.path.join(lasagna.config.fiji_target, 'tmp.tif')

    imp = lasagna.config.j.ij.IJ.getImage()
    lasagna.config.j.ij.IJ.save(imp, path)
    data = lasagna.io.read_stack(path, memmap=True)
    os.remove(path)
    
    # title = imp.getTitle()
    # if title:
    #     print 'grabbed image "%s"' % title

    if data.dtype == np.dtype('>u2'):
        data = data.astype(np.uint16)
    return data

file_pattern = [
        r'((?P<home>.*)\/)?',
        r'(?P<dataset>(?P<date>[0-9]{8}).*?)\/',
        r'(?:(?P<subdir>.*)\/)*',
        r'(MAX_)?(?P<mag>[0-9]+X).',
        r'(?:(?P<cycle>[^_\.]*).*?(?:.*MMStack)?.)?',
        r'(?P<well>[A-H][0-9]*)',
        r'(?:[_-]Site[_-](?P<site>([0-9]+)))?',
        r'(?:_Tile-(?P<tile>([0-9]+)))?',
        r'(?:\.(?P<tag>.*))*\.(tif|pkl)']

file_pattern_abs = ''.join(file_pattern)
file_pattern_rel = ''.join(file_pattern[2:])
        

def parse_filename(filename):
    """Parse filename into dictionary. Some entries in dictionary optional, e.g., cycle and tile.
    """
    filename = os.path.normpath(filename)
    filename = filename.replace('\\', '/')
    if os.path.isabs(filename):
        pattern = file_pattern_abs
    else:
        pattern = file_pattern_rel

    match = re.match(pattern, filename)
    try:
        result = {k:v for k,v in match.groupdict().items() if v is not None}
        return result
    except AttributeError:
        raise ValueError('failed to parse filename: %s' % filename)


def name(description, ext='tif', **more_description):
    """Name a file from a dictionary of filename parts. Can override dictionary with keyword arguments.
    """
    d = dict(description)
    d.update(more_description)

    assert 'tag' in d

    if 'cycle' in d:
        a = '%s_%s_%s' % (d['mag'], d['cycle'], d['well'])
    else:
        a = '%s_%s' % (d['mag'], d['well'])

    # only one
    if 'tile' in d:
        b = 'Tile-%s' % d['tile']
    elif 'site' in d:
        b = 'Site-%s' % d['site']
    else:
        b = None

    if b:
        basename = '%s_%s.%s.%s' % (a, b, d['tag'], ext)
    else:
        basename = '%s.%s.%s' % (a, d['tag'], ext)
    
    optional = lambda x: d.get(x, '')
    filename = os.path.join(optional('home'), optional('dataset'), optional('subdir'), basename)
    return os.path.normpath(filename)