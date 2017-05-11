from itertools import product
import skimage.morphology
from lasagna import config
import lasagna.utils
from glob import glob
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
from skimage.external.tifffile import TiffFile, imsave, imread

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

DEFAULT_LUTS = BLUE, GREEN, RED, MAGENTA, GRAY, GRAY, GRAY

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

    sub = np.zeros(stack.shape[:-2]+(i1-i0, j1-j0),     dtype=stack.dtype)

    i0_, j0_ = max(i0, 0), max(j0, 0)
    i1_, j1_ = min(i1, stack.shape[-2]), min(j1, stack.shape[-1])
    s = (Ellipsis, 
         slice(i0_-i0, (i0_-i0) + i1_-i0_),
         slice(j0_-j0, (j0_-j0) + j1_-j0_))

    sub[s] = stack[..., i0_:i1_, j0_:j1_]
    return sub



def offset(stack, offsets):
    """Applies offset to stack, fills with zero.
    :param stack: N-dim array
    :param offsets: list of N offsets
    :return:
    """
    if len(offsets) != stack.ndim:
        if len(offsets) == 2 and stack.ndim > 2:
            offsets = [0] * (stack.ndim - 2) + list(offsets)
        else:
            raise IndexError("number of offsets must equal stack dimensions, or 2 (trailing dimensions)")
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
    from lasagna.io import subimage, pile, read_stack

    arr = []
    for filename, bounds_ in zip(files, bounds):
        I = read_stack(filename, memmap=False) # some memory issue right now
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



default_dirs = {'raw': 'raw',
                'data': 'data',
                'analysis': 'analysis',
                'nuclei': 'analysis/nuclei',
                'stitch': 'stitch',
                'calibration': 'calibration',
                'export': 'export'}

default_file_pattern = \
        r'(?P<dataset>(?P<date>[0-9]{8}).*)[\/\\]' + \
        r'(?P<mag>[0-9]+X).' + \
        r'(?:(?P<cycle>[^_\.]*).*?(?:.*MMStack)?.)?' + \
        r'(?P<well>[A-H]([0-9]|[[01][012]))' + \
        r'(?:\.(?P<tag>.*))*\.tif'


default_file_pattern_old = '(data)/((([0-9]*X).*round([0-9]))*.*)/(((.*_([A-Z][0-9]))-Site_([0-9]*)).ome.tif)'
default_file_groups = 'data', 'set', '', 'mag', 'round', 'file', 'file_well_site', 'file_well', 'well', 'site'
default_path_formula = {'raw': '[data]/[set]/[file]',
                        'calibrated': '[data]/[set]/[file_well_site].calibrated.tif',
                        'stitched': '[data]/[set]/[file_well].stitched.tif',
                        'aligned': '[data]/aligned/[mag]/[well].aligned.tif',
                        'aligned_FFT': '[data]/aligned/[mag]/[well].aligned.FFT.tif',
                        'nuclei': '[data]/aligned/[mag]/[well].aligned.nuclei.tif',
                        }
default_table_index = {'mag': str,
                       'round': float,
                       'set': str,
                       'well': str,
                       'site': int}


class Paths(object):
    def __init__(self, dataset, lasagna_path='/broad/blainey_lab/David/lasagna',
                 sub_dirs=None):
        """Store file paths relative to lasagna_path, allowing dataset to be loaded in different
        absolute locations. Retrieve raw files, stitched files, and nuclei. Pass stitch name and
        get nuclei name. Store path to calibration information.

        Path information stored in DataFrame paths.table, e.g.,
        paths.table.find(mag='60X', round=1)

        :param dataset:
        :param lasagna_path:
        :return:
        """
        self.calibrations = []
        self.datafiles = []
        self.dirs = default_dirs if sub_dirs is None else sub_dirs
        self.table = None
        self.dataset = dataset
        # resolve path, strip trailing slashes
        self.lasagna_path = os.path.abspath(lasagna_path)

        self.update_datafiles()
        self.update_table()
        self.update_calibration()

    def full(self, *args):
        """Prepend dataset location, multiple arguments are joined. If given an absolute
        path, just return it..
        :param args:
        :return:
        """
        if args and os.path.isabs(args[0]):
            return args[0]
        return os.path.join(self.lasagna_path, self.dataset, *args)

    def export(self, *args):
        """Shortcut to export directory.
        :param args:
        :return:
        """
        return self.full(self.dirs['export'], *args)

    def relative(self, s):
        """Convert absolute paths to relative paths within dataset.
        :param s:
        :return:
        """
        return s.replace(self.full() + '/', '')

    def parent(self, s):
        """Shortcut to name of parent directory of file or directory.
        :param s:
        :return:
        """
        return os.path.basename(os.path.dirname(s))

    def update_datafiles(self):
        self.datafiles = []
        for dirpath, dirnames, filenames in os.walk(self.full(self.dirs['data'])):
            self.datafiles += [os.path.join(dirpath, f) for f in filenames]

    def update_table(self, file_pattern=default_file_pattern_old, groups=default_file_groups,
                     path_formula=default_path_formula, table_index=default_table_index):

        """Match pattern to find original files in dataset. For each raw file, apply patterns in
        self.dirs to generate new file names and add to self.table if file exists (else nan).

        E.g., raw file 'data/set/set_A1-Site1.ome.tif' is captured as ('set', 'Site1'). Pattern-matching yields:
         'aligned': '[data]/aligned/[set]/[file_well].aligned.tif'
         ==> data/aligned/set/set_A1.tif
         'calibrated': '[data]/[set]/[file_well_site].calibrated.tif'
         ==> data/set/set_A1-Site1.calibrated.tif

        """

        # one dict per raw data file, keys are regex groups and patterns after substitution
        d = []
        for f in self.datafiles:
            m = re.match(file_pattern, self.relative(f))
            if m:
                d += [{k: v for k, v in zip(groups, m.groups())}]
        for entry in d:
            for key, pattern in path_formula.items():
                for group, value in entry.items():
                    pattern = pattern.replace('[%s]' % group, str(value))
                entry.update({key: pattern})

        self.table = lasagna.utils.DataFrameFind(d)
        # skip if empty
        if d:
            for k, v in table_index.items():
                self.table[k] = self.table[k].astype(v)

            self.table.set_index(table_index.keys(), inplace=True)
            self.table.sortlevel(inplace=True)

    def update_calibration(self):
        calibration_dir = self.full(self.dirs['calibration'])
        if os.path.isdir(calibration_dir):
            _, calibrations, _ = os.walk(calibration_dir).next()
            self.calibrations = [os.path.join(self.dirs['calibration'], c) for c in calibrations]

    def make_dirs(self, files):
        """Create sub-directories for files in column, if they don't exist.
        :return:
        """
        for f in files:
            if pandas.notnull(f):
                d = self.full(os.path.dirname(f))
                if not os.path.exists(d):
                    os.makedirs(d)
                    print 'created directory', d

    def lookup(self, column, **kwargs):
        """Convenient search.
            :param column: column to return item from
            :param kwargs: search_column=value
            :return:
            """
        source, value = kwargs.items()[0]
        return self.table[self.table[source] == value][column][0]


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


def read_registered(path):
    """Reads output of Fiji Grid/Collection stitching (TileConfiguration.registered.txt),
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


def read_stack(filename, memmap=False):
    """Read a .tif file into a numpy array, with optional memory mapping.
    """
    if memmap:
        data = _get_mapped_tif(filename)
    else:
        # os.stat detects if file has been updated
        data = _imread(filename, hash(os.stat(filename)))
        while data.shape[0] == 1:
            data = np.squeeze(data, axis=(0,))
    return data

@lasagna.utils.Memoized
def _imread(filename, dummy):
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

