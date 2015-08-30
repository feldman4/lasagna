from itertools import product
import skimage.morphology
from lasagna import config
from lasagna.utils import Memoized
from glob import glob
import pandas
import struct
import os
import PIL.ImageFont
import PIL.Image
import PIL.ImageDraw
import numpy as np
import regex as re
import StringIO, pickle, zlib
from skimage.external.tifffile import TiffFile, imsave, imread

imagej_description = ''.join(['ImageJ=1.49v\nimages=%d\nchannels=%d\nslices=%d',
                              '\nframes=%d\nhyperstack=true\nmode=composite',
                              '\nunit=\\u00B5m\nspacing=8.0\nloop=false\n',
                              'min=764.0\nmax=38220.0\n'])

UM_PER_PX = {'40X': 0.44,
             '20X': 0.22}

BINNING = 2
OBJECTIVE = '40X'

RED = tuple(range(256) + [0] * 512)
GREEN = tuple([0] * 256 + range(256) + [0] * 256)
BLUE = tuple([0] * 512 + range(256))
MAGENTA = tuple(range(256) + [0] * 256 + range(256))
GRAY = tuple(range(256) * 3)
CYAN = tuple([0] * 256 + range(256) * 2)

DEFAULT_LUTS = (BLUE, GREEN, RED, MAGENTA)

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


def read_stack(filename, master=None, memmap=False):
    if master:
        TF = _load_tifffile(master)
        names = [s.pages[0].parent.filename for s in TF.series]
        index = names.index(filename.split('/')[-1])
        page = TF.series[index].pages[0]
        data = TF.asarray(series=index, memmap=memmap)
    else:
        data = imread(filename, multifile=False, memmap=memmap)
    while data.shape[0] == 1:
        data = np.squeeze(data, axis=(0,))
    return data


# save load time, will bring trouble if the TiffFile reference is closed
@Memoized
def _load_tifffile(master):
    return TiffFile(master)


def get_row_stack(row, full=False, nuclei=False, apply_offset=False, pad=(0, 0)):
    """For a given DataFrame row, get image data for full frame or cell extents only.
    :param row:
    :param full:
    :param nuclei:
    :param apply_offset:
    :return:
    """
    file_name = row[('all', 'file')]
    if nuclei:
        file_name = config.paths.lookup('nuclei', stitch=row[('all', 'file')])
    I = _get_stack(file_name)
    if full is False:
        I = I[b_idx(row, padding=(pad, I.shape))]
    if apply_offset:
        offsets = np.array(I.shape) * 0
        offsets[-2:] = 1 * row['offset x'], 1 * row['offset y']
        I = offset_stack(I, offsets)

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

    return compose_stacks(arr_)


def compose_stacks(arr_):
    """Concatenate stacks of same dimensionality along leading dimension. Values are
    filled from top left of matrix. Fills background with zero.
    :param arr_:
    :return:
    """
    shape = [max(s) for s in zip(*[x.shape for x in arr_])]
    # strange numpy limitations
    arr_out = []
    for x in arr_:
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


@Memoized
def _get_stack(name):
    data = imread(config.paths.full(name), multifile=False)
    while data.shape[0] == 1:
        data = np.squeeze(data, axis=(0,))
    return data


def save_hyperstack(name, data, autocast=True, resolution=None,
                    luts=None, display_ranges=None, compress=0,
                    auto_make_dir=True):
    """input ND array dimensions as ([time], [z slice], channel, y, x)
    leading dimensions beyond 5 could be wrapped into time, not implemented
    """
    if name[-4:] != '.tif':
        name += '.tif'
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    nchannels = data.shape[-3]
    if resolution is None:
        resolution = (1. / (UM_PER_PX[OBJECTIVE] * BINNING),) * 2
    if luts is None:
        luts = [x for x, _ in zip(DEFAULT_LUTS, range(nchannels))]
    if display_ranges is None:
        display_ranges = tuple([(x.min(), x.max())
                                for x in np.rollaxis(data, -3)])

    if len(luts) != len(display_ranges) or len(luts) != data.shape[-3]:
        raise ValueError('lookup table, display ranges, and data shape must match')

    # convert to uint16
    tmp = data.copy()
    if autocast:
        tmp = tmp.astype(np.uint16)

    # metadata encoding LUTs and display ranges
    # see http://rsb.info.nih.gov/ij/developer/source/ij/io/TiffEncoder.java.html
    description = ij_description(data.shape)
    tag_50838 = ij_tag_50838(nchannels)
    tag_50839 = ij_tag_50839(luts, display_ranges)

    if auto_make_dir and not os.path.isdir(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))

    imsave(name, tmp, photometric='minisblack',
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


# helper functions for loading
def sort_by_site(s):
    return ''.join(get_well_site(s)[::-1])


def get_well_site(s):
    match = re.search('_(..)-Site_([0-9]*)', s)
    if match:
        well, site = match.groups(1)
        return well, int(site)
    match = re.search('([A-H][0-9]*)_', s)
    if match:
        well = match.groups(1)
        return well[0], 0
    print s
    raise NameError('FuckYouError')


def get_round(s):
    match = re.search('round([0-9]*)', s)
    if match:
        return int(match.groups(1)[0])
    return 0


def get_magnification(s):
    match = re.search('([0-9]+X)', s)
    if match:
        return match.groups(1)[0]


def b_idx(row, padding=None):
    """For a given DataFrame row, get slice index to cell in original data. Assumes 4D data.
    :param row:
    :return:
    """

    bounds = row[('all', 'bounds')]
    if padding:
        pad, shape = padding
        bounds = bounds[0] - pad[0], bounds[1] - pad[1], bounds[2] + pad[0], bounds[3] + pad[1]
        bounds = max(bounds[0], 0), max(bounds[1], 0), min(bounds[2], shape[-2]), min(bounds[3], shape[-1])
    return Ellipsis, slice(bounds[0], bounds[2]), slice(bounds[1], bounds[3])


def offset_stack(stack, offsets):
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


default_dirs = {'raw': 'raw',
                'data': 'data',
                'analysis': 'analysis',
                'nuclei': 'analysis/nuclei',
                'stitch': 'stitch',
                'calibration': 'calibration',
                'export': 'export'}

default_file_pattern = '(data)/(.*)/(((([0-9]*X)_(.*)_MMStack_([A-Z][0-9]))-Site_([0-9]*)).ome.tif)'
default_file_pattern = '(data)/(.*)/(((([0-9]*X)_(.*round([0-9]))*.*_MMStack_([A-Z][0-9]))-Site_([0-9]*)).ome.tif)'
default_file_groups = 'data', 'set', 'file', 'file_well_site', 'file_well', 'mag', '', 'round', 'well', 'site'
default_path_formula = {'raw': '[data]/[set]/[file]',
                        'calibrated': '[data]/[set]/[file_well_site].calibrated.tif',
                        'stitched': '[data]/[set]/[file_well_site].stitched.tif',
                        'aligned': '[data]/aligned/[set]/[file_well].aligned.tif',
                        'nuclei': '[data]/aligned/[set]/[file_well].aligned.nuclei.tif',
                        }
default_table_index = ['mag', 'set', 'well', 'site']


class Paths(object):
    def __init__(self, dataset, lasagna_path='/broad/blainey_lab/David/lasagna',
                 sub_dirs=None):
        """Store file paths relative to lasagna_path, allowing dataset to be loaded in different
        absolute locations. Retrieve raw files, stitched files, and nuclei. Pass stitch name and
        get nuclei name. Store path to calibration information.
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
        """Prepend dataset location, multiple arguments are joined.
        :param args:
        :return:
        """
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

    def update_table(self, file_pattern=default_file_pattern, groups=default_file_groups,
                     path_formula=default_path_formula, table_index=default_table_index):

        """Match pattern to find original files in dataset. For each raw file, apply patterns in
        self.dirs to generate new file names and add to self.table if file exists (else nan).

        E.g., raw file 'data/set/set_A1-Site1.ome.tif' is captured as ('set', 'Site1'). Pattern-matching yields:
         'aligned': '[data]/aligned/[set]/[file_well].aligned.tif'
         ==> data/aligned/set/set_A1.tif
         'calibrated': '[data]/[set]/[file_well_site].calibrated.tif'
         ==> data/set/set_A1-Site1.calibrated.tif

        :return:
        """

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

        self.table = pandas.DataFrame(d)
        self.table.set_index(table_index, inplace=True).sortlevel(inplace=True)

    def update_calibration(self):
        calibration_dir = self.full(self.dirs['calibration'])
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

    def table_(self, **kwargs):
        """Convenience function to subset conditions, e.g. table_(well='A1', mag='40X', site=0).
        :param kwargs:
        :return:
        """
        index = tuple()
        for n in self.table.index.names:
            if n in kwargs:
                index += (kwargs[n],)
            else:
                index += (slice(None),)
        return self.table.loc[index, :]

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
    :param shape:
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
    if type(text) is str:
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


def mark_features(shape, features, type='box'):
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
            im += mark_features(im.shape, blobs) * 2 ** i
    return im


def compress_obj(obj):
    s = StringIO.StringIO()
    pickle.dump(obj, s)
    out = zlib.compress(s.getvalue())
    s.close()
    return out


def decompress_obj(string):
    return pickle.load(StringIO.StringIO(zlib.decompress(string)))


def load_lut(name):
    file_name = os.path.join(config.luts, name + '.txt')
    with open(file_name, 'r') as fh:
        lines = fh.readlines()
        values = [line[:-1].split() for line in lines]
    return [int(y) for x in zip(*values) for y in x]


GLASBEY = load_lut('glasbey')
