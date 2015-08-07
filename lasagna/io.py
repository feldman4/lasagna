from itertools import product
from lasagna.utils import Memoized
from glob import glob
import pandas
import struct, os
import numpy as np
import regex as re
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

DEFAULT_LUTS = (BLUE, GREEN, RED, MAGENTA)

DIR = {}


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
    return data


# save load time, will bring trouble if the TiffFile reference is closed
@Memoized
def _load_tifffile(master):
    return TiffFile(master)


def get_row_stack(row, full=False, nuclei=False, apply_offset=False):
    """For a given DataFrame row, get image data for full frame or cell extents only.
    :param row:
    :param full:
    :param nuclei:
    :param apply_offset:
    :return:
    """
    I = _get_stack(row['name'], nuclei=nuclei)
    if full is False:
        I = I[b_idx(row)]
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
    shape = [max(s) for s in zip(*[x.shape for x in arr_])]
    # strange numpy limitations
    for i, x in enumerate(arr_):
        y = np.zeros(shape, x.dtype)
        slicer = [slice(None, s) for s in x.shape]
        y[slicer] = x
        arr_[i] = y[None, ...]

    return np.concatenate(arr_, axis=0)


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
def _get_stack(name, nuclei=False):
    if '/broad/' not in name:
        name = DIR['dataset_path'] + name
    if nuclei:
        name = '/'.join(name.split('/')[:-1] + ['nuclei', name.split('/')[-1]])
        return imread(name, multifile=False)
    I = imread(name, multifile=False)
    I = I.reshape(4, I.size / (4 * 1024 * 1024), 1024, 1024)
    return I


def save_hyperstack(name, data, autocast=True, resolution=None,
                    luts=None, display_ranges=None, compress=0):
    """input ND array dimensions as ([time], [z slice], channel, y, x)
    leading dimensions beyond 5 could be wrapped into time, not implemented
    """
    nchannels = data.shape[-3]
    if resolution is None:
        resolution = (1. / (UM_PER_PX[OBJECTIVE] * BINNING),) * 2
    if luts is None:
        luts = [x for x, _ in zip(DEFAULT_LUTS, range(nchannels))]
    if display_ranges is None:
        display_ranges = tuple([(x.min(), x.max())
                                for x in np.rollaxis(data, -3)])

    # convert to uint16
    tmp = data.copy()
    if autocast:
        tmp = tmp.astype(np.uint16)

    # metadata encoding LUTs and display ranges
    # see http://rsb.info.nih.gov/ij/developer/source/ij/io/TiffEncoder.java.html
    description = ij_description(data.shape)
    tag_50838 = ij_tag_50838(nchannels)
    tag_50839 = ij_tag_50839(luts, display_ranges)

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
    raise 'FuckYouError'


def get_magnification(s):
    match = re.search('([0-9]+X)', s)
    if match:
        return match.groups(1)[0]


def b_idx(row):
    """For a given DataFrame row, get slice index to cell in original data. Assumes 4D data.
    :param row:
    :return:
    """

    bounds = row['bounds']
    return (slice(None), slice(None), slice(bounds[0], bounds[2]), slice(bounds[1], bounds[3]))


def offset_stack(stack, offsets):
    """Applies offset to stack, periodic boundaries.
    :param stack: N-dim array
    :param offsets: list of offsets
    :return:
    """
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
                'analysis': 'analysis',
                'nuclei': 'analysis/nuclei',
                'stitch': 'stitch'}


class Paths(object):
    def __init__(self, dataset, lasagna_path='/broad/blainey_lab/David/lasagna',
                 sub_dirs=None):
        """Store file paths relative to lasagna_path, allowing dataset to be loaded in different
        absolute locations. Retrieve raw files, stitched files, and nuclei. Pass stitch name and
        get nuclei name.
        :param dataset:
        :param lasagna_path:
        :param stitch_dir:
        :param nuclei_dir:
        :return:
        """
        self.dirs = default_dirs if sub_dirs is None else sub_dirs
        self.table = None
        self.dataset = dataset
        # resolve path, strip trailing slashes
        self.lasagna_path = os.path.abspath(lasagna_path)

        self.update()

    def full(self, *args):
        return os.path.join(self.lasagna_path, self.dataset, *args)

    def relative(self, s):
        return s.replace(self.full() + '/', '')

    def parent(self, s):
        return os.path.basename(os.path.dirname(s))

    def update(self):
        """Look for .tif files in stitched directory. Look for matching raw data in other subdirectories.
        :return:
        """
        raw_files = []

        # look for raw images
        raw_dir = self.full(self.dirs['raw'])
        for root, dirs, files in os.walk(raw_dir):
            parent = os.path.basename(root)
            files = [self.relative(os.path.join(root, f)) for f in files]
            raw_files += [f for f in files if parent in f and '.tif' in f]

        # look for corresponding stitched images
        stitch_files = []
        stitch_dir = self.full(self.dirs['stitch'])
        for root, dirs, files in os.walk(stitch_dir):
            parent = os.path.basename(root)
            files = [self.relative(os.path.join(root, f)) for f in files]
            stitch_files += [f for f in files if parent in f and '.tif' in f]

        stitch_dict = {}
        for f in stitch_files:
            tmp = get_magnification(f), self.parent(f), get_well_site(f)[0]
            stitch_dict.update({tmp: f})

        raw_well_sites = zip(*[get_well_site(f) for f in raw_files])
        raw_sets = [self.parent(f) for f in raw_files]

        self.table = pandas.DataFrame({'file': [os.path.basename(f) for f in raw_files],
                                        'raw': raw_files,
                                        'mag': [get_magnification(s) for s in raw_files],
                                        'well': raw_well_sites[0],
                                        'site': raw_well_sites[1],
                                        'set': raw_sets,
                                        })
        self.table = self.table.set_index(['mag', 'set', 'well', 'site']).sortlevel()

        # match stitch names based on all index values except site
        for ix, row in self.table.iterrows():
            if ix[:-1] in stitch_dict:
                self.table.loc[ix, 'stitch'] = stitch_dict[ix[:-1]]

        self.add_analysis('stitch', 'nuclei')

    def add_analysis(self, column_in, analysis_name):
        # generate nuclei file names, stitch only for now
        for ix, row in self.table.iterrows():
            name_in = row[column_in]
            if pandas.notnull(name_in):
                self.table.loc[ix, analysis_name] = os.path.join(self.dirs['analysis'],
                                                                  analysis_name, name_in)

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



def initialize_paths(dataset, subset='',
                     lasagna_dir='/broad/blainey_lab/David/lasagna/'):
    """Define paths where data and job files are stored.
    :param dataset:
    :param subset:
    :param lasagna_dir:
    :return:
    """
    if subset:
        subset = os.path.normpath(subset)
    lasagna_dir = os.path.normpath(lasagna_dir)
    dataset = os.path.normpath(dataset)

    global DIR
    DIR = {'lasagna': lasagna_dir,
           'dataset': dataset}

    DIR['job_path'] = '%s/jobs' % DIR['lasagna']
    DIR['dataset_path'] = '%s/%s' % (DIR['lasagna'], DIR['dataset'])
    DIR['analysis'] = '%s/%s/analysis' % (DIR['lasagna'], DIR['dataset'])

    DIR['data_path'] = '%s/%s/*/*.tif' % (DIR['dataset_path'], subset)
    DIR['nuclei_path'] = '%s/%s/*/nuclei/*.tif' % (DIR['dataset_path'], subset)

    DIR['stacks'] = sorted(glob(DIR['data_path']), key=get_well_site)
    DIR['nuclei'] = sorted(glob(DIR['nuclei_path']), key=get_well_site)


print 'call lasagna.io.initialize_paths(dataset, ...) to set up lasagna.io.DIR'
