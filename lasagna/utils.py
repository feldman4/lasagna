import time
import os
import functools
import regex as re

from functools import wraps
from inspect import getargspec, isfunction
from itertools import izip, ifilter, starmap, product
from collections import OrderedDict, Counter
from decorator import decorator 

import numpy as np
import pandas as pd



# PYTHON
class Memoized(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    Numpy arrays are treated specially with `copy` kwarg.
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kwargs):
        key = str(args) + str(kwargs)
        try:
            if isinstance(self.cache[key], np.ndarray):
                if kwargs.get('copy', True):
                    return self.cache[key].copy()
                else:
                    return self.cache[key]
            return self.cache[key]
        except KeyError:
            value = self.func(*args, **kwargs)
            self.cache[key] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        fn = functools.partial(self.__call__, obj)
        fn.reset = self._reset
        return fn

    def _reset(self):
        self.cache = {}


def compress_obj(obj):
    s = StringIO.StringIO()
    pickle.dump(obj, s)
    out = zlib.compress(s.getvalue())
    s.close()
    return out


def decompress_obj(string):
    return pickle.load(StringIO.StringIO(zlib.decompress(string)))


def concatMap(f, xs):
    return sum(map(f, xs), [])


def call(arg, stdin='', shell=True):
    """Call process with stdin provided (equivalent to cat), return stdout.
    :param arg:
    :param stdin:
    :return:
    """
    import subprocess
    p = subprocess.Popen(arg, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, shell=shell)
    p.stdin.write(stdin)
    p.stdin.close()
    return p.stdout.read()


def timestamp(filename=None, clock=False):
    import time
    time_format = '%Y%m%d-%H%M%S' if clock else '%Y%m%d'
    stamp = time.strftime(time_format)
    if filename:
        directory = os.path.dirname(filename)
        name = os.path.basename(filename)
        return os.path.join(directory, '{0}_{1}'.format(stamp, name))
    else:
        return stamp
    

# PANDAS
def standardize(x):
    # only standardize numeric columns
    numerics = (np.float64, np.int64)
    filt = [d[0] for d in x.dtypes.iteritems() if d[1] in numerics]
    return (x[filt] - x[filt].mean()) / x[filt].std()


def normalize_rows(x):
    """Divide rows of DataFrame by L2 norm.
    :param x:
    :return:
    """
    return x.divide(((x ** 2).sum(axis=1)) ** 0.5, axis=0)


def group_sort(x, columns, top_n=None, **kwargs):
    """Sort dataframe by column corresponding to groupby index.
    columns: {index: column}
    if index is not a tuple, index => (index,)
    """
    index = x.index.values[0]
    for k, column in columns.items():
        # e.g., index 'B2' => ('B2',)
        k = (k,) if type(k) != tuple else k
        # index may cover some but not all levels of row MultiIndex
        if all(a == b for a, b in zip(k, index)):
            output = x.sort(columns=column, **kwargs)
            break
    else:
        raise IndexError('index not in keys of `columns`')

    if top_n:
        output = output.ix[:top_n]
        output.index = output.index.droplevel(range(len(k)))
    return output


def print_table(df):
    """Call returned function to print table in IPython notebook with escaped newlines.
    Doesn't work for MultiIndex.
    :param df:
    :return:
    """
    from IPython.display import HTML
    import cgi

    def escape(a):
        return cgi.escape(a).replace('\n', '<br>')

    htm = '<table>' + \
          '<thead><tr><th></th>' + \
          ''.join(['<th nowrap>' + escape(c) + \
                   '</th>' for c in df]) + '</tr></thead>' + \
          '<tbody>' + ''.join(['<tr>' + '<th>' + str(r[0]) + \
                               '</th>' + ''.join(['<td nowrap>' + escape(c) + \
                                                  '</td>' for c in r[1]]) + '</tr>' for r in enumerate(df.values)]) + \
          '</tbody></table>'

    return lambda: HTML(htm)


def pivot(x, plus=1, minus=0, expand=False):
    """Pivot table keeping existing index. Create full product column MultiIndex from all
    columns, set entries from original table to plus, rest to minus.
    :return:
    """
    x = x.copy()
    columns = list(x.columns)
    x['dummy'] = plus
    x = x.reset_index().pivot_table(values='dummy',
                                    index=x.index.name,
                                    columns=columns)
    if expand:
        index = pd.MultiIndex.from_product(x.index.levels,
                                           names=x.index.names)
        x = x.transpose().reindex(index, fill_value=minus).transpose()

    return x.fillna(minus)


def jitter(x, r=0.1):
    """Add jitter to matrix of data, proportional to standard deviation of
    each column, with scale factor r.
    """
    return x + np.random.rand(*x.shape) * np.std(x)[None, :] * r


class DataFrameFind(pd.DataFrame):
    def __call__(self, **kwargs):
        """Add dictionary-style find function for DataFrame with MultiIndex.
        df(index1='whatever', index3=['a', 'b'])
        """
        index = tuple()
        for n in self.index.names:
            if n in kwargs:
                index += (kwargs[n],)
            else:
                index += (slice(None),)
        return self.loc[index, :]


def comma_split(df, column, split=', '):
    """Split entries in given column of strings, duplicating the 
    index and the rest of the row.
    """
    arr, index = [], []
    for ix, row in df.iterrows():
        for entry in row[column].split(split):
            r = row.copy()
            r[column] = entry
            arr += [r]
            index += [ix]
    df_out = pd.concat(arr, axis=1).T
    if isinstance(df.index, pd.MultiIndex):
        df_out.index = pd.MultiIndex.from_tuples(index, names=df.index.names)
    else:
        df_out.index = index
        df_out.index.name = df.index.name
    return df_out

def bin_join(xs, symbol):
    symbol = ' ' + symbol + ' ' 
    return symbol.join('(%s)' % x for x in xs)
        
or_join  = functools.partial(bin_join, symbol='|')
and_join = functools.partial(bin_join, symbol='&')

def groupby_reduce_concat(gb, *args, **kwargs):
    """
    df = (df_cells
          .groupby(['stimulant', 'gene'])['gate_NT']
          .pipe(groupby_reduce_concat, 
                fraction_gate_NT='mean', 
                cell_count='size'))
    """
    for arg in args:
    	kwargs[arg] = arg
    reductions = {'mean': lambda x: x.mean(),
                  'sem': lambda x: x.sem(),
                  'size': lambda x: x.size(),
                  'count': lambda x: x.size(),
                  'sum': lambda x: x.sum(),
                  'sum_int': lambda x: x.sum().astype(int),
                  'first': lambda x: x.nth(0),
                  'second': lambda x: x.nth(1)}
    
    for arg in args:
        if arg in reductions:
            kwargs[arg] = arg

    arr = []
    for name, f in kwargs.items():
        if callable(f):
            arr += [f(gb).rename(name)]
        else:
            arr += [reductions[f](gb).rename(name)]

    return pd.concat(arr, axis=1).reset_index()

def groupby_histogram(df, index, column, bins, cumulative=False):
    """Substitute for df.groupby(index)[column].histogram(bins),
    only supports one column label.
    """
    maybe_cumsum = lambda x: x.cumsum(axis=1) if cumulative else x
    column_bin = column + '_bin'
    column_count = column + ('_csum' if cumulative else '_count')
    return (df
        .assign(**{column_bin: bins[np.digitize(df[column], bins) - 1]})
        .pivot_table(index=index, columns=column_bin, values=df.columns[0], 
                     aggfunc='count')
        .reindex(labels=list(bins), axis=1)
        .fillna(0)
        .pipe(maybe_cumsum)
        .stack().rename(column_count)
        .astype(int).reset_index()
           )

# GLUE
def show_grid(z, force_fit=False):
    import qgrid
    qgrid.set_defaults(remote_js=True, precision=4)

    new = pd.DataFrame()
    for x in z.columns:
        new[' '.join(x)] = z[x]
    return qgrid.show_grid(new, grid_options={'forceFitColumns': force_fit, 'defaultColumnWidth': 120})


# IMAGEJ
def launch_queue(queue, log=None):
    import threading

    def evaluate(q):
        """Call functions in queue (doesn't return results).
        Entries in queue are formatted as (func, (args, kwargs)).
        Example:
            def f(x):
                x[0] += 1
            y = [0]
            q += [[f, ([y], {})]]
        """
        import time
        while True:
            time.sleep(0.1)
            if len(q):
                f, (args, kwargs) = q.pop()
                f(*args, **kwargs)
                if log is not None:
                    log.append([f, (args, kwargs)])

    t = threading.Thread(target=evaluate, args=(queue,))
    t.daemon = True
    t.start()
    return t


def start_client():
    """Start rpyc client connected to jython instance. Returns object representing
    base workspace of jython instance, including imported (Java) classes.
    """
    from lasagna.rpyc_utils import start_client
    client = start_client()
    import lasagna.jython_imports
    j = lasagna.jython_imports.jython_import(client.root.exposed_get_head(),
                                             client.root.exposed_execute)
    return j


def pack_contours(contours):
    """Pack contours into lists that can be sent to jython overlay_contours
    """
    # imagej origin is at top left corner, matplotlib origin is at center of top left pixel
    packed = [(0.5 + c).T.tolist() for c in contours]
    return packed



def linear_assignment(df):
    """Wrapper of sklearn linear assignment algorithm for DataFrame cost matrix. Returns
    DataFrame with columns for matched labels. Minimizes cost.
    """
    from sklearn.utils.linear_assignment_ import linear_assignment

    x = linear_assignment(df.values)
    y = zip(df.index[x[:, 0]], df.columns[x[:, 1]])
    df_out = pd.DataFrame(y, columns=[df.index.name, df.columns.name])
    return df_out


def cells_to_barcodes(ind_vars_table, cloning=None):
    """
    :param cloning: dict of DataFrames based on Lasagna Cloning sheets
    """
    cells = ind_vars_table['cells']
    virus = cloning['cell lines'].loc[cells, 'lentivirus']
    plasmids = cloning['lentivirus'].loc[virus, 'plasmid']
    plasmids = plasmids.fillna('')
    barcodes = cloning['plasmids'].loc[plasmids, 'barcode']
    barcodes = barcodes.fillna('')
    # split comma-separated list of barcodes
    ind_vars_table['barcodes'] = [tuple(x.split(', ')) for x in barcodes]


def sample(line=tuple(), plane=tuple(), scale='um_per_px'):
    import skimage.transform
    """Wrap image processing functions.
    
    **USUALLY UNNECESSARY TO SCALE KWARG FUNCTION PARAMETERS**

    Given a function defined as:
        @sample(line=('linear_arg1', 'linear_arg2), 
                plane='plane_arg')
        output_image = func(*input_images, 
                             linear_arg1=1, 
                             linear_arg2=1,
                             plane_arg=1,
                             um_per_px=0.5)
    
    input_images will be resized by a factor of 0.5 and the function 
    called as:
        output_image = func(*input_images_scaled,
                             linear_arg1=0.5,
                             linear_arg2=0.5,
                             plane_arg=0.25,
                             um_per_px=0.5)
    
    If the output is a numpy.ndarray, it will be rescaled to the shape of 
    the input image.
    """
    # pass in argument names or lists of arguments
    if isinstance(line, str):
        line = [line]
    if isinstance(plane, str):
        plane = [plane]

    def wrapper_of_f(func):
        def wrapped_f(*args, **kwargs):
            spec = getargspec(func)
            kwargs_ = dict(zip(spec.args[::-1], spec.defaults[::-1]))
            kwargs_.update(kwargs)
            kwargs = kwargs_

            if kwargs[scale] != 1:
                # rescale image arguments
                images = []
                for image in args:
                    if np.issubdtype(image.dtype, np.integer):
                        order = 0
                    else:
                        order = 1
                    scaled = skimage.transform.rescale(image, kwargs[scale],
                                                       order=order, preserve_range=True)
                    images += [scaled.astype(image.dtype)]

                # adjust parameter scaling
                for kw in kwargs:
                    if kw in line:
                        kwargs[kw] = kwargs[kw] / kwargs[scale]
                    if kw in plane:
                        kwargs[kw] = kwargs[kw] / (kwargs[scale] ** 2)
            else:
                images = args

            output = func(*images, **kwargs)

            # rescale output
            if isinstance(output, np.ndarray):
                if np.issubdtype(output.dtype, np.integer):
                    order = 0
                else:
                    order = 1
                scaled = skimage.transform.resize(output, args[0].shape,
                                                  order=order, preserve_range=True)
                return scaled.astype(output.dtype)
            return output

        # return decorator.decorate(func, wrapped_f)
        return wrapped_f

    return wrapper_of_f


def import_fcs(files, drop=lambda s: '-A$' in s):
    """Import a list of FACS files, adding well and x,y info if available.
    """
    import FlowCytometryTools as fcs
    wells = [r + str(c) for r in 'ABCDEFGH' for c in range(1, 13)]
    arr = []
    for f in files:
        d = fcs.FCMeasurement(ID=f, datafile=f)
        df = d.data
        df['Time'] -= df['Time'].min()
        df['Time'] /= df['Time'].max()
        for w in wells:
            if w in f:
                df['well'] = w
                y = 'ABCDEFGH'.index(w[0])
                x = int(w[1:]) - 1
                df['x'] = 1.2 * x + df['Time']
                df['y'] = (-1.2 * y) + np.random.rand(df.shape[0])
                break
        else:
            df['well'] = 'none'
            df['x'], df['y'] = 0, 0
        arr += [df]
    df = pd.concat(arr)
    return df.drop([c for c in df.columns if drop(c)], axis=1)


def to_row_col(s):
    pat = '([A-Z])([0-9]+)'
    match = re.findall(pat, s)
    if match:
        row, col = match[0][0], int(match[0][1])
        return row, col + 100
    else:
        raise ValueError('%s not a well' % s)


def pivot_96w(wells, values):
    rows = [(v,) + to_row_col(w) for w, v in zip(wells, values)]
    df = (pd.DataFrame(rows, columns=['value', 'row', 'column'])
          .pivot(values='value', index='row', columns='column'))
    return df


def long_to_wide(df, values, columns, index, extra_values):
    """Pivots dataframe based on `values`, `columns`, and `index`.
    Columns in `extra_values` are not pivoted, but the first value at the pivot
    index is kept.
    """

    df_long = pd.pivot_table(df, columns=columns, 
               values=values, index=index)

    if extra_values:
        # copy in columns that didn't need to be pivoted
        df2 = df.drop_duplicates(index).set_index(index)
        for col in extra_values:
            df_long[col] = df2[col]

    return df_long.reset_index()


def load_ij_roi(filename):
    import ijroi
    with open(filename, "rb") as fh:
        return ijroi.read_roi(fh).astype(int)


def int_mode(x):
    """Works for integer arrays only."""
    bc = np.bincount(x.flatten())
    return np.argmax(bc)
 
@decorator
def applyIJ(f, arr, *args, **kwargs):   
    """Apply a function that expects 2D input to the trailing two
    dimensions of an array. The function must output an array whose shape
    depends only on the input shape. 
    """
    h, w = arr.shape[-2:]
    reshaped = arr.reshape((-1, h, w))

    arr_ = []
    for frame in reshaped:
        arr_ += [f(frame, *args, **kwargs)]

    output_shape = arr.shape[:-2] + arr_[0].shape
    return np.array(arr_).reshape(output_shape)


def ndarray_to_dataframe(values, index):
    names, levels  = zip(*index)
    columns = pd.MultiIndex.from_product(levels, names=names)
    df = pd.DataFrame(values.reshape(values.shape[0], -1), columns=columns)
    return df

def categorize(df, subset=None, exclude_subset=None, **custom_order):
    from pandas.api.types import is_object_dtype
    exclude_subset = [] if exclude_subset is None else exclude_subset
    if subset is None:
        subset = set(df.columns) - set(exclude_subset)

    for col in subset:
        df[col] = df[col].astype('category')

        if col in custom_order:
            df[col] = df[col].cat.reorder_categories(custom_order[col])

    return df

def uncategorize(df):
    """Pivot and concat are weird with categories.
    """
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].astype('object')
    return df

def base_repr(x, n, width, to_str=True):
    x = int(x)
    arr = []
    for i in range(width)[::-1]:
        y = x // n**i
        arr.append(y)
        x -= n**i * y
    assert x == 0
    if to_str:
        return ''.join(str(y) for y in arr)
    return arr

def apply_subset(f, subset):
    """usage: df.pipe(apply_subset(f, columns))
    """
    def wrapped(df):
        df = df.copy()
        df[subset] = f(df[subset])
        return df
    return wrapped


def write_excel(filename, sheets):
    """
    d = {'files': df_finfo, 'read stats': df_stats}
    write_excel('summary', d.items())
    """
    if not filename.endswith('xlsx'):
        filename = filename + '.xlsx'
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    for name, df in sheets:
        df.to_excel(writer, index=None, sheet_name=name)
    writer.save()


class Mask(object):
    def __init__(self, mask):
        """Hack to avoid slow printing of DataFrame containing boolean np.ndarray.
        """
        self.mask = mask
    def __repr__(self):
        return str(self.mask.shape) + ' mask'



# NUMPY
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


def tile(arr, m, n, pad=None):
    """Divide a stack of images into tiles of size m x n. If m or n is between 
    0 and 1, it specifies a fraction of the input size. If pad is specified, the
    value is used to fill in edges, otherwise the tiles may not be equally sized.
    Tiles are returned in a list.
    """
    assert arr.ndim > 1
    h, w = arr.shape[-2:]
    # convert to number of tiles
    m_ = h / m if m >= 1 else int(np.round(1 / m))
    n_ = w / n if n >= 1 else int(np.round(1 / n))

    if pad is not None:
        pad_width = (arr.ndim - 2) * ((0, 0),) + ((0, -h % m), (0, -w % n))
        arr = np.pad(arr, pad_width, 'constant', constant_values=pad)
        print arr.shape

    h_ = int(int(h / m) * m)
    w_ = int(int(w / n) * n)

    tiled = np.array_split(arr[:h_, :w_], m_, axis=-2)
    tiled = lasagna.utils.concatMap(lambda x: np.array_split(x, n_, axis=-1), tiled)
    return tiled


def trim(arr, return_slice=False):
    """Remove i,j area that overlaps a zero value in any leading
    dimension. Trims stitched and piled images.
    """
    def coords_to_slice(i_0, i_1, j_0, j_1):
        return slice(i_0, i_1), slice(j_0, j_1)

    leading_dims = tuple(range(arr.ndim)[:-2])
    mask = (arr == 0).any(axis=leading_dims)
    coords = inscribe(mask)
    sl = (Ellipsis,) + coords_to_slice(*coords)
    if return_slice:
        return sl
    return arr[sl]


def inscribe(mask):
    """Guess the largest axis-aligned rectangle inside mask. 
    Rectangle must exclude zero values. Assumes zeros are at the 
    edges, there are no holes, etc. Shrinks the rectangle's most 
    egregious edge at each iteration.
    """
    h, w = mask.shape
    i_0, i_1 = 0, h - 1
    j_0, j_1 = 0, w - 1
    
    def edge_costs(i_0, i_1, j_0, j_1):
        a = mask[i_0, j_0:j_1 + 1].sum()
        b = mask[i_1, j_0:j_1 + 1].sum()
        c = mask[i_0:i_1 + 1, j_0].sum()
        d = mask[i_0:i_1 + 1, j_1].sum()  
        return a,b,c,d
    
    def area(i_0, i_1, j_0, j_1):
        return (i_1 - i_0) * (j_1 - j_0)
    
    coords = [i_0, i_1, j_0, j_1]
    while area(*coords) > 0:
        costs = edge_costs(*coords)
        if sum(costs) == 0:
            return coords
        worst = costs.index(max(costs))
        coords[worst] += 1 if worst in (0, 2) else -1
    return


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


def to_nd_array(x):
    """Converts DataFrame with MultiIndex rows and columns to ndarray.
    Accepts regular Index too.
    Will throw error if size doesn't match. Inner-most row level can have
    non-repeated values.
    """
    # update index to remove missing values
    x = x.set_index(pd.MultiIndex.from_tuples(x.index.values))
    levels = []
    last = -1
    for i, index in enumerate((x.columns, x.index)):
        if type(index) is pd.Index:
            add = [index]
        else:
            # reshape goes from inside out
            # x = x.sortlevel(axis=1-i)
            add = list(index.levels)[::-1]
        levels += add

    last = -1 * len(add)
    reshaper = [len(s) for s in levels]
    if np.prod(reshaper) != x.size:
        reshaper = reshaper[:last] + [-1] + reshaper[last + 1:]
        size = np.prod(reshaper[:last] + reshaper[last + 1:])
        levels[last] = pd.Index(range(x.size / size),
                                name=levels[last].name)
        print 'resetting innermost row index to [%d...%d]' % (levels[last][0], levels[last][-1])

    output = x.values.reshape(reshaper[::-1])
    return output, levels[::-1]


def argsort_nd(a, axis):
    """Format argsort result so it can be used to index original array.
    :param a:
    :param axis:
    :return:
    """
    index = list(np.ix_(*[np.arange(i) for i in a.shape]))
    index[axis] = a.argsort(axis)
    return index


def pad(array, pad_width, mode=None, **kwargs):
    """Extend numpy.pad to support negative pad width.
    """
    if type(pad_width) == int:
        if pad_width < 0:
            s = [slice(-1 * pad_width, pad_width)] * array.ndim
            return array[s]
    return np.pad(array, pad_width, mode=mode, **kwargs)


def argmax_nd(a):
    return np.unravel_index(a.argmax(), a.shape)


def binary_contours(img, fix=True, labeled=False):
    """Find contours of binary image, or labeled if flag set. For labeled regions,
    returns contour of largest area only.
    :param img:
    :return: list of nx2 arrays of [x, y] points along contour of each image.
    """

    def fixed_contour(contour):
	    """Fix contour generated from binary mask to exactly match outline.
	    """
	    # adjusts corner points based on CCW contour
	    def f(x0, y0, x1, y1):
	        d = (x1 - x0, y1 - y0)
	        if not(x0 % 1):
	            x0 += d[0]
	        else:
	            y0 += d[1]
	        return x0, y0

	    x, y = contour.T
	    xy = []
	    for k in range(len(x) - 1):
	        xy += [f(x[k], y[k], x[k + 1], y[k + 1])]

	    return np.array(xy)

    if labeled:
        regions = skimage.measure.regionprops(img)
        contours = [sorted(skimage.measure.find_contours(np.pad(r.image, 1, mode='constant'), 0.5, 'high'),
                key=lambda x: len(x))[-1] for r in regions]
        contours = [contour + [r.bbox[:2]] for contour,r in zip(contours, regions)]
    else:
        # pad binary image to get outer contours
        contours = skimage.measure.find_contours(np.pad(img, 1, mode='constant'),
                                                 level=0.5)
        contours = [contour - 1 for contour in contours]
    if fix:
        return [fixed_contour(c) for c in contours]
    return contours


def regionprops(labeled, intensity_image):
    """Supplement skimage.measure.regionprops with additional field containing full intensity image in
    bounding box (useful for filtering).
    :param args:
    :param kwargs:
    :return:
    """
    import skimage.measure

<<<<<<< HEAD
    regions = skimage.measure.regionprops(labeled, intensity_image=intensity_image)
=======
    regions = skimage.measure.regionprops(labeled, intensity_image=intensity_image[..., 0, :, :])
>>>>>>> df3e3b9b3706c9a6929681f9ad899a44aff1d3fc
    for region in regions:
        b = region.bbox
        region.intensity_image_full = intensity_image[..., b[0]:b[2], b[1]:b[3]]

    return regions