import functools
import numpy as np
import pandas as pd
import signal
import subprocess
import sklearn.utils.linear_assignment_


class Memoized(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kwargs):
        key = str(args) + str(kwargs)
        try:
            # if type(self.cache[key]) == np.ndarray:
            #                 return self.cache[key].copy()
            return self.cache[key]
        except KeyError:
            value = self.func(*args, **kwargs)
            self.cache[key] = value
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args, **kwargs)

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


class Filter2D(object):
    def __init__(self, func, window_size=200):
        """Create 2D fourier filter from 1D radial function.
        The filter itself is available as Filter2D.filter1D, .filter2D.
        :param func: 1D radial function in fourier space.
        :param window_size: can be anything, really
        :return:
        """
        self.func = func
        y = np.array([np.arange(window_size).astype(float)])
        self.filter1D = self.func(y[0])
        self.H = self.func(np.sqrt(y ** 2 + y.T ** 2))
        self.H = self.H / self.H.max()
        self.__call__(self.H)

    def __call__(self, M, pad_width=None):
        """
        :param M:
        :param pad_width: minimum pad width, pads up to next power of 2
        :return:
        """
        if pad_width:
            sz = 2 ** (int(np.log2(max(M.shape) + pad_width)) + 1)
            pad_width = [((sz - s) / 2, (sz - s) - (sz - s) / 2) for s in M.shape]
            M_ = np.pad(M, pad_width, mode='linear_ramp', end_values=(M.mean(),))
        else:
            M_ = M
        H = self.H
        M_fft = np.fft.fft2(M_)
        s, t = [s / 2 for s in M_fft.shape]
        h = np.zeros(M_.shape)
        h[:s, :t] = H[:s, :t]
        h[:-s - 1:-1, :t] = H[:s, :t]
        h[:-s - 1:-1, :-t - 1:-1] = H[:s, :t]
        h[:s, :-t - 1:-1] = H[:s, :t]
        self.M_pre_abs = np.fft.ifft2(h * M_fft)
        M_f = np.abs(self.M_pre_abs)
        self.filter2D = np.fft.fftshift(h)
        # out = M_f * (M_.sum()/M_f.sum())
        if pad_width:
            M_f = M_f[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]
        return M_f

    def build_pyramid(self, real_filter, start=1, end=10):
        pyramid = {}
        if len(real_filter) < 2 ** end:
            real_filter = np.pad(real_filter, (0, 2 ** end - len(real_filter)),
                                 mode='constant', constant_values=0)
        for width in [2 ** i for i in range(start, end + 1)]:
            pyramid[width] = np.zeros(width)
            x = [float(width) / i for i in range(1, width + 1)]
            pyramid[width] = np.interp(x, range(1, len(real_filter) + 1), real_filter)
        return pyramid


def mad(arr, axis=None, keepdims=True):
    med = np.median(arr, axis=axis, keepdims=keepdims)
    return np.median(np.abs(arr - med), axis=axis, keepdims=keepdims)


class Filter2DReal(object):
    def __init__(self, func, max_size=10):
        self.pyramid = None
        self.pyramid_2D = None
        self.max_size = max_size
        self.x = np.arange(0., 2. ** (max_size + 1), 0.2)
        self.real_filter = [func(x) for x in self.x]

        self.build_pyramid()

    def __call__(self, M, pad_width=2):
        i = int(np.ceil(np.log2(max(M.shape) + pad_width)))
        width = 2 ** i

        pad_width = [(int((width - s) / 2), int((width - s) - (width - s) / 2)) for s in M.shape]
        M_ = np.pad(M, pad_width, mode='linear_ramp', end_values=(M.mean(),))

        self.M_ = M_

        M_fft = np.fft.fft2(M_)

        slic = (slice(pad_width[0][0], -pad_width[0][1]),
                slice(pad_width[1][0], -pad_width[1][1]))

        self.M_pre_abs = np.fft.ifft2(M_fft * self.pyramid_2D[i])
        M_filt = np.abs(self.M_pre_abs)
        M_filt[M_filt < 0] = 0

        self.M_pre_abs = self.M_pre_abs[slic]

        return M_filt[slic]

    def build_pyramid(self):
        self.pyramid = {}
        self.pyramid_2D = {}

        y = np.array([self.x])

        for i in range(1, self.max_size + 1):
            width = 2. ** (i - 1)

            y = [width / x for x in np.arange(1, width + 1)]
            self.pyramid[i] = np.interp(y, self.x, self.real_filter)

            z = np.array([np.arange(1, width + 1)])
            r = np.sqrt(z ** 2 + z.T ** 2)

            f = np.interp(r, z[0], self.pyramid[i])
            self.pyramid_2D[i] = self.quadrant_to_full(f)

    def quadrant_to_full(self, m):
        n = np.r_[m, m[::-1, :]]
        return np.c_[n, n[:, ::-1]]


def plot_image_overlay(image0, image1, offset, ax=None):
    from matplotlib import pyplot as plt
    from matplotlib import cm

    sz0, sz1 = image0.shape, image1.shape
    # (x0, x1, y0, y1); flip y by convention
    extent = (0 + offset[1], sz1[1] + offset[1],
              sz1[0] + offset[0], 0 + offset[0])
    print extent, sz1
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image0, cmap=cm.Blues, alpha=0.5)
    ax.hold('on')
    ax.imshow(image1, cmap=cm.Reds, extent=extent, alpha=0.5)
    ax.axis('image')


def plot_corner_alignments(data, n=500, axs=None):
    """Superimpose corners of 3-D array. Only uses first two entries in leading dimension.
    :param data: [2, height, width] array
    :param n: corner window width
    :param axs: list of up to 4 axes to plot to (clockwise from top left)
    :return:
    """
    import matplotlib.pyplot as plt
    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    corners = _get_corners(n)
    for ax, corner in zip(axs, (0, 1, 3, 2)):
        plot_image_overlay(data[corners[corner]][0],
                           data[corners[corner]][1],
                           np.array([0, 0]), ax=ax)
    return axs


def _get_corners(n):
    """Retrieve slice index for 2-D corners of 3-D array.
    :param n:
    :return:
    """
    a, b, c = slice(None), slice(None, n), slice(-n, None)
    corners = ((a, b, b),
               (a, b, c),
               (a, c, c),
               (a, c, b))
    return corners


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

    output = x.as_matrix().reshape(reshaper[::-1])
    return output, levels[::-1]


def argsort_nd(a, axis):
    """Properly format argsort result so it can be used to index original array.
    :param a:
    :param axis:
    :return:
    """
    index = list(np.ix_(*[np.arange(i) for i in a.shape]))
    index[axis] = a.argsort(axis)
    return index


def argmax_nd(a):
    return np.unravel_index(a.argmax(), a.shape)


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


def show_grid(z, force_fit=False):
    import qgrid
    qgrid.set_defaults(remote_js=True, precision=4)

    new = pd.DataFrame()
    for x in z.columns:
        new[' '.join(x)] = z[x]
    return qgrid.show_grid(new, grid_options={'forceFitColumns': force_fit, 'defaultColumnWidth': 120})


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


def call(arg, stdin, shell=True):
    """Call process with stdin provided (equivalent to cat), return stdout.
    :param arg:
    :param stdin:
    :return:
    """
    p = subprocess.Popen(arg, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, shell=shell)
    p.stdin.write(stdin)
    p.stdin.close()
    return p.stdout.read()

def linear_assignment(df):
    """Wrapper of sklearn linear assignment algorithm for DataFrame cost matrix. Returns
    DataFrame with columns for matched labels. Minimizes cost.
    """
    x = sklearn.utils.linear_assignment_.linear_assignment(df.as_matrix())
    y = zip(df.index[x[:,0]], df.columns[x[:,1]])
    df_out = pd.DataFrame(y, columns = [df.index.name, df.columns.name])
    return df_out


class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

