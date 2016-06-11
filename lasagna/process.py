import json
import random
import os
from collections import defaultdict

from skimage import transform
import skimage
from skimage.feature import register_translation
import numpy as np
import pandas as pd
import scipy.stats
from skimage.filters import gaussian_filter, threshold_adaptive
from skimage.morphology import disk, watershed, opening
from skimage.util import img_as_uint
import skimage.measure
from skimage.feature import peak_local_max
from scipy import ndimage

from lasagna import io
from lasagna import config
from lasagna.utils import Filter2D, _get_corners
import lasagna.utils

DOWNSAMPLE = 2

default_features = {'mean': lambda region: region.intensity_image[region.image].mean(),
                    'median': lambda region: np.median(region.intensity_image[region.image]),
                    'max': lambda region: region.intensity_image[region.image].max()}

default_nucleus_features = {
    'area':     lambda region: region.area,
    'centroid': lambda region: region.centroid,
    'bounds':   lambda region: region.bbox,
    'label':    lambda region: np.median(region.intensity_image[region.intensity_image > 0]),
    'mask':     lambda region: Mask(region.image),
    'hash':     lambda region: hex(random.getrandbits(128))
}


def binary_contours(img, fix=True, labeled=False):
    """Find contours of binary image, or labeled if flag set. For labeled regions,
    returns contour of largest area only.
    :param img:
    :return: list of nx2 arrays of [x, y] points along contour of each image.
    """
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


def feature_table(data, mask, features):
    """Apply functions in features to regions in data specified by
    integer mask.
    """
    regions = lasagna.utils.regionprops(mask, intensity_image=data)
    results = {feature: [] for feature in features}
    for region in regions:
        for feature, func in features.items():
            results[feature] += [func(region)]
    return pd.DataFrame(results)

def build_feature_table(stack, mask, features, index):
    """Iterate over leading dimensions of stack. Label resulting 
    table by index = (index_name, index_values).

        >>> stack.shape 
        (3, 4, 511, 626)
        
        index = (('round', range(1,4)), 
                 ('channel', ('DAPI', 'Cy3', 'A594', 'Cy5')))
    
        build_feature_table(stack, mask, features, index) 

    """
    from itertools import product
    index_vals = list(product(*[vals for _,vals in index]))
    index_names = [x[0] for x in index]
    
    s = stack.shape
    results = []
    for frame, vals in zip(stack.reshape(-1, s[-2], s[-1]), index_vals):
        df = feature_table(frame, mask, features)
        for name, val in zip(index_names, vals):
            df[name] = val
        results += [df]
    
    return pd.concat(results)


def register_images(images, index=None, window=(500, 500), upsample=1.):
    """Register a series of image stacks to pixel accuracy.
    :param images: list of N-dim image arrays, height and width may differ
    :param index: image[index] should yield 2D array with which to perform alignment
    :param window: centered window in which to perform registration, smaller is faster
    :param upsample: align to sub-pixels of width 1/upsample
    :return list[(int)]: list of offsets
    """
    if index is None:
        index = ((0,) * (images[0].ndim - 2) + (slice(None),) * 2)

    sz = [image[index].shape for image in images]
    sz = np.array([max(x) for x in zip(*sz)])

    origin = np.array(images[0].shape) * 0.

    center = tuple([slice(s / 2 - min(s / 2, rw), s / 2 + min(s / 2, rw))
                    for s, rw in zip(sz, window)])

    def pad(img):
        pad_width = [(s / 2, s - s / 2) for s in (sz - img.shape)]
        img = np.pad(img, pad_width, 'constant')
        return img[center], np.array([x[0] for x in pad_width]).astype(float)

    image0, pad_width = pad(images[0][index])
    offsets = [origin.copy()]
    offsets[0][-2:] += pad_width
    for image in [x[index] for x in images[1:]]:
        padded, pad_width = pad(image)
        shift, error, _ = register_translation(image0, padded, upsample_factor=upsample)

        offsets += [origin.copy()]
        offsets[-1][-2:] = shift + pad_width  # automatically cast to uint64

    return offsets


class Sample(object):
    def __init__(self, rate):
        """Provides methods for downsampling and upsampling trailing XY dimensions of ndarray.
        Automatically uses original shape for upsampling.
        :param rate:
        :return:
        """
        self.rate = float(rate)
        self.sampled = {}

    def downsample(self, img, shape=None):
        """Downsample image according to Sample.rate, or shape if provided.
        :param img:
        :param shape: tuple indicating downsampled XY dimensions
        :return:
        """
        if shape is None:
            shape = tuple([int(s / self.rate) for s in img.shape[-2:]])

        new_img = np.zeros(img.shape[:-2] + shape, dtype=img.dtype)

        for idx in np.ndindex(img.shape[:-2]):
            # parameters necessary to properly transform label arrays by non-integer factors
            new_img[idx] = transform.resize(img[idx], shape, order=0,
                                            mode='nearest', preserve_range=True)

        # store correct shape for inverting
        self.sampled[(shape, self.rate)] = img.shape[-2:]

        return new_img

    def upsample(self, img, shape=None):
        if shape is None:
            s = (img.shape[-2:], self.rate)
            if s in self.sampled:
                shape = self.sampled[s]
            else:
                shape = tuple([int(s * self.rate) for s in img.shape[-2:]])

        new_img = np.zeros(img.shape[:-2] + shape, dtype=img.dtype)

        for idx in np.ndindex(img.shape[:-2]):
            # parameters necessary to properly transform label arrays by non-integer factors
            new_img[idx] = transform.resize(img[idx], shape, order=0,
                                            mode='nearest', preserve_range=True)

        return new_img


@lasagna.utils.sample()
def find_nuclei(dapi, radius=15, area_min=50, area_max=500, um_per_px=1., 
                score=lambda r: r.mean_intensity,
                threshold=skimage.filters.threshold_otsu):
    """Could downsample to consistent pixel size (40X?)
    """
    area = area_min, area_max
    smooth = 1.35 # gaussian smoothing in watershed

    mask = _binarize(dapi, radius, area[0])
    labeled = skimage.measure.label(mask, background=0) + 1
    labeled = filter_by_region(labeled, score, threshold, intensity=dapi) > 0

    # only fill holes below minimum area
    filled = ndimage.binary_fill_holes(labeled)
    change = filter_by_region(filled!=labeled, lambda r: r.area < area[0], 0)
    labeled[change] = filled[change]

    nuclei = apply_watershed(labeled, smooth=smooth)

    return filter_by_region(nuclei, lambda r: area[0] < r.area < area[1], threshold)


def _binarize(dapi, radius, min_size):
    """Apply local mean threshold to find cell outlines. Filter out 
    background shapes. Otsu threshold on list of region mean intensities will remove a few
    dark cells. Could use shape to improve the filtering.
    """
    dapi = skimage.img_as_ubyte(dapi)
    # slower than optimized disk in imagej, scipy.ndimage.uniform_filter with square is fast but crappy
    meanered = skimage.filters.rank.mean(dapi, selem=disk(radius))
    mask = dapi > meanered
    mask = skimage.morphology.remove_small_objects(mask, min_size=min_size)

    return mask


def filter_by_region(labeled, score, threshold, intensity=None):
    """Apply a filter to labeled image. The key function takes a single region as input and
    returns a score. Regions are filtered out by score using the
    provided threshold function. If scores are boolean, scores are used as a mask and 
    threshold is disregarded. 
    """
    labeled = labeled.copy()

    if intensity is None:
        regions = skimage.measure.regionprops(labeled)
    else:
        regions = skimage.measure.regionprops(labeled, intensity_image=intensity)
    scores = np.array([score(r) for r in regions])

    if all([s in (True, False) for s in scores]):
        cut = [r.label for r, s in zip(regions, scores) if not s]
    else:
        th = threshold(scores)
        cut = [r.label for r in regions if r.mean_intensity<th]

    labeled.flat[np.in1d(labeled.flat[:], cut)] = 0
    return labeled


def fill_holes(img):
    labels = skimage.measure.label(img)
    background_label = np.bincount(labels.flatten()).argmax()
    return labels != background_label


def apply_watershed(img, smooth=4):
    distance = ndimage.distance_transform_edt(img)
    distance = gaussian_filter(distance, smooth)
    local_maxi = peak_local_max(distance, indices=False, 
                                footprint=np.ones((3, 3)), 
                                exclude_border=False)
    markers = ndimage.label(local_maxi)[0]
    return watershed(-distance, markers, mask=img).astype(np.uint16)


def get_blobs(row, pad=(3, 3), threshold=0.01, method='dog'):
    channels = [x for x in row.index.levels[0] if x != 'all']
    I = io.get_row_stack(row, pad=pad)
    blobs_all = []
    for channel, img in zip(channels, I):
        if channel == 'DAPI':
            blobs_all += [[]]
            continue
        img = skimage.img_as_float(img)
        # img /= img.max()
        blobs_all += [skimage.feature.blob_dog(img, min_sigma=1.,
                                               max_sigma=3.,
                                               threshold=threshold)]
    return blobs_all


def gaussian(x, sigma):
    return np.exp(-x ** 2 / (2 * sigma ** 2))


def double_gaussian(sigma1, sigma2):
    return lambda x: gaussian(x, sigma1) * (1 - gaussian(x, sigma2))


f2d = Filter2D(double_gaussian(50, 3), window_size=2048)


def fourier_then_blob(region, pad_width=5, threshold=50):
    I = region.intensity_image_full
    I[I == 0] = I[I != 0].mean()
    I_filt = f2d(I, pad_width=pad_width)
    I_filt = I_filt[region.image]
    # blobs = skimage.feature.blob_dog(I_filt, min_sigma=1,
    #                                  max_sigma=3,
    #                                  threshold=threshold)

    return I_filt.max()


spectral_order = ['empty', 'Atto488', 'Cy3', 'A594', 'Cy5', 'Atto647']
spectral_luts = {'empty': io.GRAY,
                 'Atto488': io.CYAN,
                 'Cy3': io.GREEN,
                 'A594': io.RED,
                 'Cy5': io.MAGENTA,
                 'Atto647': io.MAGENTA}


class Calibration(object):
    def __init__(self, path, dead_pixels_file='dead_pixels.tif', illumination_correction=True):
        """Load calibration info from .json file. Looks for dead pixels file in same (calibration)
        folder by default.
        :param path: path to calibration folder. json file must have same name, e.g., .../20150101/ and .../20150101.json
        :param dead_pixels_file: 2D grayscale image of dead pixels
        :return:
        """
        self.files = {}
        if os.path.isabs(path):
            self.path = path
        else:
            self.path = lasagna.config.paths.full(path)

        self.dead_pixels_file = lasagna.config.paths.full(dead_pixels_file)
        self.name = os.path.basename(path)
        self.background = None
        self.calibration, self.calibration_norm, self.calibration_med = [pd.DataFrame() for _ in range(3)]
        self.illumination, self.illumination_mean = pd.Series(), pd.Series()
        self.dead_pixels, self.dead_pixels_pts = None, None

        self.update_dead_pixels()
        if illumination_correction:
            with open(self.path + '.json', 'r') as fh:
                self.info = json.load(fh)
            print 'loading calibration\nchannels:', self.info['channels'], \
            '\ndyes:', list(set(self.info['wells'].values()))
            self.update_files()
            self.update_illumination()

    def update_dead_pixels(self, null_pixels=10):
        """Pick a threshold so # of background pixels above threshold is about `null_pixels`.
        """
        self.dead_pixels = io.read_stack(self.dead_pixels_file)
        sigma_factor = np.log(self.dead_pixels.size/null_pixels)
        self.dead_pixels_std = (self.dead_pixels[self.dead_pixels < self.dead_pixels.mean()]).std()
        # empirical adjustment
        self.dead_pixels_threshold = 2.5 * sigma_factor * self.dead_pixels_std + self.dead_pixels.mean()
        self.dead_pixels_pts = np.where(self.dead_pixels > self.dead_pixels_threshold)

    def update_files(self):
        self.files = defaultdict(list)
        for f in os.walk(self.path).next()[2]:
            if '.tif' in f:
                _, _, well, site = io.parse_MM(f)
                channel = self.info['wells'][well]
                self.files[channel] += [os.path.join(self.path, f)]

    def update_background(self):
        self.background = np.median([io.read_stack(f) for f in self.files['empty']], axis=0)

    def fix_dead_pixels(self, frame):
        """Replace dead pixels with average of 4 nearest neighbors.
        :param frame:
        :return:
        """
        tmp = np.pad(frame.copy(), ((0, 1), (0, 1)), mode='constant', constant_values=(np.nan,)).copy()
        ii, jj = self.dead_pixels_pts
        tmp[ii, jj] = np.nanmean([tmp[ii - 1, jj], tmp[ii + 1, jj], tmp[ii, jj - 1], tmp[ii, jj + 1]], axis=0)

        return tmp[:-1, :-1]

    def update_illumination(self):
        self.calibration = pd.DataFrame()
        channels = self.info['channels']
        for dye, files in self.files.items():
            data = np.array([io.read_stack(f) for f in files])
            for frame, channel in zip(np.median(data, axis=0), channels):
                # pandas doesn't like initializing with array-like
                self.calibration.loc[channel, dye] = 'x'
                self.calibration.loc[channel, dye] = frame

        self.calibration = self.calibration[sorted(self.calibration, key=lambda x: spectral_order.index(x))]
        self.calibration_med = self.calibration.applymap(lambda x: np.median(x))

        # subtract constant background
        # normalize to matching dye + channel
        self.calibration_norm = self.calibration_med.subtract(self.calibration_med['empty'], axis=0)
        for column in self.calibration_norm:
            if column != 'empty':
                self.calibration_norm[column] /= self.calibration_norm.loc[column, column]

        # compute individual and average illumination correction, save to file
        self.illumination = pd.Series()
        for dye in self.calibration.columns:
            if dye != 'empty':
                data = self.calibration.loc[dye, dye]
                data = data - self.calibration.loc[dye, 'empty']
                self.illumination[dye] = data / np.percentile(data.flatten(), 99)
                self.illumination[dye] = data / data.mean()
        self.illumination_mean = self.illumination.mean()

        # dump illumination stack
        stack = np.array([self.illumination_mean] + list(self.illumination))
        luts = [io.GRAY] + [spectral_luts[dye] for dye in self.calibration.columns if dye != 'empty']
        save_name = os.path.join(os.path.dirname(self.path), 'illumination_correction_%s.tif' % self.name)
        io.save_stack(save_name, 10000 * stack, luts=luts)

    def fix_illumination(self, frame, channel=None):
        """Apply background subtraction and illumination correction. If no channel is provided and input
        dimension matches number of channels for which illumination correction is available, apply correction
        per channel. Otherwise, apply channel-specific correction, or median correction if no channel is provided.
        :param frame: ndarray, final two dimensions must match calibration height and width
        :param channel: name of channel, must match column in Calibration.calibration
        :return:
        """
        if channel is None:
            try:
                if frame.shape[-3] == self.illumination.shape:
                    background = np.array(self.calibration_med)[:, None, None]
                    return np.abs(frame - background) / self.illumination
                else:
                    raise IndexError
            except IndexError:
                background = np.min(self.calibration_med['empty'])
                return np.abs(frame - background) / self.illumination_mean
        return np.abs(frame - self.calibration_med.loc[channel, 'empty']) / self.illumination[channel]

    def plot_dead_pixels(self):
        from matplotlib import pyplot as plt

        m, std = self.dead_pixels.mean(), self.dead_pixels_std
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist((self.dead_pixels.flatten() - m) / std, log=True, bins=range(24))
        ax.hold('on')
        ax.plot([(self.dead_pixels_threshold - m) / std]*2, [0, 1e8], color='red')
        ax.set_title('dead pixel distribution')
        ax.set_xlabel('standardized intensity')
        ax.set_ylabel('counts')

    def plot_crosstalk(self):
        """Requires matplotlib.
        :return:
        """
        from matplotlib import pyplot as plt

        df_ = self.calibration_norm
        logged = np.log10(df_)
        logged = logged.replace({-np.infty: 100})
        logged = logged.replace({100: logged.min().min()})

        to_plot = (('crosstalk normalized to channel matching dye', df_),
                   ('crosstalk log10', logged))

        fig, axs = plt.subplots(1, len(to_plot), figsize=(12, 6))
        for (title, data), ax in zip(to_plot, axs):
            plt.colorbar(ax.imshow(data), ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticklabels(data.columns)
            ax.set_xlabel('dye')
            ax.set_ylabel('channel')
            ax.set_yticklabels(data.index)
            ax.set_title(title)
        fig.tight_layout()

    def plot_illumination(self):
        from matplotlib import pyplot as plt

        df = self.calibration.drop('empty', axis=1)
        fig, axs = plt.subplots(int(np.ceil(df.shape[1] / 2.)), 2, figsize=(12, 12))
        for ax, color in zip(axs.flatten(), df.columns):
            ax.imshow(df.loc[color, color])
            ax.set_title(color)
            ax.axis('off')


def stitch_grid(arr, overlap, upsample=1):
    """Returns offsets between neighbors, [down rows/across columns, row, column, row_i/col_i]
    :param arr: grid of identically-sized images to stitch, [row, column, height, width]
    :param overlap: fraction of image overlapping in [0, 1]
    :return:
    """
    # find true offset, assume all same shape
    overlap_dist = arr[0][0].shape[0] * (1 - overlap)

    def subset_array(offset):
        return [slice(None, o) if o < 0 else slice(o, None) for o in offset]

    offsets = []

    offset_guesses = [np.array([0., overlap_dist]),
                      np.array([overlap_dist, 0.])]

    for offset_guess, arr_ in zip(offset_guesses,
                                  [arr, np.transpose(arr, axes=[1, 0, 2, 3])]):
        for row in arr_[:-1]:
            offsets_ = []
            for a, b in zip(row[:-1], row[1:]):
                image0_ = a[subset_array(offset_guess)]
                image1_ = b[subset_array(-offset_guess)]
                shift = register_images([image0_, image1_],
                                        window=(2000, 2000), upsample=upsample)[1]
                offsets_ += [offset_guess + shift]
            offsets += [offsets_]

    cols = arr.shape[1] - 1
    offsets = np.array(offsets)
    return np.array([offsets[cols:], offsets[:cols]])


def alpha_blend(arr, positions, clip=True, edge=0.95, edge_width=0.02, subpixel=False):
    """Blend array of images, translating image coordinates according to offset matrix.
    :param arr:
    :param grid:
    :param offset_matrix:
    :return:
    """
    
    @lasagna.utils.Memoized
    def make_alpha(s, edge=0.95, edge_width=0.02):
        """Unity in center, drops off near edge
        :param s: shape
        :param edge: mid-point of drop-off
        :param edge_width: width of drop-off in exponential
        :return:
        """
        sigmoid = lambda r: 1. / (1. + np.exp(-r))

        x, y = np.meshgrid(range(s[0]), range(s[1]))
        xy = np.concatenate([x[None, ...] - s[0] / 2,
                             y[None, ...] - s[1] / 2])
        R = np.max(np.abs(xy), axis=0)

        return sigmoid(-(R - s[0] * edge/2) / (s[0] * edge_width))

    # determine output shape, offset positions as necessary
    if subpixel:
        positions = np.array(positions)
    else:
        positions = np.round(positions)

    positions -= positions.min(axis=0)
    shapes = [a.shape for a in arr]
    output_shape = np.ceil((shapes + positions[:,::-1]).max(axis=0))

    # sum data and alpha layer separately, divide data by alpha
    output = np.zeros([2] + list(output_shape), dtype=float)
    for image, xy in zip(arr, positions):
        alpha = 100 * make_alpha(image.shape, edge=edge, edge_width=edge_width)
        if subpixel is False:
            j, i = xy

            output[0, i:i+image.shape[0], j:j+image.shape[1]] += image * alpha.T
            output[1, i:i+image.shape[0], j:j+image.shape[1]] += alpha.T
        else:
            ST = skimage.transform.SimilarityTransform(translation=xy)

            tmp = np.array([skimage.transform.warp(image, inverse_map=ST.inverse,
                                                   output_shape=output_shape,
                                                   preserve_range=True, mode='reflect'),
                            skimage.transform.warp(alpha, inverse_map=ST.inverse,
                                                   output_shape=output_shape,
                                                   preserve_range=True, mode='constant')])
            tmp[0, :, :] *= tmp[1, :, :]
            output += tmp


    output = (output[0, :, :] / output[1, :, :])

    if clip:
        def edges(n):
            return np.r_[n[:4, :].flatten(), n[-4:, :].flatten(),
                         n[:, :4].flatten(), n[:, -4:].flatten()]

        while np.isnan(edges(output)).any():
            output = output[4:-4, 4:-4]

    return output.astype(arr[0].dtype)


def compress_offsets(off):
    y = off.mean(axis=(1, 2))
    return y[::-1, :].T


def replace_minimum(img):
    """Replace local minima with minimum of neighboring points. Useful to eliminate dead pixels.
    :param img:
    :return:
    """
    selem = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    mi = skimage.filters.rank.minimum(img, selem)
    img_ = img.copy()
    img_[img_ < mi] = mi[img_ < mi]
    return img_


class Mask(object):
    def __init__(self, mask):
        """Hack to avoid slow printing of DataFrame containing boolean np.ndarray.
        """
        self.mask = mask
    def __repr__(self):
        return str(self.mask.shape) + ' mask'


# find offsets using 4 corners
def get_corner_offsets(data, n=500):
    """Find offsets between images in a 3-D stack by registering corners. Uses register_images, FFT-baesd.
    Constructs skimage.transform.SimilarityTransform based on offsets and stack dimensions. Allows for scaling
     and translation, but skimage.transform.warp doesn't seem to apply translation correctly.
    :param data: 3-D stack
    :param n: height/width of window placed at corners for alignment
    :return:
    """
    h, w = data.shape[-2:]
    corners = _get_corners(n)

    offsets = [register_images(data[index]) for index in corners]
    offsets = np.array(offsets).transpose([1, 0, 2])

    src = ((n / 2, n / 2),
           (n / 2, w - n / 2),
           (h - n / 2, w - n / 2),
           (h - n / 2, n / 2))
    src = np.array((src,) * data.shape[0])
    dst = src + offsets

    # skimage.transforms uses x,y coordinates, everything else in i,j (!!)
    transforms = []
    for src_, dst_ in zip(src, dst):
        transforms += [skimage.transform.estimate_transform('similarity', src_[:, ::-1], dst_[:, ::-1])]

    return offsets, transforms


def align_scaled(x, y, scale, **kwargs):
    """Align two images taken at different magnification. The input dimensions 
    are assumed to be [channel, height, width], and the alignment is based on 
    the first channel. The first image should contain the second image.
    Additional kwargs are passed to lasagna.process.register_images.
    """
    x = x.transpose([1, 2, 0])
    y = y.transpose([1, 2, 0])

    # downsample 100X image and align to get offset  
    y_ds = skimage.transform.rescale(y, scale, preserve_range=True)
    _, offset = lasagna.process.register_images([x[..., 0], y_ds[..., 0]],
                                                **kwargs)
    ST = skimage.transform.SimilarityTransform(translation=offset[::-1])

    # warp 40X image and resize to match 100X image
    x_win = skimage.transform.warp(x, inverse_map=ST, 
                                      output_shape=y_ds.shape[:2], 
                                      preserve_range=True)
    x_win = skimage.transform.resize(x_win, y.shape)

    # combine images along new leading dimension
    return np.r_['0,4', x_win, y].transpose([0,3,1,2])

def find_cells(nuclei, mask, small_holes=100, remove_boundary_cells=True):
    """Expand labeled nuclei to cells, constrained to where mask is >0. 
    Mask is divvied up by  
    """
    import skfmm

    # voronoi
    phi = (nuclei>0) - 0.5
    speed = mask + 0.1
    time = skfmm.travel_time(phi, speed)
    time[nuclei>0] = 0

    cells = skimage.morphology.watershed(time, nuclei, mask=mask)

    # remove cells touching the boundary
    if remove_boundary_cells:
        cut = np.concatenate([cells[0,:], cells[-1,:], 
                              cells[:,0], cells[:,-1]])
        cells.flat[np.in1d(cells, np.unique(cut))] = 0

    # assign small holes to neighboring cell with most contact
    holes = skimage.measure.label(cells==0)
    regions = skimage.measure.regionprops(holes,
                intensity_image=skimage.morphology.dilation(cells))

    for reg in regions:
        if reg.area < small_holes:
            vals = reg.intensity_image[reg.intensity_image>0]
            cells[holes == reg.label] = scipy.stats.mode(vals)[0][0]

    return cells.astype(np.uint16)
