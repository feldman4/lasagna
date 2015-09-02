import json
import uuid
import os
from collections import defaultdict

from skimage import transform
import skimage
from skimage.feature import register_translation
import numpy as np
import pandas as pd
from skimage.filter import gaussian_filter, threshold_adaptive
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

default_features = {'mean': lambda region: region.intensity_image.mean(),
                    'median': lambda region: np.median(region.intensity_image),
                    'max': lambda region: region.intensity_image.max()
                    }

default_nucleus_features = {
    'area': lambda region: region.area,
    'centroid': lambda region: region.centroid,
    'bounds': lambda region: region.bbox,
    'label': lambda region: np.median(region.intensity_image[region.intensity_image > 0]),
    'mask': lambda region: io.compress_obj(region.image)
}


def binary_contours(img):
    """Find contours of binary image
    :param img:
    :return: list of nx2 arrays of [x, y] points along contour of each image.
    """
    contours = skimage.measure.find_contours(np.pad(img, 1, mode='constant'),
                                             level=0.5)
    return [contour - 1 for contour in contours]


def pad(array, pad_width, mode=None, **kwargs):
    if type(pad_width) == int:
        if pad_width < 0:
            s = [slice(-1 * pad_width, pad_width)] * array.ndim
            return array[s]
    return np.pad(array, pad_width, mode=mode, **kwargs)


def regionprops(*args, **kwargs):
    """Supplement skimage.measure.regionprops with additional field containing full intensity image in
    bounding box (useful for filtering).
    :param args:
    :param kwargs:
    :return:
    """
    regions = skimage.measure.regionprops(*args, **kwargs)
    if 'intensity_image' in kwargs:
        intensity_image = kwargs['intensity_image']
        for region in regions:
            b = region.bbox
            region.intensity_image_full = intensity_image[b[0]:b[2], b[1]:b[3]]
    return regions


def table_from_nuclei(row, index_names, source='aligned', nuclei='nuclei', channels=None,
                      features=default_features, nucleus_features=default_nucleus_features,
                      nuclei_dilation=None, data=None):
    """Build Cells DataFrame from Paths DataFrame, list of features, calibrated aligned data, and image with
    segmented nuclei.

    :param row: from Paths DataFrame, containing filenames in source and nuclei columns
    :param source: column with filename of data file, [channels, height, width]
    :param nuclei: column with filename of segmented nuclei, [height, width]
    :param channels: names of channels, used to name first column index level
    :param features: dict of functions accepting regionprops region as argument, key used to name
    second column index level
    :param nuclei_dilation: structuring element by which to dilate nuclei image before calling regionprops
    :param data: [channels, height, width] from which channel data is drawn, otherwise loaded from source
    :return:
    """
    # prefix to channel-specific features
    channels = ['channel' + str(i) for i in range(100)] if channels is None else channels

    # load nuclei file, data
    segmented = io.read_stack(config.paths.full(row[nuclei]))
    if data is None:
        data = io.read_stack(config.paths.full(row[source]))

    if nuclei_dilation is not None:
        segmented = skimage.morphology.dilation(segmented, nuclei_dilation)

    info = []
    for region in regionprops(segmented, intensity_image=segmented):
        info += [{k: v(region) for k, v in nucleus_features.items()}]

    df = pd.DataFrame(info, index=[list(x) for x in zip(*[list(row.name)] * len(info))])
    df['file'] = row[source]
    df['hash'] = [uuid.uuid4().hex for _ in range(df.shape[0])]

    df.columns = pd.MultiIndex(labels=zip(*[[0, i] for i in range(len(df.columns))]),
                               levels=[['all'], df.columns],
                               names=['channel', 'feature'])

    # add channel-specific features
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    for channel, image in zip(channels, data):
        channel_regions = regionprops(segmented, intensity_image=image)
        for name, fcn in features.items():
            df[channel, name] = [fcn(r) for r in channel_regions]

    df.index.names = index_names
    df = df.set_index(('all', 'label'), append=True)
    df.index.set_names('label', level=[('all', 'label')], inplace=True)

    return df


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


def get_nuclei(img, opening_radius=6, block_size=80, threshold_offset=0):
    s = Sample(DOWNSAMPLE)
    binary = threshold_adaptive(s.downsample(img), int(block_size / s.rate), offset=threshold_offset)
    filled = fill_holes(binary)
    opened = opening(filled, selem=disk(opening_radius / s.rate))
    nuclei = apply_watershed(opened)
    nuclei = s.upsample(nuclei)
    return img_as_uint(nuclei)


def fill_holes(img):
    labels = skimage.measure.label(img)
    background_label = np.bincount(labels.flatten()).argmax()
    return labels != background_label


def apply_watershed(img):
    distance = ndimage.distance_transform_edt(img)
    distance = gaussian_filter(distance, 4)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)))
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
    def __init__(self, full_path, dead_pixels_file='dead_pixels.tif'):
        """Load calibration info from .json file. Looks for dead pixels file in same (calibration)
        folder by default.
        :param full_path:
        :return:
        """
        self.files = {}
        self.full_path = full_path
        self.background = None
        self.calibration, self.calibration_norm, self.calibration_med = [pd.DataFrame() for _ in range(3)]
        self.illumination, self.illumination_mean = pd.Series(), pd.Series()
        self.dead_pixels, self.dead_pixels_pts = None, None
        self.dead_pixels_std_threshold = 5

        with open(full_path + '.json', 'r') as fh:
            self.info = json.load(fh)
        print 'loading calibration\nchannels:', self.info['channels'], \
            '\ndyes:', list(set(self.info['wells'].values()))
        self.magnification = io.get_magnification(os.path.basename(full_path))

        self.dead_pixels_file = os.path.join(os.path.dirname(full_path), dead_pixels_file)

        self.update_dead_pixels()
        self.update_files()
        self.update_illumination()

    def update_dead_pixels(self, std_threshold=5):
        self.dead_pixels_std_threshold = std_threshold
        self.dead_pixels = io.read_stack(self.dead_pixels_file)
        threshold = self.dead_pixels.std() * std_threshold + np.median(self.dead_pixels)
        self.dead_pixels_pts = np.where(self.dead_pixels > threshold)

    def update_files(self):
        self.files = defaultdict(list)
        for f in os.walk(self.full_path).next()[2]:
            well, site = io.get_well_site(f)
            channel = self.info['wells'][well]
            self.files[channel] += [os.path.join(self.full_path, f)]

    def update_background(self):
        self.background = np.median([io.read_stack(f) for f in self.files['empty']], axis=0)

    def fix_dead_pixels(self, frame):
        """Replace dead pixels with average of 4 nearest neighbors.
        :param frame:
        :return:
        """
        tmp = np.pad(frame, ((0, 1), (0, 1)), mode='constant', constant_values=(np.nan,)).copy()
        ii, jj = self.dead_pixels_pts
        tmp[ii, jj] = np.nanmean([tmp[ii - 1, jj], tmp[ii + 1, jj], tmp[ii, jj - 1], tmp[ii, jj + 1]])

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
        save_name = os.path.join(os.path.dirname(self.full_path), 'illumination_correction.tif')
        io.save_hyperstack(save_name, 10000 * stack, luts=luts)

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

        m, std = self.dead_pixels.mean(), int(self.dead_pixels.std())
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist((self.dead_pixels.flatten() - m) / std, log=True, bins=range(24))
        ax.hold('on')
        ax.plot([self.dead_pixels_std_threshold] * 2, [0, 1e8], color='red')
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


def alpha_blend(arr, positions, clip=True, edge=0.95, edge_width=0.02):
    """Blend array of images, translating image coordinates according to offset matrix.
    :param arr:
    :param grid:
    :param offset_matrix:
    :return:
    """

    def make_alpha(s, edge=0.95, edge_width=0.02):
        """Unity in center, drop-off near edge
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

        return sigmoid(-(R - s[0] * edge) / (s[0] * edge_width))

    positions = np.array(positions)
    # determine output shape, offset positions as necessary

    positions -= positions.min(axis=0)
    output_shape = np.ceil(positions.max(axis=0)[::-1]) + arr.shape[1:]
    # output_shape = np.r_[output_shape, [2]]

    alpha = 100 * make_alpha(arr.shape[1:], edge=edge, edge_width=edge_width)

    # store summed data and alpha layer in trailing channel dimension
    output = np.zeros(np.r_[[2], output_shape], dtype=float)
    reshape = lambda x: x.reshape(-1, *x.shape[1:])
    for image, xy in zip(reshape(arr), reshape(positions)):
        ST = skimage.transform.SimilarityTransform(translation=xy)
        # to_warp = np.r_['2,3,0', image, alpha]
        tmp = np.array([skimage.transform.warp(image, inverse_map=ST.inverse,
                                               output_shape=output_shape,
                                               preserve_range=True, mode='reflect'),
                        skimage.transform.warp(alpha, inverse_map=ST.inverse,
                                               output_shape=output_shape,
                                               preserve_range=True, mode='constant')])
        tmp[0, :, :] *= tmp[1, :, :]
        output += tmp

    output_alpha = output[1, :, :]
    output = (output[0, :, :] / output[1, :, :])

    if clip:
        def edges(n):
            return np.r_[n[:4, :].flatten(), n[-4:, :].flatten(),
                         n[:, :4].flatten(), n[:, -4:].flatten()]

        while np.isnan(edges(output)).any():
            output = output[4:-4, 4:-4]

    return output.astype(arr.dtype)


def compress_offsets(off):
    y = off.mean(axis=(1, 2))
    return y[::-1, :].T


def replace_minimum(img):
    """Replace local minima with minimum of neighboring points. Useful to eliminate dead pixels.
    :param img:
    :return:
    """
    selem = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    mi = skimage.filter.rank.minimum(img, selem)
    img_ = img.copy()
    img_[img_ < mi] = mi[img_ < mi]
    return img_


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
