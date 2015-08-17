import json
from skimage import transform
import skimage
from skimage.feature import register_translation
import numpy as np
import pandas as pd
import uuid
import os
from collections import defaultdict

from skimage.filter import gaussian_filter, threshold_adaptive
from skimage.morphology import disk, watershed, opening
from skimage.util import img_as_uint
import skimage.measure
from skimage.feature import peak_local_max
from scipy import ndimage

from lasagna import io
from lasagna import config

DOWNSAMPLE = 2


def region_fields(region):
    return {'area': region.area,
            'centroid': region.centroid,
            'bounds': region.bbox,
            'label': region.label,
            'mask': io.compress_obj(region.image)}


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


def table_from_nuclei(file_table, source='stitch', nuclei='nuclei', channels=None,
                      features=None):
    """
    :param file_table:
    :param source:
    :param nuclei:
    :return:
    """
    # prefix to channel-specific features
    channels = ['channel' + str(i) for i in range(100)] if channels is None else channels
    features = default_features if features is None else features

    dataframes = []
    for ix, row in file_table.iterrows():

        print 'processing:', row[source]
        # load nuclei file
        segmented = io.read_stack(config.paths.full(row[nuclei]))
        data = io.read_stack(config.paths.full(row[source]))

        info = [region_fields(r) for r in regionprops(segmented)]
        df = pd.DataFrame(info, index=[list(x) for x in zip(*[list(ix)] * len(info))])
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

        df.index.names = file_table.index.names
        df = df.set_index(('all', 'label'), append=True)
        df.index.set_names('label', level=[('all', 'label')], inplace=True)

        dataframes.append(df)

    df = pd.concat(dataframes)

    return df


def register_images(images, index=None, window=(500, 500)):
    """Register a series of image stacks to pixel accuracy.
    :param images: list of N-dim image arrays, height and width may differ
    :param index: image[index] should yield 2D array with which to perform alignment
    :param window: centered window in which to perform registration, smaller is faster
    :return list[(int)]: list of offsets
    """
    if index is None:
        index = ((0,) * (images[0].ndim - 2) + (slice(None),) * 2)

    sz = [image[index].shape for image in images]
    sz = np.array([max(x) for x in zip(*sz)])

    origin = np.array(images[0].shape) * 0

    center = tuple([slice(s / 2 - min(s / 2, rw), s / 2 + min(s / 2, rw))
                    for s, rw in zip(sz, window)])

    def pad(img):
        pad_width = [(s / 2, s - s / 2) for s in (sz - img.shape)]
        img = np.pad(img, pad_width, 'constant')
        return img[center], [x[0] for x in pad_width]

    image0, pad_width = pad(images[0][index])
    offsets = [origin.copy()]
    offsets[0][-2:] += pad_width
    for image in [x[index] for x in images[1:]]:
        padded, pad_width = pad(image)
        shift, error, _ = register_translation(image0,
                                               padded)
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


class Filter2D(object):
    def __init__(self, func, window_size=200):
        """Create 2D fourier filter from 1D radial function.
        The filter itself is available as Filter2D.filter1D, .filter2D.
        :param func: 1D radial function in fourier space.
        :param window_size: can be anything, really
        :return:
        """
        self.func = func
        y = np.array([range(window_size)])
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


default_features = {'mean': lambda region: region.intensity_image.mean(),
                    'median': lambda region: np.median(region.intensity_image),
                    'max': lambda region: region.intensity_image.max(),
                    'blob': lambda region: fourier_then_blob(region),
                    }

spectral_order = ['empty', 'Atto488', 'Cy3', 'A594', 'Cy5', 'Atto647']


class Calibration(object):
    def __init__(self, full_path, dead_pixels=None, dead_pixels_std=10):
        """Load calibration info from .json file. Looks for "dead_pixels.tif" in same (calibration)
        folder by default.
        :param full_path:
        :return:
        """
        self.files = {}
        self.full_path = full_path
        self.background = None

        with open(full_path + '.json', 'r') as fh:
            self.info = json.load(fh)
        self.magnification = io.get_magnification(os.path.basename(full_path))

        if dead_pixels is None:
            self.dead_pixels_file = os.path.join(os.path.dirname(full_path), 'dead_pixels.tif')
        else:
            self.dead_pixels_file = dead_pixels

        self.dead_pixels = io.read_stack(self.dead_pixels_file)
        self.dead_pixels_std = dead_pixels_std
        threshold = self.dead_pixels.std() * 10 + np.median(self.dead_pixels)
        self.dead_pixels_pts = np.where(self.dead_pixels > threshold)

        self.update_files()
        self.update_illumination()

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
        sz = self.dead_pixels.shape
        for ii, jj in zip(*self.dead_pixels_pts):
            ii_ = [ii - 1, ii - 1, (ii + 1) % sz[0], (ii + 1) % sz[0]]
            jj_ = [jj - 1, (jj + 1) % sz[1], jj - 1, (jj + 1) % sz[1]]
            frame[ii, jj] = np.mean(frame[ii_, jj_])
        return frame

    def update_illumination(self):
        self.colors = {}
        self.colors_med = {}
        for channel, files in self.files.items():
            data = np.array([io.read_stack(f) for f in files])
            self.colors[channel] = np.median(data, axis=0)
            self.colors_med[channel] = np.median(self.colors[channel], axis=[1, 2])
        self.color_table = pd.DataFrame(self.colors_med, index=self.info['channels'])
        self.color_table = self.color_table[sorted(self.colors, key=lambda x: spectral_order.index(x))]

        # subtract constant background
        self.color_table_sub = self.color_table.subtract(self.color_table['empty'], axis=0)

        # normalize to matching dye + channel
        self.color_table_norm = self.color_table_sub.copy()
        for column in self.color_table_norm:
            if column != 'empty':
                self.color_table_norm[column] /= self.color_table_norm.loc[column, column]

        # just do average illumination correction
        arr = []
        for color in self.colors:
            if color != 'empty':
                index = self.info['channels'].index(color)
                data = self.colors[color][index]
                data = data - self.color_table.loc[color, 'empty']
                arr += [data / np.percentile(data.flatten(), 99)]
        self.arr = arr
        self.illumination = np.array(arr).mean(axis=0)
        self.illumination /= self.illumination.mean()

        # TODO make individual channel correction functions
        # self.correction = np.median(self.colors.values(), axis=[0, ]
        # for i, channel in enumerate(self.info['channels']):
        #     self.illumination[channel] = self.colors[channel][]
        #     def correction(frame):
        #         # closure...
        #         frame - self.color_table.loc[channel, 'empty']
        #
        #     self.correction[channel] = correction

    def fix_illumination(self, frame):
        return frame * self.illumination

    def plot_crosstalk(self):
        """Requires matplotlib.
        :return:
        """
        from matplotlib import pyplot as plt

        df_ = self.color_table_norm
        logged = np.log10(df_)
        logged = logged.replace({-np.infty: 100})
        logged = logged.replace({100: logged.min().min()})

        to_plot = (('crosstalk normalized to channel matching dye', df_),
                   ('crosstalk log10', logged))

        fig, axs = plt.subplots(1, len(to_plot), figsize=(12, 6))
        for (title, data), ax in zip(to_plot, axs):
            plt.colorbar(ax.imshow(data), ax=ax,fraction=0.046, pad=0.04)
            ax.set_xticklabels(data.columns)
            ax.set_xlabel('dye')
            ax.set_ylabel('channel')
            ax.set_yticklabels(data.index)
            ax.set_title(title)
        fig.tight_layout()

    def plot_illumination(self):
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        colors = [c for c in self.colors if c!='empty']
        for ax, m, color in zip(axs.flatten(), self.arr, colors):
            ax.imshow(m)
            ax.set_title(color)
            ax.axis('off')

