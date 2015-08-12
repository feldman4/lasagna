from skimage import transform
import skimage
from skimage.feature import register_translation
import numpy as np
import pandas as pd
import uuid

from skimage.filter import gaussian_filter, threshold_adaptive
from skimage.morphology import disk, watershed, opening
from skimage.util import img_as_uint
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from scipy import ndimage

from lasagna import io
from lasagna import config

DOWNSAMPLE = 2


def region_fields(region):
    return {'area': region.area,
            'centroid': region.centroid,
            'bounds': region.bbox,
            'label': region.label}


default_features = {'mean': lambda region: region.intensity_image.mean(),
                    'median': lambda region: np.median(region.intensity_image),
                    'max': lambda region: region.intensity_image.max(),
                    }


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
    labels = label(img)
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


class filter2D(object):
    def __init__(self, func):
        y = np.array([[func(r) for r in range(100)]])
        self.H = y * y.T
        
    def __call__(self, M):
        H = self.H
        M_fft = np.fft.fft2(M)
        s, t = [s/2 for s in M_fft.shape]
        h = np.zeros(M.shape)
        h[:s, :t] = H[:s, :t]
        h[:-s-1:-1, :t] = H[:s, :t]
        h[:-s-1:-1, :-t-1:-1] = H[:s, :t]
        h[:s, :-t-1:-1] = H[:s, :t]
        M_f = np.absolute( np.fft.ifft2(h * M_fft))
        self.h = h
        return M_f * (M.sum()/M_f.sum())
    

def make_2D_filter(func):
    """Turn a 1D function in Fourier space into a callable 2D filter.
    """
    y = np.array([[func(r) for r in range(100)]])
    H = y * y.T
    def filter2D(M):
        M_fft = np.fft.fft2(M)
        s, t = [s/2 for s in M_fft.shape]
        h = np.zeros(M.shape)
        h[:s, :t] = H[:s, :t]
        h[:-s-1:-1, :t] = H[:s, :t]
        h[:-s-1:-1, :-t-1:-1] = H[:s, :t]
        h[:s, :-t-1:-1] = H[:s, :t]
        M_f = np.abs( np.fft.ifft2(h * M_fft))
        return M_f * (M.sum()/M_f.sum())
    return filter2D