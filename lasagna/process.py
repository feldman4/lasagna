from skimage import transform
from skimage.feature import register_translation
import numpy as np

from skimage.filter import gaussian_filter, threshold_adaptive
from skimage.morphology import disk, watershed, opening
from skimage.util import img_as_uint
from skimage.transform import resize
from skimage.measure import label
from skimage.feature import peak_local_max
from scipy import ndimage

DOWNSAMPLE = 2

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
    offsets = [origin]

    center = tuple([slice(s / 2 - min(s / 2, rw), s / 2 + min(s / 2, rw))
                    for s, rw in zip(sz, window)])

    def pad(img):
        pad_width = [(s / 2, s - s / 2) for s in (sz - img.shape)]
        img = np.pad(img, pad_width, 'constant')
        return img[center]

    image0 = pad(images[0][index])
    for image in [x[index] for x in images[1:]]:
        shift, error, _ = register_translation(image0,
                                               pad(image))
        offsets += [origin.copy()]
        offsets[-1][-2:] = shift  # automatically cast to uint64

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


def get_nuclei(img, opening_radius=20, block_size=80, threshold_offset=0):
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
