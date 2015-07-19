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

def register_images(images, index=None, window=(500, 500)):
    """Register a series of image stacks to pixel accuracy.
    :param images: list of N-dim image arrays, height and width may differ
    :param index: image[index] should yield 2D array with which to perform alignment
    :param window: centered window in which to perform registration, smaller is faster
    :return list[(int)]: list of offsets
    """
    if index is None:
        index = ((0,)*(images[0].ndim - 2) + (slice(None),)*2)

    sz = [image[index].shape for image in images]
    sz = np.array([max(x) for x in zip(*sz)])

    origin = np.array(images[0].shape) * 0
    offsets = [origin]

    center = tuple([slice(s/2 - min(s/2, rw), s/2 + min(s/2, rw))
                    for s, rw in zip(sz, window)])

    def pad(img):
        pad_width = [(s/2, s-s/2) for s in (sz - img.shape)]
        img = np.pad(img, pad_width, 'constant')
        return img[center]

    image0 = pad(images[0][index])
    for image in [x[index] for x in images[1:]]:
        shift, error, _ = register_translation(image0,
                                               pad(image))
        offsets += [origin.copy()]
        offsets[-1][-2:] = shift  # automatically cast to uint64

    return offsets

# wrapper that automatically downsamples input and upsamples output? but some outputs need to be adjusted
# could be bad to re-sample for sequential operations
# alternative is to create a sample class that tracks downsampling events and automatically adjusts upsampling rounding

class Sample(object):
    def __init__(self, rate):
        self.rate = float(rate)
        self.sampled = {}

    def downsample(self, img, shape=None):
        """Downsample image according to Sample.rate, or shape if provided.
        :param img:
        :param shape: tuple indicating downsampled XY dimensions
        :return:
        """
        if shape is None:
            shape = tuple([int(s/self.rate) for s in img.shape[-2:] ])

        new_img = np.zeros(img.shape[:-2] + shape, dtype=img.dtype)

        for idx in np.ndindex(img.shape[:-2]):
            new_img[idx] = transform.resize(img[idx], order=0, mode='nearest', preserve_range=True)

        pass

    def upsample(self, img, shape=None):
        pass




DOWNSAMPLE = 1
SHAPES = {}

# def downsample(image, factor=DOWNSAMPLE):
#     # works on stacks, roughly reversible
#     global SHAPES
#     new_size = (int(factor * s) for s in image.shape)
#     SHAPES[(image.shape, DOWNSAMPLE)] =
#     return resize(image, new_size)
#
# def upsample(image, factor=1./DOWNSAMPLE):
#     global SHAPES
#
#     key =
#     new_size =
    

# adaptive threshold and watershed, 500ms
downsample = (512, 512)
input_size = (1024, 1024)
scale_factor = downsample[0]/float(input_size[0])
block_size = 80
threshold_offset = -150
threshold_offset = 0
diameter_range = [scale_factor * r for r in (10, 100)]

def fill_holes(I):
    labels = label(I)
    bkgd_label = np.bincount(labels.flatten()).argmax()
    return labels != bkgd_label
    
def apply_watershed(I):
    distance = ndimage.distance_transform_edt(I)
    distance = gaussian_filter(distance, 4)
    local_maxi = peak_local_max(distance, indices=False, 
                               footprint = np.ones((3, 3)))
    markers = ndimage.label(local_maxi)[0]
    return watershed(-distance, markers, mask=I).astype(np.uint16)

def get_nuclei(img):
    
    binary = threshold_adaptive(resize(img, downsample), int(block_size*scale_factor), offset=threshold_offset)
    filled = fill_holes(binary)
    opened = opening(filled, selem=disk(diameter_range[0]))
    nuclei = apply_watershed(opened)
    nuclei = resize(nuclei, input_size, mode='nearest', order=0)
    return img_as_uint(nuclei)

