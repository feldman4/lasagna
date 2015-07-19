from skimage.feature import register_translation
import numpy as np


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

