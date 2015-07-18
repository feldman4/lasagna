import numpy as np
from skimage.external.tifffile import TiffFile, imsave, imread

imagej_description = ''.join(['ImageJ=1.49v\nimages=%d\nchannels=%d\nslices=%d',
                              '\nframes=%d\nhyperstack=true\nmode=composite',
                              '\nunit=\\u00B5m\nspacing=8.0\nloop=false\n',
                              'min=764.0\nmax=38220.0\n'])

UM_PER_PX = {'40X': 0.44,
             '20X': 0.22}

BINNING = 2
OBJECTIVE = '40X'

TAG_50839_BASE = (74, 73, 74, 73, 111, 102, 110, 105, 1, 0, 0, 0, 103, 110, 97, 114, 1, 0, 0,
                  0, 115, 116, 117, 108, 4, 0, 0, 0, 83, 0, 111, 0, 102, 0, 116, 0, 119, 0, 97,
                  0, 114, 0, 101, 0, 58, 0, 32, 0, 116, 0, 105, 0, 102, 0, 102, 0, 102, 0, 105,
                  0, 108, 0, 101, 0, 46, 0, 112, 0, 121, 0, 10, 0, 68, 0, 97, 0, 116, 0, 101, 0,
                  84, 0, 105, 0, 109, 0, 101, 0, 58, 0, 32, 0, 50, 0, 48, 0, 49, 0, 53, 0, 58, 0,
                  48, 0, 55, 0, 58, 0, 48, 0, 57, 0, 32, 0, 49, 0, 53, 0, 58, 0, 52, 0, 49, 0, 58,
                  0, 48, 0, 52, 0, 10, 0, 0, 0, 0, 0, 0, 144, 120, 64, 0, 0, 0, 0, 236, 168, 192,
                  64, 0, 0, 0, 0, 0, 224, 135, 64, 0, 0, 0, 0, 128, 169, 226, 64, 0, 0, 0, 0, 224,
                  108, 141, 64, 0, 0, 0, 0, 224, 168, 150, 64, 0, 0, 0, 0, 0, 224, 135, 64, 0, 0, 0,
                  0, 128, 169, 226, 64)

RED = tuple(range(256) + [0] * 512)
GREEN = tuple([0]*256 + range(256) + [0]*256)
BLUE = tuple([0] * 512 + range(256))
MAGENTA = tuple(range(256) + [0] * 512 + range(256))

DEFAULT_LUTS = (BLUE, GREEN, RED, MAGENTA)


def save_hyperstack(name, data, autocast=True, resolution=None, luts=None):
    """input ND array dimensions as ([time], [z slice], channel, y, x)
    leading dimensions beyond 5 could be wrapped into time, not implemented
    """
    nchannels = data.shape[-3]
    if resolution is None:
        resolution = (1. / (UM_PER_PX[OBJECTIVE] * BINNING),) * 2
    if luts is None:
        luts = [x for x,_ in zip(DEFAULT_LUTS, range(nchannels))]

    # save as uint16
    tmp = data.copy()
    if autocast:
        tmp = tmp.astype(np.uint16)

    description = ij_description(data.shape)
    tag_50838 = ij_tag_50838(nchannels)
    tag_50839 = ij_tag_50839(luts)

    imsave(name, tmp, photometric='minisblack',
           description=description, resolution=resolution,
           extratags=[(50838, 'I', len(tag_50838), tag_50838, True),
                      (50839, 'B', len(tag_50839), tag_50839, True),
                      ])

def ij_description(shape):
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
    px_size = (16 * nchannels,)
    luts = (256*3,)*nchannels
    return (28, 104) + px_size + luts


def ij_tag_50839(luts):
    """Return tag 50839 for imagej header (info, range, luts)
    :param luts: iterable containing 255*3 ints in range 0-255 specifying RGB, io.RED etc available
    :return:
    """
    return TAG_50839_BASE + sum([list(x) for x in luts],[])