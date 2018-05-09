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
from skimage.filters import gaussian, threshold_adaptive
from skimage.morphology import disk, watershed, opening
from skimage.util import img_as_uint
import skimage.measure
from skimage.feature import peak_local_max
from scipy import ndimage

import lasagna.io
import lasagna.utils

DOWNSAMPLE = 2

default_intensity_features = {'mean': lambda region: region.intensity_image[region.image].mean(),
                    'median': lambda region: np.median(region.intensity_image[region.image]),
                    'max': lambda region: region.intensity_image[region.image].max(),
                    'min': lambda region: region.intensity_image[region.image].min() }

default_object_features = {
    'area':     lambda region: region.area,
    'i':        lambda region: region.centroid[0],
    'j':        lambda region: region.centroid[1],
    'bounds':   lambda region: region.bbox,
    'contour':  lambda region: lasagna.io.binary_contours(region.image, fix=True, labeled=False)[0],
    'label':    lambda region: region.label,
    'mask':     lambda region: lasagna.utils.Mask(region.image),
    'hash':     lambda region: hex(random.getrandbits(128)) }


# FEATURES
def feature_table(data, mask, features, global_features=None):
    """Apply functions in features to regions in data specified by
    integer mask.
    """
    regions = regionprops(mask, intensity_image=data)
    results = {feature: [] for feature in features}
    for region in regions:
        for feature, func in features.items():
            results[feature] += [func(region)]
    if global_features:
        for feature, func in global_features.items():
            results[feature] = func(data, mask)
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


def regionprops(*args, **kwargs):
    """Supplement skimage.measure.regionprops with additional field containing full intensity image in
    bounding box (useful for filtering).
    :param args:
    :param kwargs:
    :return:
    """
    import skimage.measure

    regions = skimage.measure.regionprops(*args, **kwargs)
    if 'intensity_image' in kwargs:
        intensity_image = kwargs['intensity_image']
        for region in regions:
            b = region.bbox
            region.intensity_image_full = intensity_image[..., b[0]:b[2], b[1]:b[3]]
    return regions


# ALIGN
def register_images(images, index=None, window=(500, 500), upsample=1.):
    """Register image stacks to pixel accuracy.
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


def register_and_offset(images, registration_images=None, verbose=False):
    """Wrapper around `register_images` and `offset`.
    """
    if registration_images is None:
        registration_images = images
    offsets = register_images(registration_images)
    if verbose:
        print np.array(offsets)
    aligned = [lasagna.utils.offset(d, o) for d,o in zip(images, offsets)]
    return np.array(aligned)



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
    arr : N x I x J
    positions : N x 2 (n, i, j)
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
    # convert from ij to xy
    positions = positions[:, [1, 0]]    

    positions -= positions.min(axis=0)
    shapes = [a.shape for a in arr]
    output_shape = np.ceil((shapes + positions[:,::-1]).max(axis=0)).astype(int)

    # sum data and alpha layer separately, divide data by alpha
    output = np.zeros([2] + list(output_shape), dtype=float)
    for image, xy in zip(arr, positions):
        alpha = 100 * make_alpha(image.shape, edge=edge, edge_width=edge_width)
        if subpixel is False:
            j, i = np.round(xy).astype(int)

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


# SEGMENT
def find_nuclei(dapi, radius=15, area_min=50, area_max=500, um_per_px=1., 
                score=lambda r: r.mean_intensity,
                threshold=skimage.filters.threshold_otsu,
                verbose=False, smooth=1.35):
    """
    """
    area = area_min, area_max
    # smooth = 1.35 # gaussian smoothing in watershed

    mask = _binarize(dapi, radius, area[0])
    labeled = skimage.measure.label(mask)
    labeled = filter_by_region(labeled, score, threshold, intensity=dapi) > 0

    # only fill holes below minimum area
    filled = ndimage.binary_fill_holes(labeled)
    difference = skimage.measure.label(filled!=labeled)

    change = filter_by_region(difference, lambda r: r.area < area[0], 0) > 0
    labeled[change] = filled[change]

    nuclei = apply_watershed(labeled, smooth=smooth)

    result = filter_by_region(nuclei, lambda r: area[0] < r.area < area[1], threshold)
    if verbose:
        return mask, labeled, nuclei, result, change

    result, _, _ = skimage.segmentation.relabel_sequential(result)
    return result


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
    """Apply a filter to labeled image. The score function takes a single region as input and
    returns a score. Regions are filtered out by score using the
    provided threshold function. If scores are boolean, scores are used as a mask and 
    threshold is disregarded. 
    """
    labeled = labeled.copy().astype(int)

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
    if smooth > 0:
        distance = gaussian(distance, sigma=smooth)
    local_maxi = peak_local_max(distance, indices=False, 
                                footprint=np.ones((3, 3)), 
                                exclude_border=False)
    markers = ndimage.label(local_maxi)[0]
    return watershed(-distance, markers, mask=img).astype(np.uint16)


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

    # for reg in regions:
    #     if reg.area < small_holes:
    #         vals = reg.intensity_image[reg.intensity_image>0]
    #         cells[holes == reg.label] = scipy.stats.mode(vals)[0][0]

    return cells.astype(np.uint16)


def find_peaks(aligned, n=5):
    """At peak, max value in neighborhood and max-min
    """
    from scipy.ndimage import filters
    neighborhood_size = (1,)*(aligned.ndim-2) + (n,n)
    data_max = filters.maximum_filter(aligned, neighborhood_size)
    data_min = filters.minimum_filter(aligned, neighborhood_size)
    peaks = data_max - data_min
    peaks[aligned!=data_max] = 0
    
    # remove peaks close to edge
    mask = np.ones(peaks.shape, dtype=bool)
    mask[...,n:-n, n:-n] = False
    peaks[mask] = 0
    
    return peaks


def peak_to_region(peak, data, threshold=2000, n=5):
    selem = np.ones((n,n))
    peak = peak.copy()
    peak[peak<threshold] = 0
    
    labeled = skimage.measure.label(peak)
    regions = regionprops(labeled, intensity_image=data)
    # hack for rare peak w/ more than 1 pixel
    for r in regions:
        if r.area > 1:
            labeled[labeled==r.label] = np.array([r.label] + [0]*(r.area-1))

    # dilate labels so higher intensity regions win
    fwd = [r.max_intensity for r in regions]
    fwd = np.argsort(np.argsort(fwd)) + 1
    rev = np.argsort(fwd)

    labeled[labeled>0] = fwd

    labeled = skimage.morphology.dilation(labeled, selem)
    labeled[labeled>0] = rev[labeled[labeled>0] - 1]

    return labeled


def log_ndi(data, *args, **kwargs):
    """Apply laplacian of gaussian to each image in a stack of shape
    (..., I, J). 
    Extra arguments are passed to scipy.ndimage.filters.gaussian_laplace.
    """
    from scipy.ndimage.filters import gaussian_laplace
    h, w = data.shape[-2:]
    arr = []
    for frame in data.reshape((-1, h, w)):
        arr += [gaussian_laplace(frame, *args, **kwargs)]
    return np.array(arr).reshape(data.shape)


