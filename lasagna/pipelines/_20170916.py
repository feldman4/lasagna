from lasagna.imports import *

from scipy.spatial import Delaunay, KDTree, delaunay_plot_2d
from scipy.ndimage.filters import minimum_filter, rank_filter


def close_neighbors(points, distance):
    
    tri = Delaunay(points)
    kdt = KDTree(points)
    
    indices, indptr = tri.vertex_neighbor_vertices

    neighbors = {}
    for k in range(len(tri.points)):
        neighbors[k] = indptr[indices[k]:indices[k+1]]
        
    pairs = defaultdict(list)
    for a,b in kdt.query_pairs(distance):
        pairs[a] += [b]
        pairs[b] += [a]
    
    for a in neighbors:
        neighbors[a] = list(set(neighbors[a]) & set(pairs[a]))
    
    return neighbors

def mask_neighbors(n, labels, neighbors):
    mask = (labels > 0) * 1.
    mask[labels == n] = 3
    filt = np.in1d(labels.flat[:], neighbors[n])
    mask.flat[filt] = 2
    return mask

def sample(f, x):
    def h(img, *args, **kwargs):
        img_ = img.reshape(-1, img.shape[-2], img.shape[-1])
        arr = []
        for frame in img_:
            arr += [skimage.transform.rescale(frame, x, preserve_range=True)]
        img_ = np.array(arr).astype(img.dtype)
        result = f(img_, *args, **kwargs)
    
        arr = []
        shape = img.shape[-2:]
        for frame in result:
            arr += [skimage.transform.resize(frame, shape, preserve_range=True)]

        return np.array(arr).astype(img.dtype).reshape(img.shape)
    return h
 
def color(a, b, fore, percentile=0):
    a_ = np.percentile(a[fore], percentile)
    b_ = np.percentile(b[fore], percentile)

    c = np.log10(a - a_) - np.log10(b - b_)
    c[~fore] = 0

    c = to_int16(c)
    return c
    result = np.zeros_like(a)
    result[fore] = c
    return result

def to_int16(x):
    x = skimage.exposure.rescale_intensity(x)
    x = skimage.img_as_uint(x)
    return x

@lasagna.utils.applyXY
def get_foreground(img):
    mask = img > skimage.filters.threshold_li(img)
    mask = scipy.ndimage.morphology.binary_fill_holes(mask)
    return mask

def build_table(data, nuclei):
    features = {'median': lambda r: np.median(r.intensity_image[:])
               ,'label': lambda r: r.label}
    index = ('channel', ('A', 'B', 'A_B')),
    table = lasagna.process.build_feature_table(data[1:4], nuclei, features, index)

    features = lasagna.process.default_object_features
    table_ = lasagna.process.feature_table(nuclei, nuclei, features)

    table = table.pivot_table(values='median', columns='channel', index='label')
    table = table_.join(table, on='label')
    return table

def mark_permeabilized(table, neighbors):
    table = table.copy()
    perms = table.set_index('label')['perm']

    arr = []
    for label in table['label']:
        x = perms.loc[neighbors[label]]
        arr += [[x.all(), (~x).all(), len(x)]]

    table['all_perm']     = zip(*arr)[0]
    table['all_not_perm'] = zip(*arr)[1]
    table['neighbors']    = zip(*arr)[2]

    return table

def li_subtract(img):
    x = skimage.filters.threshold_li(img)
    img = img.copy()
    img[img<x] = x
    img = img - x
    return img

def align(img):
    offsets = lasagna.process.register_images(img)
    img = [lasagna.io.offset(x, o) for x,o in zip(img, offsets)]
    img = np.array(img)
    return img

# def color(a, b, rank=0.2, ds=0.3, radius=50):
#     """Local rank filter
#     """
#     rank = int(rank * (ds * radius)**2)

#     size = (1, int(radius*ds), int(radius*ds))
#     f = sample(partial(rank_filter, rank=rank), ds)
#     a_ = a - f(a, size=size).astype(float)
#     b_ = b - f(b, size=size).astype(float)

#     a_[a_ < 0] = 0
#     b_[b_ < 0] = 0

#     c = a_ / b_
#     c[b_ == 0] = 0
    
#     return c