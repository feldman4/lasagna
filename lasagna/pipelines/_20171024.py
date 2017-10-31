from lasagna.imports import *
from lasagna.pipelines._20171018 import parse_filenames, tagged_filename

def bmqc_to_cells(bmqc):
    threshold_bmqc = 100
    min_area = 400
    max_area = 5000
    cells = bmqc > threshold_bmqc
    cells = skimage.morphology.remove_small_objects(cells, min_size=400)
    cells = skimage.measure.label(cells)
    cells = lasagna.process.apply_watershed(cells)
    cells = skimage.morphology.remove_small_objects(cells, min_size=400)
    return cells


def p90(r):
    xs = r.intensity_image
    xs = xs[xs > 0]
    return np.percentile(xs, 90)

