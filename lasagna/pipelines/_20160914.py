from lasagna.imports import *
from lasagna.process import build_feature_table, feature_table

# name of lasagna folder and sheet in Lasagna FISH
datasets = '20160914_96W-G064',
# dead_pixels_file = 'calibration/dead_pixels_20151219_MAX.tif'
# tile_configuration = None

channels = 'DAPI', 'FITC', 'Cy3', 'TexasRed', 'Cy5'
luts = GRAY, CYAN, GREEN, RED, MAGENTA
lasagna.io.default_luts = luts
display_ranges = ((500, 20000),
                  (500, 6000),
                  (500, 6000),
                  (500, 6000),
                  (500, 6000))

DO_slices = [2, 4]
DO_thresholds = ('DO_Puro', 1000), ('DO_ActB', 500)


all_index= (('cycle',   ('c0-DO', 'c1-5B2', 'c2-5B3', 'c3-5B4'))
           ,('channel', ('DAPI', 'FITC', 'Cy3', 'TxRed', 'Cy5')))

def crop(data):
    return data[..., 30:-30, 30:-30]
    
def do_alignment(df_files):
    """Aligns data. DO and sequencing cycles are aligned using DAPI.
    Within each sequencing cycle, all channels except DAPI are aligned to FITC.
    """ 
    DO_index = [0] + DO_slices # include DAPI
    bases = 'base2_1', 'base2_common_1', 'base3_1', 'base4_1'
    index_fwd = [2, 1, 3, 4]
    index_rev = [0, 1, 2, 3]
    for well, df_ in df_files.groupby('well'):
        files = df_.sort_values('cycle')['file']
        data = []
        for f in files:
            if not any(b in f for b in bases + ('DO',)):
                raise NotImplementedError('file name not recognized: ' + f)
            data_ = read(f)
            if 'DO' in f:
                # fill in blanks
                shape = 5, data_.shape[1], data_.shape[2]
                data2 = np.zeros(shape, np.uint16)
                data2[DO_index] = data_
                data_ = data2
            data += [data_]
        
        assert len(set([d.shape for d in data])) == 1 # all data for this well is the same shape
        data = np.array(data) 

        # register DAPI
        offsets = register_images(data[:,0])
        print 'DAPI offsets:', offsets
        data = np.array([lasagna.io.offset(d, o.astype(int)) for d, o in zip(data, offsets)])
        
        # spectral alignment to FITC, susceptible to crap in images
        if data.shape[0] > 1:
            data = data.transpose([1,0,2,3])
            offsets = [np.array([0, 0])] + register_images(data[1:,1], window=[100, 100])
            print 'channel offsets:', offsets
            data = np.array([lasagna.io.offset(d, o.astype(int)) for d, o in zip(data, offsets)])
            data = data.transpose([1,0,2,3])

        dataset, mag, well = df_.iloc[0][['dataset', 'mag', 'well']]
        f = make_filename(dataset, mag, well, 'aligned')
        save(f, crop(data), luts=luts, display_ranges=display_ranges)
        print well, f
        lasagna.io._get_stack._reset()
        

def do_peak_detection():
    df = paths()
    for f in df.query('tag=="aligned"')['file']:
        aligned = read(f)
        peaks = pipeline.find_peaks(aligned)
        f2 = f.replace('aligned', 'aligned.peaks')
        if f != f2:
            save(f2, peaks, luts=luts, compress=1)
            print f
        else:
            print 'wtf',


def make_filename(dataset, mag, well, tag, cycle=None):
    if cycle:
        filename = '%s_%s_%s.%s.tif' % (mag, cycle, well, tag)
    else:
        filename = '%s_%s.%s.tif' % (mag, well, tag)
    return os.path.join(dataset, filename)


def find_nuclei_fast(dapi, smooth=0):
    t = skimage.filters.threshold_li(dapi)
    w = lasagna.process.apply_watershed(dapi > t, smooth=smooth)
    return w


def fix_cycle_labels(df):
    """Replace labels from `all_index` with labels from experiment sheet.
    """
    pass

