from lasagna.pipelines._20171018 import *

def get_cells(cy3, nuclei):
    # _, cy3, _ = read(row['file'])
    fore = cy3 - scipy.ndimage.filters.minimum_filter(cy3, size=50)
    fore = fore > 100
    fore = skimage.morphology.remove_small_objects(fore, min_size=400)
    
    cells = lasagna.process.find_cells(nuclei, fore, remove_boundary_cells=False)
    return cells

def analyze_peaks(cells, peaks, threshold_do=2500):

    features = {'max': lambda r: r.intensity_image.max()}

    arr = []
    for channel, frame in zip(['barcode', 'actb'], peaks[1:]):
        blobs = frame > 2500
        blobs = skimage.measure.label(blobs)

        df  = lasagna.process.feature_table(frame, blobs, features)
        df2 = lasagna.process.feature_table(cells, blobs, features)

        df['channel'] = channel
        df['cell']    = df2['max']
        arr += [df]

    df = pd.concat(arr)
    return df
    

def load_blob_data(df_files):
    arr = []
    for _, row in df_files.iterrows():
        f_blobs  = name(row, subdir='', tag='blobs')
        f_blobs  = f_blobs.replace('tif', 'pkl')
        
        arr += [pd.read_pickle(f_blobs)]
    return pd.concat(arr)