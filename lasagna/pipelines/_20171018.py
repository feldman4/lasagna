from lasagna.imports import *
import lasagna.bayer

file_pattern = \
        r'((?P<home>.*)[\/\\])?' + \
        r'(?P<dataset>(?P<date>[0-9]{8}).*?)[\/\\]' + \
        r'(?P<subdir>.*[\/\\])*' + \
        r'(MAX_)?(?P<mag>[0-9]+X).' + \
        r'(?:(?P<cycle>[^_\.]*).*?(?:.*MMStack)?.)?' + \
        r'(?P<well>[A-H][0-9]*)' + \
        r'(?:-Site_(?P<site>([0-9]+)))?' + \
        r'(?:-Tile_(?P<tile>([0-9]+)))?' + \
        r'(?:\.(?P<tag>.*))*\.tif'


luts = GRAY, CYAN, GREEN, RED, MAGENTA

dr = (
    (500, 65000), 
    (500, 6000), 
    (500, 6000), 
    (500, 6000), 
    (500, 6000))

dr_log = (
    (500, 65000), 
    (250, 1000), 
    (250, 3000), 
    (250, 3000), 
    (250, 3000))

good_cycles = ['c0-DO', 'c1-5B1', 'c3-5B3', 'c5-3B2', 'c6-3B3', 'c8-5B2']
cycles_designed = ['c0-DO', 'c1-5B1', 'c2-5B2', 'c3-5B3', 'c4-3B1', 'c5-3B2']

def parse_tile_configuration(txt):
    """Read output of Fiji Grid/Collection Stitching.
    """
    try:
        dimension = re.findall('dim = (.)', txt)[0]
    except IndexError:
        dimension = None

    pat = r'\n([^;\n]*);.*\((.*)?,\s*(.*)\)'
    coordinates = re.findall(pat, txt)
    coordinates = [(file, float(x), float(y)) for file, x, y in coordinates]

    return coordinates

def estimate_basis(ij):
    """Estimate basis vectors from a 2 x M x N grid of points. Instead of median,
    could use skimage to estimate transform.
    
    _, ij = load_tile_configuration(tile_config)
    ij2 = ij.T.reshape((2, 15, 15))
    e_0, e_1 = estimate_basis(ij2)
    print e_0
    print e_1

    """
    e_0 = np.median(np.diff(ij, axis=1).reshape(2, -1), axis=1)
    e_1 = np.median(np.diff(ij, axis=2).reshape(2, -1), axis=1)
    return e_0, e_1

def grid_from_basis(e_0, e_1, width, height):
    """Transform a grid of points with integer spacing into the given basis.
    """
    I, J = np.meshgrid(np.arange(width), np.arange(height))
    a = I * e_0[:, None, None]
    b = J * e_1[:, None, None]
    return  (a + b).reshape(2, -1)

def load_tile_configuration(filename):
    """Read i,j coordinate translations from output of Fijii Grid/Collection Stitching.

    """
    with open(filename, 'r') as fh:
        txt = fh.read()

    coordinates = parse_tile_configuration(txt)
    files = [f for (f, _, _) in coordinates]
    ij = np.array([[y, x] for (_, x, y) in coordinates])

    sites = [re.findall('Site_(.*?)\.', f) for (f, _, _) in coordinates]
    sites = [int(x[0]) for x in sites]

    return files, ij

def max_to_full_name(home):
    """Dictionary of max names in tile configuration to actual names.
    """
    search = os.path.join(home, 'MAX', '*', '*.ome.tif')
    return {os.path.basename(f): f for f in glob(search)}

def parse_filenames(files):

    matches, files_keep = [], []
    for f in files:
        match = re.match(file_pattern, f)
        if match:
            matches += [match.groupdict()]
            files_keep += [f]
    df = pd.DataFrame(matches)
    df['file'] = files_keep
    df = lasagna.io.well_to_row_col(df)
    df['basename'] = df['file'].apply(os.path.basename)

    return df

def annotate(df_files, tile_config, grid_shape=(15,15), tile_shape=(0.333, 0.333)):
    # convert 96-well to 6-well
    df_files = df_files.copy()
    plate = microwells()
    rename = {k: v for k,v in zip(plate['96_well'], plate['6_well'])}
    df_files['well_6'] = [rename[w] for w in df_files['well']]

    
    # convert sites to tile identity 
    height, width = grid_shape
    sites = np.arange(height * width).reshape((height, width)).astype(int)
    tiles = tile(sites, tile_shape[0], tile_shape[1])

    site_to_tile = {}
    for tile_name, tile_ in enumerate(tiles):
        for i in tile_.flatten():
            site_to_tile[i] = tile_name

    df_files['tile'] = [site_to_tile[int(s)] for s in df_files['site']]

    # relative position determined by Fiji Grid/Collection Stitching
    _, ij = load_tile_configuration(tile_config)
    positions = ij[df_files['site'].astype(int)]
    df_files['position_ij'] = [tuple(p) for p in positions]

    # position relative to A01 of a 96-well plate
    df_files['plate_ij'] = [plate_coordinate(w, int(s)) for w,s in zip(df_files['well'], df_files['site'])]
    # distance to center of 6-well
    df_files['center_distance'] = df_files.groupby('well_6')['plate_ij'].transform(center_distance)

    return df_files

def center_distance(plate_ij):
    ij = np.array(plate_ij.tolist())
    well_6 = plate_ij.name
    center = np.array(well_center(well_6))
    # assert False
    return np.sqrt(((ij - center)**2).sum(axis=1))
    
def well_center(well, plate='6'):
    df_plate = microwells()
    filt = df_plate['%s_well' % plate] == well
    wells_96 = df_plate.loc[filt, '96_well']
    ij = [plate_coordinate(w, 0, grid_shape=(1,1)) for w in wells_96]
    i, j = np.array(ij).mean(axis=0)
    return i, j

def plate_coordinate(well, site, grid_shape=(15, 15), offset=(0, 0), mag='20X'):
    spacing_96w = 9000
    if mag == '20X':
        delta = 600
    else:
        raise ValueError('mag')

    row, col = well_to_row_col(well)
    i, j = row * spacing_96w, col * spacing_96w

    height, width = grid_shape
    i += delta * int(site / width)
    j += delta * (site % width)
    
    i -= delta * (height / 2) 
    j -= delta * (width  / 2)

    i += offset[0]
    j += offset[1]

    return i, j

def name(description, **kwargs):
    d = dict(description)
    d.update(kwargs)

    basename = '%s_%s_%s-Tile_%s.%s.tif' % (d['mag'], d['cycle'], d['well'], d['tile'], d['tag'])
    subdir = d['subdir'] if d['subdir'] else ''
    return os.path.join(d['home'], d['dataset'], subdir, basename)

def microwells(base=(8, 12), downsample=((2, 2), (4, 4))):
    width = int(np.ceil(base[1] / 10))
    ds = downsample
    
    rows, cols = np.meshgrid(range(base[0]), range(base[1]))

    a, b = base
    base_rows = pd.Series(rows.flatten(), name='%s_row' % (a*b)).astype(int)
    base_cols = pd.Series(cols.flatten(), name='%s_col' % (a*b)).astype(int)
    base_wells = [row_col_to_well(r, c, width=width) for r,c in zip(base_rows, base_cols)]
    base_wells = pd.Series(base_wells, name='%d_well' % (a*b))

    arr = [base_rows, base_cols, base_wells]
    for i, j in ds:
        rows = (base_rows / i).astype(int)
        cols = (base_cols / j).astype(int)
        wells = [row_col_to_well(r, c, width=width) for r,c in zip(rows, cols)]
        wells = pd.Series(wells)

        a = base[0] / i
        b = base[1] / j
        rows.name  = '%d_row'  % (a*b)
        cols.name  = '%d_col'  % (a*b)
        wells.name = '%d_well' % (a*b)

        arr += [rows, cols, wells]

    return pd.concat(arr, axis=1)

def row_col_to_well(row, col, width=1):
    import string
    fmt = r'%s%0' + str(width) + 'd' 
    return fmt % (string.ascii_uppercase[row], col + 1)

def well_to_row_col(well):
    import string
    return string.ascii_uppercase.index(well[0]), int(well[1:]) - 1

def expand_DO(data):
    """Insert zeros in the FITC and A594 channels.
    """
    return np.insert(data, [1, 2], 0, axis=-3)

@lasagna.bayer.applyXY
def laplace_only(data, *args, **kwargs):
    from skimage.filters import laplace
    h, w = data.shape[-2:]
    arr = []
    for frame in data.reshape((-1, h, w)):
        arr += [laplace(frame, *args, **kwargs)]
    return np.array(arr).reshape(data.shape)

### actions

def do_stitch(df_files, overwrite=False):

    def process_df2(df2):
        for cycle, df3 in df2.groupby('cycle'):
            f2 = name(df3.iloc[0], subdir='', tag='stitched')
            if os.path.exists(f2) and not overwrite:
                continue

            # read in data for this tile
            data = np.array([read(f) for f in df3['file']])
            if 'DO' in df3['file'].iloc[0]:
                data = expand_DO(data)

            # stitch
            positions = np.array(df3['position_ij'].tolist())
            arr = []
            for c in range(data.shape[1]):
                result = lasagna.process.alpha_blend(data[:, c], positions)
                arr += [result]
            stitched = np.array(arr)

            # save
            save(f2, stitched)
            lasagna.io._get_stack._reset()

    n_cycles = len(set(df_files['cycle']))
    gb = df_files.groupby(['well', 'tile'])
    for (well, tile), df2 in gb:
        print 'stitching', well, tile
        files = glob('*%s-Tile_%d*.tif' % (well, tile))
        process_df2(df2)
        
def do_log(df_files):

    gb = (df_files.drop_duplicates(['well', 'tile', 'cycle'])
                  .sort_values('cycle')
                  .groupby(['well', 'tile'])
                  )
             
    for (well, tile), df_files2 in gb:
        print 'log aligning', well, tile

        cycles = ''.join([c[:2] for c in sorted(set(df_files2['cycle']))])
        row = df_files2.iloc[0]
        f = name(row, cycle=cycles, subdir='', tag='aligned')

        if os.path.exists(f):
            continue
        aligned = log_it(df_files2)


        save(f, aligned, display_ranges=dr_log)
        print 'saved', f

def log_it(df_files):
    """Save .log and .aligned for multiple cycles at a single tile.
    """

    # load data
    lasagna.io._get_stack._reset()
    name_row = lambda row: name(row, subdir='', tag='stitched')
    files = df_files.apply(name_row, axis=1)
    data = np.array([read(f) for f in files])
    lasagna.io._get_stack._reset()

    # float slow
    loged = lasagna.bayer.log_ndi(data)
    # copy DAPI
    loged[:,0] = data[:,0]

    aligned_log = lasagna.bayer.register_and_offset(loged, registration_images=data[:, 0])
    
    return aligned

def align_phenotypes(df_files, cycles=('c0-DO', 'Myc-HA')):

    gb = (df_files
         .drop_duplicates(['well', 'tile'])
         .groupby(['well', 'tile']))

    for (well, tile), row in gb:
        row = row.iloc[0]
        f_c0    = name(row_c0, cycle=cycles[0], subdir='', tag='stitched')
        f_ph    = name(row_ph, cycle=cycles[0], subdir='', tag='stitched')
        f_ph_al = name(row_ph, cycle=cycles[1], subdir='', tag='aligned')

        print f_ph_al
        if os.path.exists(f_ph_al):
            continue

        data_c0 = read(f_c0)
        data_ph = read(f_ph)

        dapi = np.array([data_c0[0], data_ph[0]])
        data = np.array([data_c0[:3], data_ph])
        aligned = lasagna.bayer.register_and_offset(data, dapi)
        aligned = aligned[1]
        save(f_ph_al, aligned)

def do_segment_nuclei_peaks(df_files):

    gb = (df_files.drop_duplicates(['well', 'tile'])
                  .groupby(['well', 'tile'])
                  )
             
    for (well, tile), df_files2 in gb:
        print 'segmenting', well, tile

        row = df_files2.iloc[0]
        f_c0     = name(row, cycle='c0-DO', subdir='', tag='stitched')
        f_peaks  = name(row, cycle='c0-DO', subdir='', tag='peaks')
        f_nuclei = name(row, cycle='c0-DO', subdir='', tag='nuclei')


        if os.path.exists(f_peaks) and os.path.exists(f_nuclei):
            continue

        
        lasagna.io._get_stack._reset()
        data_c0 = read(f_c0)
        lasagna.io._get_stack._reset()

        peaks = np.zeros_like(data_c0)
        peaks[[2, 4]] = lasagna.process.find_peaks(data_c0[[2, 4]])
        nuclei = lasagna.process.find_nuclei(data_c0[0], 
                                            area_max=1000, 
                                            threshold=lambda x: 5000)

        save(f_peaks,  peaks,  compress=1)
        save(f_nuclei, nuclei, compress=1)

def do_cell_segment(df_files):

    gb = (df_files.drop_duplicates(['well', 'tile'])
                  .groupby(['well', 'tile'])
                  )

    for (well, tile), df_files2 in gb:
        print 'segmenting cells', well, tile
        row = df_files2.iloc[0]
        f_cells  = name(row, cycle='c1c8',  subdir='', tag='cells')
        f_nuclei = name(row, cycle='c0-DO', subdir='', tag='nuclei')
        f_al = name(row, subdir='', cycle='c0c1c3c5c6c8', tag='aligned')
        f_c1 = name(row, subdir='', cycle='c1-5B1', tag='stitched')
        f_c8 = name(row, subdir='', cycle='c8-5B2', tag='stitched')


        if os.path.exists(f_cells):
            continue

        nuclei = read(f_nuclei)

        cells = segment_cells(nuclei, f_al, f_c1, f_c8)

        save(f_cells, cells, compress=1)

def segment_cells(nuclei, f_al, f_c1, f_c8):
        

    dapi_c0 = read(f_al, memmap=True)[0, 0]
    dapi_c1, fitc_c1 = read(f_c1, memmap=True)[[0, 1]]
    dapi_c8, fitc_c8 = read(f_c8, memmap=True)[[0, 1]]

    # use min of c1 and c8 fitc to determine cell outlines
    dapi = [dapi_c0, dapi_c1, dapi_c8]
    fitc = [fitc_c1, fitc_c1, fitc_c8]
    _, fitc_c1, fitc_c8 = lasagna.bayer.register_and_offset(fitc, registration_images=dapi)

    signal = np.min([fitc_c1, fitc_c8], axis=0)
    background = scipy.ndimage.filters.minimum_filter(signal, size=100)

    fore = (signal - background) > 100

    cells = lasagna.process.find_cells(nuclei, fore)

    return cells



### base calls

def do_calls(df_files, cycles):
    gb = df_files.drop_duplicates(['well', 'tile']).groupby(['well', 'tile'])

    arr = []
    for (well, tile), rows in gb:
        row = rows.iloc[0]

        f_values = name(row, cycle='c0c1c3c5c6c8', subdir='export', tag='values')
        f_values = f_values.replace('.tif', '.pkl')

        if os.path.exists(f_values):
            continue

        values, labels = extract_values(row)

        df_v = values_to_dataframe(values, good_cycles)

        df_v['cell'] = labels

        df_v['well'] = row['well']
        df_v['tile'] = row['tile']

        df_v.to_pickle(f_values)

def do_phenotype(df_files):
    from lasagna.pipelines._20170914_endo import feature_table_stack

    gb = df_files.drop_duplicates(['well', 'tile']).groupby(['well', 'tile'])

    for (well, tile), rows in gb:
        row = rows.iloc[0]
        
        f_nuclei = name(row, cycle='c0-DO', subdir='', tag='nuclei')
        f_ph_al  = name(row, cycle='Myc-HA', subdir='', tag='aligned')
        f_corr   = name(row, cycle='Myc-HA', subdir='export', tag='corr')
        f_corr   = f_corr.replace('.tif', '.pkl')

        if os.path.exists(f_corr):
            continue

        features = {
            'corr'       : correlate_dapi_myc,
            'dapi_median': lambda r: np.median(r.intensity_image_full[0]),
            'fitc_median': lambda r: np.median(r.intensity_image_full[1]),
            'myc_median' : lambda r: np.median(r.intensity_image_full[2]),
            'cell'       : lambda r: r.label
        }

        stack = read(f_ph_al)
        mask = read(f_nuclei)
        lasagna.io._get_stack._reset()

        table = feature_table_stack(stack, mask, features)

        table['well'] = row['well']
        table['tile'] = row['tile']

        table.to_pickle(f_corr)

def extract_values(row, width=5):
    """Retrieve values at blob posititions from LoG-transformed, max-expanded data.
    Uses matrix approach rather than regionprops for speed.
    """
    f_cells  = name(row, cycle='c1c8',  subdir='', tag='cells')
    f_nuclei = name(row, cycle='c0-DO', subdir='', tag='nuclei')
    f_al     = name(row,  cycle='c0c1c3c5c6c8', subdir='', tag='aligned')
    f_peaks  = name(row, cycle='c0-DO', subdir='', tag='peaks') 
    data = read(f_al)
    nuclei = read(f_nuclei)
    cells = read(f_cells)
    peaks = read(f_peaks)

    lasagna.io._get_stack._reset()

    size = 1,1,width,width
    blobs = peaks[2] > 500

    # slow
    maxd = scipy.ndimage.filters.maximum_filter(data[:,1:], size)
    blob_mask = blobs & (cells > 0)
    values = maxd[:, :, blob_mask]
    labels = cells[blob_mask]

    return values, labels

def values_to_dataframe(values, cycles):
    channels = list('TGCA')
    levels = cycles, channels
    names = 'cycle', 'channel'
    columns = pd.MultiIndex.from_product(levels, names=names)

    a,b,c = values.shape
    df_v = pd.DataFrame(values.reshape(a*b, -1).T, columns=columns)

    return df_v 

def dataframe_to_values(dataframe, cycles=None):
    """If cycles provided, ensures correct ordering of columns before converting.
    """
    if cycles:
        channels = list('TGCA')
        cycles = 'c0-DO', 'c1-5B1', 'c2-5B2', 'c3-5B3', 'c4-3B1', 'c5-3B2', 'c6-5B4'
        cycles = sorted(c for c in cycles if c in dataframe.columns.get_level_values('cycle'))
        index = list(product(cycles, channels))
        values = dataframe.sortlevel(axis=1)[index]

        i = -1
        j = len(cycles)
        k = len(channels)
    else:
        values = dataframe
        i = -1
        j = values.shape[1] / 4
        k = 4

    values = np.array(values).reshape(i, j, k)
    return values.transpose([1, 2, 0])

def correlate_dapi_myc(region):
    dapi, fitc, myc = region.intensity_image_full

    filt = dapi > 0
    dapi = dapi[filt]
    myc  = myc[filt]
    corr = (dapi - dapi.mean()) * (myc - myc.mean()) / (dapi.std() * myc.std())

    return corr.mean()

###

def cycle_to_index(cycle):
    n = int(cycle[-1])
    if '5B' in cycle:
        return n - 1
    elif '3B' in cycle:
        return -1 * n
    raise ValueError
    
def barcode_to_in_situ(barcode, cycles):
    indices = [cycle_to_index(c) for c in cycles]
    return ''.join(barcode[i] for i in indices)


def narrow_design(df_design, cycles):
    """subpool 1
    """
    df_design = df_design.drop_duplicates(['barcode'])

    f = partial(barcode_to_in_situ, cycles=cycles)
    df_design['barcode_in_situ'] = df_design['barcode'].apply(f)

    f = lambda x: len(set(x['design'])) > 1
    filt = df_design.groupby('barcode_in_situ').apply(f)
    filt = list(filt[filt].index)
    df_design['ambiguous'] = df_design['barcode_in_situ'].isin(filt)
    return df_design

