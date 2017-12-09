from lasagna.imports import *
from lasagna import tasker
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
        r'(?:\.(?P<tag>.*))*\.(tif|pkl)'
        

def parse_filename(filename):
    match = re.match(file_pattern, filename)
    try:
        return match.groupdict()
    except AttributeError:
        raise ValueError('failed to parse filename: %s' % filename)

def make_hasher(fields, remove=None):
    def hasher(filename):
        info = parse_filename(filename)
        if remove and remove(info):
            return None
        group_id = {f: info[f] for f in fields}
        return tuple(group_id.items())
    return hasher

def tag_extractor(filename):
    info = parse_filename(filename)
    return info['tag']

def make_tagger(home, ext='tif'):
    def tagger(group_id, tag):
        group_id = dict(group_id)
        group_id['dataset'] = '20171031'
        group_id['subdir'] = 'process'
        return name(group_id, home=home, tag=tag, ext=ext)
    return tagger

def name(description, ext='tif', **kwargs):
    d = dict(description)
    d.update(kwargs)
    if 'cycle' in d:
        basename = '%s_%s_%s-Tile_%s.%s.%s' % (d['mag'], d['cycle'], d['well'], d['tile'], d['tag'], ext)
    else:
        basename = '%s_%s-Tile_%s.%s.%s' % (d['mag'], d['well'], d['tile'], d['tag'], ext)
    
    subdir = d['subdir'] if d['subdir'] else ''
    return os.path.join(d['home'], d['dataset'], subdir, basename)

def make_hasher_stitch(grid_shape=(15,15), tile_shape=(0.333, 0.333)):
    """Uses site to find tile. Including tile in `group_id` ensures 
    `tagger` puts it in the filename.
    """
    def hasher(filename):
        info = parse_filename(filename)

        fields = 'cycle', 'well', 'tile', 'dataset', 'subdir', 'mag'
        group_id = tuple([(field, info[field]) for field in fields])
        
        site = info['site']
        if site is None:
            return None

        tile = site_to_tile(grid_shape, tile_shape)[int(site)]
        group_id = group_id + (('tile', tile),)
        
        return group_id
    return hasher

def site_to_tile(grid_shape, tile_shape):
        """Create dictionary from site number to tile number. Can be
        cached if necessary.
        """
        height, width = grid_shape
        sites = np.arange(height * width).reshape((height, width)).astype(int)
        tiles = tile(sites, tile_shape[0], tile_shape[1])

        site_to_tile = {}
        for tile_name, tile_ in enumerate(tiles):
            for i in tile_.flatten():
                site_to_tile[i] = tile_name
                
        return site_to_tile

def load_tile_configuration(filename):
    """Read i,j coordinate translations from output of Fijii Grid/Collection Stitching.

    """
    with open(filename, 'r') as fh:
        txt = fh.read()

    coordinates = parse_tile_configuration(txt)
    files = [f for (f, _, _) in coordinates]
    ij = np.array([[y, x] for (_, x, y) in coordinates])

    sites = [re.findall('Site[_-](.*?)\.', f) for (f, _, _) in coordinates]
    sites = [int(x[0]) for x in sites]

    return files, ij

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

def validate_cycles(good_cycles):
    """All cycles must be available, with exactly one file per cycle. 
    Returns inputs sorted by cycles as provided.
    """
    def validator(inputs):
        cycles = defaultdict(list)
        for name in inputs:
            cycles[parse_filename(name)['cycle']].append(name)
        
        inputs2 = []
        for cycle in good_cycles:
            tmp = cycles[cycle]
            if len(tmp) != 1:
                return None
            inputs2.append(tmp[0])
        if len(inputs2) == 1:
            return inputs2[0]
        return inputs2
    return validator

def check_compressibility(data, window=1000):
    midpoint = int(data.size / 2)
    left = int(max(midpoint - window/2, 0))
    right = int(min(midpoint + window/2, data.size))
    return compressibility(data.flat[left:right])
    
def compressibility(x):
    import zlib
    compressed = zlib.compress(x)
    return len(x) / float(len(compressed))

def dump_tif_or_dataframe(f, data):
    if isinstance(data, pd.DataFrame):
        data.to_pickle(f)
    else:
        compress = 0
        # compress = 1 if compressibility(data) > 3 else 0
        lasagna.io.save_stack(f, data, compress=compress)

def read_pickle(filename):
    tags = parse_filename(filename)
    df = pd.read_pickle(filename)
    for tag, value in tags.items():
        if value is not None:
            df[tag] = value # could cast to categorical
    return df
        

### process data

def fake(data):
    return data[..., :100, :100]

def identity(f):
    def inner():
        return f
    return inner

@identity
def stitch(sites, tile_config, backdoor):
    blah = 30
    _, ij = load_tile_configuration(tile_config)
    files = backdoor[0]
    positions = []
    for f in files:
        site = int(parse_filename(f)['site'])
        positions += [ij[site]]
    
    data = np.array(sites)
    arr = []
    for c in range(data.shape[1]):
        result = lasagna.process.alpha_blend(data[:, c], positions)
        arr += [result]
    stitched = np.array(arr)
    
    return stitched
    
@identity
def laplacian_of_gaussian(data):
    loged = lasagna.bayer.log_ndi(data)
    loged[:,0] = data[:,0]
    return loged

@identity
def align(data):
    """Align data using DAPI.
    """
    arr = []
    for d in data:
        # DO
        if d.shape[0] == 3:
            d = np.insert(d, [1, 2], 0, axis=0)
        arr.append(d)
    data = np.array(arr)
    dapi = data[:,0]
    return lasagna.bayer.register_and_offset(data, registration_images=data[:, 0])

@identity
def align_phenotype(data_DO, data_phenotype):
    _, offset = register_images([data_DO[0], data_phenotype[0]])
    return lasagna.io.offset(data_phenotype, offset)
   
@identity
def segment_nuclei(data, nuclei_threshold=5000, nuclei_area_max=1000):
    """Find nuclei from DAPI. Find cell foreground from aligned but unfiltered 
    data. Expects data to have shape C x I x J.
    """
    nuclei = lasagna.process.find_nuclei(data[0], 
                            area_max=nuclei_area_max, 
                            threshold=lambda x: nuclei_threshold)
    return nuclei

@identity
def segment_cells(aligned_data, nuclei, threshold=750):
    """Segment cells from aligned data. To use less than full cycles for 
    segmentation, filter the input files.
    """
    data = np.array(aligned_data)
    # no DAPI, min over cycles, mean over channels
    mask = data[:, 1:].min(axis=0).mean(axis=0)
    mask = mask > threshold

    cells = lasagna.process.find_cells(nuclei, mask)

    return cells

@identity
def find_peaks(data, cutoff=50):
    peaks = [lasagna.process.find_peaks(x) 
                if x.max() > 0 else x 
                for x in data]
    peaks = np.array(peaks)
    peaks[peaks < cutoff] = 0 # for compression
    return peaks
    
@identity
def calculate_phenotypes(data_DO, data_phenotype, nuclei):
    def correlate_dapi_myc(region):
        dapi, fitc, myc = region.intensity_image_full

        filt = dapi > 0
        if filt.sum() == 0:
            assert False
            return np.nan

        dapi = dapi[filt]
        myc  = myc[filt]
        corr = (dapi - dapi.mean()) * (myc - myc.mean()) / (dapi.std() * myc.std())

        return corr.mean()

    features = {
        'corr'       : correlate_dapi_myc,
        'dapi_median': lambda r: np.median(r.intensity_image_full[0]),
        'fitc_median': lambda r: np.median(r.intensity_image_full[1]),
        'myc_median' : lambda r: np.median(r.intensity_image_full[2]),
        'cell'       : lambda r: r.label
    }

    from lasagna.pipelines._20170914_endo import feature_table_stack
    dapi = data_DO[0]
    data = np.array([dapi] + list(data_phenotype[1:]))
    table = feature_table_stack(data, nuclei, features)
    return table
       
@identity 
def max_filter(data, width=5):
    maxed = scipy.ndimage.filters.maximum_filter(data, size=(1, 1, width, width))
    maxed[:, 0] = data[:, 0] # DAPI
    return maxed

    # assign barcode identities in the next step
    # threshold DO => matrix processing
    # values = (ID, cycles, channels)
    # positions = (ID, i, j)
    # features = (ID, num_features)

@identity
def extract_barcodes(peaks, data_max, cells, threshold_DO, index_DO, cycles):
    data_max = data_max[:, 1:] # no DAPI
    blob_mask = (peaks[index_DO] > threshold_DO) & (cells > 0)
    values = data_max[:, :, blob_mask].transpose([2, 0, 1])
    labels = cells[blob_mask]
    positions = np.array(np.where(blob_mask)).T

    index = ('cycle', cycles), ('channel', list('TGCA'))
    df = lasagna.utils.ndarray_to_dataframe(values, index)

    df_positions = pd.DataFrame(positions, columns=['position_i', 'position_j'])
    return (df.stack(['cycle', 'channel'])
        .reset_index()
        .rename(columns={0:'intensity', 'level_0': 'blob'})
        .join(pd.Series(labels, name='cell'), on='blob')
        .join(df_positions, on='blob')
        )

def call_bases(df):
    cols = ['well', 'tile', 'blob', 'cycle']
    df2 = df.pivot_table(index=cols, columns='channel', values='intensity')

    channels = sorted(set(df['channel'])) # in alphabetical order
    call = np.argmax(np.array(df2), axis=1)
    call = np.array(channels)[call]
    s = pd.Series(call, index=df2.index, name='call')
    df = df.join(s, on=cols)
    
    cols = ['well', 'tile', 'blob']
    df2 = df.pivot_table(index=cols, columns='cycle', values='call', aggfunc=lambda x: x.iat[0])

    name = 'barcode_in_situ'
    barcodes = [''.join(x) for x in np.array(df2)]
    s = pd.Series(barcodes, index=df2.index, name=name)
    df = df.join(s, on=cols)

    return df

def call_cells(df):
    cols = ['well', 'tile', 'cell']
    s = (df.drop_duplicates(['well', 'tile', 'blob'])
       .groupby(cols)['barcode_in_situ']
       .value_counts()
       .rename('count')
       .sort_values(ascending=False)
       .reset_index('barcode_in_situ')
       .groupby(cols)
        )

    df2 = \
    (df.join(s.nth(0)['barcode_in_situ'].rename('barcode_in_situ_0'), on=cols)
       .join(s.nth(0)['count']          .rename('barcode_count_0'), on=cols)
       .join(s.nth(1)['barcode_in_situ'].rename('barcode_in_situ_1'), on=cols)
       .join(s.nth(1)['count']          .rename('barcode_count_1'), on=cols)
    )
    return df2

def call(df):
    df = call_bases(df)
    mask = filter_intensity(df, lambda x: x > 5000)
    df = df[mask]
    df = call_cells(df)
    return df

def filter_intensity(df, filt_func, cycle=0, base=1):

    cycles = df['cycle'].value_counts()
    assert len(set(cycles)) == 1
    n_cycles = len(cycles)
    x = np.array(df['intensity']).reshape(-1, n_cycles, 4)

    mask = np.zeros(x.shape, dtype=bool)
    mask[...] = filt_func(x[:,cycle,base])[:, None, None]
    mask = mask.flatten()
    return mask

### dataset-specific

home = '/Volumes/Samsung_T5/lasagna/'

common_fields = {'well', 'tile', 'dataset', 'subdir', 'mag'}

tagger = make_tagger(home)
tagged = partial(tasker.tagged, tagger=tagger, tag_extractor=tag_extractor)

def fake_load(data):
    return np.zeros((3, 100, 100), dtype=np.uint16)

tasker.load_data = lasagna.io.read_stack
# tasker.load_data = fake_load
tasker.dump_data = dump_tif_or_dataframe

def find_files():
    ome_inputs = glob(os.path.join(home, '20171018_6W-G126A/MAX/*/*ome.tif'))
    process_inputs = glob(os.path.join(home, '20171031/process/*.tif'))
    return ome_inputs + process_inputs + glob(os.path.join(home, '20171031/*.tif'))

def task_stitch(grid_shape, tile_shape):
    """custom hasher for site info
    """
    ome_tag = ('ome', lambda xs: xs if len(xs) == 25 else None)
    hasher = make_hasher_stitch(grid_shape, tile_shape)
    return tagged([ome_tag], ['stitched'], hasher)(stitch())

def task_LoG(cycles_seq):
    hasher = make_hasher(common_fields | {'cycle'},
                     remove=lambda info: info['cycle'] not in cycles_seq)
    return tagged(['stitched'], ['log'], hasher)(laplacian_of_gaussian())

def task_align(cycles_seq):
    # select cycles in the validator
    hasher = make_hasher(common_fields)
    log_tag = ('log', validate_cycles(cycles_seq))
    return tagged([log_tag], ['aligned'], hasher)(align())

def task_align_phenotype():
    hasher = make_hasher(common_fields)
    DO_tag = ('stitched', validate_cycles(['c0-DO']))
    phenotype_tag = ('stitched', validate_cycles(['Myc-HA']))
    return tagged([DO_tag, phenotype_tag], ['aligned_phenotype'], hasher)(align_phenotype())
 
def task_segment_nuclei():
    # select cycle in the hasher
    hasher = make_hasher(common_fields)
    stitched_tag = ('stitched', validate_cycles(['c0-DO']))
    return tagged(['stitched'], ['nuclei'], hasher)(segment_nuclei())

def task_segment_cells(cycles_cell):
    # select cycles in the validator
    cell_input_tag = ('stitched', validate_cycles(cycles_cell))
    hasher = make_hasher(common_fields)
    return tagged([cell_input_tag, 'nuclei'], ['cells'], hasher)(segment_cells())

def task_find_peaks():
    hasher = make_hasher(common_fields)
    stitched_tag = ('stitched', validate_cycles(['c0-DO']))
    return tagged([stitched_tag], ['peaks'], hasher)(find_peaks())

def task_calculate_phenotypes():
    tagger = make_tagger(home, ext='pkl')
    hasher = make_hasher(common_fields)
    stitched_tag = ('stitched', validate_cycles(['c0-DO']))
    return (tagged([stitched_tag, 'aligned_phenotype', 'nuclei'], ['phenotype'], hasher, tagger=tagger)
                    (calculate_phenotypes()))

def task_max_filter(width):
    hasher = make_hasher(common_fields)
    return tagged(['aligned'], ['max%d' % width], hasher=hasher)(max_filter())

def task_extract_barcodes():
    tagger = make_tagger(home, ext='pkl')
    hasher = make_hasher(common_fields)
    return (tagged(['peaks', 'max5', 'cells'], ['barcodes'], hasher, tagger=tagger)
             (extract_barcodes()))

def tasks():
    """
    Make tasks with different hashers
    for special cases plus a generic hasher that returns `group_id` only if special 
    case hashers all return `None`.
 
    # tagged(tags, hasher...)(function)(*args, **kwargs) 
    # vs 
    # tagged2(partial(function, *args, **kwargs), tags..., hasher..., )
    #
    # wrote the first to support decorator style...
    # can support both!


    """
    # make task generators
    cycles_cell = ['c1-5B1', 'c3-5B3']
    cycles_seq = ['c0-DO', 'c1-5B1', 'c3-5B3', 'c5-3B2', 'c6-3B3', 'c8-5B2']

    search = os.path.join(home, '20171031/*TileConfig*registered.txt')
    tile_config = glob(search)[0]
    grid_shape = (15, 15)
    tile_shape = (0.333, 0.333)
    index_DO, threshold_DO = 1, 500 # can threshold DO further later on

    # parameterize task generator, then supply non-data arguments
    return [task_stitch(grid_shape, tile_shape)(tile_config, backdoor=None),
            task_align_phenotype()(),
            task_segment_nuclei()(),
            task_calculate_phenotypes()(),

            task_LoG(cycles_seq)(), 
            task_align(cycles_seq)(),
            task_segment_cells(cycles_cell)(),
            task_find_peaks()(),

            task_max_filter(width=5)(),
            task_extract_barcodes()(threshold_DO, index_DO, cycles_seq)]


# custom hasher for site info
# ome_tag = ('ome', lambda xs: xs if len(xs) == 25 else None)
# hasher = make_hasher_stitch(grid_shape, tile_shape)
# task_stitch = tagged([ome_tag], ['stitched'], hasher)(stitch())


# hasher = make_hasher(common_fields | {'cycle'},
#                  remove=lambda info: info['cycle'] not in cycles_seq)
# task_LoG = tagged(['stitched'], ['log'], hasher)(laplacian_of_gaussian())
