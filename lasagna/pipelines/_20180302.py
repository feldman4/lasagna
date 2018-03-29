import networkx as nx
from lasagna.imports import *
from lasagna.bayer import register_and_offset
from lasagna.pipelines._20170914_endo import feature_table_stack


def displacement(x):
    d = np.sqrt(np.diff(x['x'])**2 + np.diff(x['y'])**2)
    return d

def add_max_displacement(df):
    d = (df.groupby(['well', 'cell'])
           .apply(lambda y: displacement(y).max())
           .rename('max_displacement'))
    return df.join(d, on=['well', 'cell'])

def frame_to_timestamp_20180302(frame):
    first_interval = 300 # seconds
    second_interval = 600
    
    if frame == 0:
        return 0
    if frame < 11:
        return frame * first_interval
    else:
        return 10 * first_interval + (frame - 10) * second_interval
    
def extract_phenotype_live_translocation(data_phenotype, nuclei, perimeter):
    import lasagna.snakes.firesnake
    extract = lasagna.snakes.firesnake.Snake._extract_phenotype_translocation
    
    data = data_phenotype[:, [1, 0]]
    
    it = enumerate(zip(data, nuclei, perimeter))
    
    arr = []
    for frame, (d, n, p) in it:
        arr.append(extract(d, n, p, {})
                   .assign(frame=frame)
                   .assign(time=frame_to_timestamp_20180302(frame)))
    
    return pd.concat(arr)

def get_nuclear_perimeter(nuclei, width=5):
    print 'width -------', width
    print 'nuclei shape---------', nuclei.shape
    from skimage.morphology import dilation, disk
    selem = disk(width)
    if nuclei.ndim == 3:
        dilated = np.array([dilation(x, selem) for x in nuclei])
    else:
        dilated = dilation(nuclei, selem)
    dilated[nuclei > 0] = 0
    return dilated

def analyze_stitched_nuclei(f, num_frames=30, tolerance_per_frame=1.):
    
    motion_threshold=num_frames * tolerance_per_frame
    
    d = parse(f)
    d['subdir'] = 'process'
    f2 = name(d, tag='nuclei_%d' % num_frames)
    f3 = name(d, tag='nuclei_%d_track' % num_frames)

    flag1 = os.path.exists(f2)
    flag2 = os.path.exists(f3)
    if flag1 and flag2:
        return
    
    hoechst = read(f)[:num_frames, 1]
    
    if not flag1:
        rescaling = hoechst.mean(axis=(1, 2)) / 2000
        hoechst_norm = (hoechst * rescaling[:, None, None]).astype(np.uint16)
        nuclei = np.array(map(get_nuclei, hoechst_norm))
        luts = (GLASBEY,)*len(nuclei)
        save(f2, nuclei, luts=luts)
    else:
        nuclei = read(f2)
        
    arr_df = get_nuclei_features(hoechst, nuclei)
    G = initialize_graph(arr_df)
    cost, path = analyze_graph(G)
    relabel = filter_paths(cost, path, threshold=motion_threshold)
    nuclei_ = relabel_nuclei(nuclei, relabel)

    save(f3, nuclei_, luts=luts)

def load_and_concat(files):
    """expects stitched files
    """
    a, b, c= [read(f) for f in files]
    a = a[None]
    b = b[:-1]

    xs = pile(list(a) + list(b) + list(c))
    i0, i1, i2 = 0, 1, 1 + len(b)

    offsets = register_images([xs[i0, 1], xs[i1, 1], xs[i2, 1]])
    aligned = [lasagna.io.offset(xs[:i1], offsets[0]),
               lasagna.io.offset(xs[i1:i2], offsets[1]),
               lasagna.io.offset(xs[i2:], offsets[2])]

    data = np.array(list(aligned[0]) + list(aligned[1]) + list(aligned[2]))
    mask = data.min(axis=(0, 1)) > 0
    i, j = np.where(mask)
    data = data[:, :, i.min():i.max(), j.min():j.max()]

    return data

def concat_and_save(files):
    data = load_and_concat(files)
    
    d = parse(df['file'].iloc[0])
    f2 = name(d, subdir=os.path.join('process', dataset))

    dr = (2000, 10000), (500, 10000)
    save(f2, data, luts=(GREEN, RED), display_ranges=dr)

def get_nuclei(dapi):
    from lasagna.process import find_nuclei
    return find_nuclei(dapi, 
                       threshold=lambda x: 2400, 
                       area_min=0.25*150, 
                       area_max=0.25*800).astype(np.uint16)


def get_nuclei_features(hoechst, nuclei):
    features = {'label': lambda r: r.label,
            'area': lambda r: r.area,
            'i': lambda region: region.centroid[0],
            'j': lambda region: region.centroid[1],
           }

    arr = []
    for i, (h, n) in enumerate(zip(hoechst, nuclei)):
        arr.append(lasagna.process.feature_table(h, n, features)
                   .assign(frame=i))

    return arr


def get_edges(df1, df2):
    from scipy.spatial.kdtree import KDTree
    get_label = lambda x: tuple(int(y) for y in x[[2, 3]])

    x1 = df1[['i', 'j', 'frame', 'label']].as_matrix()
    x2 = df2[['i', 'j', 'frame', 'label']].as_matrix()
    
    kdt = KDTree(df1[['i', 'j']])
    points = df2[['i', 'j']]

    result = kdt.query(points, 3)
    edges = []
    for i2, (ds, ns) in enumerate(zip(*result)):
        end_node = get_label(x2[i2])
        for d, i1 in zip(ds, ns):
            start_node = get_label(x1[i1])
            w = d
            edges.append((start_node, end_node, w))

    return edges


def initialize_graph(arr_df):
    df_all = pd.concat(arr_df)
    nodes = df_all[['frame', 'label']].as_matrix()
    nodes = [tuple(x) for x in nodes]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    edges = []
    for df1, df2 in zip(arr_df, arr_df[1:]):
        edges = get_edges(df1, df2)
        G.add_weighted_edges_from(edges)
    
    return G

def analyze_graph(G):
    start_nodes = [n for n in G.nodes if n[0] == 0]
    max_frame = max([frame for frame, _ in G.nodes])
    
    cost, path = nx.multi_source_dijkstra(G, start_nodes, cutoff=100)
    cost = {k:v for k,v in cost.items() if k[0] == max_frame}
    path = {k:v for k,v in path.items() if k[0] == max_frame}
    return cost, path

def filter_paths(cost, path, threshold=35):
    """returns list of one [(frame, label)] per trajectory
    """
    node_count = Counter(sum(path.values(), []))
    bad = set(k for k,v in node_count.items() if v > 1)

    too_costly = [k for k,v in cost.items() if v > threshold]
    bad = bad | set(too_costly)

    relabel = [v for v in path.values() if not (set(v) & bad)]
    assert(len(relabel) > 0)
    return relabel

def relabel_nuclei(nuclei, relabel):
    nuclei_ = nuclei.copy()
    max_label = nuclei.max() + 1
    for i, nodes in enumerate(zip(*relabel)):
        labels = [n[1] for n in nodes]
        table = np.zeros(max_label).astype(int)
        table[labels] = range(len(labels))
        nuclei_[i] = table[nuclei_[i]]

    return nuclei_


def preprocess_sites():
    # site names snake by row...
    remap = {'0': '0', '1': '1', '2': '3', '3': '2'}
    files = glob('10X*/*.tif')

    for f in tqdm(files):
        d = parse(f)
        d['subdir'] = os.path.join('tmp', d['subdir'])
        d['site'] = remap[d['site']]
        f2 = name(d)
        save(f2, read(f).reshape(-1, 2, 1024, 1024))

        
def preprocess_stitched():
    dataset = '10X_plateB'

    files = glob('tmp/*stitched.tif')
    df_files = pd.DataFrame(map(parse, files)).assign(file=files)

    gb = df_files.sort_values('file')
    for ws, df in tqdm(df_files.groupby(['well'])):
        concat_and_save(df['file'])
