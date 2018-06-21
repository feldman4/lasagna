from lasagna.imports import *

def relabel_array(arr, new_label_dict):
    """new_labels is a dictionary {old: new}
    """
    n = arr.max()
    arr_ = np.zeros(n+1)
    for k, v in new_label_dict.items():
        if k <= n:
            arr_[k] = v
    
    return arr_[arr]

def outline_mask(arr, direction='outer'):
    from skimage.morphology import erosion, dilation
    arr = arr.copy()
    if direction == 'outer':
        mask = erosion(arr)
        arr[mask > 0] = 0
        return arr
    elif direction == 'inner':
        mask1 = erosion(arr) == arr
        mask2 = dilation(arr) == arr
        arr[mask1 & mask2] = 0
        return arr
    else:
        raise ValueError(direction)
    

def bitmap_label(labels, positions, colors=None):
    positions = np.array(positions).astype(int)
    if colors is None:
        colors = [1] * len(labels)
    i_all, j_all, c_all = [], [], []
    for label, (i, j), color in zip(labels, positions, colors):
        if label == '':
            continue
        i_px, j_px = np.where(lasagna.io.bitmap_text(label))
        i_all += list(i_px + i)
        j_all += list(j_px + j)
        c_all += [color] * len(i_px)
        
    shape = max(i_all) + 1, max(j_all) + 1
    arr = np.zeros(shape, dtype=int)
    arr[i_all, j_all] = c_all
    return arr

def index_singleton_clusters(clusters):
    clusters = clusters.copy()
    filt = clusters == -1
    n = clusters.max()
    clusters[filt] = range(n, n + len(filt))
    return clusters

def build_GRMC():
    import seaborn as sns
    colors = (0, 1, 0, 1), 'red', 'magenta', 'cyan'
    lut = []
    for color in colors:
        lut.append([0, 0, 0, 1])
        lut.extend(sns.dark_palette(color, n_colors=64 - 1))
    lut = np.array(lut)[:, :3]
    RGCM = np.zeros((256, 3), dtype=int)
    RGCM[:len(lut)] = (lut * 255).astype(int)
    return tuple(RGCM.T.flatten())

def build_glasbey2():
    glasbey = lasagna.io.read_lut('glasbey_inverted')
    glasbey = np.array(glasbey).T.reshape((256, 3))
    # saturate?
    glasbey[glasbey < 100] = 100
    glasbey[0] = 0
    return glasbey.T.astype(int).flatten()

def combine_annotated(data_ph, data_sbs, labels_reads, labels_cells, labels_phenotype):
    """Assign phenotype and SBS independent color channels. 
    Use Z dimension for SBS cycles.
    """
    t = [1, 0, 2, 3]
    non_sbs_index = [0, 1, 7, 8]
    
    z = data_sbs.shape[0]
    c = 9
    h, w = data_ph.shape[-2:]
    arr = np.zeros((z, c, h, w))
    
    arr[:,  :2] = data_ph
    arr[:, 2:6] = data_sbs
    arr[:, 6]   = labels_reads
    arr[:, 7]   = labels_cells
    arr[:, 8]   = labels_phenotype

    # data_ph = data_ph[:, None]
    # data_sbs = data_sbs.transpose(t)
    # labels_cells = labels_cells[None, None]
    # labels_reads = labels_reads[:, None]
    # labels_phenotype = labels_phenotype[None, None]
    
    # arr = sum([list(x) for x in (data_ph, data_sbs, labels_cells)], [])
    # arr = pile(arr).transpose(t)
    # arr[:, non_sbs_index] = arr[0, non_sbs_index]
    # arr = np.concatenate([arr, labels_reads], axis=1)

    return arr

def annotate_cells(df_cells, use_nuclei_phenotype=False):
    from skimage.morphology import dilation
    assert len(df_cells.drop_duplicates(['well', 'tile'])) == 1
    df_cells = df_cells.assign(cluster=lambda x: x['cluster'].pipe(index_singleton_clusters))

    well = df_cells['well'].iloc[0]
    tile = df_cells['tile'].iloc[0]
    
    description = {'subdir': 'process', 'well': well, 'tile': tile,
             'mag': '10X', 'ext': 'tif'}
        
    data_ph   = read(name(description, tag='phenotype_aligned'))
    data_sbs  = read(name(description, tag='log'))
    cells     = read(name(description, tag='cells'))
    if use_nuclei_phenotype:
        nuclei_labels = df_cells.set_index('cell_ph')['cell'].to_dict()
        nuclei_ph = read(name(description, tag='nuclei_phenotype'))
        nuclei = relabel_array(nuclei_ph, nuclei_labels)
    else:
        nuclei    = read(name(description, tag='nuclei'))

    cluster_labels = (df_cells
                      .set_index('cell')
                      .query('cluster > -0.5')
                      .eval('cluster + 1').to_dict())

    cells = relabel_array(cells, cluster_labels)
    nuclei_mask = outline_mask(nuclei).astype(float)
    nuclei_mask = nuclei.astype(float) / 10000
    labels_cells = outline_mask(cells, direction='inner')
    labels_cells[labels_cells == 0] += nuclei_mask[labels_cells == 0]
    
    gb_cluster = df_cells.groupby(['sgRNA_name', 'cluster'])
    labels = gb_cluster['gene'].nth(0)
    positions = gb_cluster[['i_SBS', 'j_SBS']].mean().as_matrix()
    positions[:,1] += 10
    colors = gb_cluster['cluster'].nth(0) + 1
    
    labels = [l if isinstance(l, str) else '' for l in labels]
    labeled = bitmap_label(labels, positions, colors)
    h = min(labels_cells.shape[0], labeled.shape[0])
    w = min(labels_cells.shape[1], labeled.shape[1])
    
    # gene labels go on top
    mask = np.zeros(labels_cells.shape, dtype=bool)
    mask[:h, :w] = dilation(labeled > 0, selem=np.ones((3,3)))[:h, :w]
    labels_cells[mask] = 0
    labels_cells[:h, :w] = labels_cells[:h, :w] + labeled[:h, :w]
    
    return data_ph, data_sbs, labels_cells

def annotate_reads(df_reads, shape=(1024, 1024)):
    """Filter to a single (well, tile) before calling.
    """
    from skimage.morphology import dilation
    assert len(df_reads.drop_duplicates(['well', 'tile'])) == 1
    n = len(df_reads['barcode'].iloc[0])
    shape = n, shape[0], shape[1]
    
    base_to_channel = {k: i for i, k in enumerate(in_situ.IMAGING_ORDER)}
    barcodes = (np.array([base_to_channel[c] for b in df_reads['barcode'] for c in b])
                .T.reshape(-1, n))

    ij = df_reads[['i', 'j']].as_matrix()

    arr = np.zeros(shape, dtype=int)
    for (i, j), barcode in zip(ij, barcodes):
        arr[:, i, j] = (barcode + 1) * 64 - 1

    labels_reads = np.array([dilation(x, selem=np.ones((3, 3))) for x in arr])
    return labels_reads

def annotate_phenotype(df_cells, value):
    assert len(df_cells.drop_duplicates(['well', 'tile'])) == 1

    well = df_cells['well'].iloc[0]
    tile = df_cells['tile'].iloc[0]
    description = {'subdir': 'process', 'well': well, 'tile': tile,
         'mag': '10X', 'ext': 'tif'}

    cells = read(name(description, tag='cells'))
    cells = outline_mask(cells, 'inner')
    
    cells_to_phenotype = df_cells.set_index('cell')[value]
    phenotype = relabel_array(cells, cells_to_phenotype)
    
    return phenotype

def save_annotated(*args, **kwargs):
    index = kwargs.pop('index', None) # default
    dr_ph = (0, 15), (0, 2.5)
    dr_ph = None, None
    dr_sbs = (50, 10000), (50, 10000), (50, 7000), (50, 7000)
    dr_labels = (1, 255), None,  None
    luts_ph = GRAY, GREEN
    luts_sbs = GREEN, RED, MAGENTA, CYAN
    luts_labels = build_GRMC(), build_glasbey2(), GRAY
    luts = np.array(luts_ph + luts_sbs + luts_labels)
    dr = np.array(dr_ph + dr_sbs + dr_labels)
    if index:
        luts, dr = luts[index], dr[index]
    kwargs['luts'] = luts
    kwargs['display_ranges'] = dr
    return save(*args, **kwargs)
