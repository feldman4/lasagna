from lasagna.imports import *

def load_well_site_list():
    well_site_list = map(tuple, pd.read_pickle('well_site_list.pkl').as_matrix())       
    return well_site_list


def filter_well_site_filename(f, well_site_list):
    """
    files = glob('10X_c1-SBS-1_1/*tif')
    wsl = pipeline.load_well_site_list()
    filt = partial(pipeline.filter_well_site_filename, well_site_list=set(wsl))
    files = filter(filt, files)
    """
    d = parse(f)
    key = d['well'], d['site']
    return key in well_site_list


def copy_tif_to_process_dir(f):
    d = parse(f)
    d['site'] = lasagna.plates.remap_snake(d['site'])
    d['subdir'] = 'process/{mag}_{cycle}'.format(**d)
    f_SBS = name(d, tag='max')

    d['cycle'] = 'c0-HA-488'
    d['subdir'] = 'process/{mag}_{cycle}'.format(**d)
    f_ph = name(d, tag='max')
    
    if 'c1-SBS-1' not in f:
        if not os.path.exists(f_SBS):
            save(f_SBS, read(f))
    else:
        if not os.path.exists(f_SBS):
            data = read(f)
            data_SBS = data[[0, 2, 3, 4, 5]]
            save(f_SBS, data_SBS)
            
        if not os.path.exists(f_ph):
            data = read(f)
            data_ph = data[:2]
            save(f_ph, data_ph)