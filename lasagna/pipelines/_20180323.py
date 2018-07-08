from lasagna.imports import *
from lasagna.in_situ import load_NGS_hist

def load_well_site_list():
    well_site_list = map(tuple, pd.read_pickle('well_site_list_MM.pkl').values())       
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

def copy_tif_to_process_dir(f, rows, cols):
    d = parse(f)
    d['site'] = lasagna.plates.remap_snake(d['site'], rows=rows, cols=cols)
    d['subdir'] = 'process/{mag}_{cycle}'.format(**d)
    f_SBS = name(d, tag='max')

    d['cycle'] = 'c0-HA-488'
    d['subdir'] = 'process/{mag}_{cycle}'.format(**d)
    f_ph = name(d, tag='max')
    
    save(f_SBS, read(f))

def select(x):
    return ''.join(x[i-1] for i in (1,2,3,6,7,8,9,10,11,12))

def load_gDNA_NGS_hists():
    """need to select out the bases used
    """
    search = '/Users/feldman/lasagna/NGS/20180325_AS/cLas41_46/T1_B0[123]*hist'
    files = glob(search)
    wells = ['A1_NGS', 'A2_NGS', 'A3_NGS']
    file_to_well = {f: w for f,w in zip(files, wells)}
    return (map(load_NGS_hist, glob(search))
        .assign(barcode=lambda x: x['barcode_full'].apply(select))
        .assign(well=lambda x: x['file'].map(file_to_well)))


def load_pDNA_NGS_hists():
    """need to select out the bases used
    """
    search = '/Users/feldman/lasagna/NGS/20180325_AS/pLL_plasmid/T1_A0[12]*hist'
    files = glob(search)
    wells = ['A1_NGS', 'A2_NGS', 'A3_NGS']
    file_to_well = {f: w for f,w in zip(files, wells)}
    return (map(load_NGS_hist, glob(search))
        .assign(barcode=lambda x: x['barcode_full'].apply(select))
        .assign(well=lambda x: x['file'].map(file_to_well)))


def calculate_barcode_stats(df_reads, df_cells):
    gb = df_reads.groupby('barcode')
    a = gb['Q_min'].mean().rename('Q_min')
    b = gb.size().rename('read_count')
    gb = df_cells.groupby('barcode')
    c = gb['cell_barcode_count_0'].mean().rename('reads_per_cell')
    d = gb.size().rename('cell_count')
    return pd.concat([a,b,c,d], axis=1)

def calculate_barcode_stats_NGS(df_ngs):
    cols = {'A1_NGS': 'NGS_count_A1',
            'A2_NGS': 'NGS_count_A2',
            'A3_NGS': 'NGS_count_A3',
            }
    return (df_ngs.pivot_table(index='barcode', columns='well', 
                           values='count', aggfunc=sum)
            .rename(columns=cols))

def barcode_stats_by_well(df_reads, df_cells, wells=('A1', 'A2', 'A3')):
    arr = []
    for well in ('A1', 'A2', 'A3'):
        gate = 'well == @well'
        (calculate_barcode_stats(
            df_reads.query(gate), 
            df_cells.query(gate))
            .rename(columns=lambda x: x + '_' + well)
            .pipe(arr.append))

    return pd.concat(arr, axis=1).sort_index(axis=1).fillna(0)

def barcode_stats(df_reads, df_cells, good_barcodes):

    stats_ngs = (load_NGS_hists()
                 .pipe(calculate_barcode_stats_NGS))

    gate = 'subpool == "pool2_1"'
    stats_in_situ = barcode_stats_by_well(
                        df_reads.query(gate), 
                        df_cells.query(gate))

    stats = (pd.concat([stats_ngs, stats_in_situ], axis=1)
             .query('index == @good_barcodes'))

    # normalize
    baseline = 1e-5
    cols = stats.filter(regex='count').columns
    normed = stats[cols] / stats[cols].sum()
    stats[cols] = np.log10(baseline + normed.fillna(0))

    wells = 'A1', 'A2', 'A3'
    for well in wells:
        cell_col = 'cell_count_' + well
        ngs_col  = 'NGS_count_'  + well
        stats['cells_per_NGS_' + well] = stats.eval(cell_col + ' - ' + ngs_col)

    return stats

def calculate_barcode_stats(df_cells, df_design):
    """moved from lasagna.in_situ_plotting
    """
    gb = (df_cells
      .query('subpool == "pool2_1"')
      .groupby('cell_barcode_0'))
    A = (gb.size().rename('cells per barcode'))
    B = (gb['FR_pos']
         .mean().clip(upper=0.13)
         .rename('fraction positive cells'))

    s = df_design.set_index('barcode')[['sgRNA_design', 'sgRNA_name']]
    stats = (pd.concat([A, B], axis=1).reset_index()
               .join(s, on='cell_barcode_0'))

    rename = lambda x: 'targeting' if x == 'FR_GFP_TM' else 'nontargeting'
    stats['sgRNA type'] = stats['sgRNA_design'].apply(rename)
    return stats
