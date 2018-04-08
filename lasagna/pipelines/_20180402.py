from lasagna.imports import *

gate_cells = '(3000 < dapi_nuclear_max < 12000)&(60 < area_nuclear < 140)&(2000 < gfp_cell_median < 8000)'
gate_NT = 'dapi_gfp_nuclear_corr < -0.5'

stimulant = {'A': 'TNFa', 'B': 'IL1b'}

def combine_phenotypes(df_ph_full, df_ph_perimeter):
    """Combine phenotype data from `Snake._extract_phenotype_translocation` and 
    `Snake._extract_phenotype_translocation_ring`.
    """
    key_cols = ['well', 'tile', 'cell']

    val_cols = [
        "dapi_gfp_nuclear_corr",
        "dapi_nuclear_int",
        "dapi_nuclear_max",
        "dapi_nuclear_median",
        "gfp_nuclear_int",
        "gfp_nuclear_max",
        "gfp_nuclear_mean",
        "gfp_nuclear_median",
        "x",
        "y",
        "dapi_gfp_cell_corr",
        "gfp_cell_mean",
        "gfp_cell_median",
        "gfp_cell_int"
    ]
    
    df_ph_perimeter = (df_ph_perimeter
                       .set_index(key_cols)[val_cols]
                       .rename(columns=lambda x: x + '_perimeter'))
    
    return df_ph_full.join(df_ph_perimeter, on=key_cols)
    

def add_phenotype_cols(df_ph):
    return (df_ph
        .assign(gcm=lambda x: x.eval('gfp_cell_median - gfp_nuclear_median')))

def annotate_cells(df_cells):
    def get_gene(sgRNA_name):
        if sgRNA_name is np.nan:
            return sgRNA_name
        if sgRNA_name.startswith('LG'):
            return 'LG'
        pat = 'sg_(.*?)_'
        return re.findall(pat, sgRNA_name)[0]

    def get_targeting(sgRNA_name):
        if sgRNA_name is np.nan:
            return False
        else:
            return 'LG_sg' not in sgRNA_name

    return (df_cells
        .assign(gate_NT=lambda x: x.eval(gate_NT))
        .assign(gene=lambda x: x['sgRNA_name'].apply(get_gene))
        .assign(targeting=lambda x: x['sgRNA_name'].apply(get_targeting))
        )

