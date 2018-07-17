from lasagna.imports import *

import seaborn as sns
from matplotlib import patches

def cells_to_distributions(df_cells, col='dapi_gfp_corr_nuclear', scale=10):
    """drop na first
    """
    index = ['gene', 'sgRNA_name', 'replicate', 'stimulant']
    return (df_cells
     .assign(bin=lambda x: (x[col] * scale).round().astype(int))
     .pivot_table(index=index, columns='bin', values='cell', aggfunc='count').fillna(0).astype(int)
    )

def normalized_cdf(df, show_sampling=False):
    df_n = df.cumsum(axis=1).divide(df.sum(axis=1), axis=0)
    if show_sampling:
        df_n[12] = (np.log(df.sum(axis=1)) / 10).clip_upper(1)
    return df_n

def plot_distribution(df):

    hue_order = df.reset_index()['sgRNA_name'].value_counts().pipe(lambda x: natsorted(set(x.index)))
    colors = iter(sns.color_palette(n_colors=10))
    palette, legend_data = [], {}
    for name in hue_order:
        palette += ['black' if name.startswith('LG') else colors.next()]
        legend_data[name] = patches.Patch(color=palette[-1], label=name)


    
    def plot_lines(**kwargs):
        df = kwargs.pop('data')
        color = kwargs.pop('color')
        ax = plt.gca()
        (df
         .filter(regex='\d')
         .T.plot(ax=ax, color=color)
        )

    fg = (df
     .pipe(normalized_cdf)
     .reset_index()
     .pipe(sns.FacetGrid, row='stimulant', hue='sgRNA_name', col='replicate', 
           palette=palette, hue_order=hue_order)
     .map_dataframe(plot_lines)
     .set_titles("{row_name} rep. {col_name}")
     .add_legend(legend_data=legend_data)
    )
    return fg

def export_distributions(gene):
    f = 'figures/distributions/{0}.pdf'.format(gene)
    if not os.path.exists(f):
        select_genes = [gene, 'sg0', 'sg1', 'sg2']
        fg = (df_dist
         .query('gene == @select_genes')
         .pipe(plot_distribution))

        fg.savefig(f)
        plt.close(fg.fig)


def integrated_diff(df_dist, min_cells=15):
    arr = []
    index_cols = ['replicate', 'stimulant', 'gene', 'sgRNA_name']
    gb_cols = ['replicate', 'stimulant']
    for _, df in df_dist.set_index(index_cols).groupby(gb_cols):
        nontargeting = ['sg0', 'sg1', 'sg2', 'sg3', 'sg4']
        reference = (df
         .pipe(normalized_cdf)
         .query('gene == @nontargeting')
         .mean(axis=0)
        )
        (df
        .loc[lambda x: x.sum(axis=1) >= min_cells]
        .pipe(normalized_cdf)
         .subtract(reference)
         .sum(axis=1)
        .pipe(arr.append))

    df = pd.concat(arr, axis=0).rename('int_diff').reset_index()
#     return df
    df_int_stats = (df
     .sort_values('int_diff', ascending=False)
     .groupby(['gene', 'stimulant'])
     ['int_diff'].nth(1)
     .reset_index()
     .pivot_table(index='gene', columns='stimulant', values='int_diff')
    )
    
    return df_int_stats        