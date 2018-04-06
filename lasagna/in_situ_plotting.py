from lasagna.imports_ipython import *

def plot_reads_per_cell(df_cells, **line_kwargs):
    ax = \
    (df_cells
     ['cell_barcode_count_0'].value_counts()
     .pipe(lambda x: np.cumsum(x / x.sum()))
     .rename('cumulative fraction of cells')
     .rename_axis('reads per cell', axis=0)
     [:10]
     .plot.line(**line_kwargs)
    )
    ax.set_ylim([0, 1.05])
    return ax


def plot_red_blue_scatter(df_cells, df_design):
    stats = calculate_barcode_stats(df_cells, df_design)
    fg = (stats
     .pipe(sns.FacetGrid, hue='sgRNA type', palette=['blue', 'red'], size=6)
     .map(plt.scatter, 'cells per barcode', 'fraction positive cells', 
          s=20, alpha=0.7))

    fg.add_legend()
    ax = fg.axes.flat[0]
    ax.set_xscale('log')
    ax.set_xlim([10, 3000])
    return fg

def plot_sgRNA_phenotype_box(df_cells, df_design):
    stats = calculate_barcode_stats(df_cells, df_design)
    cell_threshold = 100

    ax = \
    (stats
     .sort_values('sgRNA_name')
     .pipe(lambda x: x[x['cells per barcode'] > cell_threshold])
     .pipe(lambda x: sns.boxplot(data=x, x='sgRNA_name', 
                     y='fraction positive cells', 
                     hue='sgRNA type'))
    )
    num = stats['cells per barcode'] > cell_threshold

    ax.set_xticklabels([])
    ax.set_xlabel('sgRNA')
    ax.set_title('showing %d/%d barcodes above %d cells' % (num.sum(), len(num), cell_threshold))
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    return ax


def calculate_barcode_stats(df_cells, df_design):
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


####





def plot_mean_quality_per_tile(df_reads):
    stats_q = (df_reads
               .filter(regex='Q_\d+|well|tile')
               .groupby(['well', 'tile'])
               .mean().dropna()
               .stack().rename('mean quality per tile').reset_index()
               .rename(columns={'level_2': 'cycle'})
              )

    ax = sns.boxplot(data=stats_q, x='cycle', y='mean quality per tile', 
                     whis=[10, 90])
    ax.set_ylim([0.3, 1])
    ax.set_xticklabels(range(1, 11))
    ax.figure.tight_layout()

    return stats_q, ax



def groupby_barcode(df_reads):
    gb = df_reads.groupby(['barcode', 'well_6'])
    s1 = gb['FR_pos'].mean()
    s2 = gb.size().rename('num_cells')
    s3 = gb['design'].nth(0)
    df_plot = pd.concat([s1, s2, s3], axis=1).reset_index()
    df_plot['num_cells_log'] = np.log10(df_plot['num_cells'])
    df_plot['FR_pos_rescale'] = df_plot['FR_pos']**0.3
    return df_plot

def plot_FR_vs_barcode(df_reads):
    df_plot = groupby_barcode(df_reads)
    bins = np.linspace(0, 0.8, 100)
    g = sns.FacetGrid(df_plot, hue='design',
                      col='well_6',
                      hue_order=sorted(set(df_plot['design'])),
                      size=5)
    g.map(sns.distplot, 'FR_pos', hist=True, norm_hist=False, 
          kde=False,  bins=bins)

    ax = g.axes[0, 0]
    ax.set_ylabel('barcodes')
    # g.add_legend()
    return g

def plot_roc(y_true, y_score, **kwargs):
    from sklearn.metrics import roc_curve
    # y_true, y_score = df_plot['design'], df_plot['FR_pos']
    fpr,tpr,thresholds = roc_curve(y_true, y_score, pos_label='FR_GFP_TM')
    plt.plot(fpr, tpr, **kwargs)


# from https://github.com/pandas-dev/pandas/issues/18124
from functools import wraps

def monkey_patch_series_plot():
    """Makes Series.plot to show series name as ylabel/xlabel according to plot kind."""
    f = pd.Series.plot.__call__
    @wraps(f)
    def _decorator(*kargs, **kwargs):
        res = f(*kargs, **kwargs)
        s = kargs[0].__dict__['_data']
        if s.name:
            try:
                kind = kwargs['kind']
            except KeyError:
                kind = 'line'
            if kind == 'line' or kind == 'scatter':
                plt.ylabel(s.name)
            elif kind == 'hist':
                plt.xlabel(s.name)
        return res
    pd.Series.plot.__call__ = _decorator

