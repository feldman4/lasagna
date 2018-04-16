from lasagna.imports_ipython import *


def plot_reads_per_cell(df_cells, ax=None):
    ax = \
    (df_cells
     ['cell_barcode_count_0'].value_counts()
     .pipe(lambda x: np.cumsum(x / x.sum()))
     .rename('cumulative fraction of cells')
     .rename_axis('reads per cell', axis=0)
     [:10]
     .plot.line(ax=ax)
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

def plot_quality_tile_clustermap(df_reads):
    def color_unique(s, cmap):
        uniq = s.unique()
        pal = sns.color_palette(cmap, len(uniq))
        return list(s.map(dict(zip(uniq, pal))))
    
    cols = list(df_reads.filter(axis=1, regex='Q_\d+').columns)
    df_plot = df_reads.groupby(['well', 'tile'])[cols].mean()
    colors = color_unique(df_plot.reset_index()['well'], 'viridis')

    cg = sns.clustermap(df_plot, 
                   row_colors=colors,
                   col_cluster=False)

    cg.ax_heatmap.set_title('tiles clustered by mean quality')
    cg.ax_row_colors.set_xlabel('well')
    cg.ax_row_dendrogram.set_visible(False)
    return cg

def plot_combined_clustermap(df_reads, df_ph, n_clusters=5):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import scale

    x = (df_ph.fillna(0)
              .groupby(['well', 'tile']).mean()
        )

    cols = list(df_reads.filter(axis=1, regex='Q_\d+').columns)
    y = df_reads.groupby(['well', 'tile'])[cols].mean()

    df_plot = pd.concat([x, y], axis=1, join='inner')
    # return df_plot
    X = standardize(df_plot)
    ag = AgglomerativeClustering(n_clusters)
    ag.fit(X)
    labels = ag.labels_
    ix = np.argsort(labels)
    palette = np.array(sns.color_palette('Set2', n_clusters))
    colors = palette[labels]
    row_colors = colors[ix]

    cg = sns.clustermap(df_plot.iloc[ix], 
                   standard_scale=True,
                   row_colors=row_colors,
                   row_cluster=False,
                   col_cluster=False)

    
    cg.ax_row_colors.set_xlabel('well', rotation=90)
    cg.ax_row_dendrogram.set_visible(False)    

    cg.cax.set_visible(False)

    ax = cg.fig.add_axes([-.27, 0.2, 0.5, 0.5])
    cluster_series = (
        pd.Series(labels, index=X.index)
         .rename('cluster')
         .reset_index()
         .pipe(lasagna.plates.add_global_xy, '6w', grid_shape=(25, 25))
         .assign(global_x=lambda x: x['global_x'] / 1000, 
                 global_y=lambda x: x['global_y'] / 1000)
         )
    cluster_series.plot.scatter(x='global_x', y='global_y', 
                       c=colors, s=40, ax=ax)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('tiles clustered by quality and phenotype')
    ax.axis('equal')

    return cg, cluster_series


def plot_mean_quality_per_tile(df_reads, ax=None):
    """compatible with sns.FacetGrid
    """
    stats_q = (df_reads
               .filter(regex='Q_\d+|^well$|^tile$')
               .groupby(['well', 'tile'])
               .mean().dropna()
               .stack().rename('mean quality per tile').reset_index()
               .rename(columns={'level_2': 'cycle'})
              )

    ax = sns.boxplot(data=stats_q, x='cycle', y='mean quality per tile', 
                     whis=[10, 90], ax=ax)
    ax.set_ylim([0.3, 1])
    num_cycles = len(stats_q['cycle'].value_counts())
    ax.set_xticklabels(range(1, num_cycles + 1))
    ax.set_xlabel('cycle')
    ax.set_ylabel('mean quality per tile')

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


### UTILITIES

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

def sns_wrap(f):
    def restrict_kwargs(kwargs, f):
        import inspect
        f_kwargs = set(get_kwarg_defaults(f).keys()) | set(get_arg_names(f))
        keys = f_kwargs & set(kwargs.keys())
        return {k: kwargs[k] for k in keys}

    argspec = inspect.getargspec(f)
    arg_df = argspec.args[0]
    def inner(data, **kwargs):
        kwargs_ = firesnake.restrict_kwargs(kwargs, f)
        return f(data, **kwargs_)
    return inner   