from lasagna.imports_ipython import *

def plot_reads_per_cell(df_reads, df_ph):
    cols = ['well', 'tile', 'cell']
    x = df_reads.set_index(cols)['cell_barcode_count_0']
    df = df_ph.join(x, on=cols).fillna(0)
    s = df['cell_barcode_count_0'].value_counts()
    
    t = np.cumsum(s / s.sum())
    ax = t.reset_index().plot(kind='scatter', x=0, y=1)
    ax.set_xlabel('reads per cell')
    ax.set_ylabel('cumulative fraction of cells')
    ax.set_ylim([0, 1.05])
  
    return ax

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