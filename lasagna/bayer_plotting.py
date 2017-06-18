from lasagna.bayer import *
import seaborn as sns
import matplotlib.pyplot as plt

def plot_primary_secondary(df_, rows, cols, size=2.5):

    pal = 'blue', 'green', 'red', 'purple'

    short_barcode = 'barcode_'
    second_base = 'second_base'
    # unique barcodes up to # of cycles (cols)
    df_[short_barcode] = df_['barcode'].apply(lambda x: x[:cols])

    select_barcodes = df_[short_barcode].value_counts().index[:rows]
    select_cycles = sorted(df_['cycle'].value_counts().index)[:cols]
    filt = df_[short_barcode].isin(select_barcodes)
    filt &= df_['cycle'].isin(select_cycles)

    with sns.plotting_context(font_scale=size * 0.45):
        
        fg = sns.FacetGrid(df_[filt], row=short_barcode, col='cycle'
                           , hue=second_base, hue_order=list('TGCA')
                           , palette=pal, size=size
                           , row_order=select_barcodes
                           , col_order=sorted(select_cycles))

        (fg.map(plt.scatter, 'first', 'ratio', s=2, alpha=0.5, linewidth=0)
           .set_titles("{col_name}"))
        [ax.set_title('') for ax in fg.axes[1:].flat[:]];
        # set axis color to barcode color
        for i, (barcode, axes) in enumerate(zip(fg.row_names, fg.axes)):
            axes[0].set_ylabel(barcode)
            for j, (c, ax) in enumerate(zip(barcode, axes)):
                color = pal['TGCA'.index(c)]
                if i != rows - 1:
                    ax.spines['bottom'].set_color('white')
                else:
                    ax.set_xlabel('log10(primary)')
                if j != 0:
                    ax.spines['left'].set_color('white')
                else:
                    ax.set_ylabel('log10(secondary/primary)')
                ax.plot([0, 5], [0, 0], c=color, linewidth=3)
                ax.set_xlim([2.5, 5])
                ax.set_ylim([-1.5, 0.3])
                ax.text(3.75, 0.05, c, color=color, fontsize=size * (24./3))

        plt.subplots_adjust(hspace=0.1, wspace=0.1)
    return fg


def legend_min_lines(fg, n):
    """Include legend for lines with peaks at smallest x values.
    """
    find_max = lambda line: line.get_xdata()[np.argmax(line.get_ydata())]
    lines_sort = lambda ax: sorted(ax.lines, key=find_max)

    for ax in fg.axes.flat[:]:
        if ax.lines:
            lines = lines_sort(ax)[:n]
            ax.legend(lines, [line.get_label() for line in lines])


def plot_base_cycle_distributions(df_, cycles=4, size=2.5, codes=24, lines=2):
    """
    Plots up to most `codes` abundant barcodes. Includes a per-base and cycle
    mean calculated over the full input data. Legend labels first `lines` 
    density estimates, ordered by peak location.
    """
    short_barcode = 'barcode_'
    cycle_col = 'cycle'
    base_col = 'first_base'
    value_col = 'first'
    
    grey_for_means = (0.4, 0.4, 0.4, 1)
    
    cycle_names = sorted(set(df_['cycle']))[:cycles]
    df_[short_barcode] = df_['barcode'].apply(lambda x: x[:cycles])
    # over all barcodes
    df_mean = df_.copy()
    df_mean[short_barcode] = ['mean_%s' % x for x in df_mean[base_col]]
    means = sorted(set(df_mean[short_barcode]))
    barcodes = list(df_[short_barcode].value_counts().index[:codes])

    df_plot = pd.concat([df_, df_mean])
    filt = df_plot[cycle_col].isin(cycle_names)
    filt &= df_plot[short_barcode].isin(barcodes + means)

    sns.set_context(font_scale=size * 0.45)
    palette = sns.color_palette('Set2', codes)
    palette += [grey_for_means,]*len(means)
    fg = sns.FacetGrid(data=df_plot[filt], row='cycle', col=base_col
                       , row_order=cycle_names, col_order=list('TCGA')
                       , hue=short_barcode, size=size
                       , palette=palette)

    (fg.map(sns.distplot, value_col, kde=True, hist=False)
       .set_titles("{col_name}"))
    [ax.set_title('') for ax in fg.axes[1:].flat[:]]
    for name, ax in zip(fg.row_names, fg.axes[:,0]):
        ax.set_ylabel(name)
    legend_min_lines(fg, lines)
    fg.fig.tight_layout()
    return fg


def plot_base_correlations(df, low=2, high=4):

    def log_scale(x):
        x *= 1.
        x = np.log10(1 + x)
        mask = x < low
        x[mask] = np.nan
        return x

    cycle_names = set(df.columns.get_level_values('cycle'))
    cycle_names = sorted([x for x in cycle_names if x.startswith('c')])

    base_palette = sns.xkcd_palette(['dark sky blue', 'medium green', 'light red', 'orchid'])
    palette      = sns.color_palette('Set1', len(cycle_names))

    df_plot = df[cycle_names].apply(log_scale).stack('cycle').reset_index()

    x_vars = list('TGCA')
    pg = sns.PairGrid(data=df_plot.sample(frac=0.005), hue='cycle'
                      , hue_order=cycle_names
                      , palette=palette
                      , x_vars=x_vars, y_vars=x_vars)

    def f(x, y, color=None, **kwargs):
        kwargs['cmap'] = None
        keep = ~(np.isnan(x) | np.isnan(y))
        return sns.kdeplot(y[keep], x[keep], colors=(color,), **kwargs)

    def g(*args, **kwargs):
        return plt.plot([low, high], [low, high], c='grey', lw=1)

    pg.map_upper(plt.scatter, s=4, linewidth=0, alpha=0.7)
    pg.map_lower(f, alpha=0.5)
    pg.map_upper(g)
    pg.map_lower(g)
    # pg.map_diag(sns.distplot, hist=False)

    for i, color in enumerate(base_palette):
        pg.axes[-1, i].xaxis.label.set_color(color)
        pg.axes[i,  0].yaxis.label.set_color(color)

    for ax in pg.axes.flat[:]:
        ax.set_xlim([low - 0.1, high])
        ax.set_ylim([low - 0.1, high])
    pg.fig.tight_layout()
    pg.add_legend()
    # fix legend marker size
    handles = pg.fig.legends[0].legendHandles
    for handle in handles:
        handle._sizes = [30]
    return pg
