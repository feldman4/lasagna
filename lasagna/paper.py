import json
import os
import shutil
import re
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.patheffects
import matplotlib.pyplot as plt
import seaborn as sns
import imageio


from lasagna.schema import *
import lasagna.utils
from lasagna.utils import groupby_reduce_concat

stimulant_order = 'IL1b', 'TNFa'

f_87_in = '/Users/feldman/lasagna/20180402_6W-G161/cells.csv'
f_87 = '/Users/feldman/lasagna/paper/data/87_genes.cells.csv'

f_FR_ngs_stats_in = '/Users/feldman/lasagna/NGS/20180424_DF/stats_rep1.csv'
f_FR_ngs_stats = '/Users/feldman/lasagna/paper/data/FR_ngs_stats.csv'

f_FR_cLas501_facs_in = '/Users/feldman/lasagna/FACS/20180419_FR_FACS/50.1_[2 Way] Data Source - 1.fcs'
f_FR_cLas501_facs = '/Users/feldman/lasagna/paper/data/20180419_cLas50.1.fcs'

DARK_GRAY   = '#555555'
DARK_GREEN  = '#66A763'
STRONG_GRAY = '#919191'
ORANGE      = '#F7941D'
BLUE        = '#4F74AC'
RED         = '#EF3F5A'
GRAY        = '#D6D6D6'
GREEN       = '#1EBB13'
LIGHT_BLUE  = '#8BC5EE'
LIGHT_GREEN = '#9EF28A'
VERY_LIGHT_GREEN = '#E8FCE3'
VERY_LIGHT_BLUE  = '#E1ECF6'

IMAGEJ_GREEN = '#29FD2F'
IMAGEJ_BLUE  = '#0A23F6'
MAGENTA = '#D453CD'

nucleotide_colors = {'A': MAGENTA, 'G': 'green', 
                     'T': 'red', 'C': 'blue'}

NFKB_axis_label = 'Non-translocated cells (%)'
NFKB_TNFa_axis_label = u'Non-translocated cells \nafter TNF\u03b1 stimulation (%)'
NFKB_IL1b_axis_label = u'Non-translocated cells \nafter IL1\u03b2 stimulation (%)'
NFKB_yticks = 0, 20, 40, 60, 80
TNFa_label = u'TNF\u03b1'
IL1b_label = u'IL1\u03b2'
NFKB_regulator_labels = {'both': u'IL1\u03b2 and TNF\u03b1 \nregulators', 
    'IL1b': u'IL1\u03b2 regulators', 
    'TNFa': u'TNF\u03b1 regulators'}
NFKB_phenotype_rename = {'dapi_gfp_nuclear_corr': 
    # 'mNeon:DAPI correlation'
    'translocation score'
    }

GENE_CLASS_COLORS = GREEN, ORANGE, RED, GRAY
STIMULANT_COLORS = GREEN, ORANGE

FR_design_order = 'FR_LG', 'FR_GFP_TM'

custom_rcParams = {
    'legend.handletextpad': 0,
    'legend.columnspacing': 0.6,
    'legend.fontsize': 14,
    'font.sans-serif': 'Helvetica',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.labelsize': 16,
}

RCP = custom_rcParams

def apply_custom_rcParams():
    mpl.rcParams.update(custom_rcParams)

def copy(src, dst):
    if os.path.exists(dst):
        message = 'skipping copy {dst} to {src}'
    else:
        message = 'creating {dst} from {src}'
        shutil.copy(src, dst)

    print(message.format(src=src, dst=dst))
    
        
def copy_data():
    if not os.path.exists(f_87):
        # 87 gene set
        from lasagna.pipelines._20180402 import annotate_cells, gate_cells
        df_cells = (pd.read_csv(f_87_in)
              .query(gate_cells)
              .pipe(annotate_cells)
              )

        dataset = '20180402_6W-G161'
        index = [STIMULANT, GENE_SYMBOL, SGRNA_NAME] 
        col = 'dapi_gfp_nuclear_corr'

        df_cells[index + [col]].assign(dataset=dataset).to_csv(f_87, index=None)

    copy(f_FR_ngs_stats_in, f_FR_ngs_stats)
    copy(f_FR_cLas501_facs_in, f_FR_cLas501_facs)

def load_cells():
    def load(f):
        return (pd.read_csv(f)
                .assign(stimulant=lambda x: x['stimulant']
                    .astype('category', order=stimulant_order)))

    files = f_87,
    df_cells = pd.concat(map(pd.read_csv, files))
    return df_cells

def load_sgRNA_stats():
    from lasagna.pipelines._20180422 import label_gene_classes
    df_cells = load_cells()

    feature = 'dapi_gfp_nuclear_corr'
    cell_statistics = 'mean', 'count'
    sg_statistic = 'mean'
    count_threshold = 30
    quantile = 0.03
    index = [DATASET, STIMULANT, GENE_SYMBOL, SGRNA_NAME]
    thresholds = df_cells.groupby(STIMULANT)[feature].quantile(quantile)
    assign_class = lambda x: x[GENE_SYMBOL].pipe(label_gene_classes)

    df_sg_stats = (df_cells
        .assign(threshold=list(thresholds[df_cells[STIMULANT]]))
        .assign(NT_positive=lambda x: x.eval('{0} < threshold'.format(feature)))
        .groupby(index)['NT_positive']
        .pipe(groupby_reduce_concat, *cell_statistics)
        .assign(gene_class=assign_class) 
        )


    df_gene_stats = (df_sg_stats
        .query('count > @count_threshold')
        # pandas bug if a categorical column is used as pivot_table index
        .drop([GENE_CLASS], axis=1)
        .pivot_table(index=GENE_SYMBOL, columns='stimulant', 
                  values='mean', aggfunc=sg_statistic)
        .reset_index()
        .assign(gene_class=assign_class) 
        )

    return df_sg_stats, df_gene_stats

# def condense_4K_data():

def analyze_barcodes_20180424(ngs_home, fillna=1e-4):
    """Filters barcodes that map to pool2_1, then 
    calculates fractions for each FACS/NGS replicate.
    
    Stats are calculated for the first FACS replicate, 
    averaging over the NGS replicates.
    """
    def load_hist(f):
        try:
            return (pd.read_csv(f, sep='\s+', header=None)
                .rename(columns={0: 'count', 1: 'seq'})
                .query('count > 3')
                .assign(fraction=lambda x: x['count']/x['count'].sum())
                .assign(log10_fraction=lambda x: np.log10(x['fraction']))
                .assign(file=f)
               )
        except:
            return None

    f = '/Users/feldman/lasagna/libraries/pool2_design.csv'
    df_design = pd.read_csv(f)

    
    f = os.path.join(ngs_home, 'sample_info.tsv')
    df_info = pd.read_csv(f, sep='\t')

    files = glob(os.path.join(ngs_home, 'FACS/*pL42*hist'))

    cols = ['sgRNA_name', 'sgRNA_design', 'subpool']
    get_sample = lambda x: re.findall('(T._...)', x)[0]
    df_bcs = (pd.concat(map(load_hist, files))
              .join(df_design.set_index('barcode')[cols], on='seq')
              .assign(mapped=lambda x: ~x['subpool'].isnull())
              .assign(sample=lambda x: x['file'].apply(get_sample))
              .join(df_info.set_index('sample'), on='sample')
              .query('subpool == "pool2_1"')
              .assign(sample_count=lambda x: 
                     x.groupby('sample')['count'].transform('sum'))
              .assign(fraction=lambda x: x.eval('count / sample_count'))         
              .assign(log10_fraction=lambda x: np.log10(x['fraction']))
              .rename(columns={'seq': 'barcode'})
             )
    
    
    df_summary = (df_bcs
     .groupby(['sample', 'sgRNA_design', 'gate'])
     ['fraction']
     .pipe(groupby_reduce_concat, 'sum', 'count')
     .reset_index()
    )

    df_stats = (df_bcs
     .query('FACS_replicate == 1')
     .groupby(['barcode', 'sgRNA_design', 'sgRNA_name', 'gate'])['count'].sum()
     .reset_index()
     .pivot_table(index=['barcode', 'sgRNA_design', 'sgRNA_name'], columns='gate', values='count')
     .fillna(fillna)
     .assign(enrichment=lambda x: np.log10(x.eval('positive / negative')))
     .assign(total=lambda x: x.eval('positive + negative'))
     .reset_index()
    )
    
    return df_stats, df_bcs, df_summary


# PLOTTING

def plot_gene_scatter_87(df_gene_stats, size=5):
    palette = GENE_CLASS_COLORS
    class_names = {'TNFa': TNFa_label, 'IL1b': IL1b_label}
    fix_class_names = lambda x: class_names.get(x, x)
    fg = (df_gene_stats
     .query('gene_symbol != "RELA"')
     .assign(TNFa=lambda x: 100 * x['TNFa'])
     .assign(IL1b=lambda x: 100 * x['IL1b'])
     .assign(gene_class=lambda x: x[GENE_CLASS].apply(fix_class_names))
     .pipe(sns.FacetGrid, hue=GENE_CLASS, palette=palette, size=size)
     .map(plt.scatter, 'TNFa', 'IL1b', s=40)
     .add_legend(title=None, ncol=2)
    )
    ax = fg.axes.flat[0]
    ax.set_xlim(100 * np.array([-0.02, 0.51]))
    ax.set_ylim(100 * np.array([-0.02, 0.51]))

    cols = ['TNFa', 'IL1b', GENE_SYMBOL, GENE_CLASS]
    for x,y,s,c in df_gene_stats[cols].values:
        if c != 'negative':
            ax.text(x=100*x,y=100*(y + 0.005),s=s, fontdict=dict(fontsize=11))
    
    fg._legend.set_title('Gene category')
    plt.setp(fg._legend.get_title(), fontsize=14)

 
    ax.set_xlabel(NFKB_TNFa_axis_label)
    ax.set_ylabel(NFKB_IL1b_axis_label)

    fg._legend.set_bbox_to_anchor((0, 0, 0.57, 2))

    return fg

def plot_sg_scatter_87(df_sg_stats, ax=None, flip=True):
    flip_TNFa = '((stimulant == "TNFa") * -2 + 1.) * mean'
    stimulant_labels = {'TNFa': TNFa_label, 'IL1b': IL1b_label}
    fix_stimulant_labels = lambda x: stimulant_labels.get(x, x)

    palette = STIMULANT_COLORS
    hue_order = IL1b_label, TNFa_label
    ax = (df_sg_stats
     .query('gene_class != "negative"')
     .sort_values([GENE_CLASS, 'mean'], ascending=[True, False])
     .assign(mean=lambda x: x.eval(flip_TNFa) if flip else x['mean'])
     .assign(stimulant=lambda x: x[STIMULANT].apply(fix_stimulant_labels))
     .pipe(lambda data: 
           sns.swarmplot(data=data, x=GENE_SYMBOL, y='mean', 
                         hue=STIMULANT, dodge=not flip, ax=ax,
                         palette=palette, hue_order=hue_order))
    )
    
    plt.sca(ax)
    if flip:
        plt.legend(title='Stimulant', bbox_to_anchor=(-1.6, 1), 
            loc=2, borderaxespad=0., ncol=2)

    else:
        plt.legend(title='Stimulant', bbox_to_anchor=(.45, 1), 
            loc=2, borderaxespad=0.)
    ax.set_xlabel('')
    ax.set_ylabel(NFKB_axis_label)
    plt.xticks(rotation=45)

    if flip:
        ax.set_ylim(100 * np.array([-0.9, 0.9]))
    else:
        ax.set_ylim(100 * np.array([-0.1, 0.9]))

    return ax

def plot_sg_scatter_87_combined(df_sg_stats, classes=('IL1b', 'TNFa', 'both'),
                                flip=False):
    classes = list(classes)

    fig, axs = plt.subplots(figsize=(12, 5))

    sizes = 1 + (df_sg_stats
         .drop_duplicates(GENE_SYMBOL)
         .groupby(GENE_CLASS)
         .size()[classes])

    gs = mpl.gridspec.GridSpec(1, sizes.sum())
    sizes_ = [0] + list(sizes.cumsum() - 1)

    for (i,c) in enumerate(classes):
        
        ax = plt.subplot(gs[:, sizes_[i]:sizes_[i + 1]])
        (df_sg_stats
            .assign(mean=lambda x: 100 * x['mean'])
            .query('gene_class == @c')
            .pipe(plot_sg_scatter_87, ax=ax, flip=flip)
        )
        if i > 0:
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False)
        if i < 2:
            ax.legend_.remove()
        else:
            plt.setp(ax.legend_.get_title(),fontsize=14)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        xlabel_pad = {'both': 10, 'IL1b': 18, 'TNFa': 0}
        ax.set_xlabel(NFKB_regulator_labels[c], color=GENE_CLASS_COLORS[i], fontweight='bold')
        ax.xaxis.labelpad = xlabel_pad[c]
        if flip:
            yticks = sorted(set(list(NFKB_yticks) + [-1 * t for t in NFKB_yticks]))
            plt.yticks(yticks, map(abs, yticks))
        else:
            ax.set_yticks(NFKB_yticks)

        ax.spines['left'].set_position(('outward', 8))
        
        

    return fig

def plot_sg_scatter_vs_cells(df_sg_stats): 
    def category_map(x, f):
        return x.cat.reorder_categories(f(x.cat.categories))

    def reverse_categories(series):
        return category_map(series, lambda x: x[::-1])

    palette = GENE_CLASS_COLORS[::-1]
    label_order = df_sg_stats[GENE_CLASS].cat.categories
    fg = (df_sg_stats
        .assign(mean=lambda x: 100 * x['mean'])
        .assign(gene_class=lambda x: reverse_categories(x[GENE_CLASS]))
        # .assign(log_count=lambda x: np.log10(x['count']))
        .pipe(sns.FacetGrid, size=6, hue=GENE_CLASS, palette=palette)
        .map(plt.scatter, 'count', 'mean')
        .add_legend(title='Gene category', label_order=label_order)
    )
    ax = fg.axes.flat[0]
    ax.set_ylim(100 * np.array([-0.02, 0.81]))
    ax.set_xlabel('cells (log10)')
    ax.set_ylabel(NFKB_axis_label)
    ax.set_xscale('log')
    ax.set_xlim([25, 1200])
    ax.set_yticks(NFKB_yticks)

    return fg

def plot_pathway_map(stimulant, svg_filename):
    """Top level function to generate pathway svg.
    """
    pathways = ['IL-1b', 'IL-1b_simple', 'TNFa', 'TNFa_simple']

    cols = ['gene_0', 'gene_1', 'interaction', 'pathway']
    df_raw = pd.read_csv('NFKB/pathway/Lasagna-Nougat - p65_pathway.csv')
    df = df_raw[cols].dropna(axis=0).query('pathway == @pathways')
    df['gene_0_symbol'] = df['gene_0'].apply(lambda x: x.split(',')[0])
    df['gene_1_symbol'] = df['gene_1'].apply(lambda x: x.split(',')[0])
    G = nx.from_pandas_edgelist(df, source='gene_0_symbol', 
                                target='gene_1_symbol', 
                                edge_attr='interaction', create_using=nx.DiGraph())

    set1_genes = [x.split(',')[0] for x in df_raw['set1'].dropna()]
    G2 = nx.DiGraph(G.subgraph(set1_genes))
    nx.write_graphml(G2, 'set1_v5.graphml')

    # load hits

    # TODO: single source of gene stats
    df_stats = (pd.read_csv('/Users/feldman/lasagna/20180402_6W-G161/gene_stats.csv'))
    duplicate_symbols = ('CHUK', 'IKBKA'),
    for s1, s2 in duplicate_symbols:
        row = df_stats.query('gene == @s1').copy()
        row['gene'] = s2
        df_stats = pd.concat([df_stats, row])

    # extract gene positions from cyjs
    f = 'NFKB/pathway/set1_v7.cyjs'
    positions = load_cyjs_positions(f)
    positions['p65'] = positions['p65/p50']

    G3 = color_graph(G2, positions, df_stats, stimulant)

    G3.draw(svg_filename, prog='neato', args='-n2')

    return G3

def luminance(rgb):
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])

def color_graph(G, positions, df_stats, stimulant):
    edge_labels = nx.get_edge_attributes(G, 'interaction')
    A = nx.nx_agraph.to_agraph(G)
    for node in A.nodes():
        node.attr['pos'] = '%f,%f)'%positions[node]
        node.attr['color'] = 'black'
        node.attr['style'] = 'filled'
        if node == 'nucleus':
            node.attr['fillcolor'] = 'white'
        elif 'p65' in node:
            node.attr['fillcolor'] = LIGHT_GREEN
        else:
            node.attr['fillcolor'] = PathwayColorMap().color_node(df_stats, node, stimulant)            

        if luminance(sns.color_palette([node.attr['fillcolor']])[0]) < 0.7:
            node.attr['fontcolor'] = 'white'


        node.attr['shape'] = 'box'
        node.attr['fontname'] = 'helvetica'
        node.attr['height'] = 0.05
        node.attr['width'] = 0.4
        

    for edge in A.edges():
        label = edge_labels[edge]
        edge.attr['color'] = PathwayColorMap.edge_colors.get(label, 'gray')
        if label == 'T':
            edge.attr['style'] = 'dashed'
        elif label == 'B':
            edge.attr['style'] = 'invis'
        else:
            edge.attr['style'] = 'solid'
    return A

def load_cyjs_positions(filename):
    with open(filename, 'r') as fh:
        js = json.load(fh)
    positions = {}
    for node in js['elements']['nodes']:
        name = node['data']['name']
        x, y = node['position']['x'], node['position']['y']
        positions[name] = (x, -1*y)
        
    for k in positions.keys():
        positions[k.split(',')[0]] = positions[k]
    return positions

def update_cyjs_positions(filename, positions):
    with open(filename, 'r') as fh:
        js = json.load(fh)
    for node in js['elements']['nodes']:
        name = node['data']['name']
        if name in positions:
            node['position']['x'] = positions[name][0]
            node['position']['y'] = -1 * positions[name][1]
    
    with open(filename, 'w') as fh:
        json.dump(js, fh)

class PathwayColorMap(object):
    colors = sns.color_palette('Blues', 3).as_hex() 
    colors = [GRAY] + colors
    thresholds = 0, 0.05, 0.1, 0.2
    edge_key = {'B': 'binding'
                    ,'D': 'degradation'
                    ,'P': 'phosphorylation'
                    ,'A': 'activation'
                    ,'U': 'ubiquitination'
                    ,'T': 'translocation'}
    edge_colors = OrderedDict([('P', 'red'), 
                               ('U', 'darkgreen'), 
                               ('A', 'purple'),
                               ('T', 'black')])
    
    def color_node(self, df_stats, gene, stimulant):
        try:
            x = df_stats.query('gene == @gene')[stimulant].iloc[0]
        except Exception as e:
            return 'white'
        for threshold, b in zip(self.thresholds, self.colors)[::-1]:
            if x > threshold:
                return b
        raise ValueError
    
    def node_legend(self):
        handles = []
        for color, threshold in zip(self.colors, self.thresholds):
            label = '> %d%% positive' % int(100 * threshold)
            patch = mpl.patches.Patch(color=color, label=label)
            handles += [patch]
        return handles
    
    def edge_legend(self):
        handles = []
        for letter, color in self.edge_colors.items():
            label = self.edge_key[letter]
            patch = mpl.patches.Patch(color=color, label=label)
            handles += [patch]
        return handles

def save_svg(filename, svg):
    svg.save(filename)
    with open(filename, 'r') as fh:
        txt = fh.read()

    # txt = txt.replace('<svg', '<svg fill="none"')

    with open(filename, 'w') as fh:
        fh.write(txt)

def show_svg(x):
    import os
    import IPython.display
    save_svg('tmp.svg', x)
    y = IPython.display.SVG('tmp.svg')
    os.remove('tmp.svg')
    return y

def plot_image_grid(images, x_labels, y_labels, scale=2.5):
    """images has shape (rows, columns, I, J, RGB)
    """
    assert images.ndim == 5
    h, w = images.shape[2:4]
    h_grid, w_grid = images.shape[:2]
    figsize = scale * np.array([float(w)/h * w_grid,
                             1 * h_grid, ])


    fig, axs = plt.subplots(nrows=h_grid, ncols=w_grid,
                           figsize=figsize)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.05, hspace=0.05)

    if h == 1 & w == 1:
        axs = np.array([axs])
    axs = axs.reshape(h_grid, w_grid)

    for i in range(h_grid):
        for j in range(w_grid):
            axs[i,j].imshow(images[i,j])
            axs[i,j].xaxis.set_ticks([])
            axs[i,j].yaxis.set_ticks([])        

    for i in range(h_grid):
        axs[i, 0].set_ylabel(y_labels[i])

    for j in range(w_grid):
        axs[0, j].set_xlabel(x_labels[j])
        axs[0, j].xaxis.set_label_position('top') 

    return fig

def get_20180325_image_files():
    files = glob('paper/NFKB_figure/20180325/*.crop.jpg')
    pat = 'DAPI-mNeon_(.*?)_(.*?)_(.*?).stitched'
    get_conditions = lambda x: re.findall(pat, x)[0]
    cols = 'well', 'KO', 'stimulation'
    df_files = pd.DataFrame(map(get_conditions, files), 
                 columns=cols).assign(file=files)

    stimulation = ['no-stimulation', 'IL1b-40min', 'TNFa-40min']
    KO = ['wt', 'IL1R1']

    df_files['stimulation'] = (df_files['stimulation']
                               .astype('category', categories=stimulation))
    df_files['KO'] = (df_files['KO']
                               .astype('category', categories=KO))
    df_files = df_files.sort_values(['KO', 'stimulation'])
    
    return df_files['file'].tolist()

def load_20180325_images(width):

    images = [img[:width, :width] 
              for img in map(imageio.imread, get_20180325_image_files())]

    images = lasagna.utils.pile(images).reshape(2, 3, width, width, 3)
    return images

def plot_NFKB_image_grid(images):

    x_labels = 'no stimulation', u'IL1\u03b2', u'TNF\u03b1',
    y_labels = 'non-targeting', 'IL1R1', 'TNFSF1A', 'MAP3K7'
        
    fig = plot_image_grid(images, x_labels, y_labels, scale=1.8)

    plt.figtext(0, 0.55, 'sgRNA', rotation=90, size=16)

    ax2 = plt.axes([0,0,1,1], axisbg=(1,1,1,0))
    ax2.axis('off')
    line = mpl.lines.Line2D([0.05, 0.05], 
                            [0.25, 0.75], lw=1, color='black')
    ax2.add_line(line)

    return fig

def plot_phenotype_distribution(df_cells, vertical=True):

    df_cells = df_cells.copy()

    positive_gene = 'MAP3K7'
    df_cells['status'] = 'all cells'
    df_cells.loc[df_cells[GENE_SYMBOL] == positive_gene, 'status'] = positive_gene

    bins = np.linspace(-1, 1, 30)
    palette = STRONG_GRAY, RED

    order = ['IL1b', 'TNFa']
    if vertical:
        kwargs = {'row': 'stimulant', 'row_order': order, 'size': 2.8}
    else:
        kwargs = {'col': 'stimulant', 'col_order': order, 'size': 2.5}

    fg = (df_cells
     .rename(columns=NFKB_phenotype_rename)
     .pipe(sns.FacetGrid, hue='status', 
           palette=palette, sharey=False, **kwargs)
     .map(plt.hist, NFKB_phenotype_rename.values()[0],
          normed=True, cumulative=True, bins=bins, 
          lw=2, histtype='step', alpha=0.85)
     )

    ax0, ax1 = fg.axes.flat[:]

    if vertical:
        ax0.set_ylabel('cumulative fraction')
        ax1.set_ylabel('cumulative fraction')
        # ax0.yaxis.set_label_coords(-0.2, -0.2)
    else:
        ax1.yaxis.set_visible(False)
        ax1.set_xlabel('')
        ax0.xaxis.set_label_coords(1.3, -0.25)

    for i, ax in enumerate(fg.axes.flat[:]):
        ax.set_yticks([])
        ax.set_xlim([-1, 0.95])
        ax.set_xticks([-1, 0, 0.95])
        ax.set_xticklabels([-1, 0, 1])
        ax.margins(y=0.1)
        ax.autoscale(enable=True, axis='y', tight=False)
        a,b = ax.get_ylim()
        ax.set_ylim([0, b])
        rect = mpl.patches.Rectangle((-1, 0), 0.5, 10, zorder=-10,
                                    color=VERY_LIGHT_GREEN)
        ax.add_patch(rect)


    fg.axes.flat[0].text(0.55, 0.15, 'all cells', size=14, color=DARK_GRAY)
    fg.axes.flat[0].text(-0.2, 0.7, 'MAP3K7', size=14, color=RED)
    fg.axes.flat[1].text(0.3, 0.2, 'all cells', size=14, color=DARK_GRAY)
    fg.axes.flat[1].text(-0.4, 0.68, 'MAP3K7', size=14, color=RED)

    fg.axes.flat[0].set_title(IL1b_label)
    fg.axes.flat[1].set_title(TNFa_label)

    return fg

def load_SBS_images(expand=10):
    files = sorted(glob('paper/data/G135B/F2_1.log-60*.png'))

    images = []
    for f in files:
        images += [np.kron(imageio.imread(f), 
                           np.ones((expand, expand, 1)))]
    images = np.array(images)/255
    h, w, _ = images.shape[-3:]
    return images

def plot_grid_triangles(fig, coords, colors, triangle_size=150, 
                        triangle_angle = 0.25 * 3.14, number=False):
    theta = triangle_angle
    M = np.matrix([[np.cos(theta), np.sin(theta)],
                  [np.sin(theta), -np.cos(theta)]])
    triangle = np.array([[0, 0], [1, 0], [0.5, 1]])
    triangle = triangle * M
    triangle = triangle_size * triangle 

    number_offset = np.array([0, -0.6 * triangle_size])

    triangles = []
    for i, (coord, color) in enumerate(zip(coords, colors)):
        for ax in fig.axes:
            patch = mpl.patches.Polygon(triangle + coord, 
                       closed=True, color=color)
            ax.add_patch(patch)
            if number:
                text = ax.annotate(str(i + 1), coord + number_offset, 
                    color=color, fontsize=RCP['xtick.labelsize'], fontweight='bold')

                text.set_path_effects([mpl.patheffects.Stroke(linewidth=4,
                                                   foreground='black'),
                               mpl.patheffects.Normal()])

def plot_SBS_image_grid():
    images = load_SBS_images()
    h, w = images.shape[-3], images.shape[-2]
    images = images.reshape(3, 4, h, w, 3)

    x_labels = '', '', '', ''
    y_labels = '', '', ''
    fig = plot_image_grid(images, x_labels, y_labels, scale=2.1)

    coords = ((80, 320),
              (330, 630))
    plot_grid_triangles(fig, coords, colors=('white', 'white'), number=True)

    plot_annotate_cycles(fig)

    return fig

def plot_annotate_cycles(fig):
    char_width = 0.09

    for i, ax in enumerate(fig.axes):
        label = 'cycle %d' % (i + 1)
        width = len(label) * char_width

        x0, y0 = 1 - width, 0
        dx, dy = width, 0.2

        rect_coords = [(x0, y0), (x0 + dx, y0),
                       (x0 + dx, y0 + dy), (x0, y0 + dy)]

        ax.text(1.05 - width, 0.15, label, transform=ax.transAxes,
          fontsize=16, fontweight='bold', color='white', va='top',
          fontname='sans-serif')

        rect = mpl.patches.Polygon(rect_coords, 
                       closed=True, color='black', transform=ax.transAxes)

        ax.add_patch(rect)

def plot_colored_nucleotides(sequence):
    width = 0.023
    fig, ax = plt.subplots()
    for i, c in enumerate(sequence):
        ax.text(i*width, 0.95, c, color=nucleotide_colors[c],
               fontsize=14, fontname='consolas')
    ax.axis('off')
    return fig

def plot_primed_DNA(lw=20, color=DARK_GRAY):
    fig, ax = plt.subplots()
    ax.plot([0.2, 1], [0.3, 0.3], color, lw=lw)
    ax.plot([0.2, 1], [0.5, 0.5], color, lw=lw)
    ax.plot([1, 0.75], [0.5, 0.85], color, lw=lw)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    
    ax.axis('off')
    return fig

def plot_quality_vs_mapping_rate(df_reads, downsample=1e5):
    """
    """
    def calc_quality_vs_mapping_rate(df_reads):
        arr = []
        num_reads = len(df_reads)
        for th in range(31):
            q = df_reads.query('Q_min >= @th')
            mapped = (q['subpool'] != 'unmapped')
            arr += [[th, mapped.mean(), mapped.sum(), 
                     len(q) / float(num_reads) ]]

        return np.array(arr)

    num_reads = int(min(downsample, len(df_reads)))
    df_reads_ = df_reads.sample(num_reads)
    arr = calc_quality_vs_mapping_rate(df_reads_)
    
    fig, ax0 = plt.subplots(figsize=(5,3.5))
    ax1 = ax0.twinx()
    max_q = np.where(arr[:, 2] < 1000)[0][0]
    labels = 'mapping rate (%)', 'reads above threshold (%)'

    left_color = BLUE
    right_color = DARK_GREEN
    line0 = ax0.plot(arr[:max_q,0], arr[:max_q, 1] * 100. * .92, label=labels[0], color=left_color)
    line1 = ax1.plot(arr[:,0], arr[:, 3] * 100., label=labels[1], color=right_color)
    # ax0.legend(line0 + line1, labels, loc='lower center')
    
    ax0.set_xlabel('read quality threshold')
    ax0.set_ylabel(labels[0], labelpad=0)
    
    # ax0.set_title('max mapping rate: %.2f%%' % (100 * np.max(arr[:max_q, 1])))
    
    ax1.set_ylabel(labels[1])
    # ax1.set_ylim([0, 100])
    # ax1.yaxis.set_visible(False)

    fig.tight_layout()


    ax0.yaxis.label.set_color(left_color)
    ax1.yaxis.label.set_color(right_color)


    return fig

def plot_annotate_FR(fig):
    ax0, ax1, ax2, ax3 = fig.axes
    
    HA_width = 0.55
    label = 'HA-488'
    color = IMAGEJ_GREEN
    ax0.text(1.05 - HA_width, 0.15, label, transform=ax0.transAxes,
          fontsize=16, fontweight='bold', color=color, va='top',
          fontname='sans-serif')
    
    DAPI_width = 0.4
    label = 'DAPI'
    color = IMAGEJ_BLUE
    ax1.text(1.05 - DAPI_width, 0.15, label, transform=ax1.transAxes,
          fontsize=16, fontweight='bold', color=color, va='top',
          fontname='sans-serif')
    
    
    label = 'DAPI'
    color = IMAGEJ_BLUE
    ax2.text(1.05 - DAPI_width, 0.3, label, transform=ax2.transAxes,
          fontsize=16, fontweight='bold', color=color, va='top',
          fontname='sans-serif')
    
    
    HA_width = 0.55
    label = 'HA-488'
    color = IMAGEJ_GREEN
    ax2.text(1.05 - HA_width, 0.15, label, transform=ax2.transAxes,
          fontsize=16, fontweight='bold', color=color, va='top',
          fontname='sans-serif')
    
    label = 'DAPI'
    color = IMAGEJ_BLUE
    ax3.text(1.05 - DAPI_width, 0.3, label, transform=ax3.transAxes,
          fontsize=16, fontweight='bold', color=color, va='top',
          fontname='sans-serif')
    
    label = 'cycle 1'
    color = 'white'
    cycle_width = 0.53
    ax3.text(1.05 - cycle_width, 0.15, label, transform=ax3.transAxes,
          fontsize=16, fontweight='bold', color=color, va='top',
          fontname='sans-serif')

def plot_image_grid_FR():
    files = glob('paper/FR_figure/G167C/[1234]_*png')

    i0, i1 = 10, 110
    j0, j1 = 20, 120
    images = np.array(map(imageio.imread, files))
    images = (images[:, i0:i1, j0:j1]
     .reshape(2, 2, i1 - i0, j1 - j0, 3)
     [:, :, ::-1]
             )

    x_labels = '', ''
    y_labels = '', ''
    fig = plot_image_grid(images, x_labels, y_labels)
    plot_annotate_FR(fig)

    return fig

def plot_dummy_validation_vertical():

    f_dummy = 'paper/NFKB_figure/20180325/20X_DAPI-mNeon_A4_wt_TNFa-40min.stitched.crop.jpg'

    genes = ['MYD88', 'TRAF6', 'IRAK1', 'IRAK2',
             'TNFRSF1A', 'TRADD']

    fakedata_x  = [0, 1, 2, 3]
    fakedata_y  = np.array([0, 0.1, 0.8, 1]) * 100.
    fakedata_y_ = np.array([0, 0.05, 0.4, 0.6]) * 100.


    fig, axs = plt.subplots(nrows=len(genes), ncols=3,
                            figsize=(6,10))

    plt.subplots_adjust(right=0.85)

    for i, gene in enumerate(genes):
        img = imageio.imread(f_dummy)[:200, :200]
        axs[i, 0].imshow(img)
        axs[i, 0].xaxis.set_visible(False)
        axs[i, 0].set_ylabel(gene, fontsize=12)
        axs[i, 0].set_yticks([])

        axs[i, 1].plot(fakedata_x, fakedata_y, color=STIMULANT_COLORS[0])
        axs[i, 1].plot(fakedata_x, fakedata_y_, color=STIMULANT_COLORS[1])

        axs[i, 1].yaxis.set_visible(False)
        if i < (len(genes) - 1):
            axs[i, 1].xaxis.set_visible(False)

        axs[i, 2].plot(fakedata_x, fakedata_y, color=STIMULANT_COLORS[0])
        axs[i, 2].plot(fakedata_x, fakedata_y_, color=STIMULANT_COLORS[1])

        if i < (len(genes) - 1):
            axs[i, 2].xaxis.set_visible(False)

        axs[i, 2].yaxis.tick_right()
        axs[i, 2].set_yticks([10, 50, 90])
        axs[i, 2].set_yticklabels([0, 50, 100], fontsize=12)

    axs[-1, 1].set_xticks([0, .8, 1.7, 2.6])
    axs[-1, 1].set_xticklabels([0, 30, 300, 3000], rotation=40, fontsize=12)
    axs[-1, 1].tick_params(axis='x', which='major', pad=2)

    axs[-1, 2].set_xticks([0, .8, 1.7, 2.6])
    axs[-1, 2].set_xticklabels([0, 20, 40, 360], rotation=40, fontsize=12)    
    axs[-1, 2].tick_params(axis='x', which='major', pad=2)

    axs[-1, 1].set_xlabel('stimulant \nconcentration (ng/uL)', fontsize=12, labelpad=0)
    axs[-1, 2].set_xlabel('time (min)', fontsize=12)

    ax2 = plt.axes([0,0,1,1], axisbg=(1,1,1,0))
    ax2.axis('off')
    ax2.annotate('cells translocated (%)', (0.95, 0.6), ha='center', fontsize=14, rotation=90)

    return fig

def plot_dummy_validation_horizontal():
    f_dummy = 'paper/NFKB_figure/20180325/20X_DAPI-mNeon_A4_wt_TNFa-40min.stitched.crop.jpg'

    genes = ['IL1R1', 'MYD88', 'IRAK1', 'IRAK4', 'TRAF6',
             'TNFRSF1A', 'TRADD', 'TRAF2', 'RIPK1',
             'MAP3K7', 'NFKBIA']

    fakedata_x  = [0, 1, 2, 3]
    fakedata_y  = np.array([0, 0.1, 0.8, 1]) * 100.
    fakedata_y_ = np.array([0, 0.05, 0.4, 0.6]) * 100.

    
    fig, axs = plt.subplots(nrows=1, ncols=len(genes), 
                            figsize=(16,6), sharey=False)

    for i, gene in enumerate(genes):
        img = imageio.imread(f_dummy)
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(gene, fontsize=12)


    fig2, axs = plt.subplots(nrows=2, ncols=len(genes), 
                            figsize=(16,4), sharey=False)
    plt.subplots_adjust(hspace=0.8, bottom=0.17)

    for i, gene in enumerate(genes):
        axs[0, i].plot(fakedata_x, fakedata_y, color=STIMULANT_COLORS[0])
        axs[0, i].plot(fakedata_x, fakedata_y_, color=STIMULANT_COLORS[1]) 

        if i > 0:
            axs[0, i].yaxis.set_visible(False)

        axs[0, i].set_xticks([0, .8, 1.7, 2.6])
        axs[0, i].set_xticklabels([0, 20, 40, 360], rotation=40)    
        axs[0, i].tick_params(axis='x', which='major', pad=2)

        axs[1, i].plot(fakedata_x, fakedata_y, color=STIMULANT_COLORS[0])
        axs[1, i].plot(fakedata_x, fakedata_y_, color=STIMULANT_COLORS[1])

        if i > 0:
            axs[1, i].yaxis.set_visible(False)

        axs[1, i].set_xticks([0, .8, 1.7, 2.6])
        axs[1, i].set_xticklabels([0, 30, 300, 3000], rotation=40)
        axs[1, i].tick_params(axis='x', which='major', pad=2)

    for ax in axs[:, 0]:
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel('translocated \ncells (%)', fontsize=12, labelpad=-5)

    for ax in axs.flat[:]:
        ax.tick_params(axis='both', which='major', labelsize=12, pad=5)


    ax2 = plt.axes([0,0,1,1], axisbg=(1,1,1,0))
    ax2.axis('off')
    ax2.annotate('stimulant concentration (ng/uL)', (0.5, 0), ha='center', fontsize=14)
    ax2.annotate('time (min)', (0.5, 0.49), ha='center', fontsize=14)
    
    return fig, fig2

def letter_case(c):
    return c.upper()

def plot_base_quality_per_cycle(df_reads, log_y, num_cycles=12, 
                            figsize=(4, 3.2), per_col=4):
    
    fig, axs = plt.subplots(nrows=min(num_cycles, per_col), 
                            ncols=num_cycles/per_col, 
                            sharex=True, 
                            figsize=figsize)

    plt.subplots_adjust(wspace=0.4, bottom=0.15, top=1)
    bins = range(0, 33, 3)
    palette = sns.color_palette('husl', int(num_cycles * 1.2))
    for i, ax in enumerate(axs.T.flat[:]):
        ax.set_yticks([])
        ax.yaxis.set_label_coords(-0.15, 0.2) 
        ax.set_ylabel(i + 1, rotation=0, fontsize=RCP['ytick.labelsize'])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        col = 'Q_{0:02d}'.format(i)
    #     df_reads[col].pipe(ax.hist, bins=bins, color=palette[i])
        df_reads[col].pipe(sns.distplot, ax=ax, kde=False, hist=True, bins=bins, color=palette[i])    
        ax.set_xlabel('')
        if log_y:
            ax.set_yscale('log')
            ax.set_yticks([])

    for ax in axs[-1, :]:
        ax.set_xticks([0, 15, 30])

    ax2 = plt.axes([0,0,1,1], axisbg=(1,1,1,0))
    ax2.axis('off')
    ax2.annotate('base quality', (0.5, 0), ha='center', fontsize=RCP['axes.labelsize'])
    ax2.annotate('cycle', (0.04, 0.5), ha='center', fontsize=RCP['axes.labelsize'], rotation=90)

    return fig

def plot_read_quality_mapped_histogram(df_reads, size=3.5):
    palette = STRONG_GRAY, RED
    bins = range(0, 33, 3)
    fg = (df_reads
     .pipe(sns.FacetGrid, hue='mapped', size=size, palette=palette)
     .map(plt.hist, 'Q_min', bins=bins, histtype='step', lw=2)
    )

    fg.ax.set_yticks([])
    fg.ax.set_xticks([0, 10, 20, 30])
    fg.ax.set_xlim([0, 30.5])
    fg.ax.set_xlabel('read quality')
    fg.ax.set_ylabel('reads')

    unmapped = mpl.lines.Line2D([], [], color=palette[0], marker='s', 
                    linestyle='None', markersize=10, label='unmapped')

    mapped = mpl.lines.Line2D([], [], color=palette[1], marker='s', 
                    linestyle='None', markersize=10, label='mapped')

    plt.legend(handles=[mapped, unmapped], loc=(0.16, 0.75))    
    
    return fg

def plot_reads_per_cell(df_reads, figsize=(4, 3)):
    counts = df_reads.groupby(['well', 'tile', 'cell']).size().value_counts()
    num_cells = df_reads.groupby(['well', 'tile'])['cell'].max().sum()

    ys = [num_cells - counts.sum()]
    bins = ['0']
    last_bin = 5
    for i in range(1, last_bin):
        ys += [counts.loc[i]]
        bins += [str(i)]

    ys += [counts.loc[last_bin:].sum()]
    bins += ['{0}+'.format(last_bin)]

    df = pd.DataFrame({'count': ys, 'reads per cell': bins}).assign()
    df['count'] = 100 * (df['count'] / df['count'].sum())

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x='reads per cell', y='count', 
                ax=ax, palette='Greens')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('cells (%)')
    return fig

def plot_FR_scatter_barcodes(df_cells, count_threshold=10, figsize=(4,3)):

    size=figsize[0]
    aspect = figsize[0] / float(figsize[1])
    palette = DARK_GREEN, BLUE
    fg = (df_cells
     .groupby(['cell_barcode_0', 'sgRNA_design'])['FR_pos']
     .pipe(groupby_reduce_concat, 'mean', 'count')
     .query('count > @count_threshold')
     .assign(mean_pct=lambda x: x['mean'] * 100)
     .pipe(sns.FacetGrid, hue='sgRNA_design', 
           size=size, aspect=aspect,
          palette=palette)
     .map(plt.scatter, 'count', 'mean_pct'))

    fg.ax.set_yticks([0, 25, 50, 75, 100])
    fg.ax.set_xlim([count_threshold, 1200])
    fg.ax.set_ylim([-5, 100])
    fg.ax.set_xscale('log')
    fg.ax.set_ylabel('in situ \nHA+ cells (%)',
                    labelpad=0)
    fg.ax.set_xlabel('in situ, cells per barcode')

    targeting = mpl.lines.Line2D([], [], color=palette[0], marker='s', 
                    linestyle='None', markersize=10, label='targeting')

    control = mpl.lines.Line2D([], [], color=palette[1], marker='s', 
                    linestyle='None', markersize=10, label='control')

    fg.ax.legend(handles=[targeting, control], ncol=2, loc=(0.15, 0.95))

    
    return fg

def calc_sgRNA_stats(df_cells, count_threshold=30):
    return (df_cells
     .groupby(['cell_barcode_0', 'sgRNA_name', 'sgRNA_design'])['FR_pos']
     .pipe(groupby_reduce_concat, 'mean', 'count')
     .query('count > @count_threshold')
     .assign(mean_pct=lambda x: x['mean'] * 100))

def plot_FR_scatter_sgRNAs(df_cells, figsize=(5.5,2.5)):
    size=figsize[0]
    aspect = figsize[0] / float(figsize[1])

    palette = DARK_GREEN, BLUE
    order = sorted(df_cells['sgRNA_name'].value_counts().index)
    fg = (df_cells
        .pipe(calc_sgRNA_stats)
        .pipe(sns.FacetGrid,
            size=size, aspect=aspect,
            palette=palette)
        .map_dataframe(lambda data, color:
            sns.swarmplot(data=data, x='sgRNA_name', y='mean_pct',
                    order=order))
         )


    colors, labels = [], []
    for i, tick in enumerate(fg.ax.xaxis.get_ticklabels()):
        if 'LG' in tick.get_text():
            color = DARK_GREEN
        else:
            color = BLUE

        label = 'sgRNA {0}'.format(i + 1)
        colors += [color]
        labels += [label]

    fg.ax.set_xticks(np.arange(-0.3, 9.7))
    fg.ax.set_xticklabels(labels, rotation=30)
    [t.set_color(c) for c, t in zip(colors, fg.ax.xaxis.get_ticklabels())]

    fg.ax.set_yticks([0, 25, 50, 75, 100])
    fg.ax.set_ylim([-5, 100])
    fg.ax.set_ylabel('HA-488-positive cells \nper barcode (%)',
                    labelpad=0)

    ax2 = plt.axes([0,-0.1,1,1], axisbg=(1,1,1,0))
    ax2.axis('off')
    ax2.text(0.2, 0, 'targeting', fontsize=16, color=DARK_GREEN)
    ax2.text(0.65, 0, 'control', fontsize=16, color=BLUE)
    
    return fg

def plot_FR_scatter_sgRNAs_insitu_NGS(df_cells, df_stats_ngs, figsize=(4,4)):

    palette = DARK_GREEN, BLUE
    order = sorted(df_cells['sgRNA_name'].value_counts().index)
    
    df_insitu_stats = df_cells.pipe(calc_sgRNA_stats)

    df_stats_ngs = df_stats_ngs.assign(ratio=lambda x: 10**x['enrichment'])


    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=figsize)

    sns.boxplot(ax=ax0, data=df_insitu_stats, 
        x='sgRNA_name', y='mean_pct', hue='sgRNA_design', dodge=False,
        order=order, hue_order=FR_design_order, palette=palette)

    sns.boxplot(ax=ax1, data=df_stats_ngs, 
        x='sgRNA_name', y='ratio', hue='sgRNA_design', dodge=False,
        order=order, hue_order=FR_design_order, palette=palette)

    for ax in (ax0, ax1):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('')
        ax.set_xticklabels([])

    # y axis
    ax0.set_ylabel('in situ \nHA+ cells (%)')
    ax0.set_yticks([0, 50, 100])
    ax1.set_ylabel('FACS \nHA enrichment    ')
    ax1.set_yscale('log')
    ax1.set_ylim([10**-1.7, 10**1.7])

    ax0.legend_.remove()
    ax1.legend_.remove()

    # legend
    targeting = mpl.lines.Line2D([], [], color=palette[0], marker='s', 
                    linestyle='None', markersize=10, label='targeting')

    control = mpl.lines.Line2D([], [], color=palette[1], marker='s', 
                    linestyle='None', markersize=10, label='control')

    ax0.legend(handles=[targeting, control], loc=(0.52, 0.7))

    # x axis
    # ax2 = plt.axes([0,0,1,1], axisbg=(1,1,1,0))
    # ax2.axis('off')
    # ax2.text(0.31, 0.04, 'targeting sgRNAs', color=DARK_GREEN,
    #     fontsize=16, ha='center')
    # ax2.text(0.72, 0.04, 'control sgRNAs', color=BLUE,
    #      fontsize=16, ha='center')

    ax1.set_xlabel('sgRNA')

    return fig


    fg = (df
     .pipe(sns.FacetGrid,
           size=size, aspect=aspect, row='method',
           palette=palette)
     .map_dataframe(lambda data, color:
        sns.swarmplot(data=data, x='sgRNA_name', y='mean_pct',
                     order=order))
         )


    colors, labels = [], []
    for i, tick in enumerate(fg.ax.xaxis.get_ticklabels()):
        if 'LG' in tick.get_text():
            color = DARK_GREEN
        else:
            color = BLUE

        label = 'sgRNA {0}'.format(i + 1)
        colors += [color]
        labels += [label]

    fg.ax.set_xticks(np.arange(-0.3, 9.7))
    fg.ax.set_xticklabels(labels, rotation=30)
    [t.set_color(c) for c, t in zip(colors, fg.ax.xaxis.get_ticklabels())]

    fg.ax.set_yticks([0, 25, 50, 75, 100])
    fg.ax.set_ylim([-5, 100])
    fg.ax.set_ylabel('HA-488-positive cells \nper barcode (%)',
                    labelpad=0)


    
    return fg

def plot_abundance_NGS_vs_in_situ(df_cells, size=5):

    def load_hist(f):
        try:
            return (pd.read_csv(f, sep='\s+', header=None)
                .rename(columns={0: 'count', 1: 'seq'})
                .query('count > 3')
                .assign(fraction=lambda x: x['count']/x['count'].sum())
                .assign(log10_fraction=lambda x: np.log10(x['fraction']))
                .assign(file=f)
               )
        except:
            return None

    # load data from 20180418_AS_RJC_DF_AJG
    well = 'B1'
    cell_line = 'cLas50.1'

    df_cell_counts = (df_cells
     .query('subpool == "pool2_1"')
     .query('well == @well')
     .groupby(['barcode', 'subpool']).size()
     .rename('insitu_counts')
     .reset_index()
    )

    files = glob('/Users/feldman/lasagna/NGS/samples/cLas50.?.rep?.pL42.hist')

    get_cell_line = lambda x: re.findall('(cLas50..)', x)[0]
    get_rep = lambda x: re.findall('(rep.)', x)[0]

    df_ngs = (pd.concat(map(load_hist, files))
     .assign(cell_line=lambda x: x['file'].apply(get_cell_line))
     .assign(NGS_replicate=lambda x: x['file'].apply(get_rep))
     .rename(columns={'seq': 'barcode'})
     )

    df_ngs_cLas501 = (df_ngs
     .query('cell_line == @cell_line')
     .groupby(['barcode']) # average replicates
     ['fraction'].mean())

    df = (df_cell_counts
     .join(df_ngs_cLas501, on='barcode')
     .assign(in_situ=lambda x: x['insitu_counts'] / x['insitu_counts'].sum())
     .assign(NGS=lambda x: x['fraction'] / x['fraction'].sum())
     .assign(in_situ_log=lambda x: np.log10(x['in_situ']))
     .assign(NGS_log=lambda x: np.log10(x['NGS']))
     .dropna()
    )

    bins = np.logspace(-5, -2, 15)
    jg = sns.jointplot(data=df, x='NGS', y='in_situ', size=size,
                       marginal_kws=dict(bins=bins), stat_func=None,
                       color='black', joint_kws=dict(s=5))
    jg.ax_joint.set_xscale('log')
    jg.ax_joint.set_yscale('log')
    jg.ax_joint.set_xlim([3e-5, 1e-2])
    jg.ax_joint.set_ylim([3e-5, 1e-2])
    jg.ax_joint.set_xlabel('barcode abundance, NGS')
    jg.ax_joint.set_ylabel('barcode abundance, in situ')
    
    print df.corr()

    return jg

def plot_FR_insitu_FACS_phenotype_hist(df_cells, figsize=(4, 4)):
    
    import FlowCytometryTools as FCT

    df_facs = FCT.FCMeasurement('', datafile=f_FR_cLas501_facs).data

    cols =[u'FSC-A', u'FSC-H', u'FSC-W', u'SSC-A',
           u'Brilliant Violet 421-A', u'FITC-A', u'PE-A', u'APC-A']
    df = df_facs[cols].abs().pipe(np.log10)


    bins_in_situ = np.logspace(2.15, 4.5, 30)
    bins_facs = np.logspace(1.8, 5, 30)
    facs_deflate = 2

    in_situ_threshold = 2200
    facs_threshold = 2700

    in_situ_ha = df_cells.query('ha_median > 0')['ha_median']
    facs_ha = df_facs['APC-A']

    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=figsize)
    ax0.hist(in_situ_ha, bins=bins_in_situ, color=DARK_GREEN)
    ax1.hist(facs_ha,    bins=bins_facs,    color=DARK_GREEN)


    ax0.set_xlim([bins_in_situ[0], bins_in_situ[-1]])
    ax1.set_xlim([bins_facs[0], bins_facs[-1]])
    ax0.set_xscale('log')
    ax1.set_xscale('log')
    ax0.set_ylabel('in situ')
    ax1.set_ylabel('FACS')
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax1.set_yticks([])

    ax1.set_xlabel('HA signal (arbitrary units)')

    lims = ax0.get_ylim(), ax1.get_ylim()
    ax0.plot([in_situ_threshold]*2, [0, 1e5], color=DARK_GRAY, ls='--')
    ax1.plot([facs_threshold   ]*2, [0, 1e5], color=DARK_GRAY, ls='--')
    ax0.set_ylim(lims[0])
    ax1.set_ylim(np.array(lims[1]) * facs_deflate)

    for ax in ax0, ax1:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    return fig

def plot_insitu_barcode_scaling():
    xs = np.arange(1, 13)

    ys = 4**xs

    ys_d2 = (xs + 1)**8 / 2000
    ys_d3 = (xs + 1)**6 / 100

    fig, (ax, ax_) = plt.subplots(ncols=2, figsize=(5, 3))
    ax_.set_visible(False)

    scale_labels = ('all barcodes', 
                    'error detecting (d=2)',
                    'error correcting (d=3)')

    h1, = ax.plot(xs, ys, color=DARK_GRAY)
    h2, = ax.plot(xs, ys_d2, color=DARK_GRAY, ls='--')
    h3, = ax.plot(xs, ys_d3, color=DARK_GRAY, ls=':')

    screens = (('genome-scale with \nerror correction', 
                    12, 50000, BLUE),
               ('genome-scale with \nerror detection',
                    9, 50000, LIGHT_BLUE),
               ('focused screen \n(1,000 barcodes)', 
                    5, 1000, STRONG_GRAY),           
              )

    screen_handles, screen_labels = [], []
    for label, x,y, color in screens:
        screen_handles += [ax.scatter(x, y, color=color, 
                                label=label, s=300, zorder=10)]
        screen_labels += [label]

    # guide lines
    ax.plot([0, 12], [50000, 50000], color=STRONG_GRAY)
    ax.plot([9, 9], [0, 50000], color=STRONG_GRAY)
    ax.plot([12, 12], [0, 50000], color=STRONG_GRAY)

    ax.set_yscale('log')
    ax.set_xlim([4, 12.6])
    ax.set_xticks([4, 6, 8, 10, 12])
    ax.set_ylim([100, 1e6])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('barcode length')
    ax.set_ylabel('library size')

    plt.sca(ax)

    leg2 = plt.legend(screen_handles, screen_labels, 
                      loc=(0.93, -0.2), handletextpad=0.3)
    leg1 = plt.legend([h1, h2, h3], scale_labels,
                      loc=(0.95, 0.6), handlelength=1.4,
                      handletextpad=0.6)

    for lh in leg2.legendHandles:
        lh._sizes = [200]

    ax.add_artist(leg2)

    return fig

class ValidationFigure():
    @staticmethod
    def plot_gene(df_pos, df_neg, gene, bins, figsize=(3, 2.5)):

        hist_kwargs = dict(bins=bins, cumulative=True, 
                           density=True, histtype='step', lw=2)

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)

        colors = {'TNFa': STIMULANT_COLORS[1], 'IL1b': STIMULANT_COLORS[0], 'negative': STRONG_GRAY}
        stimulants = 'IL1b', 'TNFa'
        for stimulant, ax in zip(stimulants, axs[1, :]):
            positive = (df_pos.query('stimulant == @stimulant')
                       ['dapi_gfp_nuclear_corr'])

            negative = (df_neg.query('stimulant == @stimulant')
                        ['dapi_gfp_nuclear_corr'])

            ax.hist(negative, color=colors['negative'], **hist_kwargs)
            ax.hist(positive, color=colors[stimulant], **hist_kwargs)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticks([])

        axs[0, 0].set_title(stimulants[0], fontsize=14)
        axs[0, 1].set_title(stimulants[1], fontsize=14)

        fig.subplots_adjust(hspace=0.7, wspace=0.25)
        return fig, axs
    
    @staticmethod
    def add_snapshots(axs, gene):
        snapshots = glob('paper/snapshots/*{0}*.tif'.format(gene))
        for stimulant, ax in zip(['IL1b', 'TNFa'], axs[0, :]):
            dapi, mNeon = lasagna.io.read_stack([f for f in snapshots if stimulant in f][0])
            n = 75
            vmin, vmax = 700, 7000
            rgb = np.zeros((n, n, 3), dtype=float)
            rgb[:, :, 1] =((mNeon[:n, :n].astype(float) - vmin) / (vmax - vmin))
            ax.imshow(rgb)
            ax.axis('off')

            box = ax.get_position()
            s = 0.3
            ax.set_position([box.x0 - box.width * s, box.y0 - box.height * s, 
                             box.width * (1 + 2*s) , box.height * (1 + 2*s)])

        for ax in axs[1]:
            y_offset = 0.07
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + y_offset, box.width, box.height])

    @staticmethod
    def plot_validation(df_ph, gene):
        df_pos = df_ph.query('gene == @gene')
        df_neg = df_ph.query('gene == "NTC"')
        bins = np.linspace(-1, 1.04, 21)
        fig, axs = ValidationFigure.plot_gene(df_pos, df_neg, gene, bins=bins)
        ValidationFigure.add_snapshots(axs, gene)
        return fig

    @staticmethod
    def plot_hist(data, color, label, bins, df_NTC, vertical=True):
        """Supplemental
        """
        hist_kwargs = dict(bins=bins, cumulative=True, density=True, histtype='step', lw=2)
        ax = plt.gca()
        negative = df_NTC.query('stimulant == @label')['dapi_gfp_nuclear_corr']
        _, _, (p_,) = ax.hist(negative, color='gray', **hist_kwargs)
        _, _, (p,)  = ax.hist(data['dapi_gfp_nuclear_corr'],  color=color, label=label, **hist_kwargs)
        move_up = lambda y: y * 0.4 + 0.5
        move_down = lambda y: y * 0.4
        move_left = lambda x: x * 0.4
        move_right = lambda x: x * 0.4 + 0.5
        if label == 'IL1b':
            if vertical:
                p.xy[:, 1]  = move_up(p.xy[:, 1])
                p_.xy[:, 1] = move_up(p_.xy[:, 1])
            else:
                p.xy[:, 0]  = move_left(p.xy[:, 0])
                p_.xy[:, 0] = move_left(p_.xy[:, 0])  
                p.xy[:, 1]  = move_down(p.xy[:, 1])
                p_.xy[:, 1] = move_down(p_.xy[:, 1])  
        else:
            if vertical:
                p.xy[:, 1]  = move_down(p.xy[:, 1])
                p_.xy[:, 1] = move_down(p_.xy[:, 1])
            else:
                p.xy[:, 0]  = move_right(p.xy[:, 0])
                p_.xy[:, 0] = move_right(p_.xy[:, 0])

    @staticmethod
    def fix_ax(ax, x0, x1):
        """Supplemental
        """
        ax.set_xlim([x0, x1 - 0.04])
        ax.plot([x0, x1], [0.5, 0.5], color='black', lw=1)
        ax.plot([x0, x0], [0.5, 0.95], color='black', lw=1)
        ax.plot([x0, x0], [0, 0.45], color='black', lw=1)
        ax.set_title(ax.get_title().replace('gene = ', ''))
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])


    @staticmethod
    def save_group_distributions(df_ph):
        """Supplemental
        """
        df_manual = ValidationFigure.load_df_manual()
        gene_groups = (df_manual.groupby('manual_call'))

        for label, group in gene_groups:
            genes = group['gene'].tolist()
            
            fg = (df_ph
             .query('gene == @genes')
             .sort_values('gene')
             .pipe(sns.FacetGrid, hue='stimulant', col='gene', col_wrap=6, aspect=0.6,
                  palette=lasagna.paper.STIMULANT_COLORS)
             .map_dataframe(ValidationFigure.plot_hist, bins=np.linspace(-1, 1.04, 21), 
                            df_NTC=df_ph.query('gene == "NTC"'),
                            vertical=True)
            )

            for ax in fg.axes.flat[:]:
                ValidationFigure.fix_ax(ax=ax, x0=-1, x1=1.04)
            
            f = 'paper/{label}_distributions.svg'
            fg.fig.tight_layout()
            fg.savefig(f.format(label=label))

    @staticmethod
    def load_df_manual():
        f = 'libraries/validation_manual_calls.csv'
        df_manual = (pd.read_csv(f, sep='\t')
         .assign(display=lambda x: x['display_group'].str.extract('(.*)_\\d')))
        return df_manual



def compose_figure_in_situ(x='17.4cm', y='11cm'):
    from svgutils.compose import Figure, Panel, SVG, Text, Image, Line

    def primed_DNA(seq, i):
        return Panel(
            Text('barcode {0}'.format(i), 0, 0, size=8).move(63, 23),
            SVG('paper/in_situ_figure/{0}.svg'.format(seq)).scale(0.9).move(0, 0),
            SVG('paper/in_situ_figure/primed_DNA.svg').scale(0.09).move(13, 18),
            )

    SBS_scale = Panel(Line(((0, 0), (61, 0))).move(9, 0),
                Text('1', 0, 0, size=10).move(0, 4),
                Text('12', 0, 0, size=10).move(72, 4),
                Text('sequencing cycle', 0, 0, size=8).move(6, 12),
                )

    DNA_subpanel = Panel(
                    primed_DNA('CTGTTATGCACT', 1).move(0, 5),
                    primed_DNA('AACAGTCTTGCG', 2).move(0, 33),
                    SBS_scale.move(49, 77)
                    )

    FOV_subpanel = Panel(
        (Image(300, 300,
            'paper/data/G135B/F2_1.log-2.FOV_DAPI.png')
            ).scale(0.3),
        Text('10X objective; 1,623 cells', -7, 100, size=8, weight='light')
        )


    panel_A = Panel(
            SVG('paper/in_situ_figure/cloning_workflow.svg').scale(0.5).move(-160, -50),
            Text(letter_case('A'), 0, 10, size=12, weight='bold'),
        )

    panel_B = Panel(
            SVG('paper/in_situ_figure/SBS_grid.svg').scale(0.5).move(90, -22),
            DNA_subpanel.move(-13, 88),
            FOV_subpanel.move(22, 0),
            Text(letter_case('B'), 0, 10, size=12, weight='bold'),
        )

    panel_C = Panel(
            SVG('paper/in_situ_figure/base_quality_per_cycle.svg').scale(0.5).move(5, -5),
            Text(letter_case('C'), 0, 10, size=12, weight='bold'),
            )

    panel_D = Panel(
            SVG('paper/in_situ_figure/mapping_rate.svg').scale(0.5).move(10, -5),
            Text(letter_case('D'), 0, 10, size=12, weight='bold'),
        )

    panel_E = Panel(
            SVG('paper/in_situ_figure/reads_per_cell.svg').scale(0.5).move(15, -7),
            Text(letter_case('E'), 0, 10, size=12, weight='bold'),
            )

    panel_F = Panel(
            SVG('paper/in_situ_figure/barcode_scaling.svg').scale(0.5).move(0, 0),
            Text(letter_case('F'), 0, 10, size=12, weight='bold'),
        )


    return Figure(x, y, panel_A, panel_B.move(25, 0), Panel(), Panel(),
                        panel_C, panel_D.move(-30, 0), panel_E.move(-30, 0), panel_F.move(-80, 0))

def compose_figure_FR(x='17.4cm', y='8.5cm'):
    from svgutils.compose import Figure, Panel, SVG, Text, Image

    panel_A = Panel(
            SVG('paper/FR_figure/FR_reporter.svg').scale(0.5).move(0, 0),
            Text(letter_case('A'), 0, 10, size=12, weight='bold'),
        )
    
    panel_B = Panel(
            SVG('paper/FR_figure/FR_image_grid.svg').scale(0.48).move(-5, -15),
            Text(letter_case('B'), 0, 10, size=12, weight='bold'),
        )

    panel_C = Panel(
            SVG('paper/FR_figure/HA_hist_insitu_FACS.svg').scale(0.5).move(3, -10),
            Text(letter_case('C'), 0, 10, size=12, weight='bold'),
        )

    panel_D = Panel(
            SVG('paper/FR_figure/FR_scatter_barcodes.svg').scale(0.5).move(10, -5),
            Text(letter_case('D'), 0, 10, size=12, weight='bold')
            )

    panel_E = Panel(
            SVG('paper/FR_figure/insitu_FACS_FR_sgRNA.svg').scale(0.5).move(30, -10),
            Text(letter_case('E'), 0, 10, size=12, weight='bold')
            )

    panel_F = Panel(
        SVG('paper/FR_figure/abundance_insitu_NGS.svg').scale(0.5).move(5, 0),
        Text(letter_case('F'), 0, 10, size=12, weight='bold')
        )

    return Figure(x, y, panel_A, panel_B, panel_C.move(-30, 0),
        panel_D, panel_E, panel_F.move(-30, 0))

def compose_figure_NFKB(x='17.4cm', y='6.5cm'):
    from svgutils.compose import Figure, Panel, SVG, Text

    panel_A = Panel(
            SVG('paper/NFKB_figure/IL1b_pathway_graph.svg').scale(0.5).move(15, 0),
            SVG('paper/NFKB_figure/pathway_node_legend.svg').scale(0.45).move(-100, 120),
            SVG('paper/NFKB_figure/pathway_edge_legend.svg').scale(0.45).move(20, 120),
            Text(letter_case('A'), 0, 10, size=12, weight='bold'),
        )

    panel_B = Panel(
            SVG('paper/individual_KO_grid.svg').scale(0.5).move(-15, -15),
            Text(letter_case('B'), 0, 10, size=12, weight='bold'),
        )

    panel_C = Panel(
            SVG('paper/NFKB_figure/correlation_distribution_vertical.svg').scale(0.5),
            Text(letter_case('C'), 0, 10, size=12, weight='bold'),
        )

    panel_D = Panel(
            SVG('paper/NFKB_figure/87_gene_scatter.svg').scale(0.5).move(10, -30),
            Text(letter_case('D'), 0, 10, size=12, weight='bold'),
            )

    panel_E = Panel(
            SVG('paper/NFKB_figure/87_gene_signal_per_sg_flip.svg').scale(0.5),
            Text(letter_case('E'), 0, 10, size=12, weight='bold'),
            )

    return Figure(x, y, panel_A, panel_B, panel_C, panel_D, panel_E)

def compose_figure_NFKB_2(x='17.4cm', y='6.5cm'):
    from svgutils.compose import Figure, Panel, SVG, Text

    panel_A = Panel(
            SVG('paper/NFKB_figure/workflow_slide.svg').scale(0.31).move(-35, 15),
            Text(letter_case('A'), 0, 10, size=12, weight='bold'),
        )

    panel_B = Panel(
            SVG('paper/NFKB_figure/individual_KO_grid_1.8.svg').scale(0.5).move(-20, -5),
            Text(letter_case('B'), 0, 10, size=12, weight='bold'),
            Text('p65-mNeon', 24, 75, size=7, weight='bold', color=GREEN)
        )

    panel_C = Panel(
            SVG('paper/NFKB_figure/correlation_distribution_horizontal.svg').scale(0.5).move(0, 10),
            Text(letter_case('C'), 0, 10, size=12, weight='bold'),
        )

    panel_D = Panel(
            SVG('paper/NFKB_figure/87_gene_scatter_4.9.svg').scale(0.5).move(-15, -10),
            Text(letter_case('D'), 0, 10, size=12, weight='bold'),
            )

    panel_E = Panel(
            SVG('paper/NFKB_figure/87_gene_signal_per_sg_flip.svg').scale(0.5).move(0, 5),
            Text(letter_case('E'), 0, 10, size=12, weight='bold'),
            )

    return Figure(x, y, 
        panel_A, panel_B.move(-35, 0), panel_C.move(-45, 0),
        Panel(), Panel(), panel_D.move(-40, -50),
        Panel(), panel_E.move(-30, -20))

def compose_figure_validation(x='17.4cm', y='8cm'):
    from svgutils.compose import Figure, Panel, SVG, Text, Image

    panel_A_horizontal = Panel(
            SVG('paper/validation_figure/dummy_horizontal_1.svg').scale(0.5).move(-50, -75),
            SVG('paper/validation_figure/dummy_horizontal_2.svg').scale(0.5).move(-50, 50),
            Text(letter_case('A'), 0, 10, size=12, weight='bold'),
        )

    panel_A_vertical = Panel(
        SVG('paper/validation_figure/dummy_vertical.svg').scale(0.5).move(-15, -27),
            Text(letter_case('A'), 0, 10, size=12, weight='bold'),
        ).move(20, 0)

    panel_B = Panel(
            SVG('paper/NFKB_figure/IL1b_pathway_graph.svg').scale(0.5).move(15, 0),
            SVG('paper/NFKB_figure/pathway_node_legend.svg').scale(0.45).move(-100, 120),
            SVG('paper/NFKB_figure/pathway_edge_legend.svg').scale(0.45).move(20, 120),
            Text('IL1b stimulation', 0, 10, size=10).move(60, -15),
            Text(letter_case('B'), 0, 10, size=12, weight='bold'),
        )

    panel_C = Panel(
        SVG('paper/NFKB_figure/TNFa_pathway_graph.svg').scale(0.5).move(15, 0),
        SVG('paper/NFKB_figure/pathway_node_legend.svg').scale(0.45).move(-100, 120),
        SVG('paper/NFKB_figure/pathway_edge_legend.svg').scale(0.45).move(20, 120),
        Text('TNFa stimulation', 0, 10, size=10).move(60, -15),
        Text(letter_case('C'), 0, 10, size=12, weight='bold'),
    )

    # layout = Figure(x, y, panel_A_vertical, panel_B.move(10, 0), panel_C.move(5, 0)).tile(3, 1)
    layout = Figure(x, '12cm', panel_A_horizontal.move(15, 0), Panel(), 
        panel_B.move(30, 0), panel_C.move(-30, 0)).tile(2, 2)

    return layout

