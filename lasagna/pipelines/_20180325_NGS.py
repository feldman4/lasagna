from lasagna.imports import *

samples = {'cLas41_46': 'cLas4[123456].pL42.hist',
        'cLas34': 'cLas34_[123].pL42.hist',
        'pLL31_35': 'pLL3[15].pL42.hist',
        'pLL46_47': 'pLL4[67].pL42.hist',
        'pLL47_RJC': 'pLL47_RJC*.pL42.hist'
}

def load_abundances(home='NGS/samples'):
    arr = []
    for tag, search in samples.items():
        files = glob(os.path.join(home, search))
        file_to_library = {f: re.findall('.*/(.*).pL42.hist', f)[0] for f in files}
        df_NGS = (pd.concat(map(lasagna.in_situ.load_NGS_hist, files))
                 .query('count > 5')
                 .assign(library=lambda x: x['file'].map(file_to_library))
                 .assign(tag=tag))
        arr += [df_NGS]

    return pd.concat(arr)

def plot_and_export_abundance(df_NGS):
    for tag, df in df_NGS.groupby('tag'):
        fig, ax = plt.subplots()
        df.reset_index(drop=True).groupby('library', as_index=False)['fraction'].plot(ax=ax)
        ax.set_xlabel('barcode rank')
        ax.set_ylabel('log10 fraction')
        plt.legend()
        fig.tight_layout()
        f = '/Users/feldman/lasagna/NGS/samples/{tag}_abundance_filt.pdf'
        fig.savefig(f.format(tag=tag))
