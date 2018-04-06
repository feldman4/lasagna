import pandas as pd
from glob import glob
import re
import seaborn as sns
from lasagna.designs.pool0 import load_enst_ncbi

def load_endo(hist_glob='20170827_DF/*hist'
			  , info_path='20170827_DF/samples.nfo'
			  , combine_info=(('cell line', 'library'), ('dox', 'live'))):
	df_info = load_info(info_path)
	for a,b in combine_info:
		col = '_'.join([a,b])
		df_info[col] = zip(df_info[a], df_info[b])

	files = glob(hist_glob)
	get_well = lambda s: re.findall('G_(...)_', s)[0]
	get_sgRNA = lambda s: s[5:-5]

	arr = []
	for f in files:
	    df = pd.read_csv(f, sep='\s+')
	    df.columns ='count', 'sgRNA'
	    df['sgRNA'] = df['sgRNA'].apply(get_sgRNA)
	    df['file'] = f
	    df['well'] = get_well(f)
	    df['fraction'] = df['count'] / df['count'].sum()
	    arr += [df]
	df = pd.concat(arr)

	filt = df['well'].isin(df_info.index)
	df = df[filt].join(df_info, on='well')

	df_sgRNAs = (pd.read_csv('../libraries/Feldman_12K_Array_pool1_table.csv')
               .drop_duplicates('sgRNA')
               .set_index('sgRNA'))

	df_genes = (load_enst_ncbi(path='../libraries/')
		          .dropna()
				  .drop_duplicates('gene_id')
				  .set_index('gene_id'))

	df_sgRNAs = df_sgRNAs.join(df_genes['gene_symbol'], on='gene_id')

	df = df.join(df_sgRNAs[['gene_id', 'gene_symbol']], on='sgRNA')

	df['mapped'] = df['sgRNA'].isin(df_sgRNAs.index)

	return df


def mapping_stats(df):
	df_info = load_info()
	a = df.groupby('well').apply(lambda x: (x['mapped'] * x['fraction']).sum())
	b = df.groupby('well').apply(lambda x: (x['mapped'] * x['count']).sum())
	x = pd.concat([a,b], axis=1)
	x.columns = 'mapping %', 'mapped reads'
	return pd.concat([x, df_info], axis=1)


def load_info(path):
	return pd.read_csv(path, sep='\t|,', engine='python').set_index('well')