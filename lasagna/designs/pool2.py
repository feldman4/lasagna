from lasagna.imports import *

home = '/Users/feldman/lasagna/libraries/'

def load_barcodes():
	f = os.path.join(home, '20171211_12bp_dist3_97304.csv')
	return pd.read_csv(f)

def load_oligos_layout():
	f = os.path.join(home, '20171211_oligos_layout.csv')
	f = os.path.join(home, 'Lasagna Oligos - pools.csv')
	df_layout = (pd.read_csv(f, skiprows=4)
		.filter(regex='^(?!Unnamed).*')
		.dropna(subset=['pool']))
	return df_layout

def run():
	df_barcodes = (load_barcodes()
		.query('hp < 5 & gc < 0.11 & div2 > 2')
		.reset_index(drop=True))

	df_design = load_oligos_layout()

	return df_design

	