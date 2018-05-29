import os
import shutil
import pandas as pd
from lasagna.schema import *

home = '/Users/feldman/lasagna/'

def locate(filename):
	return os.path.join(home, filename)

def copy(src, dst):
	if os.path.exists(dst):
		message = 'overwriting {dst} from {src}'
	else:
		message = 'creating {dst} from {src}'
	print(message.format(src=src, dst=dst))
	shutil.copy(src, dst)
		
def copy_data():
	# 87 gene set
	f1 = '20180402_6W-G161/sgRNA_stats.csv'
	f2 = 'paper/data/sgRNA_stats_87_genes.csv'
	info = {DATASET: '6W-161'}
	df = (pd.read_csv(f1)
		  .assign(**info)
		  .)



# def load_sgRNA_stats():

