from lasagna.io import read_stack
from lasagna.io import save_stack
from lasagna.io import montage
from lasagna.io import pile
from lasagna.io import subimage
from lasagna.io import parse_MM
from lasagna.io import offset
from lasagna.io import read_registered

from lasagna.process import feature_table
from lasagna.process import build_feature_table
from lasagna.process import alpha_blend
from nose.tools import assert_raises

import numpy as np
import pandas as pd
import os
import hashlib 

def hash_np(arr):
	h = hashlib.sha1(arr).hexdigest()
	return hash((h, arr.shape))

home = lambda s: os.path.join(os.path.dirname(__file__), s)
nuclei = home('nuclei.tif')
nuclei_compressed = home('nuclei_compressed.tif')
stack = home('stack.tif')
tmp = (home(str(i)) for i in range(1000))

def test_read_stack():
	data = read_stack(nuclei)
	assert hash_np(data) == -856296963688120929

	data = read_stack(nuclei_compressed)
	assert hash_np(data) == -856296963688120929

	data = read_stack(stack)
	assert data.shape == (3, 4, 511, 626)
	assert hash_np(data) == 4530181413177493733

	# test memory mapping, endian
	data_ = read_stack(stack, memmap=True)
	assert (data == data_).all()


def test_write_stack():
	data = read_stack(stack)

	saveto = tmp.next() + '.tif'

	# test compression
	for compress in (0, 1):
		save_stack(saveto, data, compress=compress)
		data_ = read_stack(saveto)

		assert hash_np(data) == hash_np(data_)

	# test data types
	for dtype in (np.bool, np.uint8, np.uint16, np.float32, np.float64):
		save_stack(saveto, data.astype(dtype))
		data_ = read_stack(saveto)
		if dtype == np.bool:
			assert data_.max() == 255
		else:
			assert data_.max() == data.astype(dtype).max()

		if dtype == np.bool:
			assert data_.dtype == np.uint8
		elif dtype == np.float64:
			assert data_.dtype == np.float32
		else:
			assert data_.dtype == dtype 

	# test resolution
	# test LUTS, display_ranges

	os.remove(saveto)

def test_montage():
	data = read_stack(stack)
	data = data[...,:400,:500]
	
	n = 100
	
	arr = []
	for i in range(4):
	    for j in range(5):
	        arr += [data[:,:,i*n:i*n+n, j*n:j*n+n]]

	assert (montage(arr, shape=(4,5)) == data).all()


def test_pile():
	data = read_stack(stack)

	arr = []
	for n in (30, 40, 50):
		arr += [data[...,:n,:n]]

	data_ = pile(arr)

	# same tiles in vertical stack
	assert ((data_[..., :30, :30] - data_[0,...,:30,:30]) == 0).all()

	# number of padded entries
	assert (data_ == 0).sum() == 30000

def test_subimage():
	data = read_stack(stack)

	p = 10
	data_ = subimage(data, [10, 5, 15, 30], pad=p)
	assert (data_[...,5:] == data[..., 0:25, :40]).all()

	assert (data_==0).sum() == 1500

def test_offset():
	data = read_stack(stack)

	offsets = [1, -1, 4, 8]
	data_ = offset(data, offsets)

	assert data.shape == data_.shape
	assert data_[1,0,4,8] == data[0,1,0,0]

def test_parse_MM():

	cases = (("100X_round1_1_MMStack_A1-Site_15.ome.tif", 
				('100X', 1,  'A1', 15)),
			 ("blah/100X_scan_1_MMStack_A1-Site_15.ome.tif",
     			('100X', None, 'A1', 15)))
	for s, output in cases:
		assert parse_MM(s) == output

def test_feature_table():
	features = {
		    'area':     lambda region: region.area,
		    'bounds':   lambda region: region.bbox,
		    'label':    lambda region: np.median(region.intensity_image[region.intensity_image > 0]),
	}

	data = read_stack(stack)
	mask = read_stack(nuclei)

	results = feature_table(data[0][0], mask, features)
	
	results_ = pd.read_pickle(os.path.join(home, 'feature_table.pkl'))
	assert (results == results_).all().all()
  

def test_build_feature_table():
	features = {
			'mean': lambda region: region.intensity_image[region.image].mean(),
            'median': lambda region: np.median(region.intensity_image[region.image]),
         	'max': lambda region: region.intensity_image[region.image].max()
    }

	index = (('round', range(1,4)), ('channel', ('DAPI', 'Cy3', 'A594', 'Cy5')))

	data = read_stack(stack)
	mask = read_stack(nuclei)

	df = build_feature_table(data, mask, features, index)
	df.index.name = 'cell'

	df = df.set_index(['round', 'channel'], append=True).stack().unstack('cell').T

	df_ = pd.read_pickle(home('build_feature_table.pkl'))

	assert (df == df_).all().all()


def test_alpha_blend():
	tiles = ['tile_%d.tif' % i for i in range(4)]
	arr = [read_stack(home(t)) for t in tiles]
	translations = read_registered(home('tiles_registered.txt'))

	fused = alpha_blend(arr, translations, clip=False)
	fused_ = read_stack(home('fused.tif'))
	average_diff = np.abs(fused.astype(float) - fused_.astype(float)).sum() / fused.size

	assert average_diff < 1.

