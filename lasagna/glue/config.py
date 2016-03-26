from __future__ import absolute_import, division, print_function

from glue.external import echo
from glue import custom_viewer
import glue
import glue.config
import glue.core
import glue.qt.custom_viewer
import pandas as pd
import numpy as np
from natsort import natsorted

import lasagna.glueviz
# from glue.config import qt_client, data_factory, link_function


"""Declare any extra link functions like this"""
#@link_function(info='translates A to B', output_labels=['b'])
#def a_to_b(a):
#    return a * 3


"""Data factories take a filename as input and return a Data object"""
#@data_factory('JPEG Image')
#def jpeg_reader(file_name):
#    ...
#    return data


"""Extra qt clients"""
#qt_client(ClientClass)


class Fiji(glue.qt.custom_viewer.CustomViewer):
	name = 'Fiji Viewer'
	source = 'att(file)'
	contours = 'att(contour)'
	bounds = 'att(bounds)'

	y = 'att'

	def setup(self, axes):
		self.setup_output = lasagna.glueviz.FijiViewer.setup(self, axes)

	def plot_data(self, axes, source, y, contours, bounds):
		self.plot_output = lasagna.glueviz.FijiViewer.plot_data(self, axes, source, y, contours, bounds)

	def plot_subset(self, axes, source, y, contours, style):
		self.subset_output = lasagna.glueviz.FijiViewer.plot_subset(self, axes, source, y, contours, style)


	def make_selector(self, roi, source, y):
		self.select_state = lasagna.glueviz.FijiViewer.make_selector(self, roi, source, y)
		return self.select_state
		

grid = custom_viewer('Grid Viewer',
                      Fiji = False,
                      show_nuclei = False,
                      source='att(file)',
                      contours='att(contour)',
                      bounds = 'att(bounds)',
                      sort_by = 'att',
                      padding = 50,
                      limit = 256)

@grid.setup
def a(self, axes):
	self.setup_output = lasagna.glueviz.FijiGridViewer.setup(self, axes)

@grid.plot_data
def b(self, axes):
	# set up ImagePlus
	self.plot_output = lasagna.glueviz.FijiGridViewer.plot_data(self, axes)

@grid.plot_subset
def c(self, axes, Fiji, show_nuclei, source, contours, bounds, sort_by, padding, limit, style):
	# if this is the first visible/enabled layer, show a grid of cells in Fiji
	self.plot_subset_output = lasagna.glueviz.FijiGridViewer.plot_subset(self, axes, Fiji, show_nuclei, source, contours, bounds, sort_by, padding, limit, style)

# @bball.make_selector
# def make_selector(self, roi, sort_by):
# 	# transform subsets on grid into constraints on sort variable by mapping xy
# 	# to range of 
# 	self.select_state = lasagna.glueviz.FijiGridViewer.make_selector(self, roi, sort_by)
# 	return self.select_state


# class Grid(glue.qt.custom_viewer.CustomViewer):
# 	name = 'Fiji Grid Viewer'
# 	Fiji = False
# 	source = 'att(file)'
# 	contours = 'att(contour)'
# 	# nucleus = True
# 	# cell_outline = False
# 	bounds = 'att(bounds)'
# 	sort_by = 'att'
# 	padding = 50

# 	def setup(self, axes):
# 		self.setup_output = lasagna.glueviz.FijiGridViewer.setup(self, axes)

# 	def plot_data(self, axes):
# 		# set up ImagePlus
# 		self.plot_output = lasagna.glueviz.FijiGridViewer.plot_data(self, axes)

# 	def plot_subset(self, axes, Fiji, source, contours, bounds, sort_by, padding, style):
		
# 		# if this is the first layer, show a grid of cells in Fiji
# 		# after making grid, record shape so we can 
# 		self.subset_output = lasagna.glueviz.FijiGridViewer.plot_subset(self, axes, Fiji, source, contours, bounds, sort_by, padding, style)

# 	def make_selector(self, roi, sort_by):
# 		# transform subsets on grid into constraints on sort variable by mapping xy
# 		# to range of 
# 		self.select_state = lasagna.glueviz.FijiGridViewer.make_selector(self, roi, sort_by)
# 		return self.select_state




# def pyqt_set_trace():
#     '''Set a tracepoint in the Python debugger that works with Qt'''
#     from PyQt4.QtCore import pyqtRemoveInputHook
#     import pdb
#     import sys
#     pyqtRemoveInputHook()
#     # set up the debugger
#     debugger = pdb.Pdb()
#     debugger.reset()
#     # custom next to get outside of function scope
#     debugger.do_next(None) # run the next command
#     users_frame = sys._getframe().f_back # frame where the user invoked `pyqt_set_trace()`
#     debugger.interaction(users_frame, None)


# uniques = glue.custom_viewer('Uniques',
#                       well='att(well)',
#                       counts='att(counts)',
#                       sgRNA='att(sgRNA)',
#                       threshold=40,
#                       sort=['plate order', 'reads', 'unique sgRNAs'])

# import pdb

# @uniques.plot_data
# def show_line(axes, well, counts, sgRNA, threshold, sort, state):
	
# 	# prepare data
# 	# non-numerical types converted to AttributeInfo containing index into categories
# 	well = [well.categories[i] for i in well]
# 	df = pd.DataFrame({'well': well, 'counts': counts,
# 					   'sgRNA': sgRNA})
# 	df = df.pivot_table(values='counts', columns='well', index='sgRNA')
# 	df.columns = natsorted(df.columns)

# 	# collapse and sort
# 	well_stats = pd.DataFrame()
# 	well_stats['unique sgRNAs'] = (df > threshold).sum(axis=0)
# 	well_stats['reads'] = df.sum(axis=0)

	
# 	if sort in ('reads', 'unique sgRNAs'):
# 		well_stats = well_stats.sort(columns=sort)
	
# 	print(3)
# 	# plot
# 	axes2 = axes.twinx()
# 	well_stats['unique sgRNAs'].plot(ax=axes, xticks=range(95), rot=90, c='b')
# 	well_stats['reads'].plot(ax=axes2, c='g')

# 	axes.set_ylabel('unique sgRNAs')
# 	axes2.set_ylabel('reads', color='g')
# 	axes2.set_xticklabels(well_stats.index)



# @uniques.plot_subset
# def show_points(axes, well, counts, sgRNA, threshold):
# 	df = pd.DataFrame({'well': well, 'counts': counts,
# 					   'sgRNA': sgRNA})
# 	df = df.pivot_table(values='counts', columns='well', index='sgRNA')
# 	unique_counts = (df > threshold).sum(axis=0)
# 	axes.plot(unique_counts)



# from skimage.external.tifffile import imread

# def is_tif(filename, **kwargs):
#     return filename.endswith('.tif')

# @glue.config.data_factory('3D image loader', is_tif)
# def read_tif(file_name):
#     im = imread(file_name)
#     return glue.core.Data(cube=im)