import lasagna.io
import lasagna.config
import numpy as np
import glue.core
from lasagna import echo


luts = None
display_ranges = None

default_color = lambda: lasagna.config.j.java.awt.Color.GRAY

fiji_label = 1

class FijiViewer(object):

	@staticmethod
	def setup(self, axes):
		j = lasagna.config.j

		self.imp = lasagna.io.show_hyperstack(np.zeros((100,100)), title='viewer')
		self.displayed_file = None
		
	@staticmethod
	def plot_data(self, axes, source, y, contours, bounds):
		j = lasagna.config.j

		if y.size:
			axes.scatter(source, y, s=10, c='g', marker='.')
		
		# can recolor these as necessary
		bounds = bounds.categories[bounds.astype(int)]
		self.contours = np.array([x + y[:2] for x,y in zip(contours.id._unhashable, bounds)])

		self.files = source.categories[source.astype(int)]
		self.source_id = source.id

		if self.imp.getOverlay():
			self.imp.getOverlay().setStrokeColor(j.java.awt.Color.GRAY)

		# make a key listener for selection events
		j.add_key_typed(j.make_selection_listener(update_selection, self, 
			lasagna.config.queue_appender, key='`'), self.imp)

	@staticmethod
	def plot_subset(self, axes, source, y, contours, style):
		j = lasagna.config.j
		lasagna.config.self = self

		# decide on a file
		if not source.size:	
			return
		
		# take first selected file, only update display if changed
		self.source_val = source.astype(int)[0]
		file_to_show = source.categories[self.source_val]

		att_index = np.where(self.files == file_to_show)
		if file_to_show != self.displayed_file:
			data = lasagna.io.read_stack(file_to_show)
			self.imp = lasagna.io.show_hyperstack(data, imp=self.imp, 
								luts=luts, display_ranges=display_ranges)

			self.displayed_file = file_to_show

			# overlay all contours for the new file
			all_contours = self.contours[att_index]
			packed = [(1 + contour).T.tolist() for contour in all_contours]
			j.overlay_contours(packed, imp=self.imp)
			self.imp.getOverlay().setStrokeColor(default_color())
			# lasagna.config.j.ij.IJ.log('changed to %s' % file_to_show)

		# lasagna.config.j.ij.IJ.log('displayed %s' % str(style.parent))
		overlay = self.imp.getOverlay()
		color = j.java.awt.Color.decode(style.color)
		# only show contours that apply to this file
		rois = np.where(np.in1d(att_index, contours))[0]
		lasagna.config.selection = np.intersect1d(contours, att_index)

		j.set_overlay_contours_color(rois, self.imp, color)
		self.imp.updateAndDraw()

		# reset contour when mpl artist removed
		if not y.size:
			artist = axes.scatter([1], [1])
			# lasagna.config.j.ij.IJ.log('y was empty %s' % style.parent.label)
		else:
			artist = axes.scatter(source, y, s=100, c=style.color, marker='.', alpha=0.5)
		rois = np.intersect1d(contours, att_index)
		rois = np.where(np.in1d(att_index, contours))[0]
		def do_first(g, imp=self.imp, rois=rois, style=style):
			def wrapped(*args, **kwargs):
				# if overlay is gone (new file loaded), skip
				if imp.getOverlay():
					lasagna.config.rois = rois
					j.set_overlay_contours_color(rois, 
												 imp, default_color())
					# lasagna.config.j.ij.IJ.log('removed %d rois from %s' % (len(rois), style.parent.label))
				return g(*args, **kwargs)
			return wrapped
		artist.remove = do_first(artist.remove)


	@staticmethod
	def make_selector(self, roi, source, y):
		state = glue.core.subset.RoiSubsetState()
		state.roi = roi
		# selections on image will always be along x and y axes
		# coud imagine propagating more complex selection criteria from image (e.g.,
		# magic wand for nearest neighbors on barcode simplex)
		state.xatt = source.id
		state.yatt = y.id
		return state
		# update selection, bypassing callback


class FijiGridViewer(object):

	@staticmethod
	def setup(self, axes):
		"""Set up ImagePlus.
		"""
		j = lasagna.config.j

		self.imp = lasagna.io.show_hyperstack(np.zeros((100,100)), title='grid viewer')

	@staticmethod
	def plot_data(self, axes):
		"""Add any callbacks. Opportunity to store attributes of full data, as opposed to 
		subsets.
		"""
		axes.invert_yaxis()
		pass
		

	@staticmethod
	def plot_subset(self, axes, Fiji, show_nuclei, source, contours, bounds, sort_by, padding, style):
		"""Show a grid of cells, when called by the first visible layer.
		"""
		j = lasagna.config.j

		lasagna.config.artists = self.widget.layers
		artists = self.widget.layers
		layers = [a.layer for a in artists]
		active_artists = [a for a in artists if a.enabled and a.visible]
		active_layers = [a.layer for a in active_artists]
		lasagna.config.this_layer = this_layer = style.parent

		# 1. we are redrawing the first visible layer only
		# 2. redrawing all, this is the first visible layer (supposedly)
		try:
			j.ij.IJ.log('artists: %s' % artists)
		except TypeError:
			pass
		if active_artists:
			flag1 = this_layer == active_layers[0]
			flag2 = layers.index(this_layer) < min(layers.index(x) for x in active_layers)
			j.ij.IJ.log('flag1: %s flag2: %s' % (flag1, flag2))
			flag3 = False
		else:
			flag3 = True
			j.ij.IJ.log('flag3: %s' % flag3)
		# j.ij.IJ.log('this artist: %s' % (this_artist))
		j.ij.IJ.log("---%s---" % style.parent.label)
		for ca in self.widget.layers:
			j.ij.IJ.log("%s: enabled (%s) visible (%s)" % (ca.layer.label, ca.enabled, ca.visible))

		if flag3 or flag2 or flag1:
			lasagna.config.sort_by = sort_by
			# pick coordinates
			index = np.argsort(sort_by)
			width = np.ceil(np.sqrt(len(index)))
			x = np.arange(len(index)) % width
			y = (np.arange(len(index)) / width).astype(int)
			axes.scatter(x, y, c=style.color)
			axes.axis('tight')

			if Fiji:
				files = source.categories[source.astype(int)][index]
				bounds = bounds.categories[bounds.astype(int)][index]
				lasagna.config.bounds = bounds
				data = grid_view(files, bounds, padding=padding)
				shape = data.shape
				data = lasagna.io.montage(data)
				self.imp = lasagna.io.show_hyperstack(data, imp=self.imp, 
									luts=luts, display_ranges=display_ranges)

				if show_nuclei:
					# offset contours to match grid spacing
					offsets = (np.array([y, x]).T * shape[-2:]) + padding
					# need to apply subset to unhashable, then sort order
					c = contours.id._unhashable[contours.astype(int)][index]
					contours_offset = np.array([x + y[:2] for x,y in zip(c, offsets)])
					packed = [(1 + contour).T.tolist() for contour in contours_offset]

					j.overlay_contours(packed, imp=self.imp)
					self.imp.getOverlay().setStrokeColor(default_color())
				else:
					self.imp.setOverlay(j.ij.gui.Overlay())


	def make_selector(self, roi, sort_by):
		# transform subsets on grid into constraints on sort variable by mapping xy
		# to range of 
		pass


def update_selection(selection, viewer):
	"""Assumes first dataset contains 'x' and 'y' components.
	Selection consists of (xmin, xmax, ymin, ymax)
	"""
	
	xmin, xmax, ymin, ymax = selection
	roi = glue.core.roi.RectangularROI(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
	xatt = lasagna.config.app.data_collection[0].data.find_component_id('x')
	yatt = lasagna.config.app.data_collection[0].data.find_component_id('y')
	xy_state = glue.core.subset.RoiSubsetState(xatt=xatt , yatt=yatt, roi=roi)
	file_state = glue.core.subset.CategorySubsetState(viewer.source_id, viewer.source_val)
	# selection only applies to data in displayed file
	subset_state = glue.core.subset.AndState(xy_state, file_state)

	data_collection = lasagna.config.app.data_collection

	# if no subset groups selected, make a new one
	layers = (lasagna.config.app.centralWidget()
					 .layerWidget.layerTree.selected_layers())
	subset_groups = [s for s in layers if isinstance(s, glue.core.subset_group.SubsetGroup)]
	if len(subset_groups) == 0:
		global fiji_label
		new_label = 'Fiji %d' % fiji_label
		fiji_label += 1
		data_collection.new_subset_group(label=new_label, subset_state=subset_state)
	else:
		edit_mode = glue.core.edit_subset_mode.EditSubsetMode()
		edit_mode.update(data_collection, subset_state)


def make_selection_listener(viewer, key='u'):
	def selection_listener(event):
		if event.getKeyChar().lower() == key:
			imp = event.getSource().getImage()
			roi = imp.getRoi()
			if roi:
				if roi.getTypeAsString() == 'Rectangle':
					rect = roi.getBounds()
					selection = (rect.getMinX(), rect.getMaxX(), 
								 rect.getMinY(), rect.getMaxY())
					update_selection(selection, viewer)
	return selection_listener

def grid_view(files, bounds, padding=40):
    from lasagna.io import b_idx, compose_stacks, get_mapped_tif
    
    arr = []
    for filename, bounds_ in zip(files, bounds):
        I = get_mapped_tif(filename)
        I_cell = I[b_idx(None, bounds=bounds_, padding=((padding,padding), I.shape))]
        arr.append(I_cell.copy())

    return compose_stacks(arr)
				
# lasagna.config.style = style
# print "---%s---" % style.parent.label
# for ca in lasagna.config.self.widget.layers:
	
# 	print "%s: enabled (%s) visible (%s)" % (ca.layer.label, ca.enabled, ca.visible)


# artists = self.widget.layers
# layers = [a.layer for a in artists]
# # determine if this is the first layer
# # layer currently being plotted is not visible
# active_artists = [] # subsets, excluding currently plotted one
# for i, a in enumerate(artists):
# 	if a.enabled and a.visible and not isinstance(a.layer, glue.core.data.Data):
# 		active_artists += [a]
# print active_artists

# lasagna.config.d[style.parent.label] = active_artists
# # TODO fix the index checking
# lasagna.config.d1[style.parent.label] = layers
# lasagna.config.d2[style.parent.label] = style.parent
# current_artist_position = layers.index(style.parent)
# if all(layers.index(style.parent) < layers.index(a.layer) for a in active_artists):
# 	lasagna.config.reset += [style.parent]
# 	# reset overlay
# 	overlay = self.imp.getOverlay()
# 	if overlay:
# 		overlay.setStrokeColor(default_color())
# 	print 'reset on %s' % style.parent
# # lasagna.config.style += [style]


