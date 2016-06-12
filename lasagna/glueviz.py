from glue.core.data import Data
from glue.core.component import CategoricalComponent
from glue.core.component_id import ComponentID
from glue.core.roi import RectangularROI
from glue.core.subset import RoiSubsetState, CategorySubsetState, AndState
from glue.core.subset_group import SubsetGroup
from glue.core.edit_subset_mode import EditSubsetMode

import numpy as np

import lasagna.config
import lasagna.io
import lasagna.utils

luts = None
display_ranges = None

default_color = lambda: lasagna.config.j.java.awt.Color.GRAY

fiji_label = 1

# used by pandas_to_glue for consistent naming
default_name_map = (
    (('all', 'bounds', 1.), 'bounds'),
    (('all', 'file', 1.), 'file'),
    (('all', 'contour', 1.), 'contour'),
    (('well', '', ''), 'well'),
    (('row', '', ''), 'row'),
    (('column', '', ''), 'column'),
    (('all', 'x', 1.), 'x'),
    (('all', 'y', 1.), 'y'),
    (('barcode x', '', ''), 'barcode x'),
    (('barcode y', '', ''), 'barcode y'),
    (('positive', '', ''), 'positive'),
    (('signal', '', ''), 'signal'),
)


class FijiViewer(object):
    @staticmethod
    def setup(self, axes):
        """Creates ImagePlus tied to this viewer.
        :param self:
        :param axes:
        :return:
        """
        if lasagna.config.j is None:
            print 'connecting to Fiji via rpyc...'
            lasagna.config.j = lasagna.utils.start_client()
        self.imp = lasagna.io.show_IJ(np.zeros((100, 100)), title='FijiViewer')
        self.displayed_file = None

    @staticmethod
    def plot_data(self, axes, source, y, contours, bounds):
        """
        :param self:
        :param axes:
        :param source:
        :param y:
        :param contours:
        :param bounds:
        :return:
        """
        j = lasagna.config.j
        if y.size:
            axes.scatter(source, y, s=10, c='g', marker='.')

        # can recolor these as necessary
        bounds = bounds.categories[bounds.astype(int)]
        self.contours = np.array([x + y[:2] for x, y in zip(contours.id._unhashable, bounds)])

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
            self.imp = lasagna.io.show_IJ(data, imp=self.imp,
                                             luts=luts, display_ranges=display_ranges)

            self.displayed_file = file_to_show

            # overlay all contours for the new file
            # all_contours = self.contours[att_index]
            # packed = [(1 + contour).T.tolist() for contour in all_contours]
            packed = lasagna.utils.pack_contours(self.contours[att_index])
            self.packed = packed
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
        from glue.core.subset import RoiSubsetState
        state = RoiSubsetState()
        state.roi = roi
        # selections on image will always be along x and y axes
        # could imagine propagating more complex selection criteria from image (e.g.,
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

        self.imp = lasagna.io.show_IJ(np.zeros((100, 100)), title='GridViewer')

    @staticmethod
    def plot_data(self, axes):
        """Add any callbacks. Opportunity to store attributes of full data, as opposed to
		subsets.
		"""
        pass

    @staticmethod
    def plot_subset(self, axes, Fiji, show_nuclei, source, contours, bounds, sort_by, padding, limit, style):
        """Show a grid of cells, when called by the first visible layer.
		"""
        j = lasagna.config.j
        artists = lasagna.config.artists = self.widget.layers
        this_layer = lasagna.config.this_layer = style.parent

        if check_artist(artists, this_layer):

            lasagna.config.sort_by = sort_by
            # pick coordinates
            index = np.argsort(sort_by)
            # limit the total # of cells displayed
            _, keep = np.unique(np.linspace(0, limit - 1, len(index)).astype(int), return_index=True)
            lasagna.config.index = index.copy()
            lasagna.config.keep = keep
            index = index[keep]
            width = np.ceil(np.sqrt(len(index)))
            x = np.arange(len(index)) % width
            y = (np.arange(len(index)) / width).astype(int)
            axes.scatter(x, y, c=style.color)
            if len(x) > 0:
                axes.set_xlim(min(x) - 0.5, max(x) + 0.5)
                axes.set_ylim(max(y) + 0.5, min(y) - 0.5)

            if Fiji:
                files = source.categories[source.astype(int)][index]
                bounds = bounds.categories[bounds.astype(int)][index]
                lasagna.config.bounds = bounds
                data = grid_view(files, bounds, padding=padding)
                shape = data.shape
                data = lasagna.io.montage(data)
                self.imp = lasagna.io.show_IJ(data, imp=self.imp,
                                              luts=luts, display_ranges=display_ranges)

                if show_nuclei:
                    # offset contours to match grid spacing
                    offsets = (np.array([y, x]).T * shape[-2:]) + padding
                    # need to apply subset to unhashable, then sort order
                    c = contours.id._unhashable[contours.astype(int)][index]
                    contours_offset = np.array([x + y[:2] for x, y in zip(c, offsets)])
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
    roi = RectangularROI(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    xatt = lasagna.config.app.data_collection[0].data.find_component_id('x')
    yatt = lasagna.config.app.data_collection[0].data.find_component_id('y')
    xy_state = RoiSubsetState(xatt=xatt, yatt=yatt, roi=roi)
    file_state = CategorySubsetState(viewer.source_id, viewer.source_val)
    # selection only applies to data in displayed file
    subset_state = AndState(xy_state, file_state)

    data_collection = lasagna.config.app.data_collection

    # if no subset groups selected, make a new one
    layers = (lasagna.config.app._layer_widget.selected_layers())
    subset_groups = [s for s in layers if isinstance(s, SubsetGroup)]
    if len(subset_groups) == 0:
        global fiji_label
        new_label = 'Fiji %d' % fiji_label
        fiji_label += 1
        data_collection.new_subset_group(label=new_label, subset_state=subset_state)
    else:
        edit_mode = EditSubsetMode()
        lasagna.config.self.subset_state=subset_state
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
    from lasagna.io import subimage, pile, read_stack

    arr = []
    for filename, bounds_ in zip(files, bounds):
        I = read_stack(filename, memmap=False)
        I_cell = subimage(I, bounds_, pad=padding)
        arr.append(I_cell.copy())

    return pile(arr)


def pandas_to_glue(df, label='data', name_map=default_name_map):
    """Convert dataframe to glue.core.data.Data. Glue categorical variables require hashing,
    store array of unhashable components in ComponentID._unhashable. Override column names
    in name_map with dictionary values.

    """

    name_map = dict(name_map)
    data = Data(label=label)
    for c in df.columns:
        if c in name_map:
            c_name = name_map[c]
        else:
            c_name = str(c)
        try:
            data.add_component(df[c], c_name)
        except TypeError:
            # some fucking pd.factorize error with int list input to CategoricalComponent
            r = ['%09d' % i for i in range(len(df[c]))]
            cc = CategoricalComponent(r)
            c_id = ComponentID(c_name)
            c_id._unhashable = df[c]
            data.add_component(cc, c_id)
    return data


def lasagna_to_glue(df, label='data', name_map=default_name_map):

    data = pandas_to_glue(df, label='data', name_map=default_name_map)
    assert (data.get_component('x'))
    assert (data.get_component('y'))
    assert (data.get_component('contour'))
    assert (data.get_component('file'))
    assert (data.get_component('bounds'))
    return data


def map_barcode(digits, k, spacer=0.5):
    flag = False
    if len(digits) % 2:
        digits = list(digits) + [0]
        flag = True

    scale = 1. / k
    x, y = 0., 0.
    for x_, y_ in zip(digits[::2], digits[1::2]):
        x += x_ * scale
        y += y_ * scale
        scale /= k + spacer

    if flag:
        y /= k
    return x, y


def check_artist(artists, this_layer):
    """Try to figure out whether this is the first visible, enabled layer, based on
	status of artists/layers during custom viewer callback
	"""

    layers = [a.layer for a in artists]
    active_artists = [a for a in artists if a.enabled and a.visible]
    active_layers = [a.layer for a in active_artists]

    # 1. we are redrawing the first visible layer only
    # 2. redrawing all, this is the first visible layer (supposedly)
    # try:
    # 	j.ij.IJ.log('artists: %s' % artists)
    # except TypeError:
    # 	pass
    if active_artists:
        flag1 = this_layer == active_layers[0]
        flag2 = layers.index(this_layer) < min(layers.index(x) for x in active_layers)
        # j.ij.IJ.log('flag1: %s flag2: %s' % (flag1, flag2))
        flag3 = False
    else:
        flag3 = True
    # j.ij.IJ.log('flag3: %s' % flag3)
    # j.ij.IJ.log('this artist: %s' % (this_artist))
    # j.ij.IJ.log("---%s---" % style.parent.label)
    # for ca in self.widget.layers:
    # 	j.ij.IJ.log("%s: enabled (%s) visible (%s)" % (ca.layer.label, ca.enabled, ca.visible))

    return flag3 or flag2 or flag1


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
# 	lasagna.config.reset += [style.parent]~
# 	# reset overlay
# 	overlay = self.imp.getOverlay()
# 	if overlay:
# 		overlay.setStrokeColor(default_color())
# 	print 'reset on %s' % style.parent
# # lasagna.config.style += [style]


def debug_trace():
    '''Set a tracepoint in the Python debugger that works with Qt'''
    from PyQt4.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()
