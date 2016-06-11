from __future__ import absolute_import, division, print_function


from glue import custom_viewer
from glue.viewers.custom.qt import CustomViewer
import lasagna.glueviz

"""Declare any extra link functions like this"""
# @link_function(info='translates A to B', output_labels=['b'])
# def a_to_b(a):
#    return a * 3


"""Data factories take a filename as input and return a Data object"""
# @data_factory('JPEG Image')
# def jpeg_reader(file_name):
#    ...
#    return data


"""Extra qt clients"""
# qt_client(ClientClass)


# workaround due to glue's bizarre import and introspection scheme
# or maybe it allows live updates to callbacks by reloading lasagna.glueviz
class Fiji(CustomViewer):
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
                     Fiji=False,
                     show_nuclei=False,
                     source='att(file)',
                     contours='att(contour)',
                     bounds='att(bounds)',
                     sort_by='att',
                     padding=50,
                     limit=256)


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
    self.plot_subset_output = lasagna.glueviz.FijiGridViewer.plot_subset(self, axes, Fiji, show_nuclei, source,
                                                                         contours, bounds, sort_by, padding, limit,
                                                                         style)
