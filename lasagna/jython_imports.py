"""
Functions written for jython interpreter in Fiji. Contains decorators to transparently
pickle arguments in cpython and unpickle in jython (mostly to accelerate transfer of
lists).

Example:
	client = rpyc_stuff.start_client()
	import jython_imports
	j = jython_imports.jython_import(client.root.exposed_get_head(),
	                                 client.root.exposed_execute)

	# use j.X to access packages, functions and classes of jython_import defined in jython
	# if a new attribute Y is defined, must be explicitly accessed once as j.Y before
	# it appears in introspection.

	# j is built from jython_import.__dict__ after importing module in jython interpreter

Issues:
- Doesn't preserve function signatures. Should be able to copy from cpython import; tried to use
decorator module but it has difficulties with optional arguments in jython. Jython methods can be
identified as crashing inspect.getargspec().

- Requires assignment to another module (e.g., lasagna.config.j = j) for global access. Would be
cleaner to override attributes in the cpython module itself with corresponding jython attributes.

"""

import sys
import functools

if sys.subversion[0] == 'Jython':
    import java.awt.event.MouseAdapter
    import java.awt.event.KeyAdapter
    import java.awt.Color
    import ij.gui.PolygonRoi
    import ij.gui.TextRoi
    import ij.gui.Overlay
    import ij.gui
    import ij.ImagePlus
    import ij.ImageStack
    import ij.process.ImageProcessor
    import ij.process.ByteProcessor
    import ij.process.ShortProcessor
    import ij.LookUpTable
    import ij.IJ
    import array
    import pickle


def UnPickler(*args_to_pickle):
    """Jython side, unpickles specified args. Couldn't get signatures preserved with
	jython (decorator module fails).
	"""
    def UnPickledDecorator(f):
        argnames = f.func_code.co_varnames

        @functools.wraps(f)
        def unpickled_f(*args, **kwargs):
            import pickle
            args_ = []
            for (x, name) in zip(args, argnames):
                if name in args_to_pickle:
                    args_ += [pickle.loads(x)]
                else:
                    args_ += [x]
            return f(*args_, **kwargs)

        unpickled_f._argnames = argnames
        unpickled_f._args_to_pickle = args_to_pickle
        return unpickled_f

    return UnPickledDecorator


def Pickler(f):
    """Cpython side, pickles corresponding arguments in already-decorated function.
	"""
    import numpy as np
    argnames = f._argnames
    args_to_pickle = f._args_to_pickle

    @functools.wraps(f)
    def pickled_f(*args, **kwargs):
        import pickle
        args_ = []
        for (x, name) in zip(args, argnames):
            if name in args_to_pickle:
                # no numpy on the jython side
                if isinstance(x, np.ndarray):
                    x = x.tolist()
                args_ += [pickle.dumps(x)]
            else:
                args_ += [x]
        return f(*args_, **kwargs)

    return pickled_f


def caller(f, *args, **kwargs):
    """Workaround for calling java functions from cpython with lists as arguments. Automatically
    converts these to jython lists, which are compatible with java arrays.
    :param f: function to call, e.g., j.ij.gui.PolygonRoi
    :param args:
    :param kwargs:
    :return:
    """
    args2 = []
    for arg in args:
        if str(type(arg)) == "<netref class '__builtin__.list'>":
            arg = list(arg)
        args2 += [arg]
    return f(*args2, **kwargs)


def show_polygon(x, y, imp=None, name=None):
    """
    show_polygon(x, y, imp=None, name=None)
    Displays a polygon Roi. Faster than storing output of ij.gui.PolygonRoi in
	cpython interpreter.
	"""
    if imp is None:
        imp = ij.IJ.getImage()
    imp.setRoi(make_polygon(x, y))


def make_polygon(x, y, name=None):
    """
    make_polygon(x, y, name=None)
    Displays a polygon ROI from the coordinates listed in x and y.
    :param x:
    :param y:
    :param name:
    :return:
    """
    x, y = list(x), list(y)
    poly = ij.gui.PolygonRoi(x, y, len(x), ij.gui.Roi.POLYGON)
    if name:
        poly.setName(name)
    return poly


@UnPickler('contours')
def overlay_contours(contours, names=None, imp=None, overlay=None):
    """overlay_contours(contours, names=None, imp=None, overlay=None)
	"""
    overlay = overlay or ij.gui.Overlay()
    imp = imp or ij.IJ.getImage()
    names = names or [None for _ in contours]

    arr = []
    for (y, x), name in zip(contours, names):
        poly = make_polygon(x, y)
        if name:
            poly.setName(name)
        overlay.add(poly)
        arr += [poly]

    imp.setOverlay(overlay)
    return overlay


@UnPickler('rois')
def set_overlay_contours_color(rois, imp, color):
    """set_overlay_contours_color(rois, imp, color)
    Set stroke color of existing ROIs based on integer index.
	"""
    overlay = imp.getOverlay()
    for i in rois:
        roi = overlay.get(int(i))
        if roi:
            roi.setStrokeColor(color)


def mouse_pressed(f):
    """Returns a listener that can be attached to java object with addMouseListener.
	"""

    class ML(java.awt.event.MouseAdapter):
        def mousePressed(self, event):
            pass

    listener = ML()
    listener.mousePressed = f
    return listener


def add_key_typed(f, imp, re_add=True):
    """ Removes existing KeyListeners on Canvas associated to provided ImagePlus.
    Attaches new KeyListener with provided callback. Reattaches old KeyListeners.
    """
    class KL(java.awt.event.KeyAdapter):
        def keyTyped(self, event):
            pass

    listener = KL()
    listener.keyTyped = f

    canvas = imp.getWindow().getCanvas()
    kls = canvas.getKeyListeners()
    [canvas.removeKeyListener(x) for x in kls]
    canvas.addKeyListener(listener)
    if re_add:
        [canvas.addKeyListener(x) for x in kls]


class Head(object):
    def __init__(self, head):
        # TODO: point directly to value in head
        # this is an issue because jython globals() is a stringmap, not dict
        self.__dict__ = dict(head)
        self._head = head

    def __getattr__(self, key):
        if key in self._head:
            self.__dict__.update(dict(self._bhead))
            return self.__dict__[key]
        raise KeyError(key)


def jython_import(head, executor):
    """Call from cpython. Causes jython interpreter to import this module, producing
    Head object with netrefs to both jython functions and Java in __main__ of jython server.
	"""
    import os
    module_path = os.path.dirname(__file__)
    executor('import sys\nsys.path.append("%s")' % module_path)
    executor('import jython_imports')
    executor('reload(jython_imports)')
    j = Head(head['jython_imports'].__dict__)
    # identify pickled functions and wrap with pre-pickler
    for key, val in j.__dict__.items():
        if isinstance(val, type(lambda: 0)):
            if '_args_to_pickle' in val.func_dict:
                setattr(j, key, Pickler(val))
    return j


def make_selection_listener(update_selection, viewer, queue_append, key='u'):
    """ Called from FijiViewer setup. Creates a callback to be attached to 
    KeyListener (?) on ImagePlus (?). The callback is executed in CPython.
    """
    def selection_listener(event):
        if event.getKeyChar().lower() == key:
            imp = event.getSource().getImage()
            roi = imp.getRoi()
            if roi:
                if roi.getTypeAsString() == 'Rectangle':
                    rect = roi.getBounds()
                    selection = (rect.getMinX(), rect.getMaxX(),
                                 rect.getMinY(), rect.getMaxY())

                    # add to lasagna.config.queue
                    queue_append([update_selection, ([selection, viewer], {})])

    return selection_listener
