"""
Functions written for jython interpeter in Fiji. Contains decorators to transparently
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

import functools 
import sys

if sys.subversion[0] == 'Jython':
	import java.awt.event.MouseAdapter
	import java.awt.event.KeyAdapter
	import java.awt.Color
	import ij.gui.PolygonRoi
	import ij.gui.Overlay
	import ij.ImagePlus
	import ij.ImageStack
	import ij.process.ImageProcessor
	import ij.process.ByteProcessor
	import ij.process.ShortProcessor
	import ij.LookUpTable
	import ij.IJ
	import array
	import pickle
	import functools


def UnPickler(*args_to_pickle):
	"""Jython side, unpickles specified args. Couldn't get signatures preserved with
	jython (decorator module failed while working in cpython).
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


def show_polygon(x, y, imp=None, name=None):
	"""Displays a polygon Roi. Faster than storing output of ij.gui.PolygonRoi in
	cpython interpreter.
	"""
	if imp is None:
		imp = ij.IJ.getImage()
	imp.setRoi(make_polygon(x,y))

def make_polygon(x, y, name=None):
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

	for (y, x), name in zip(contours, names):
		poly = make_polygon(x, y)
		if name:
			poly.setName(name)
		overlay.add(poly)

	imp.setOverlay(overlay)

@UnPickler('rois')
def set_overlay_contours_color(rois, imp, color):
	overlay = imp.getOverlay()
	for i in rois:
		overlay.get(int(i)).setStrokeColor(color)
	

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
		#TODO: point directly to value in head
		# this is an issue because jython globals() is a stringmap, not dict
		self.__dict__ = dict(head)
		self._head = head
	def __getattr__(self, key):
		if key in self._head:
			self.__dict__.update(dict(self._bhead))
			return self.__dict__[key]
		raise KeyError(key)


def jython_import(head, executor):
	"""Call from cpython.
	"""
	import os
	module_path = os.path.dirname(__file__)
	executor('import sys\nsys.path.append("%s")' % module_path)
	executor('import jython_imports')
	j = Head(head['jython_imports'].__dict__)
	# identify pickled functions and wrap with pre-pickler
	for key, val in j.__dict__.items():
		if isinstance(val, type(lambda:0)):
			if '_args_to_pickle' in val.func_dict:
				setattr(j, key, Pickler(val))
	return j



# def jython_import(head, executor, exclude=('extract_def', 'jython_import', 'PrePickled')):
# 	"""Top level function. Executes imports and function definitions from this
# 	module in namespace using executor function ("exec" in jython interpreter).
# 	Converts head (jython globals() dict) to object for easy access.
# 	Example:
# 		client = rpyc_stuff.start_client()
# 		import jython_imports
# 		j = jython_imports.jython_import(client.root.exposed_get_head(),
# 										 client.root.exposed_execute)
# 	"""
# 	module = globals()
# 	with open(__file__, 'r') as fh:
# 		txt = fh.read()
# 	keys = [key for key in module.keys() if not key.startswith('_')]
# 	keys = sorted(keys, key=lambda s: txt.index(s))

# 	# imports go first
# 	cmd = extract_def(module['imports'])
# 	executor(cmd)

# 	head = dict(head)
# 	for key in keys:
# 		field = module[key]
# 		if key[0] is not '_' and isinstance(field, (type, type(lambda:0))):
# 			if key not in exclude:
# 				cmd = extract_def(module[key])
# 				executor(cmd)
# 				if cmd.startswith('@SuperPickled'):
# 					head[key] = PrePickledDecorator(head[key], globals()[key])

# 	return Head(head)


# def extract_def(f):
# 	"""Extracts source for given function (must be defined in a file). If
# 	the function is named "imports", returns block of import statements only
# 	(useful for hiding java imports destined for jython interpreter).
# 	"""
# 	decorators = []
# 	# class, assumes it's defined in this file
# 	name = __file__.replace('pyc', 'py')
# 	with open(name, 'r') as fh:
# 		txt = fh.read().split('\n')
# 	if isinstance(f, type(lambda:0)):
# 		# function, co_firstlineno doesn't work with decorators
# 		# name = f.func_code.co_filename
# 		# lineno = f.func_code.co_firstlineno
# 		with open(name, 'r') as fh:
# 			txt = fh.read().split('\n')
# 		lineno = [line.startswith('def %s' % f.__name__) for line in txt].index(True)
# 		lineno += 1
# 		dummy = lineno - 2
# 		while txt[dummy].startswith('@'):
# 			decorators += [txt[dummy]]
# 			dummy -= 1
# 		decorators = decorators[::-1]

# 	elif isinstance(f, type):

# 		# doesn't capture decorators
# 		lineno = [line.startswith('class %s' % f.__name__) for line in txt].index(True)
# 		lineno += 1

# 	out = decorators + [txt[lineno - 1]]
# 	for line in txt[lineno:]:
# 		if line and line[0] not in ' \t':
# 			break
# 		out += [line]
	
# 	if 'def imports():' in out[0]:
# 		out = [line.strip() for line in out]
# 		return '\n'.join(line for line in out if line.startswith('import '))
# 	return '\n'.join(out) + '\n'


