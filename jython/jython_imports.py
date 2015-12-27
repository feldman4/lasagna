def imports():
	"""Dummy function, do not call!
	"""
	import java.awt.event.MouseAdapter
	import ij.gui.PolygonRoi
	import ij.ImagePlus
	import ij.ImageStack
	import ij.process.ImageProcessor
	import ij.process.ByteProcessor
	import ij.process.ShortProcessor
	import ij.LookUpTable
	import array


def make_polygon(x, y, imp):
	"""Makes a polygon Roi. Faster than storing output of ij.gui.PolygonRoi in
	cpython interpreter.
	"""
	x, y = list(x), list(y)
	poly = ij.gui.PolygonRoi(x, y, len(x), ij.gui.Roi.POLYGON)
	imp.setRoi(poly)

def mouse_pressed(f):
	"""Returns a listener that can be attached to java object with addMouseListener.
	"""
	class ML(java.awt.event.MouseAdapter):
		def mousePressed(self, event):
			pass
	listener = ML()
	listener.mousePressed = f
	return listener


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


def extract_def(f):
	"""Extracts source for given function (must be defined in a file). If
	the function is named "imports", returns block of import statements only
	(useful for hiding java imports destined for jython interpreter).
	"""
	name = f.func_code.co_filename
	lineno = f.func_code.co_firstlineno
	with open(name, 'r') as fh:
		txt = fh.read().split('\n')
	out = [txt[lineno - 1]]
	for line in txt[lineno:]:
		if line and line[0] not in ' \t':
			break
		out += [line]
	
	if 'def imports():' in out[0]:
		out = [line.strip() for line in out]
		return '\n'.join(line for line in out if line.startswith('import '))
	return '\n'.join(out) + '\n'


def jython_import(head, executor, exclude=('extract_def', 'jython_import')):
	"""Top level function. Executes imports and function definitions from this
	module in namespace using executor function ("exec" in jython interpreter).
	Converts head (jython globals() dict) to object for easy access.
	Example:
		client = rpyc_stuff.start_client()
		import jython_imports
		j = jython_imports.jython_import(client.root.exposed_get_head(),
										 client.root.exposed_execute)
	"""
	module = globals()
	for key in module:
		field = module[key]
		if key[0] is not '_' and isinstance(field, type(lambda:0)):
			if key not in exclude:
				cmd = extract_def(module[key])
				executor(cmd)
	# h = Head()
	# [h.__setattr__(key, head[key]) for key in head]
	return Head(head)


