from ij import IJ
from glob import glob 

rows = 'ABCDEF'
columns='12345'
wells = []
for r in rows:
	for c in columns:
		wells.append(r+c)

home_dir = 'D:\User Folders\David\lasagna\\20150711 smurf ish\\'
data_dirs = ['stitch intermediate']

channels, slices, frames = 2, 2, 8

for dd in data_dirs:
	files = glob(home_dir + dd + '\*.tif')
	
	this_wells = [w for w in wells if any(w in x for x in files)]
	print this_wells
	for well in this_wells:
		cmd = "open=[%s] file=%s_stitch sort" % (files[0], well)
		print cmd
		IJ.run("Image Sequence...", cmd);
		cmd_StH = "order=xyczt(default) channels=%d slices=%d frames=%d display=Composite"
		IJ.run("Stack to Hyperstack...", cmd_StH % (channels, slices, frames));
		IJ.saveAs("Tiff", home_dir + '%s_stack.tif' % well);
		IJ.run("Close All");