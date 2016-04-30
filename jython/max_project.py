from ij import IJ, ImagePlus, WindowManager
from ij.measure import Calibration
import ij.io.FileSaver
import time, os
from glob import glob

home_dir = 'D:\\User Folders\\David\\lasagna\\20160428_96W-G032\\'
#home_dir = '//Users//feldman//Downloads//20151122_96W-G020//data

sub_dirs = ['60X_scan_4']
filesep = '//'

# if acquisition is still running, this needs to be specified
channels, slices, stack = 4, 5, True
stack = False
rows, columns = 'ABCDEFGH', [str(x) for x in range(1, 13)]

wells = [r + c for r in rows for c in columns]
wells = set(wells) - set(['A1', 'A2', 'A3'])

for sub_dir in sub_dirs:
	files = glob(home_dir + sub_dir + filesep + '*.tif')
	files = [f for f in files if any(well in f for well in wells)]
	max_dir = home_dir + 'MAX' + filesep + sub_dir + filesep

	if not os.path.isdir(max_dir):
		os.makedirs(max_dir)
	
	print files
	for f in files:
		print f
		imp = IJ.openImage(f)
		imp.show()
		if stack:
			IJ.run("Stack to Hyperstack...", "order=xyzct channels=%d slices=%d frames=1 display=Color" % (channels, slices));
		IJ.run("Z Project...", "projection=[Max Intensity]");
		
		imp.close()
		imp = IJ.getImage()
		cal = Calibration()
		imp.setCalibration(cal)
		ij.io.FileSaver(imp).saveAsTiff(max_dir + imp.title)
		imp.close()