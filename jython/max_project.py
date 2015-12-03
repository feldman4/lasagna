from ij import IJ, ImagePlus, WindowManager
from ij.measure import Calibration
import ij.io.FileSaver
import time, os
from glob import glob

home_dir = 'D:\\User Folders\\David\\lasagna\\20151122_96W-G020\\'
home_dir = '//Users//feldman//Downloads//20151122_96W-G020//data

sub_dirs = ['60X_round4_3']

filesep = '//'

for sub_dir in sub_dirs:
	files = glob(home_dir + sub_dir + filesep + '*.tif')

	max_dir = home_dir + 'MAX' + filesep + sub_dir + filesep

	if not os.path.isdir(max_dir):
		os.makedirs(max_dir)
	
	print files
	for f in files:
		print f
		imp = IJ.openImage(f)
		imp.show()
		IJ.run("Z Project...", "projection=[Max Intensity]");
		imp.close()
		imp = IJ.getImage()
		cal = Calibration()
		imp.setCalibration(cal)
		ij.io.FileSaver(imp).saveAsTiff(max_dir + imp.title)
		imp.close()