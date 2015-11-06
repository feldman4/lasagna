from ij import IJ, ImagePlus, WindowManager
from ij.measure import Calibration
import ij.io.FileSaver
import time, os
from glob import glob

home_dir = 'D:\\User Folders\\David\\lasagna\\20151103_96W-G015\\'

sub_dirs = ['100X_round1_1', '100X_round1_2', '100X_round1_3', '100X_round1_6', '100X_round1_8', '100X_round1_9']

max_dir = home_dir + '100X_MAX\\'

for sub_dir in sub_dirs:
	files = glob(home_dir + sub_dir + '\\*.tif')
	
	
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