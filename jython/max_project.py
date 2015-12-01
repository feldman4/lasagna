from ij import IJ, ImagePlus, WindowManager
from ij.measure import Calibration
import ij.io.FileSaver
import time, os
from glob import glob

home_dir = 'D:\\User Folders\\David\\lasagna\\20151122_96W-G020\\'

sub_dirs = ['60X_round4_3']


for sub_dir in sub_dirs:
	files = glob(home_dir + sub_dir + '\\*.tif')

	max_dir = home_dir + 'MAX\\' + sub_dir + '\\'

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