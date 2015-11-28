from ij import IJ, ImagePlus, WindowManager
from ij.measure import Calibration
import ij.io.FileSaver
import time, os
from glob import glob

colors = ['Grays', 'Magenta', 'Cyan', 'Green']

imp = IJ.getImage()

for i, color in enumerate(colors): 
	imp.setC(i + 1)
	IJ.run(color)

IJ.run('8-bit')
IJ.run("RGB Color");
IJ.run("Invert");