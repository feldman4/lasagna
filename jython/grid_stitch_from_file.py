
# automate grid/collection stitching from tiled MM stacks
# channels are automatically set to 

from ij import IJ, ImagePlus, WindowManager
import ij.io.FileSaver
from ij.measure import Calibration
import time, os
from glob import glob

# C
#channel_luts = (('Grays', (400, 8000)),)
channel_luts = (('Blue', (400, 50000)),
                ('Green', (2000, 5000)),
                ('Red', (800, 5000)),
                ('Magenta', (800, 5000)))
channel_luts = (('Grays', (400, 60000)),
				('Blue', (400, 4000)))

channels = len(channel_luts)
slices = 1  # Z
frames = 43;  # T

#### 40X
tiles, overlap = (5, 5), int(100 * (1. - 300. / 350))
pixel_width = 0.175 * 2

#### 20X
tiles, overlap = (3, 3), int(100 * (1. - 500. / 750))
pixel_width = 0.35 * 2

### 4X
#tiles, overlap = (3, 3), int(100*(1. - 1800./3379))
#pixel_width = 1.64

### 100X
#tiles, overlap = (7, 7), int(100*(1. - 100./135))
#pixel_width = 0.066 * 2

### 60X
#tiles, overlap = (7, 7), int(100*(1. - 200./225.3))
#pixel_width = 0.110 * 2

print tiles, overlap
nuclei_singleton = False

if True:
	# osx, unix
    filesep = '/'
    home_dir = '/broad/blainey_lab/David/lasagna/20150817 6 round/data/'
    home_dir = '/Users/feldman/Downloads/20151209/'
else:
	# windows
    home_dir = 'D:\\User Folders\\David\\lasagna\\20151122_96W-G020\\'
    # home_dir = '\\\\neon-cifs\\blainey_lab\\David\\lasagna\\20150817 6 round\\analysis\\calibrated\\raw\\'
    filesep = '\\'

data_dirs = ['stack']

cal = Calibration()
cal.setUnit('um')
cal.pixelWidth = pixel_width
cal.pixelHeight = pixel_width

def savename(well, data_dir):
    # TODO better naming convention, use Site_0?
    return home_dir + data_dir + '_MMStack_' + well + '.stitched.tif'


def stitch_cmd_file(directory, layout_file):
	s = """type=[Positions from file] order=[Defined by TileConfiguration] directory=%s layout_file=%s fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]"""
	return s % (directory, layout_file)

for data_dir in data_dirs:
    print home_dir + data_dir + filesep + '*.registered.tif'
    files = glob(home_dir + data_dir + filesep + '*.registered.txt')
    files = [f.split(filesep)[-1] for f in files]
	
    print 'wells to stitch:', '\n'.join(files)
    for filename in files:
    	
        IJ.run("Grid/Collection stitching", stitch_cmd_file(home_dir + data_dir + filesep, filename))

        if any(x > 1 for x in (channels, slices, frames)):
            print channels, slices, frames
            IJ.run("Stack to Hyperstack...",
                   "order=xyzct channels=%d slices=%d frames=%d display=Composite" % (channels, slices, frames));

        ip = IJ.getImage()
        for i, (color, display_range) in enumerate(channel_luts):
            ip.setC(i + 1)
            ip.setDisplayRange(*display_range)
            IJ.run(color)

        #        ij.io.FileSaver(ip).saveAsTiff(savename(well, data_dir))
        ip.setCalibration(cal)
        IJ.saveAs("Tiff", home_dir + filename.replace('registered.txt', 'stitched.tif'))
        IJ.run("Close All")

print 'completed without error'
