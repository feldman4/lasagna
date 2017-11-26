from ij import IJ, ImagePlus, WindowManager
from ij.measure import Calibration
import ij.io.FileSaver
import time, os
from glob import glob

home_dir = 'D:\\David\\lasagna\\20171121_6W-133\\'
#home_dir = '//Users//feldman//Downloads//20151122_96W-G020//data

sub_dirs = ['20X_c0-DO_2']
filesep = '//'

# if acquisition is still running, this needs to be specified
channels, slices, stack = 2, 6, True
stack = True
rows, columns = 'ABCDEFGH', [str(x) for x in range(1, 13)]

wells = [r + c for r in rows for c in columns]

wells = [r + c for r in rows[:6] for c in columns]
print 'rows', rows[:6]
wells = ['B10', 'B11']

for sub_dir in sub_dirs:
    search = home_dir + sub_dir + filesep + '*.tif'
    print search
    files = glob(search)
    files = [f for f in files if any(well + '-' in f for well in wells)]
    max_dir = home_dir + 'MAX' + filesep + sub_dir + filesep

    if not os.path.isdir(max_dir):
        os.makedirs(max_dir)
    
    print files
    for f in files:
        print f
        imp = IJ.openImage(f)
        imp.show()
        if imp.getNSlices() == 12:
            IJ.run("Stack to Hyperstack...", "order=xyzct channels=%d slices=%d frames=1 display=Color" % (channels, slices));
        IJ.run("Z Project...", "projection=[Max Intensity]");
        
        imp.close()
        imp = IJ.getImage()
        cal = Calibration()
        imp.setCalibration(cal)
        print 'title', imp.title
        # ij.io.FileSaver(imp).saveAsTiff()
        ij.io.FileSaver(imp).saveAsTiffStack(max_dir + imp.title)
        imp.close()
        IJ.run("Close All");