# automate grid/collection stitching from tiled MM stacks
# channels are automatically set to 

from ij import IJ, ImagePlus, WindowManager
import ij.io.FileSaver
import time, os
from glob import glob

# C
channel_luts = (('Grays', (400, 8000)),)
channel_luts = (('Blue', (600, 2000)), 
				('Green', (800,3500)),
				('Red', (800,3500)),
                ('Magenta', (800, 3500)))
#channel_luts = (('Blue', (400, 60000)),
#				('Green', (400, 1200)))
                
channels = len(channel_luts)
slices = 1  # Z
frames = 1;  # T
# 40X
tiles, overlap = (5, 5), int(100*(1. - 300./350))
# 4X
# tiles, overlap = (3, 3), int(100*(1. - 1800./3379))
print tiles, overlap
nuclei_singleton = False

if False:
    filesep = '/'
    home_dir = '/broad/blainey_lab/David/lasagna/20150817 6 round/analysis/calibrated/raw/'
else:
    home_dir = 'D:\\User Folders\\David\\lasagna\\20150817\\to_stitch\\'
    # home_dir = '\\\\neon-cifs\\blainey_lab\\David\\lasagna\\20150817 6 round\\analysis\\calibrated\\raw\\'
    filesep = '\\'


data_dirs = ['40X_round4_1']


def savename(well, data_dir):
    # TODO better naming convention, use Site_0?
    return home_dir + data_dir + '_stitch_' + well + '-Site_0.tif'


def stitch_cmd(grid_size, overlap, directory, file_pattern):
    s = """type=[Grid: row-by-row] order=[Right & Down                ]
    grid_size_x=%d grid_size_y=%d tile_overlap=%d first_file_index_i=0 directory=[%s]
    file_names=%s output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending]
    regression_threshold=0.30 max/avg_displacement_threshold=2.50
    absolute_displacement_threshold=3.50 compute_overlap subpixel_accuracy
    computation_parameters=[Save computation time (but use more RAM)]
    image_output=[Fuse and display]"""
    return s % (grid_size[0], grid_size[1], overlap, directory, file_pattern)


rows = 'ABCDEFGH'
columns = '123456789'

wells = [r + c for r in rows for c in columns]

for data_dir in data_dirs:
    print home_dir + data_dir + filesep + '*.tif'
    files = glob(home_dir + data_dir + filesep + '*.tif')
#    print files
    this_wells = [w for w in wells if any(w in x for x in files)]
    print 'wells to stitch:', this_wells
    for well in this_wells:
        file_pattern = [f for f in files if well + '-Site_0' in f][0].replace('Site_0', 'Site_{i}')
        file_pattern = file_pattern.split(filesep)[-1]
        print file_pattern
        IJ.run("Grid/Collection stitching", stitch_cmd(tiles, overlap, home_dir + data_dir, file_pattern))

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
        IJ.saveAs("Tiff", savename(well, data_dir))
        IJ.run("Close All")

print 'completed without error'
