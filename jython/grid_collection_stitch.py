# automate grid/collection stitching from tiled MM stacks
# channels are automatically set to 

from ij import IJ, ImagePlus, WindowManager
import ij.io.FileSaver
from ij.measure import Calibration
import time, os
from glob import glob

if True:
    # osx, unix
    filesep = '/'
    home_dir = '/broad/blainey_lab/David/lasagna/20150817 6 round/data/'
    home_dir = '/Users/feldman/Downloads/20151219_96W-G024/data/'
else:
    # windows
    home_dir = 'D:\\User Folders\\David\\lasagna\\20151219_96W-G024\\'
    # home_dir = '\\\\neon-cifs\\blainey_lab\\David\\lasagna\\20150817 6 round\\analysis\\calibrated\\raw\\'
    filesep = '\\'

# C
#channel_luts = (('Grays', (400, 8000)),)
channel_luts = (('Blue', (400, 50000)),
                ('Green', (2000, 8000)),
                ('Red', (800, 8000)),
                ('Magenta', (800, 8000)))
#channel_luts = (('Grays', (400, 60000)),
#               ('Blue', (400, 4000)))
#channel_luts = (('Grays', (400, 40000)),)

channels = len(channel_luts)
slices = 2  # Z
frames = 1;  # T

#### 40X
tiles, overlap = (4, 4), int(100 * (1. - 300. / 350))
pixel_width = 0.175 * 2

##### 20X
#tiles, overlap = (3, 3), int(100 * (1. - 500. / 750))
#pixel_width = 0.35 * 2

### 4X
#tiles, overlap = (3, 3), int(100*(1. - 1800./3379))
#pixel_width = 1.64

### 100X
#tiles, overlap = (7, 7), int(100*(1. - 100./135))
#pixel_width = 0.066 * 2

### 60X
#tiles, overlap = (3, 3), int(100*(1. - 200./225.3))
#pixel_width = 0.110 * 2

print tiles, overlap
nuclei_singleton = False

use_template = True
template = None

data_dirs = ['40X_round3_1']

# usually xyzct, except on bad days when it's xyczt(default)
order = 'xyzct'
#order = 'xyczt(default)'

rows = 'ABCDEFGH'
columns = [str(x) for x in range(1, 13)]

wells = [r + c for r in rows for c in columns]
wells = ['A9']


cal = Calibration()
cal.setUnit('um')
cal.pixelWidth = pixel_width
cal.pixelHeight = pixel_width


def savename(well, data_dir):
    # TODO better naming convention, use Site_0?
    return home_dir + data_dir + '_MMStack_' + well + '.stitched.tif'

def tile_config_name(well, data_dir):
    return 

def make_template(well, data_dir):
    template_path = os.path.join(home_dir, data_dir, 
    'TileConfiguration_%s_%s.registered.txt' % (data_dir, well))
    fh = open(template_path, 'r')
    template = fh.read()
    fh.close()
    def f(new_well, new_data_dir):
        txt = template.replace(well, new_well).replace(data_dir, new_data_dir)
        txt_path = template_path.replace(well, new_well).replace(data_dir, new_data_dir)
        fh = open(txt_path, 'w')
        fh.write(txt)
        fh.close()
        return txt_path
    return f

def stitch_cmd(grid_size, overlap, directory, file_pattern, config):
    s = """type=[Grid: row-by-row] order=[Right & Down                ]
    grid_size_x=%d grid_size_y=%d tile_overlap=%d first_file_index_i=0 directory=[%s]
    file_names=%s output_textfile_name=TileConfiguration_%s.txt fusion_method=[Linear Blending]
    regression_threshold=0.30 max/avg_displacement_threshold=2.50
    absolute_displacement_threshold=3.50 compute_overlap subpixel_accuracy
    computation_parameters=[Save computation time (but use more RAM)]
    image_output=[Fuse and display]"""
    return s % (grid_size[0], grid_size[1], overlap, directory, file_pattern, config)

def stitch_from_file_cmd(layout_file_path):
    s = """type=[Positions from file] order=[Defined by TileConfiguration] 
    directory=%s layout_file=%s fusion_method=[Linear Blending] 
    regression_threshold=0.30 max/avg_displacement_threshold=2.50 
    absolute_displacement_threshold=3.50  
    computation_parameters=[Save computation time (but use more RAM)] 
    image_output=[Fuse and display]"""
    return s % (os.path.dirname(layout_file_path), os.path.basename(layout_file_path))
        
### MAIN LOOP ###

for data_dir in data_dirs:
    print home_dir + data_dir + filesep + '*.tif'
    files = glob(home_dir + data_dir + filesep + '*.tif')
    #    print files
    this_wells = [w for w in wells if any(w in x for x in files)]
    print 'wells to stitch:', this_wells
    for well in this_wells:
        print sorted(files)[0]
        file_pattern = [f for f in files if well + '-Site_0' in f][0].replace('Site_0', 'Site_{i}')
        print file_pattern
        file_pattern = file_pattern.split(filesep)[-1]
        print file_pattern
        config = data_dir + '_' + well

        if template:
            txt_path = template(well, data_dir)
            IJ.run("Grid/Collection stitching", stitch_from_file_cmd(txt_path))
        else:
            IJ.run("Grid/Collection stitching", stitch_cmd(tiles, overlap, home_dir + data_dir, file_pattern, config))
		
        if any(x > 1 for x in (channels, slices, frames)):
            print channels, slices, frames
            IJ.run("Stack to Hyperstack...",
                   "order=%s channels=%d slices=%d frames=%d display=Composite" % (order, channels, slices, frames));

        ip = IJ.getImage()
        for i, (color, display_range) in enumerate(channel_luts):
            ip.setC(i + 1)
            ip.setDisplayRange(*display_range)
            IJ.run(color)

        ip.setCalibration(cal)
        IJ.saveAs("Tiff", savename(well, data_dir))
        IJ.run("Close All")

        if use_template is True:
            template = make_template(well ,data_dir)

print 'completed without error'
