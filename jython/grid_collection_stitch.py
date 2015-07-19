# automate grid/collection stitching from tiled MM stacks
# channels are automatically set to 

from ij import IJ, ImagePlus, WindowManager
from glob import glob

# C
channel_luts = (('Grays', (400, 10000)))
channel_luts = (('Blue', (800, 50000)),
                ('Red', (800, 4000)),
                ('Magenta', (800, 20000)))
channels = len(channel_luts)
slices = 2  # Z
frames = 1;  # T
tiles = (5, 5)
overlap = 20  # %

nuclei_singleton = False

home_dir = 'D:\User Folders\David\lasagna\\20150716\\'
data_dirs = ['20X_readout_pre37wash_A2-C2_5x5_1', '20X_readout_pre37wash_B4-C4_5x5_1']


def savename(well, data_dir):
    # TODO better naming convention, use Site_0?
    return home_dir + data_dir + '_stitch_' + well + '-Site_0.tif'


def stitch_cmd(grid_size, overlap, directory, file_pattern):
    s = """type=[Grid: row-by-row] order=[Right & Down                ]
    grid_size_x=%d grid_size_y=%d tile_overlap=%d first_file_index_i=0 directory=[%s]
    file_names=%s output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending]
    regression_threshold=0.30 max/avg_displacement_threshold=2.50
    absolute_displacement_threshold=3.50 compute_overlap
    computation_parameters=[Save memory (but be slower)]
    image_output=[Fuse and display]"""
    return s % (grid_size[0], grid_size[1], overlap, directory, file_pattern)


rows = 'ABCDEFGH'
columns = '12345678'
wells = [r + c for r in rows for c in columns]

for data_dir in data_dirs:
    files = glob(home_dir + data_dir + '\*.tif')

    this_wells = [w for w in wells if any(w in x for x in files)]
    print 'wells to stitch:', this_wells
    for well in this_wells:
        file_pattern = [f for f in files if well + '-Site_0' in f][0].replace('Site_0', 'Site_{i}')
        file_pattern = file_pattern.split('\\')[-1]
        print file_pattern
        IJ.run("Grid/Collection stitching", stitch_cmd(tiles, overlap, home_dir + data_dir, file_pattern))

        if nuclei_singleton is True:
            ip = WindowManager.getActiveWindow().getImagePlus()
            ims, imp = ip.getImageStack(), ip.getProcessor()
            for _ in range(slices):
                ims.addSlice('', imp, 0)

        if any(x > 1 for x in (channels, slices, frames)):
            IJ.run("Stack to Hyperstack...",
                   "order=xyzct channels=%d slices=%d frames=%d display=Composite" % (channels, slices, frames));
        ip = WindowManager.getActiveWindow().getImagePlus()
        for i, (color, display_range) in enumerate(channel_luts):
            ip.setC(i + 1)
            ip.setDisplayRange(*display_range)
            IJ.run(color)
        IJ.saveAs("Tiff", savename(well, data_dir))
        IJ.run("Close All")

print 'completed without error'
