"""
To view help: python stitch.py -h
"""

import sys, os, time, shutil, re
import argparse
from collections import defaultdict

def force_symlink(src, dst):
    if os.path.isfile(dst):
        os.remove(dst)
    os.symlink(src, dst)

def make_tmp_dir():
    tmp_dir = os.environ['HOME'] + '/tmp_%s' % hash(time.time())
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir) 
    return tmp_dir

def make_tmp_tifs(images, tmp_dir, copy=False):
    file_pattern = '{i}.tif'
    files = []
    for i, image in enumerate(images):
        files += [tmp_dir + '/%d.tif' % i]
        if copy:
            from ij import IJ
            IJ.open(os.path.abspath(image))
            IJ.saveAs("Tiff", files[-1])
            IJ.run("Close All")
        else:
            force_symlink(os.path.abspath(image), files[-1])


    return files, file_pattern

def make_calibration(magnification):
    cal = Calibration()
    cal.setUnit('um')
    cal.pixelWidth  = args.magnification
    cal.pixelHeight = cal.pixelWidth
    return cal

def check_mag(m):
    mags = {
            '4X'  : 1.640 * 2,
            '10X' : 0.700 * 2,
            '20X' : 0.350 * 2,
            '40X' : 0.175 * 2,
            '60X' : 0.110 * 2,
            '100X': 0.066 * 2,
    }
    if m in mags:
        return mags[m]
    else:
        return float(m)

def check_lut(s): 
    luts = 'Grays', 'Blue', 'Green', 'Cyan', 'Red', 'Magenta', 'Yellow', 'glasbey'
    lower_luts = [l.lower() for l in luts]
    try:
        i = lower_luts.index(s.lower())
        return luts[i]
    except ValueError:
        raise ValueError('Color LUT must be one of: %s' % ', '.join(luts))

def check_order(order):
    order_strings = {'xyzct': 'xyzct',
                     'xyczt': 'xyczt(default)'}
    try:
        order_strings[order]
    except IndexError:
        raise IndexError('Order must be one of %s' % order_strings.keys())

def check_output_path(path):
    if os.path.isdir(path):
        return os.path.join(path, '') # trailing slash
    else:
        raise ValueError('Output path does not exist: %s' % path)
    

def make_template(well, data_dir):
    template_path = os.path.join(home_dir, data_dir, 
    'TileConfiguration_%s_%s.registered.txt' % (data_dir, well))
    fh = open(template_path, 'r')
    template = fh.read()
    fh.close()
    def f(new_well, new_data_dir):
        """Make template file and return path to it.
        """
        txt = template.replace(well, new_well).replace(data_dir, new_data_dir)
        txt_path = template_path.replace(well, new_well).replace(data_dir, new_data_dir)
        fh = open(txt_path, 'w')
        fh.write(txt)
        fh.close()
        return txt_path
    return f

def match_files(files, regex):
    matches = defaultdict(list)
    for f in files:
        match = re.findall(regex, f)
        if match:
            matches[match[-1]].append(f)
        else:
            print 'regex %s failed to match %s' % regex, f
    return sorted(matches.items())

def savename(well, data_dir):
    # TODO better naming convention, use Site_0?
    return home_dir + data_dir + '_MMStack_' + well + '.stitched.tif'


def macro_dir(s):
    """Wrap directory string so ImageJ macro accepts it as parameter.
    """
    return '[%s]' % (s.replace('\\', '\\\\'))

def stitch_cmd(grid, overlap, directory, file_pattern, config):
    s = """type=[Grid: row-by-row] order=[Right & Down                ]
    grid_size_x=%d grid_size_y=%d tile_overlap=%d first_file_index_i=0 directory=[%s]
    file_names=%s output_textfile_name=%s fusion_method=[Linear Blending]
    regression_threshold=0.30 max/avg_displacement_threshold=2.50
    absolute_displacement_threshold=3.50 compute_overlap subpixel_accuracy
    computation_parameters=[Save computation time (but use more RAM)]
    image_output=[Fuse and display]"""
    return s % (grid[0], grid[1], overlap, directory, file_pattern, config)

def stitch_from_file_cmd(layout_file_path):
    s = """type=[Positions from file] order=[Defined by TileConfiguration] 
    directory=[%s]" layout_file=[%s] fusion_method=[Linear Blending] 
    regression_threshold=0.30 max/avg_displacement_threshold=2.50 
    absolute_displacement_threshold=3.50  
    computation_parameters=[Save computation time (but use more RAM)] 
    image_output=[Fuse and display]"""
    return s % (os.path.dirname(layout_file_path) + '\\', os.path.basename(layout_file_path))
        


def parse_args():
    parser = argparse.ArgumentParser(
    description='Use Fiji\'s Grid/Collection Stitching plugin to stitch images or groups of images. Run from the command line: fiji gcs.py ARGS')

    parser.add_argument('images', type=str, nargs='+',
                        help='Images to stitch. If there are multiple groups, specify REGEX below.')

    parser.add_argument('output', type=check_output_path, 
                        help='Directory where stitched images are saved.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-t', '--template', type=str, default='',
                        help='Template file, produced by previous run.')

    group.add_argument('-k', '--reuse-template', action='store_true', 
                        help="""Create a template file from the first group of images 
                        and reuse coordinates on subsequent groups. 
                        If neither -t nor -k are set, each group of images is stitched 
                        independently.""")

    parser.add_argument('-r', '--regex', type=str, default='',
                        help='Regex that captures group name from input image name. E.g., "MMStack_(.*?)_"')

    parser.add_argument('-g', '--grid', type=int, nargs=2, default=(3,3),
                        help='(x,y) size of grid, e.g., "-g 3 3"')

    parser.add_argument('-m', '--magnification', type=check_mag, default='20X',
                        help='Pixel size in um/pixel, or magnification, e.g., -m 10X')
    
    parser.add_argument('-l', '--luts', type=check_lut, nargs='+', default=('Grays',),
                        help='Colors for each channel. Must be valid ImageJ LUTs, e.g., -l Grays Green Red.')

    parser.add_argument('-s', '--slices', type=int, default=1, 
                        help='Number of Z slices.')

    parser.add_argument('-f', '--frames', type=int, default=1,
                        help='Number of T frames.')

    parser.add_argument('-o', '--overlap', type=int, default=10, 
                        help='Percent overlap (0-100)')

    parser.add_argument('-e', '--order', type=check_order, default='xyzct',
                        help='Order of images in input stack. Options are "xyzct" (default) and "xyczt"')

    parser.add_argument('-p', '--pattern', type=str, default='row-by-row',
                        help='Grid layout, e.g., "row-by-row" or "snake by rows"')

    parser.add_argument('--ome', action='store_true', help='Use if input is in .ome format. Is automatically turned on if images end with "ome.tif".')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print out runtime information.')

    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    tmp_dir = make_tmp_dir()

    args = parse_args()
    # python gcs.py -h
    from ij import IJ, ImagePlus, WindowManager
    import ij.io.FileSaver
    from ij.measure import Calibration

    calibration = make_calibration(args.magnification)
    args.luts = args.luts + ('Grays',)*100

    if args.regex:
        groups = match_files(args.images, args.regex)
    else:
        groups = [('image', args.images)]

    print 'found groups', groups
    for group, group_images in groups:
        print 'processing group %s with %d images' % (group, len(group_images))
        output_image = os.path.join(args.output, group + '.stitched.tif')
        print 'saving output to %s' % output_image

        copy_flag = args.ome or any('.ome.tif' in s for s in group_images)
        files, file_pattern = make_tmp_tifs(group_images, tmp_dir, copy=copy_flag)

        if args.template:
            # symlink provided template, use to stitch
            tmp_template = tmp_dir + '/template.txt'
            force_symlink(os.path.abspath(args.template), tmp_template)
            cmd = stitch_template_cmd(tmp_template)
        else:
            # create a new template, this will be reused if -k
            tmp_template = 'TileConfiguration_%s.txt' % group
            cmd = stitch_cmd(args.grid, args.overlap, tmp_dir, 
                            file_pattern, tmp_template)

        print 'running %s' % cmd
        IJ.run("Grid/Collection stitching", cmd)

        if any(x > 1 for x in (len(args.luts), args.slices, args.frames)):
            IJ.run("Stack to Hyperstack...",
                   "order=%s channels=%d slices=%d frames=%d display=Composite" % 
                    (args.order, len(args.luts), args.slices, args.frames));

        ip = IJ.getImage()
        for i, color in enumerate(args.luts):
            ip.setC(i + 1)
            # ip.setDisplayRange(*display_range)
            IJ.run(color)

        if not args.template:
            # if we made a new template, save it
            registered = tmp_template.replace('.txt', '.registered.txt')
            registered = os.path.join(tmp_dir, registered)
            registered_keep = os.path.join(args.output, '%s.registered.txt' % group)
            shutil.copyfile(registered, registered_keep)

        ip.setCalibration(calibration)
        IJ.saveAs("Tiff", output_image)
        IJ.run("Close All")

        if args.reuse_template is True:
            args.template = tmp_template

    shutil.rmtree(tmp_dir)

    print 'Output written to %s in %d seconds.' % (args.output, time.time() - start_time)
