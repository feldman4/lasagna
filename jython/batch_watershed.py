from ij import IJ, ImagePlus
from ij.measure import Measurements
from ij.process import ImageProcessor
from ij.plugin.filter import EDM, BackgroundSubtracter, Filters, ParticleAnalyzer
from ij.plugin import Thresholder
from mpicbg.ij.clahe import FastFlat
from fiji.threshold import Auto_Threshold
import os
from glob import glob
import java.io.File

SHOW_IMAGES = False

# job file with list of stacks is claimed by process, renamed before and after processing
job_dir = '/broad/blainey_lab/David/lasagna/jobs/'
job_name = glob(job_dir + 'watershed_job_*')[0]
progress_name, done_name = job_name.replace('job_', 'in_progress_'), job_name.replace('job_', 'done_')

fh = open(job_name, 'r')
stacks_to_process = fh.read().split('\n')
fh.close()

CLAHE = (('block_radius', 127), ('bins', 256), ('slope', 3), ('mask', None), ('composite', False))
bkgd_sub = (('radius', 40), ('createBackground', False), ('lightBackground', False),
			('useParabaloid', False), ('doPresmooth', False), ('correctCorners', False))

particle_analyzer = lambda (um_per_pixel): \
					(('options', ParticleAnalyzer.SHOW_MASKS | ParticleAnalyzer.IN_SITU_SHOW),
					 ('measurement options', 0),
					 ('results table', None),
					 ('min size', 40. / um_per_pixel**2),
					 ('max size', 300. / um_per_pixel**2),
					 ('min circularity', float(0.5)),
					 ('max circularity', float(1)) )

# rename job file, os.rename doesn't work in jython
def rename_file(f1, f2): return java.io.File(f1).renameTo(java.io.File(f2))
def savename(s): return '/'.join(s.split('/')[:-1] + ['nuclei'] + [s.split('/')[-1]])

print job_name
rename_file(job_name, progress_name)

for stack in stacks_to_process:
	# 1 - Obtain an image
	composite_img = IJ.openImage(stack)
	# Get the first DAPI slice
	dapi = composite_img.createImagePlus()
	composite_img.close()
	# happens to get the first slice
	dapi.setProcessor("nuclei", composite_img.getProcessor().duplicate())
	dapi_ip = dapi.getProcessor()
	
	if SHOW_IMAGES:
		dapi.show()
	
	# IJ.run("Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=3 mask=*None* fast_(less_accurate)");
	FastFlat.getInstance().run(dapi, *[a[1] for a in CLAHE])
	# IJ.run("Subtract Background...", "rolling=40");
	BS = BackgroundSubtracter()
	BS.rollingBallBackground(dapi_ip,
						*[a[1] for a in bkgd_sub])
	if SHOW_IMAGES:
		dapi.updateAndDraw()
	
	# IJ.setAutoThreshold("Otsu");
	# IJ.run("Convert to Mask");
	dapi_ip.autoThreshold()
	dapi.setProcessor(dapi_ip.convertToByteProcessor(False))
	dapi_ip = dapi.getProcessor()
	# IJ.run("Watershed");
	EDM().toWatershed(dapi_ip)
	dapi_ip.invert()
	
	# IJ.run("Analyze Particles...", "size=30-400 show=Masks");
	pa_options = particle_analyzer(dapi.getCalibration().pixelWidth)
	PA = ParticleAnalyzer(*[a[1] for a in pa_options])
	PA.analyze(dapi, dapi_ip)

	if SHOW_IMAGES:
		dapi.updateAndDraw()
	
	IJ.save(dapi, savename(stack))
	
	
	dapi.close()


rename_file(progress_name, done_name)
print 'done'
 	