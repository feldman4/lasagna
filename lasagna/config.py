import os

up = os.path.dirname
home = up(up(__file__))
paths = None
calibration = None
calibration_short = None
visitor_font = '/broad/blainey_lab/David/packages/lasagna/resources/fonts/visitor1.ttf'
fonts = os.path.join(home, 'resources', 'fonts')
luts = os.path.join(home, 'resources', 'luts')