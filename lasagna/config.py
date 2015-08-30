import os

up = os.path.dirname
home = up(up(__file__))
paths = None
calibration = None
calibration_short = None

fonts = os.path.join(home, 'resources', 'fonts')
luts = os.path.join(home, 'resources', 'luts')

visitor_font = os.path.join(fonts, 'visitor1.ttf')
