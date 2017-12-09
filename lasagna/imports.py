# common imports for interactive work

import os
from natsort import natsorted
from collections import OrderedDict, Counter, defaultdict
from itertools import product
from glob2 import glob
from functools import partial

import regex as re

import numpy as np
import pandas as pd
from pandas import IndexSlice as pdx
import skimage
import scipy.stats
np.mode = scipy.stats.mode

import lasagna.io
import lasagna.process
import lasagna.utils
from lasagna.utils import standardize, int_mode
from lasagna.io import save_stack as save
from lasagna.io import read_stack as read
from lasagna.io import show_IJ as show
from lasagna.io import grab_image as grab
from lasagna.io import BLUE, GREEN, RED, MAGENTA, GRAY, CYAN, GLASBEY 
from lasagna.io import pile, montage, grid_view, tile
from lasagna.io import parse_filename as parse
from lasagna.io import name
from lasagna.process import register_images, trim
import lasagna.config as config
from lasagna.utils import start_client







