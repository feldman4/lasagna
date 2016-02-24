# common imports for interactive work

import os
from natsort import natsorted
from collections import OrderedDict, Counter
from itertools import product
from glob import glob

import regex as re

import numpy as np
import pandas as pd
from pandas import IndexSlice as pdx
import skimage
np.mode = scipy.stats.mode

# import scipy.ndimage as ndi
# from scipy.ndimage.interpolation import zoom

# import matplotlib.pyplot as plt
import lasagna.io
import lasagna.process
import lasagna.utils
from lasagna.io import save_hyperstack as save
from lasagna.io import read_stack as read
from lasagna.io import show_hyperstack as show
import lasagna.config as config
from lasagna.utils import start_client









