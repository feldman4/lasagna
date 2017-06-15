from lasagna.imports import *

import IPython
IPython.get_ipython().magic('matplotlib inline')
IPython.get_ipython().magic('load_ext autoreload')
IPython.get_ipython().magic('autoreload 2')

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML

sns.set(style='white', font_scale=1.5)