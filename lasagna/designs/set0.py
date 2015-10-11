import csv, time
import os
import pandas as pd
import numpy as np
import sys
from lasagna.design import *

spacer = 'TT'
home = '/broad/blainey_lab/David/lasagna/probes/stellaris/'




files = {'reference_probes': home + 'reference_probes.csv',
         'reference_probes_uncut': home + 'reference_probes_uncut.csv',
         'probes': home + '20150923_822_probes_ordered.csv',
         'probes_export': home + '20150923_probes_export.csv',
         'tiles_export': home + '20150923_tiles_export.csv',
         'barcodes_export': home + '20150923_barcodes.fasta',
         'reference_export': home + '20150923_reference_export.csv',
         'recombination_primers': home + '20150923_recombination_primers.csv'}

EcoRI = 'GAATTC'
LHS, RHS = 'ctcagaACCGGT', rc('ctcagaGGTACC') # contain AgeI, KpnI sites

colors = ('#F58A5E', '#FAAC61', '#FFEF86', '#F8D3A9', '#B1FF67', '#75C6A9', '#B7E6D7',
          '#85DAE9', '#84B0DC', '#9EAFD2', '#C7B0E3', '#FF9CCD', '#D6B295', '#D59687',
          '#B4ABAC', '#C6C9D1')

BsmBI_overhangs = 'ctcg', 'gcct', 'ggtg', 'tcgc', 'cgcc'
BsmBI_site = 'gacgatcCGTCTCc'

benchling_headers = ['name', 'sequence', 'type', 'color']

tile_size = 14
tiles_per_barcode = 3
num_barcodes = 18


naming = {'reference': lambda x: 'ref%2d' % x}

class BarcodeSet(object):
    def __init__(self, probes, reference_probes, name='set0'):
        self.home = os.path.join(home, name)
        self.probes = probes
        self.reference_probes = reference_probes



    def make_reference(self):
        self.refs = pd.DataFrame(self.reference_probes, columns='sequence')
        self.refs['name'] = [naming['reference'](i) for i, _ in self.iterrows()]




