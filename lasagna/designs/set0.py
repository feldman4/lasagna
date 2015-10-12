import csv, time
import os
import pandas as pd
import numpy as np
import sys
from lasagna.design import *
from itertools import cycle, product
import Bio.SeqIO.AbiIO

# bottom strand
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
LHS, RHS = 'ctcagaACCGGT', rc('ctcagaGGTACC')  # contain AgeI, KpnI sites

colors = ('#F58A5E', '#FAAC61', '#FFEF86', '#F8D3A9', '#B1FF67', '#75C6A9', '#B7E6D7',
          '#85DAE9', '#84B0DC', '#9EAFD2', '#C7B0E3', '#FF9CCD', '#D6B295', '#D59687',
          '#B4ABAC', '#C6C9D1')

BsmBI_overhangs = 'ctcg', 'gcct', 'ggtg', 'tcgc', 'cgcc'
BsmBI_site = 'gacgatcCGTCTCc'

benchling_headers = ['name', 'sequence', 'type', 'color']

naming = {'reference': lambda x: 'ref%02d' % x,
          'tiles': lambda x: 'set0_t%02d' % x,
          'barcodes': lambda x: ('set0_' + '-'.join(['t%02d'] * len(x))) % tuple(x)
          }


class BarcodeSet(object):
    def __init__(self, probes, reference_probes, num_barcodes,
                 tiles_per_barcode, tile_size, name='set0', overhangs=BsmBI_overhangs):
        self.tile_size = tile_size
        self.tiles_per_barcode = tiles_per_barcode
        self.num_barcodes = num_barcodes
        self.reference_probes = reference_probes
        self.home = os.path.join(home, name)
        self.overhangs = overhangs

        self.refs = None
        self.tiles = None
        self.tiles_arr = None

        self.make_probes(probes)
        self.make_reference()
        self.make_tiles()

        self.ref_order = ['ref00', 'ref02', 'ref03', 'ref04', 'ref05', 'ref01']

        self.make_barcodes()

    def make_probes(self, probes):
        self.probes = pd.DataFrame({'sequence': probes})
        self.probes['name'] = ['set0_%03.d' % i for i, _ in enumerate(probes)]
        self.probes['color'] = [colors[(i / self.tile_size) % len(colors)]
                                for i, _ in enumerate(probes)]
        self.probes['type'] = 'stellaris (sense)'
        self.probes = self.probes[benchling_headers]

    def make_reference(self):
        self.refs = pd.DataFrame(self.reference_probes, columns=['sequence'])
        self.refs['name'] = [naming['reference'](i) for i, _ in self.refs.iterrows()]
        self.refs['color'] = 'tomato'
        self.refs['type'] = 'stellaris (sense) reference'
        self.refs = self.refs[benchling_headers]

    def make_tiles(self):
        probes = list(self.probes['sequence'])
        n = self.tiles_per_barcode * self.num_barcodes * self.tile_size
        tiles = [spacer.join(probes[i:i + self.tile_size]) for i in range(0, n, self.tile_size)]
        self.tiles_arr = np.array(tiles).reshape(self.num_barcodes, self.tiles_per_barcode)
        self.tiles = pd.DataFrame({'sequence': self.tiles_arr.flatten()})
        self.tiles['name'] = [naming['tiles'](i) for i, _ in self.tiles.iterrows()]
        self.tiles['color'] = 'gray'
        self.tiles['type'] = 'tile'
        self.tiles = self.tiles[benchling_headers]

    def make_barcode(self, tiles, refs=None, overhangs=None, code='<rstsrorstsrorstsr>'):
        if refs is None:
            refs = self.refs.set_index('name').loc[self.ref_order, 'sequence']
        overhangs = self.overhangs if overhangs is None else overhangs

        tokens = {'r': iter(refs),
                  't': iter(tiles),
                  'o': iter([rc(ov.upper()) for ov in overhangs]),
                  's': cycle([spacer]),
                  '<': cycle([rc(LHS)]),
                  '>': cycle([rc(RHS)])}

        return ''.join([rc(tokens[c].next()) for c in code])

    def make_barcodes(self):
        barcodes = []
        numbers = []
        for i in range(self.num_barcodes):
            barcodes += [self.make_barcode(self.tiles_arr[i])]
            numbers += [[i * self.tiles_per_barcode] + np.arange(self.tiles_per_barcode)]
        self.barcodes = pd.DataFrame({'sequence': barcodes})
        self.barcodes['name'] = [naming['barcodes'](x) for x in numbers]
        self.barcodes['color'] = 'white'
        self.barcodes['type'] = 'barcode'
        self.barcodes = self.barcodes[benchling_headers]

    def get_sequences(self):
        return pd.concat([self.probes, self.reference_probes, self.tiles, self.barcodes])

    def get_barcodes_fasta(self):
        return '\n'.join(['>%s\n%s' % (bc['name'], bc['sequence'])
                          for _, bc in self.barcodes.iterrows()])

    def score_seqs(self, seqs):
        """Find barcodes that best fit sequences.
        :param seqs:
        :return:
        """
        all_barcodes = []
        tiles = [self.tiles_arr[:6, i] for i in range(self.tiles_arr.shape[1])]
        for t1, t2, t3 in product(*tiles):
            all_barcodes += [self.make_barcode([t1, t2, t3])]

        best = []
        for seq in seqs:
            arr = []
            for barcode in all_barcodes:
                barcode = barcode.upper()
                arr += [sum(x in seq for x in barcode.split(rc(spacer)))]
            best += [np.argmax(arr)]
        return [all_barcodes[i] for i in best]






def load_abi(f):
    with open(f, 'rb') as fh:
        h = Bio.SeqIO.AbiIO.AbiIterator(fh).next()
        s = Bio.SeqIO.AbiIO._abi_trim(h)
    return str(s.seq)




