import time, os
import colour
import pandas as pd
from lasagna.design import *
from itertools import cycle, product
import Bio.SeqIO.AbiIO
from collections import defaultdict

# bottom strand
spacer = 'TT'
home = '/broad/blainey_lab/David/lasagna/probes/stellaris/set0/'

files = {'reference_probes': home + 'reference_probes.csv',
         'reference_probes_uncut': home + 'reference_probes_uncut.csv',
         'probes': home + '20150923_822_probes_ordered.csv',
         'probes_export': home + '20150923_probes_export.csv',
         'tiles_export': home + '20150923_tiles_export.csv',
         'barcodes_export': home + '20150923_barcodes.fasta',
         'reference_export': home + '20150923_reference_export.csv',
         'recombination_primers': home + '20150923_recombination_primers.csv',
         'BTI_order': home + 'BTI_Oligo_Order-Columns_20151007.csv'}

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


def load_set0_probes():
    reference_probes = open(files['reference_probes']).read().splitlines()
    probes = open(files['probes']).read().splitlines()
    probes = [p.split(',')[1] for p in probes]
    probes = [p for p in probes if p not in reference_probes]
    print 'loaded %d probes, %d reference probes' % (len(probes), len(reference_probes))
    return probes, reference_probes


class BarcodeSet(object):
    def __init__(self, probes, reference_probes, num_barcodes,
                 tiles_per_barcode, tile_size, name='set0', overhangs=BsmBI_overhangs):
        """Describe set of barcodes assembled from probes and reference probes. Tiles correspond to probe
        sets, and are recombined to create barcodes. Cloning uses directional overhangs between tiles.
        Reference probes at the boundaries of tiles are used for cloning, FISH quality control, and RT-qPCR.

        Sanger sequencing can be matched to barcodes using the score_seqs method.

        :param probes: list of probes, at least M x N x P
        :param reference_probes: conserved sequence at tile edges, typically need 2 * N
        :param num_barcodes: M barcodes to generate
        :param tiles_per_barcode: N tiles per barcode, typically between 3 and 6
        :param tile_size: P probes per tile, typically between 8 and 16
        :param name:
        :param overhangs: used for directional cloning, typically 4 bp, 50% GC, and non-palindromic
        :return:
        """
        self.tile_size = tile_size
        self.tiles_per_barcode = tiles_per_barcode
        self.num_barcodes = num_barcodes
        self.overhangs = overhangs
        self.tile_colors = 'purple', 'lightpink'

        self.refs = None
        self.tiles = None
        self.tiles_arr = None

        self.make_probes(probes)
        self.make_reference(reference_probes)
        self.make_tiles()

        self.ref_order = ['ref00', 'ref02', 'ref03', 'ref04', 'ref05', 'ref01']

        self.load_order(files['BTI_order'])

        self.make_barcodes()

    def load_order(self, filepath):
        self.ordered = pd.read_csv(filepath)

        probes2tiles=defaultdict(str)
        probes2tiles.update({p: row['name'] for _,row in self.tiles.iterrows()
                       for p in row['probes']})

        self.ordered['tile'] = [probes2tiles[p] for p in self.ordered['Sequence Name']]

    def save_probe_layout(self, dirpath=None):
        """Save probe and tile layouts on ordered plate to .xls, located in dirpath
        (defaults to set0.home).
        """
        dirpath = dirpath or home
        print dirpath
        plates_by_probe = {}
        plates_by_tile = {}
        for plate, df in self.ordered.groupby('Plate Name'):
            plates_by_probe[plate] = df.pivot_table(values='Sequence Name', index='Row', columns='Column', 
                                   aggfunc=lambda x: x)
            plates_by_tile[plate] = df.pivot_table(values='tile', index='Row', columns='Column', 
                                   aggfunc=lambda x: x)
        with pd.ExcelWriter(dirpath + 'plates_by_probe.xls') as writer:
            for name, plate in plates_by_probe.items():
                plate.to_excel(writer, sheet_name=name)
            
        with pd.ExcelWriter(dirpath + 'plates_by_tile.xls') as writer:
            for name, plate in plates_by_tile.items():
                plate.to_excel(writer, sheet_name=name)



    def make_probes(self, probes):
        """Turn iterable of probe sequences into named entries in table, benchling format. Color
        probes by parent tile.
        :param probes:
        :return:
        """
        self.probes = pd.DataFrame({'sequence': probes})
        self.probes['name'] = ['set0_%03.d' % i for i, _ in enumerate(probes)]
        self.probes['color'] = [colors[(i / self.tile_size) % len(colors)]
                                for i, _ in enumerate(probes)]
        self.probes['type'] = 'stellaris (sense)'
        self.probes = self.probes[benchling_headers]

    def make_reference(self, reference_probes):
        """Turn iterable of reference probes into named entries in table, benchling format.
        :return:
        """
        self.refs = pd.DataFrame(reference_probes, columns=['sequence'])
        self.refs['name'] = [naming['reference'](i) for i, _ in self.refs.iterrows()]
        self.refs['color'] = 'tomato'
        self.refs['type'] = 'stellaris (sense) reference'
        self.refs = self.refs[benchling_headers]

    def make_tiles(self):
        """Use current values in tiles_per_barcode, num_barcodes, and tile_size fields to make
        a numpy.ndarray of tile sequences, and a table in benchling format. Color tiles by position
        in barcode.
        :return:
        """
        index = np.arange(self.tiles_per_barcode * self.num_barcodes * self.tile_size)
        index = index.reshape(-1, self.tile_size)

        probes = np.array(self.probes['sequence'])[index]
        tiles = [spacer.join(x) for x in probes]
        self.tiles_arr = np.array(tiles).reshape(self.num_barcodes, self.tiles_per_barcode)

        self.tiles = pd.DataFrame({'sequence': self.tiles_arr.flatten()})
        self.tiles['name'] = [naming['tiles'](i) for i, _ in self.tiles.iterrows()]

        color_cycle = cycle(colour.Color(self.tile_colors[0])
                            .range_to(colour.Color(self.tile_colors[1]), self.tiles_per_barcode))

        self.tiles['color'] = [c.get_hex_l() for c, _ in zip(color_cycle, range(self.tiles.shape[0]))]
        self.tiles['type'] = 'tile'
        self.tiles['probes'] = [tuple(row) for row in np.array(self.probes['name'])[index]]


    def make_barcode(self, tiles, refs=None, overhangs=None, code='<rstsrorstsrorstsr>'):
        """Make a single barcode from given tiles according to code. Tokens are:
        r = ref
        t = tile
        o = overhang
        s = spacer (e.g., AA)
        <, > = left- and right-hand sides

        :param tiles:
        :param refs:
        :param overhangs:
        :param code:
        :return:
        """
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
        """Generate all sequential barcodes (e.g., tile0-tile1-tile2) from tiles in barcode set. Insert into
        table in benchling format.
        :return:
        """
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
        """Concatenate tables in benchling format, can be imported into benchling as a feature library.
        :return:
        """
        return pd.concat([self.probes[benchling_headers], 
                          self.refs[benchling_headers], 
                          self.tiles[benchling_headers], 
                          self.barcodes[benchling_headers]])

    def get_barcodes_fasta(self):
        """Return barcodes in FASTA format, useful for ordering gblocks from IDT.
        :return:
        """
        return '\n'.join(['>%s\n%s' % (bc['name'], bc['sequence'])
                          for _, bc in self.barcodes.iterrows()])

    def score_seq(self, seq):
        """Find barcodes that best fit given sequences, on a tile-by-tile basis.
        :param seq:
        :return:
        """
        g = np.vectorize(lambda tile: sum(rc(x).lower() in seq.lower() for x in tile.split(spacer)))
        matches = g(self.tiles_arr)
        hits = matches.argmax(axis=0)
        match_tiles = [x[i] for i, x in zip(hits, self.tiles_arr.transpose())]
        match_barcode = self.make_barcode(match_tiles)
        return match_barcode, matches

    def score_files(self, files, reverse_complement=False):
        """Apply BarcodeSet.score_seq to each item provided, which can be a sequence or a filename. 
        rc can be a a bool or list of bools determining which sequences to reverse complement.
        """
        if isinstance(reverse_complement, bool):
            reverse_complement = [reverse_complement] * len(files)

        parent_dir = os.path.basename(os.path.dirname(files[0]))
        reference = '>%s\n' % parent_dir
        match_seqs, matches_arr = [], []
        for f, rc in zip(files, reverse_complement):
            seq = load_abi(f) if not rc else lasagna.design.rc(load_abi(f))
            match_barcode, matches = self.score_seq(seq)
            print '%s: %s, %s' % (os.path.basename(f), matches.argmax(axis=0), matches.max(axis=0))
            match_seqs += [match_barcode]
            matches_arr += [matches.argmax(axis=0)]
        reference += ''.join(np.unique(match_seqs))
        reference_name = os.path.join(os.path.dirname(files[0]),
                                      '%s_%s_reference.fasta' % (parent_dir, time.strftime('%Y%m%d')))
        with open(reference_name, 'w') as fh:
            fh.write(reference)

        return matches_arr



def load_abi(f):
    with open(f, 'rb') as fh:
        h = Bio.SeqIO.AbiIO.AbiIterator(fh).next()
        s = Bio.SeqIO.AbiIO._abi_trim(h)
    return str(s.seq)

