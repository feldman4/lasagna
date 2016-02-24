import time, os
import colour
import pandas as pd
from lasagna.design import *
from itertools import cycle, product
import Bio.SeqIO.AbiIO
from collections import defaultdict


home = '/broad/blainey_lab/David/lasagna/probes/stellaris/set1'
date = time.strftime('%Y%m%d')
barcode_set_name = os.path.basename(__file__).split('.')[0]

files = {'reference_probes': '%s/reference_probes.csv' % home,
         'reference_probes_uncut': '%s/reference_probes_uncut.csv' % home,
         'probes': '%s/stellaris_probes.csv' % home,
         'probes_export': '%s/%s_probes_export.csv' % (home, date),
         'tiles_export': '%s/%s_tiles_export.csv' % (home, date),
         'barcodes_export': '%s/%s_barcodes.fasta' % (home, date),
         'reference_export': '%s/%s_reference_export.csv' % (home, date),
         'recombination_primers': '%s/%s_recombination_primers.csv' % (home, date),
         'BTI_order': '%s/BTI_Oligo_Order-Columns_20151007.csv' % home}

# common sequences
LHS, RHS = 'ctcagaACCGGT', rc('ctcagaGGTACC')  # contain AgeI, KpnI sites
spacer = 'TT'
BsmBI_overhangs = 'ctcg', 'gcct', 'ggtg', 'tcgc', 'cgcc'
BsmBI_site = 'gacgatcCGTCTCc'

# for export to benchling
benchling_headers = ['name', 'sequence', 'type', 'color']
colors = ('#F58A5E', '#FAAC61', '#FFEF86', '#F8D3A9', '#B1FF67', '#75C6A9', '#B7E6D7',
          '#85DAE9', '#84B0DC', '#9EAFD2', '#C7B0E3', '#FF9CCD', '#D6B295', '#D59687',
          '#B4ABAC', '#C6C9D1')

# formulas for naming sequences
naming = {'reference': lambda x: 'ref%02d' % x,
          'tiles': lambda x: 'set1_t%02d' % x,
          'barcodes': lambda x: ('set1_' + '-'.join(['t%02d'] * len(x))) % tuple(x)
          }


def load_probes():
    reference_probes = pd.read_csv(files['reference_probes'], header=None)
    reference_probes = list(reference_probes[1])
    probes = pd.read_csv(files['probes'], header=None)
    probes = [p for p in probes[1] if p not in reference_probes]
    print 'loaded %d probes, %d reference probes' % (len(probes), len(reference_probes))
    return probes, reference_probes


def cloning_primers(barcode_set, UMI=0):
    refs = barcode_set.refs.set_index('name')
    refs = refs.loc[barcode_set.ref_order, 'sequence']
    primers = {}
    internal_refs = refs[1:-1]
    for i, ((name, seq), sense) in enumerate(zip(internal_refs.iteritems(),
                                                 cycle(['REV', 'FWD']))):


        priming = seq if sense is 'REV' else rc(seq)
        overhang = barcode_set.overhangs[int(i/2)].upper()
        overhang = overhang if sense is 'FWD' else rc(overhang)
        primers['%s_%s' % (name, sense)] = BsmBI_site + overhang + 'N'*UMI + priming
    return primers


class BarcodeSet(object):
    def __init__(self, probes, reference_probes, num_barcodes,
                 tiles_per_barcode, probes_per_tile, name=barcode_set_name, 
                 overhangs=BsmBI_overhangs):
        """Describe set of barcodes assembled from probes and reference probes. Tiles correspond to probe
        sets, and are recombined to create barcodes. Cloning uses directional overhangs between tiles.
        Reference probes at the boundaries of tiles are used for cloning, FISH quality control, and RT-qPCR.
           
        Exported fasta files contain barcodes in sense orientation (top strand).
        Probes are taken literally from input and used in anti-sense orientation (bottom strand).

        5' --------------------------- 3'
        3' --------------------------- 5'
                    3' <----- 5' FISH probe

        Sanger sequencing can be matched to barcodes using the score_seqs method.

        :param probes: list of probes, at least M x N x P
        :param reference_probes: conserved sequence at tile edges, typically need 2 * N
        :param num_barcodes: M barcodes to generate
        :param tiles_per_barcode: N tiles per barcode, typically between 3 and 6
        :param probes_per_tile: P probes per tile, typically between 8 and 16
        :param name:
        :param overhangs: used for directional cloning, typically 4 bp, 50% GC, and non-palindromic
        :return:
        """
        self.probes_per_tile = probes_per_tile
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

        self.ref_order = ['ref00', 'ref02', 'ref03', 'ref04', 'ref05', 'ref01'] + \
                         ['ref06', 'ref07', 'ref08', 'ref09', 'ref10', 'ref11']

        self.load_order(files['BTI_order'])

        self.make_barcodes()

    def load_order(self, filepath):
        """Load oligo order spreadsheet and define tile for each oligo. Use 
        before calling save_probe_layout.
        """
        self.ordered = pd.read_csv(filepath)

        probes2tiles=defaultdict(str)
        probes2tiles.update({p: row['name'] for _,row in self.tiles.iterrows()
                       for p in row['probes']})

        self.ordered['tile'] = [probes2tiles[p] for p in self.ordered['Sequence Name']]

    def save_probe_layout(self, dirpath=None):
        """Save probe and tile layouts on ordered plate to .xls, located in dirpath
        (defaults to set1.home).
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
        with pd.ExcelWriter('%s/plates_by_probe.xls' % dirpath) as writer:
            for name, plate in plates_by_probe.items():
                plate.to_excel(writer, sheet_name=name)
            
        with pd.ExcelWriter('%s/plates_by_tile.xls' % dirpath) as writer:
            for name, plate in plates_by_tile.items():
                plate.to_excel(writer, sheet_name=name)

    def save_primers(self):
        primers = cloning_primers(self)
        savename = files['recombination_primers']
        pd.Series(primers).to_csv(savename, header=None)

    def make_probes(self, probes):
        """Turn iterable of probe sequences into named entries in table, benchling format. Color
        probes by parent tile.
        :param probes:
        :return:
        """
        self.probes = pd.DataFrame({'sequence': probes})
        self.probes['name'] = ['set1_%03.d' % i for i, _ in enumerate(probes)]
        self.probes['color'] = [colors[(i / self.probes_per_tile) % len(colors)]
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
        """Use current values in tiles_per_barcode, num_barcodes, and probes_per_tile fields to make
        a numpy.ndarray of tile sequences, and a table in benchling format. Color tiles by position
        in barcode.
        :return:
        """
        index = np.arange(self.tiles_per_barcode * self.num_barcodes * self.probes_per_tile)
        index = index.reshape(-1, self.probes_per_tile)

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


    def make_barcode(self, tiles, refs=None, overhangs=None, code=('<' + 'rstsro'*5 + 'rstsr>')):
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

