import IPython.display
import numpy as np
import os
import regex as re
# import RNA
# import matplotlib.pyplot as plt
# import pandas as pd
#
# from pprint import pprint as pp
#
# import csv
# import platform, time
# # import pygraphviz as pgv
import networkx as nx
import lasagna.utils


def greedyTSP(G):
    """greedy nearest-neighbor TSP, trying all starting points
    :param G:
    :return:
    """
    all_paths = []
    for start in range(len(G)):
        dupe = G.to_undirected()
        path = []
        cost = 0
        path.append(start)
        while len(dupe) > 1:
            # get the edges and weights for current endpoint
            edges = dupe.edges(path[-1], data=True)
            assert (len(edges) > 0)
            dupe.remove_node(path[-1])
            weights = [e[2]['weight'] for e in edges]
            # choose the next node
            v = edges[weights.index(min(weights))][1]
            cost = cost + min(weights)
            path.append(v)
        # path.append(dupe.nodes()[0])
        cost = cost + G[path[0]][path[-1]]['weight']
        all_paths.append((cost, path))
    return min((c, p) for (c, p) in all_paths)


def filter_cut_sites(probe, cut_sites):
    for enzyme, cut_site in cut_sites.items():
        cut_site, probe = cut_site.lower(), probe.lower()
        if cut_site in probe or rc(cut_site) in probe:
            return False
    return True


def generate_sequence(GC_content, length):
    """generate a random DNA sequence with fixed GC content and length
    :param GC_content:
    :param length:
    :return:
    """
    DNA = np.array(['A', 'T', 'C', 'G'])
    GC = int(np.ceil(GC_content * length))
    seq = [0 for i in range(length - GC)] + \
          [2 for i in range(GC)]
    np.random.shuffle(seq)
    seq = [np.random.random_integers(0, 1) + s for s in seq]
    return ''.join(DNA[seq])


watson_crick = {'A': 'T',
                'T': 'A',
                'C': 'G',
                'G': 'C',
                'N': 'N'}
watson_crick.update({k.lower(): v.lower() for k, v in watson_crick.items()})


def rc(seq):
    return ''.join(watson_crick[x] for x in seq)[::-1]


def energy(a):
    return np.vectorize(lambda x: x.energy if x else 0.)(a)


duplex_re = re.compile('(\S*)\s*([0-9]+,[0-9]+)\s*:\s*([0-9]+,[0-9]+)\s*\(\s*(.*)\)')
fold_re = re.compile('\s*(.*)\n(.*)\s\(\s*(.*)\)')


def RNAduplex(a, b, full=True):
    """Returns RNAResult from folding query strands. Provide full=True to save
    full sequence, otherwise truncated to window around bound bases.
    :param a: single str or list of str
    :param b: single str or list of str; if list and a is str, folds a against each str in b
    :return:
    """
    ix = slice(None)
    is_str = np.lib._iotools._is_string_like
    if is_str(a):
        if is_str(b):
            a = [a]
            b = [b]
            ix = slice(0)
        else:
            a = [a] * len(b)

    arg = ['RNAduplex']
    stdin = '\n'.join(sum(zip(a, b), tuple()))
    out = lasagna.utils.call(arg, stdin)
    arr = []
    for a_i, b_i, (structure, i0, i1, energy) in zip(a, b, duplex_re.findall(out)):
        i0, i1 = i0.split(','), i1.split(',')
        i0 = slice(int(i0[0]) - 1, int(i0[1]))
        i1 = slice(int(i1[0]) - 1, int(i1[1]))

        sequence = a_i[i0] + '.' + b_i[i1]

        if full:
            sequence = a_i + '.' + b_i
            a_, b_ = structure.split('&')
            s_a, s_b = ['.'] * len(a_i), ['.'] * len(b_i)
            s_a[i0] = a_
            s_b[i1] = b_
            structure = ''.join(s_a + ['&'] + s_b)
        arr += [RNAResult(sequence=sequence, structure=structure,
                          energy=float(energy), ix=(i0, i1))]

    return arr[ix]


def RNAfold(a):
    """Returns dot structure and energy.
    :param a:
    :return:
    """
    ix = slice(None)
    is_str = np.lib._iotools._is_string_like
    if is_str(a):
        a = [a]
        ix = slice(0)

    # default behavior is to save rna.ps
    arg = ['RNAfold --noPS']
    stdin = '\n'.join(a)
    out = lasagna.utils.call(arg, stdin)

    arr = []
    for sequence, structure, energy in fold_re.findall(out):
        arr += [RNAResult(sequence=sequence, structure=structure,
                          energy=float(energy))]

    return arr


class RNAResult(object):
    def __init__(self, sequence=None, structure=None, energy=None,
                 ix=None, duplex=None):
        self.sequence = sequence
        self.structure = structure
        self.energy = energy
        self.ix = ix
        self.duplex = duplex

    def plot(self, path, output_format='ps'):
        name = os.path.basename(path)
        dir_name = os.path.dirname(path)
        arg = ['RNAplot -o %s' % output_format]
        stdin = '>%s\n%s\n%s\n' % (name, self.sequence, self.structure)
        if dir_name:
            cwd = os.getcwd()
            os.chdir(dir_name)
            print stdin
            print os.getcwd()
            lasagna.utils.call(arg, stdin)
            os.chdir(cwd)
        else:
            lasagna.utils.call(arg, stdin)

    def svg(self):
        self.plot('dummy', output_format='svg')
        svg = IPython.display.SVG('dummy_ss.svg')
        os.remove('dummy_ss.svg')

        # polyline representing backbone is automatically split at '.' in ps but not svg
        dot = re.findall('(<text x=\"(.*)\" y=\"(.*)\">\.<\/text>\n)', svg.data)
        if dot:
            svg.data = svg.data.replace(dot[0][0], '')
            coordinates = ','.join(dot[0][1:3])
            svg_patch = ' " style="stroke: black; fill: none; stroke-width: 1.5"/> \n' + \
                        '   <polyline id="outline2" points=" '
            svg.data = svg.data.replace(coordinates, svg_patch)

        return svg

    def __repr__(self):
        return "%s\n%s\n(%.2f)\t%s" % (self.sequence, self.structure,
                                       self.energy, self.ix)


def suboptimal_clique(energy, threshold):
    """Find (sub-optimal) clique with no energy below threshold.
    :param energy: matrix
    :param threshold:
    :return:
    """
    off_target_G = nx.Graph(energy)

    # set up graph
    for i in off_target_G.node:
        off_target_G.node[i] = i
    edges = off_target_G.edges(data=True)

    # try to remove troublesome nodes
    idx = np.array([e[2]['weight'] for e in edges]).argsort()
    edges = np.array(edges)[idx]
    for e in edges:
        if e[2]['weight'] > threshold:
            break
        # remove a node if edge is still in the graph
        if (e[0] in off_target_G) and (e[1] in off_target_G):
            # calculate total energy for each node
            n1 = sum(x[2]['weight'] for x in off_target_G.edges(e[0],
                                                                data=True))
            n2 = sum(x[2]['weight'] for x in off_target_G.edges(e[1],
                                                                data=True))
            if n1 < n2:
                off_target_G.remove_node(e[0])
            else:
                off_target_G.remove_node(e[1])

    return off_target_G.nodes()


def plot_energies(antisense_fold, energy_thresholds=tuple(range(-30, 0)), ax=None,
                  bins=tuple(range(-40, -1))):
    """
    :param antisense_fold:
    :param energy_thresholds:
    :param ax:
    :param bins:
    :return:
    """
    import matplotlib.pyplot as plt

    on_target = np.diag(energy(antisense_fold))
    off_target = np.tril(energy(antisense_fold), -1)
    keep_n = [len(suboptimal_clique(off_target, et)) for et in energy_thresholds]
    if ax is None:
        _, ax = plt.subplots()

    ax.hist(on_target, bins=bins, label='on-target',
            log=True, bottom=0.5, color='g', zorder=10)
    ax.hist(off_target[off_target != 0].flatten(), bins=bins,
            log=True, bottom=0.5, color='b', label='off-target')
    ax.plot(energy_thresholds, keep_n, c='r', zorder=20)

    ax.legend(loc='best')
    ax.set_xlabel('energy')
    ax.set_ylabel('count')
    ax.set_title('probes remaining after iterative exclusion by off-target energy')

    return ax
