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
# # import networkx as nx
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
        while len(dupe)>1:
            # get the edges and weights for current endpoint
            edges = dupe.edges(path[-1],data=True)
            assert(len(edges)>0)
            dupe.remove_node(path[-1])
            weights = [e[2]['weight'] for e in edges]
            # choose the next node
            v = edges[weights.index(min(weights))][1]
            cost = cost + min(weights)
            path.append(v)
#         path.append(dupe.nodes()[0])
        cost = cost + G[path[0]][path[-1]]['weight']
        all_paths.append((cost,path))
    return min((c,p) for (c,p) in all_paths)


def filter_cut_sites(probe, cut_sites):
    for enzyme, cut_site in cut_sites.items():
        cut_site, probe = cut_site.lower(), probe.lower()
        if cut_site in probe or rc(cut_site) in probe:
            return False
    return True


def generate_sequence(GC_content,length):
    """generate a random DNA sequence with fixed GC content and length
    :param GC_content:
    :param length:
    :return:
    """
    DNA = np.array(['A', 'T', 'C', 'G'])
    GC = int(np.ceil(GC_content*length))
    seq = [0 for i in range(length-GC)] + \
           [2 for i in range(GC)]
    np.random.shuffle(seq)
    seq = [np.random.random_integers(0,1)+s for s in seq]
    return ''.join(DNA[seq])


watson_crick = {'A': 'T',
                'T': 'A',
                'C': 'G',
                'G': 'C',
                'N': 'N'}
watson_crick.update({k.lower(): v.lower() for k,v in watson_crick.items()})


def rc(seq):
    return ''.join(watson_crick[x] for x in seq)[::-1]


def RNAduplex(a, b):
    """Returns dot structure, energy, and slices into query strands
    corresponding to bases in structure.
    Save dot structure:
    RNAplot(a[i0] + '.' + b[1], structure, name)
    :param a:
    :param b:
    :return:
    """
    arg = ['RNAduplex']
    out = lasagna.utils.call(arg, '\n'.join([a,b]))
    pat = '(\S*)\s*([0-9]+,[0-9]+)\s*:\s*([0-9]+,[0-9]+)\s*\(\s*(.*)\)'
    structure, i0, i1, energy = re.findall(pat, out)[0]
    energy = float(energy[1:-1])
    i0, i1 = i0.split(','), i1.split(',')
    i0 = slice(int(i0[0]) - 1, int(i0[1]))
    i1 = slice(int(i1[0]) - 1, int(i1[1]))

    return structure, energy, (i0, i1)


def RNAfold(a):
    """Returns dot structure and energy.
    :param a:
    :return:
    """
    # default behavior is to save rna.ps
    arg = ['RNAfold --noPS']
    out = lasagna.utils.call(arg, a)
    sequence, structure, energy = re.findall('(.*)\n(.*)\s\(\s*(.*)\)', out)[0]
    energy = float(energy[:-1])
    return sequence, structure, energy


def RNAplot(sequence, structure, path, output_format='ps'):
    name = os.path.basename(path)
    dir_name = os.path.dirname(path)
    arg = ['RNAplot', '-o', output_format]
    stdin = '>%s\n%s\n%s\n' % (name, sequence, structure)
    if dir_name:
        cwd = os.getcwd()
        os.chdir(dir_name)
        print stdin
        print os.getcwd()
        lasagna.utils.call(arg, stdin)
        os.chdir(cwd)

    else:
        lasagna.utils.call(arg, stdin)

