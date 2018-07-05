from collections import Counter
import copy
import numpy as np
import pandas as pd


class Population(object):
    def __init__(self, cells):
        """Pass a dict of Cells, or a list of other things that get made into
        Cells with default fitness 1.
        """
        if isinstance(cells, dict):
            self.cells = Counter(cells)
            return
        if any(not isinstance(cell, Cell) for cell in cells):
            cells = [Cell(cell) for cell in cells]
        self.cells = Counter(cells)

    def __add__(self, other):
        if not isinstance(other, Population):
            raise TypeError('can only add Population to Population')
        new_cells = copy.copy(self.cells)
        new_cells.update(other.cells)
        return Population(new_cells)
    
    def __sub__(self, other):
        """Removes all Cells present in second Population, or in an iterable of Cells.
        """
        if isinstance(other, Population):
            other = other.cells
        return Population(Counter({k: v for k, v in self.cells.items() if k not in other}))
    
    def __contains__(self, other):
        return other in self.cells
               
    def expand(self, num, generations=4):
        """Simulate expansion by accumulating division times of all cells through
        fixed number of generations, then sorting by division time. 
        Stochastic division times can be implemented through Cell.fitness.
        kwarg "generations" indicates # of generations to simulate forward.
        """
        if num < 1:
            num *= self.population_size
        if num < 0:
            raise ValueError('requires positive value')
        while num > 0:
            divisions = np.zeros((2 ** generations - 1) * self.population_size, dtype=complex)
            i = 0
            cells = np.array(self.cells.keys())
            for i_cell, count in enumerate(self.cells.values()):
                for gen in range(generations ):
                    # uses same noise across generation, amplifies variance
                    division_time = float(gen + 1) / (cells[i_cell].fitness + cells[i_cell].fitness_noise())
                    icount = count * 2 ** gen
                    divisions[i:i + icount] = division_time + i_cell * 1j
                    i += icount
                    
            # sort potential new cells by division time
            divisions.sort()
            new_cells = cells[divisions.imag.astype(int)]
            
            if len(new_cells) > num:
                self.cells += Counter(new_cells[:num])
                return
            else:
                self.cells += Counter(new_cells)
                num -= len(new_cells)

    def split(self, num):
        """If num is in [0, 1]  treat as split fraction. If num is >= 1, 
        treat as # of cells to split.
        If fast=True, sample with replacement.
        """
        population_size = sum(self.cells.values())
        population_size = 3000
        # split_size = 1e10
        if num < 0:
            raise ValueError('argument must be positive')
        if num > self.population_size:
            raise ValueError('not enough cells')
        if num < 1:
            split_size = np.random.binomial(self.population_size, num)
        if num >= 1:
            split_size = np.random.binomial(self.population_size,
                                            float(num) / self.population_size)
        
        x = [a for a,b in self.cells.items() for _ in range(b)]
        np.random.shuffle(x)
        return Population(x[:split_size])


    def __repr__(self):
        return 'population of %d cell types, %d cells' % (
            len(self.cells),
            self.population_size)

    def get_weights(self):
        weights = np.array([v for k, v in self.cells.items()])
        return weights / float(sum(weights))

    @property
    def population_size(self):
        return sum(self.cells.values())

    @property
    def diversity(self):
        return len(self.cells.keys())

    def stats(self):
        return 'diversity: %d\npopulation size: %d' % (self.diversity,
                                                       self.population_size)


class Cell(object):
    def __init__(self, cell, fitness=1, fitness_noise=lambda: 0):
        """Mimic a generic object with additional fitness parameter.
        :param cell:
        :param fitness: time to division
        :param fitness_noise: function that returns noise on fitness
        :return:
        """
        self.fitness = fitness
        self.fitness_noise = fitness_noise
        self.cell = cell

    def __repr__(self):
        return self.cell.__repr__()

    def __getitem__(self, key):
        return self.cell[key]

    def __setitem__(self, key, value):
        self.cell[key] = value

    def __hash__(self):
        return self.cell.__hash__()

    def __eq__(self, other):
        return hash(self) == hash(other)


def make_matrices(wells):
    """Based on list of Populations where Cells are tuples (barcode, HR_event)
    binary DataFrames M[barcode, well], N[HR_event, well]
    
    """
    # total observed barcodes, HR events
    barcodes, HR_events = zip(*sum(wells[1:], wells[0]).cells.keys())
    
    well_index = pd.Index(range(len(wells)), name='well')
    M = pd.DataFrame(index=pd.Index(set(barcodes), name='barcode'), 
                     columns=well_index).fillna(0)
    N = pd.DataFrame(index=pd.Index(set(HR_events), name='HR event'), 
                     columns=well_index).fillna(0)
    for i, well in enumerate(wells):
        bc, HR = zip(*well.cells.keys())
        M.loc[bc, i] += 1
        N.loc[HR, i] += 1
    return M, N


def score_matrices(M, N):
    """Find size of logical intersection, complement, differences of DataFrames with 
    shared columns and return as list of DataFrames.
    """
    M_, N_ = M.values(), N.values()
    S = np.zeros((M.shape[0], N.shape[0], 4))
    
    not_M = 1 - M_
    not_N = 1 - N_
    for i in range(N_.shape[0]):
        S[:, i, 0] = (M_    & N_[i]   ).sum(axis=1)
        S[:, i, 1] = (not_M & N_[i]   ).sum(axis=1)
        S[:, i, 2] = (M_    & not_N[i]).sum(axis=1)
        S[:, i, 3] = (not_M & not_N[i]).sum(axis=1)
    
    S_ = [pd.DataFrame(S[:,:,i], index=M.index, columns=N.index) for i in range(4)]
    
    return S_

    
class LinearModel(object):
    def __init__(self):
        """Linear model describing specific and non-specific signal, and
        auto-fluorescence.
        :return:
        """
        self.A = 0.
        self.b_p = 0.

        self.B = None
        self.C = None
        self.D = None
        
        self.P = None
        self.X = None

        self.indices = {}
        self.tables = {}
        self.X_table = None

    def evaluate(self, M, b):
        """Evaluate linear model for a single sample.
        :param M:
        :return:
        """
        j, l, m = [self.indices[x] for x in 'jlm']

        # reshape M and b to match indices, need matrices for np.einsum
        if isinstance(M, pd.DataFrame):
            M = M.loc[j, l].values()
        if isinstance(b, pd.Series):
            b = b.loc[m].values()

        assert(M.shape == (len(j), len(l)))
        assert(b.shape == (len(m),))

        md = np.einsum('jl,lk->jlk', M, self.D)
        
        self.P = np.einsum('jlk,kn', md, self.C)
        bbr = np.einsum('lm,m', self.B, b)
        self.X = self.A + np.einsum('jln,l', self.P, bbr + self.b_p)

        row_index = pd.Index(self.indices['j'], name='round')
        column_index = pd.Index(self.indices['n'], name='channel')
        self.X_table = pd.DataFrame(self.X, index=row_index, columns=column_index)
        return self.X_table

    def matrices_from_tables(self):
        """Set LinearModel.indices and LinearModel.tables first. Index j set separately.
        Matrices used are subsets of tables derived using indices.
        :return:
        """
        k, l, m, n = [self.indices[x] for x in 'klmn']
        self.B = self.tables['B'].loc[l, m].values()
        self.C = self.tables['C'].loc[k, n].values()
        self.D = self.tables['D'].loc[l, k].values()

        # verify correct dimensions (DataFrame.loc is permissive of missing indices)
        assert(self.B.shape == (len(l), len(m)))
        assert(self.C.shape == (len(k), len(n)))
        assert(self.D.shape == (len(l), len(k)))
