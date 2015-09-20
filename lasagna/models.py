from collections import Counter
import copy
import numpy as np
import pandas as pd


class Population(object):
    def __init__(self, cells):
        """Pass a Counter of Cells, or a list of other things that get made into
        Cells with default fitness 1.
        """
        if type(cells) == Counter:
            self.cells = cells
            return
        if any(type(cell) != Cell for cell in cells):
            cells = [Cell(cell) for cell in cells]
        self.cells = Counter(cells)

    def __add__(self, other):
        if type(other) != Population:
            raise TypeError('can only add Population to Population')
        new_cells = copy.deepcopy(self.cells)
        new_cells.update(other.cells)
        return Population(new_cells)

    def expand(self, num, generations=5):
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
            divisions = np.zeros((2 ** generations - 2) * self.population_size, dtype=complex)
            i = 0
            cells = np.array(self.cells.keys())
            for i_cell, count in enumerate(self.cells.values()):
                for gen in range(1, generations):
                    division_time = (float(gen) / cells[i_cell].fitness) + 0.01 * np.random.randn()
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

    def split(self, num, fast=False):
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

        # sample with replacement, but conserve # of cells
        if fast:
            split_cells = Counter()
            while split_size:
                weights = self.get_weights()
                split_cells_ = np.random.choice(self.cells.keys(),
                                                size=split_size, p=weights)
                split_size = 0
                for cell, count in Counter(split_cells_).items():
                    removed = min(self.cells[cell], count)
                    self.cells[cell] -= removed
                    if self.cells[cell] == 0:
                        del self.cells[cell]
                    if removed < count:
                        split_size += count - removed
                    split_cells[cell] += removed

                # terminate if all cells removed
                if self.population_size == 0:
                    break
            return Population(split_cells)

        # sample without replacement
        split_cells = []
        for _ in range(split_size):
            if sum(self.cells.values()) == 0:
                raise ValueError('out of cells')

            weights = self.get_weights()
            split_cell = np.random.choice(self.cells.keys(), size=1,
                                          p=weights)[0]
            self.cells[split_cell] -= 1
            if self.cells[split_cell] == 0:
                del self.cells[split_cell]
            split_cells += [split_cell]

        return Population(split_cells)

    def __repr__(self):
        return 'population of %d cell types, %d cells:\n%s' % (
            len(self.cells),
            self.population_size,
            self.cells.__repr__())

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
    def __init__(self, cell, fitness=1):
        """Mimic a generic object with additional fitness parameter.
        :param cell:
        :param fitness:
        :return:
        """
        self.fitness = fitness
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
        return self.cell.__eq__(other)


class LinearModel(object):
    def __init__(self):
        """Linear model describing specific and non-specific signal, and
        auto-fluorescence.
        :return:
        """
        self.A = 0.
        self.B = None
        self.C = None
        self.D = None
        self.b = None
        self.b_p = None

        self.P = None
        self.X = None

        self.indices = {}
        self.tables = {}
        self.X_table = None

    def evaluate(self, M):
        md = np.einsum('jl,lk->jlk', M, self.D)
        self.P = np.einsum('jlk,kn', md, self.C)
        bbr = np.einsum('lm,m', self.B, self.b)
        self.X = self.A + np.einsum('jln,l', self.P, bbr + self.b_p)

        if 'j' in self.indices and 'k' in self.indices:
            self.X_table = pd.DataFrame(self.X,
                                        index=self.indices['j'],
                                        columns=self.indices['n'])

    def matrices_from_tables(self):
        """Set LinearModel.indices and LinearModel.tables first. Ignore j index.
        :return:
        """
        k, l, m, n = [self.indices[x] for x in 'klmn']
        self.B = self.tables['B'].loc[l, m].as_matrix()
        self.C = self.tables['C'].loc[k, n].as_matrix()
        self.D = self.tables['D'].loc[l, k].as_matrix()

        # verify correct dimensions (DataFrame.loc is permissive of missing indices)
        assert(self.B.shape == (len(l), len(m)))
        assert(self.C.shape == (len(k), len(n)))
        assert(self.D.shape == (len(l), len(k)))

