from collections import Counter
import copy
import numpy as np

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
    
    def expand(self, num):
        while num > 0:
            divisions = np.ones(((32-2)*self.population_size, 2))
            i = 0
            cells = self.cells.keys()
            for i_cell, count in enumerate(self.cells.values()):
                for gen in range(1,5):
                    division_time = (float(gen) / cells[i_cell].fitness) + 0.01*np.random.randn()
                    icount = count * 2**gen
                    divisions[i:i+icount,:] = np.array([[division_time, i_cell]] * icount)
                    i += icount
            divisions = np.array(sorted(divisions, key=lambda row: row[0]))
            new_cells = [cells[int(j)] for j in divisions[:,1]]
            if len(new_cells) > num:
                self.cells += Counter(new_cells[:num])
                return
            else:
                self.cells += Counter(new_cells)
                num -= len(new_cells)
            
    
    def expand_(self, num, expand_by=1, fast=False):
        """Split cells from the population and add them back in to expand.
        Approximate by expanding more than one cell at a time.
        Accepts "fast" kwarg for calls to split().
        """
        if num < 0:
            raise ValueError('can only expand by factor in [0, 1], or number > 1')
        if num < 1:
            num = self.population_size * num
        if self.population_size == 0:
            raise ValueError('cannot expand population of size 0')
        while num:
            expand_by_ = expand_by if expand_by <= self.population_size else self.population_size
            if num < expand_by_:
                expand_by_ = num
            new_cells = Population(Counter(self.cells)).split(expand_by_,
                                                             fast=fast)
            num -= expand_by_
            self.cells += new_cells.cells
            
    def split(self, num, fast=False):
        """If num is in [0, 1]  treat as split fraction. If num is >= 1, 
        treat as # of cells to split.
        If fast=True, sample with replacement.
        """
        population_size = sum(self.cells.values())
        split_size = 1e10
        if num < 0:
            raise ValueError('argument must be positive')
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
        weights = np.array([v for k,v in self.cells.items()])
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
        