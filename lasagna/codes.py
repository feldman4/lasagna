import numpy as np
from itertools import product
from collections import defaultdict, OrderedDict
from Levenshtein import distance, editops
from lasagna.designs.pool0 import GC_penalty, homopolymer_penalty, contiguous
from scipy.sparse import coo_matrix

### VT

def mod_sum(x, q=4):
    return int(np.sum(x) % q)

def syndrome(x):
    n = len(x)
    S = (np.array(x) * np.arange(1,n+1)).sum() 
    return int(S % (n + 1))

def auxiliary(digits):
    arr = []
    for i in range(len(digits) - 1):
        if digits[i+1] >= digits[i]:
            arr.append(1)
        else:
            arr.append(0)
    return arr

def stringy(digits):
    return ''.join(str(d) for d in digits)

def make_barcodes(n):
    digits = [range(4)] * n
    return product(*digits)

def VT_code(n):
    print 'VT max, no filter      :   ', 4**(n-1) / n
    xs = make_barcodes(n)
    d = defaultdict(list)
    for i, x in enumerate(xs):
        if i % 100000 == 0:
            print i
        a = syndrome(auxiliary(x))
        b = mod_sum(x)
        d[(a,b)].append(x)

    d = OrderedDict(sorted(d.items(), key=lambda x: -1 * len(x[1])))
    print 'largest set:               ', max(map(len, d.values()))

    return d

### VT tests

def same_by_deletion(a,b):
    for i in range(len(a)):
        a_ = a[:i] + a[i+1:]
        b_ = b[:i] + b[i+1:]        
        if a_ == b_:
            return True
    return False

### khash

def khash(s, k):
    n = len(s)
    window = int(np.ceil((n - k) / float(k)))
    s = s + s
    arr = []
    for i in range(n):
        # arr += [s[i:i+window]]
        for j in (0, 1):
            arr += [((i + j) % n, s[i:i+window])]
    return arr

def build_khash(xs, k):
    D = defaultdict(list)
    for x in xs:
        for h in khash(x, k):
             D[h].append(x)

    D = {k: sorted(set(v)) for k,v in D.items()}
    return D

def build_khash2(xs, k):
    D = defaultdict(list)
    for x in xs:
        for h in khash(x, k):
             D[h].append(x)
    arr = []
    for v in D.values():
        v = sorted(set(v))
        for i, a in enumerate(v):
            for b in v[i+1:]:
                arr.append((a,b))  
    return sorted(set(arr))

def sparse_dist(D, threshold, D2=None):
    """Entries less than or equal to threshold only.
    """
    if D2 is None:
        D2 = defaultdict(int)
    for xs in D.values():
        for i, a in enumerate(xs):
            for b in xs[i+1:]:
                d = distance(a,b)
                if d < threshold:
                    key = tuple(sorted((a,b)))
                    D2[key] = d
    return D2

### khash tests

def khash_speedup(s, k, q=4):
    hashes = khash(s, k)
    m = len(hashes[0])
    return q**m / float(len(hashes))
    
def test_khash(xs, D2, attempts=1e6):
    n = len(xs)
    tests = 0
    for _ in range(int(attempts)):
        i, j = np.random.randint(n, size=2)
        a, b = xs[i], xs[j]
        d = distance(a, b)
        if 0 < d < 3:
            key = tuple(sorted((a,b)))
            if key not in D2:
                print 'fuckyou'
            else:
                tests += 1
    return tests


### utility

def array_to_str(arr):
    """14e6 in 60s
    """
    arr = (np.array(arr).astype(np.uint8) + 48).view(dtype='S1')
    return [''.join(x) for x in arr]


def sparse_view(xs, D2):
    """string barcodes
    """

    mapper = {x: i for i, x in enumerate(xs)}
    f = lambda x: mapper[x]
    i,j,data = zip(*[(f(a), f(b), v) for (a,b),v in D2.items()])
    data = np.array(data) > 0
    i = np.array(i)
    j = np.array(j)

    n = len(xs)
    cm = coo_matrix((data, (i, j)), shape=(n, n))
    return cm


def maxy_clique(cm, start=None):
    """sparse matrix of missing edges (1 - adjacency)
    """
    # combine things not in the graph
    n, m = cm.shape
    assert n == m
    flatten = lambda x: np.array(x)[0]

    if start is None:
        arr = [np.random.randint(n)]
        arr = [0]
    else:
        arr = start

    unused = np.array(sorted(set(range(n)) - set(arr)))

    while True:
        if len(arr) % 100 == 0:
            print len(arr), 'barcodes;', candidates.shape, 'remaining'
        # TODO: remember as `partial`
        candidates = flatten(cm[arr, :][:, unused].sum(axis=0))
        
        # remove the useless
        # TODO: slow
        unused = list(np.array(unused)[candidates == 0])
        if len(unused) == 0:
            return arr
        # TODO: subtract off `total - partial`
        ix = np.argmin(cm[unused, :][:, unused].sum(axis=0))
        arr.append(unused.pop(ix))
        assert cm[arr, :][:, arr].sum() == 0



def maxy_clique2(cm, start=None):
    """sparse matrix of missing edges (1 - adjacency)
    """
    # combine things not in the graph
    n, m = cm.shape
    assert n == m
    flatten = lambda x: np.array(x)[0]

    if start is None:
        arr = [np.random.randint(n)]
        arr = [0]
    else:
        arr = start

    unused = np.ones(n, dtype=bool)
    unused[arr] = False

    # indices, best to worst
    # total = flatten(cm.sum(axis=0))
    # total = np.argsort(total)

    last_added = arr[0]

    r = np.arange(n)
    last_printed = 0
    while True:
        if len(arr) % 100 == 0:
            x = unused.sum()
            delta = last_printed - x
            print len(arr), 'barcodes;', x, 'remaining; +%d' % delta
            last_printed = x
        # just the neighbors of the last added barcode
        nix_these = cm[last_added, :][:, unused].indices
        
        # remove the useless
        ix = np.where(unused)[0][nix_these]
        unused[ix] = False

        if unused.sum() == 0:
            return arr

        # even less smart...
        # better to take barcode with lowest edit distance/most at 3...
        w = np.where(unused)[0]
        last_added = w[0]
        unused[last_added] = 0
        arr.append(last_added)
        # assert cm[arr, :][:, arr].sum() == 0