import decorator
import functools
import hashlib
import joblib
import tempfile
import os

from collections import defaultdict


def default_mem(verbose=1):
    cachedir = os.path.join(tempfile.gettempdir(), 'joblib')
    return joblib.Memory(cachedir=cachedir, verbose=verbose)

def cache_file_output(mem, *args, **kwargs):
    """Manipulate joblib cache to work for functions that write to a file.
    
    Decorate the function like this:
    
    mem = joblib.Memory(cachedir='/tmp/joblib', verbose=1)

    @cache_file_output(mem)
    def f(fname, ...):
        ...

    When you call the decorated function, execution is skipped and a cached version 
    of the python output is returned only if (a) the arguments are the same and
    (b) the function has run with these arguments before, yielding a file with the
    same checksum as the one on disk.
    """
    def inner(f):
        g = checksum3(mem.cache(f, *args, **kwargs))
        return functools.wraps(f)(g)
    return inner


def checksum3(f):
    def wrapper(*args, **kwargs):
        fname = args[0]
        f_arg_hash = f._get_output_dir(*args, **kwargs)[0]
        checksum = md5sum(fname)
        if checksum is None:
            output, _ = check_cache.call(f_arg_hash, checksum, args=args, kwargs=kwargs)
        output = check_cache(f_arg_hash, checksum, args=args, kwargs=kwargs)
        checksum = md5sum(fname)
        check_cache(f_arg_hash, checksum, args=args, kwargs=kwargs, skip=True)
        return output
        
    @mem.cache(ignore=['skip', 'args', 'kwargs'])
    def check_cache(f_arg_hash, checksum, skip=False, args=[], kwargs={}):
        if not skip:
            return f.call(*args, **kwargs)
    
    return wrapper
    

@decorator.decorator
def none_on_IOError(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except IOError: 
        return None
        

@none_on_IOError
def md5sum(filename, blocksize=65536):
    """Faster than calling OSX md5 utility via subprocess.
    """
    hsh = hashlib.md5()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(blocksize), b''):
            hsh.update(block)
    return hsh.hexdigest()



### CHECKSUM ROUGH DRAFTS ###

# f = '20170410_96W-G088/20X_D1.aligned.tif'
# mem = joblib.Memory(cachedir='/tmp/joblib', verbose=1)

# def run(data):
#     """
#     evaluate a series of functions
#     slow functions are cached
#     - inspect intermediate results
#         - need to reoutput if
#             - upstream functions were re-run
#             - file on disk was changed
#         - alternate file format (.tif)
#     - 
#     """
#     @mem.cache
#     def calculate_means(data):
#         return data.mean()

#     # want to persist only if args were different
#     # can mem.cache call a provided function on a cache miss?
#     # could also wrap the inner function providing 
    
#     # separate function to validate files
#     # 1. generate validation over a set of files
#     # 2. return a list of files and checksums
    
    
#     def checksum2(f):
#         store = defaultdict(list)
#         def wrapper(fname, *args, **kwargs):
#             print store
#             checksum = md5sum(fname)
#             arg_hash = f._get_argument_hash(fname, *args, **kwargs)
#             if checksum in store:
#                 if arg_hash in store[checksum]:
#                     # output file is the same, let the joblib cache 
#                     # handle producing any output arguments
#                     return f(fname, *args, **kwargs)
                
#             # otherwise force evaluation
#             output, metadata = f.call(fname, *args, **kwargs)
#             checksum = md5sum(fname)
#             # right behavior is to accept None?
#             store[checksum].append(arg_hash)
#             return output
#         return wrapper
    
#     def checksum(f):
#         """functions whose only effect is to write a file need to be re-run iff 
#         the file changed. calculate checksum, throwing it in as an extra argument to
#         cached function.
#         """
#         # find the checksum
#         # if it's None, call the function, add a key that is:
#         #   - hashed arguments + checksum
#         # if checksum is in the cache with the same hashed arguments,
#         # call wrapped function (ends up hashing args twice)
#         # if the checksum is in the cache with different hashed args,
#         # call and update
#         # otherwise, call and update
#         # map checksums to keys
#         # if we 
#         @functools.wraps(f)
#         def g(fname, *args, **kwargs):
#             try:
#                 checksum = md5sum(fname)
#             except IOError:
#                 output = f(fname, *args, checksum=checksum, **kwargs)
#                 checksum = np.random.rand() # bad, need a value that is never equal
#             return f(fname, *args, checksum=checksum, **kwargs)
#         return g
    
#     def add_checksum_arg(f):
#         @functools.wraps(f)
#         def wrapper(*args, **kwargs):
#             return f(*args, **kwargs)
#         return wrapper
    
#     def cache_checksum(f, *args, **kwargs):
#         return checksum(mem.cache(add_checksum_arg(f), *args, **kwargs))

#     def cache_checksum2(f, *args, **kwargs):
#         return checksum2(mem.cache(f, *args, **kwargs))
    
#     def cache_checksum3(f, *args, **kwargs):
#         """decorator.decorator preserves signature but hates closures
#         """
#         g = checksum3(mem.cache(f, *args, **kwargs))
#         return functools.wraps(f)(g)


#     @cache_checksum
#     def save_result(fname, means, checksum=None):
#         """a docstring
#         """
#         s = str(means)
#         print 'ran'
#         with open(fname, 'w') as fh:
#             fh.write(s)
        
#     @cache_file_output(mem)
#     def save_result3(fname, means):
#         """a docstring
#         """
#         s = str(means)
#         print 'ran'
#         with open(fname, 'w') as fh:
#             fh.write(s)
        
#     data = read(f)
#     means = calculate_means(data)
#     save_result3('dumping.ground', means)
# #     !rm 'dumping.ground'
#     save_result3('dumping.ground', means)

#     return save_result3
    
# data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# save_result = run(data)