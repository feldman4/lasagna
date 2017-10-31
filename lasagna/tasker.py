import cloudpickle
import hashlib
import os
import pickle
import functools
from collections import defaultdict

# used by `tagged` only
info_hasher = lambda info: info['sample']
info_tagger = lambda key, tag: '.'.join([key, tag])

# used by `find_file_info` only
info_parser = lambda f: {'sample': f.split('.')[0], 'tag': f.split('.')[1]}

def tagged(input_tags, output_tags):
    """Returns a function that can analyze a list of `file_info` and provide
    valid tasks that look like [(function2, inputs, outputs)], where `function2`
    operates on the non-tagged .

    Relies on:
    - info_hasher to determine if two files share a key and should be processed together
    - info_tagger to generate output filenames

    """
    
    def tagged_net(function1):
        def kwarg_net(**kwargs):
            def task(file_info):
                # find all valid groups
                # a function of the metadata evaluates to the same key
                # doesn't cover all cases: could reuse an input for multiple subtasks
                # for a particular function, might convert file_info into a table, groupby, 
                # filter for complete groups, then do more logic to find remaining pieces

                # group existing files
                # e.g., files grouped by site, tile or well
                grouped_file_info = defaultdict(list)
                for f in file_info:
                    grouped_file_info[info_hasher(f)].append(f)

                tasks = []
                # each group may generate a task if it can satisfy all input_tags
                for key, group in grouped_file_info.items():
                    inputs = []
                    for element in group:
                        for tag in input_tags:
                            if tag == element['tag']:
                                # relies on bi-directional tagging...
                                inputs.append(info_tagger(key, tag))
                                continue
                    if len(inputs) == len(input_tags):
                        # outputs for this group
                        outputs = [info_tagger(key, tag) for tag in output_tags]
                        # function that generates outputs from inputs
                        function2 = wrap_store(inputs, outputs)
                        # needed to determine when function should run
                        tasks.append((function2, inputs, outputs))

                return tasks

            def wrap_store(inputs, outputs):
                def inner():
                    # load inputs
                    input_args = [load_data(f) for f in inputs]
                    # run
                    output_data = function1(*input_args, **kwargs)
                    # save outputs
                    for name, data in zip(outputs, output_data):
                        dump_data(name, data)
                    # no return value
                inner = functools.update_wrapper(inner, function1)
                return inner

            task = functools.update_wrapper(task, function1)
            return task
        return kwarg_net
    return tagged_net

### 

def is_cached(home, function, inputs, outputs):
    """Checks if the result for this function is already on the filesystem.
    """
    for input_ in inputs:
        if not os.path.exists(input_):
            print 'the fuck'
            return False

    for output in outputs:
        if not check_all_entries(home, function, inputs, output):
            return False
            
    return True

def run(home, function, inputs, outputs):
    """Run a function, then make an entry to cache
    filesystem output. The function must be stateless except for files 
    listed in inputs and outputs.
    """
    
    function()
    
    for output in outputs:   
        entry = make_entry(function, inputs, output)
        add_entry(home, entry)

def make_entry(function, inputs, output):
    """
    function gets hashed
    inputs is a list of filenames
    output is a filename
    """
    
    return {'flash': make_flash(function), 
            'inputs': [(f, os.stat(f)) for f in inputs],
            'output': (output, os.stat(output))
           }

def check_entry(function, inputs, output, entry):
    """Can we skip running this function? Before running this, check
    whether the inputs exist.
    - function hasn't changed
    - inputs haven't changed
    - output hasn't changed
    """
    # function changed?
    if make_flash(function) != entry['flash']:
        return False
    
    # inputs changed?
    old_inputs = dict(entry['inputs'])
    # same inputs, assume the files exist
    for f in inputs:
        if not same_stat(os.stat(f), old_inputs[f]):
            return False
        
    # output changed?
    old_output, old_stat = entry['output']
    # not the same output
    if output != old_output:
        return False
    # the file itself changed
    if not os.path.exists(output):
        return False
    if not same_stat(os.stat(output), old_stat):
        return False
                      
    return True

def check_all_entries(home, function, inputs, output):
    entries = load_entries(home)
    for entry in entries:
        if check_entry(function, inputs, output, entry):
            return True
    return False

def add_entry(home, entry):
    entries = load_entries(home)
    create_entries(home, [entry] + entries)

def delete_entries(home):
    f = os.path.join(home, '.tasker')
    os.remove(f)
        
def create_entries(home, entries):
    f = os.path.join(home, '.tasker')
    with open(f, 'w') as fh:
        pickle.dump(entries, fh)
        
def load_entries(home):
    f = os.path.join(home, '.tasker')
    with open(f, 'r') as fh:
        return pickle.load(fh)    
    
def make_flash(f, *args, **kwargs):
    """Don't even bother hashing. Can look at function code later if 
    we really want to?
    """
    return cloudpickle.dumps((f, args, kwargs))

def partial(f, args, **kwargs):
    g = functools.partial(f, args, **kwargs)
    return functools.update_wrapper(g, f)

# DOMAIN-SPECIFIC

def same_stat(a, b):
    return a == b

def load_data(f):
    """Generic, replace with domain-specific.
    """
    with open(f, 'r') as fh:
        return pickle.load(fh)
    
def dump_data(f, data):
    """Generic, replace with domain-specific.
    """
    with open(f, 'w') as fh:
        pickle.dump(data, fh)

# example

def seed_files():
    for i, key in enumerate('ABC'):
        f = info_tagger(key, 'cells')
        dump_data(f, i)
        if key != 'A':
            f = info_tagger(key, 'nuclei')
            dump_data(f, -1 * i)

def find_jobs(home, task):
    file_info = find_file_info(home)
    return task(file_info)

def find_file_info(home):
    from glob import glob
    # doesn't grab .tasker
    files = glob('*')
    file_info = []
    for f in files:
        try:
            file_info.append(info_parser(f))
        except:
            pass
        
    return file_info