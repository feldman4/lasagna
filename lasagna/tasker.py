import cloudpickle
import hashlib
import os
import pickle
import functools
from collections import defaultdict, namedtuple

# used by `tagged` only
demo_hasher = lambda f: demo_parser(f)['sample']
demo_tag_extactor = lambda f: demo_parser(f)['tag']
demo_tagger = lambda key, tag: '.'.join([key, tag])

# used by `find_file_info` only
demo_parser = lambda f: {'sample': f.split('.')[0], 'tag': f.split('.')[1]}

Job = namedtuple('Job', 'function inputs outputs group_id')

def demo_tagged():
    return partial(tagged, hasher=demo_hasher, tagger=demo_tagger, tag_extractor=demo_tag_extactor)

def tagged(input_tags, output_tags, hasher, tagger, tag_extractor):
    """Returns a function that can analyze a list of `file_info` and provide
    valid tasks that look like [(function2, inputs, outputs)], where `function2`
    operates on the non-tagged .

    Relies on:
    - hasher to determine if two files share a key and should be processed together
    - tagger to generate output filenames

    """
    # tag is a pair (tag_value, [input] -> Maybe [input])
    # it filters and validates matching filenames
    # default validator takes the first matching input
    default_tag = lambda s: (s, lambda xs: xs[0] if len(xs) > 0 else None)
    input_tags = [default_tag(tag) if isinstance(tag, str) else tag 
                    for tag in input_tags]


    def tagged_net(function1):
        def arg_net(*args, **kwargs):
            task = tagger_task(function1, input_tags, output_tags, 
                                hasher, tagger, tag_extractor, 
                                args, kwargs)
            task = functools.update_wrapper(task, function1)
            return task
        arg_net = functools.update_wrapper(arg_net, function1)
        return arg_net
    return tagged_net

def tagger_task(function1, input_tags, output_tags, hasher, tagger, tag_extractor, args, kwargs):
    """Returns a task function, which can process a list of filenames into a
    list of jobs [(function2, inputs, outputs)]. In general, task functions have
    to perform some kind of pattern matching. Making an object of the task 
    generator would help higher level functions build a graph out of a set of 
    task generators. 

    A tagged task generator could be labeled only by input and output tags, 
    ignoring the work done by `hasher`, `tagger`, and `parser`. Lasagna nuance 
    comes from `hasher`, which may combine files...
    """

    def task(filenames):
        # find all valid groups
        # a function of the metadata evaluates to the same key
        # doesn't cover all cases: could reuse an input for multiple subtasks
        # for a particular function, might convert file_info into a table, groupby, 
        # filter for complete groups, then do more logic to find remaining pieces

        # group existing files
        # e.g., files grouped by site, tile or well
        # file_info = [demo_parser(name) for name in filenames]
        grouped_filenames = defaultdict(list)
        for name in filenames:
            group_id = hasher(name)
            if group_id:
                grouped_filenames[group_id].append(name)

        tasks = []
        # each group may generate a task if it can satisfy all input_tags
        for group_id, filenames in sorted(grouped_filenames.items()):
            inputs = []
            skip_group = False
            for tag, validator in input_tags:
                inputs_ = []
                for name in filenames:
                    if tag_extractor(name) == tag:
                        # relies on bi-directional tagging...
                        inputs_.append(name)

                tmp = validator(inputs_)
                if tmp is None:
                    skip_group = True
                else:
                    inputs.append(tmp)

            # if any validation failed, skip this group
            if skip_group:
                continue

            # outputs for this group
            outputs = [tagger(group_id, tag) for tag in output_tags]
            # function that generates outputs from inputs
            function2 = execute_from_files(function1, inputs, outputs, args, kwargs)
            # needed to determine when function should run
            job = Job(function=function2, inputs=inputs, outputs=outputs, group_id=group_id)
            tasks.append(job)
        return tasks
    return task

def execute_from_files(function1, inputs, outputs, args, kwargs):
    """Add steps before and after function call. Load args from 
    files in `inputs`. Call function with loaded args, then provided args,
    then provided kwargs.
    Iterate over results, saving them to outputs. 
    """
    def inner():
        # load inputs
        input_args = nested_map(load_data, inputs)
        # run
        input_args = input_args + list(args)
        kwargs2 = kwargs.copy()
        if 'backdoor' in kwargs2:
            kwargs2['backdoor'] = inputs
        output_data = function1(*input_args, **kwargs2)
        if not isinstance(output_data, tuple):
            output_data = output_data,
        # save outputs
        for name, data in zip(outputs, output_data):
            dump_data(name, data)
        # no return value
    inner = functools.update_wrapper(inner, function1)
    return inner

### 

def is_cached(home, function, inputs, outputs, group_id, entries=None):
    """Checks if the result for this function is already on the filesystem.
    """
    inputs = flatten(inputs)
    for input in inputs:
        if not os.path.exists(input):
            return False

    if entries is None:
        entries = load_entries(home)
    for output in outputs:
        if not check_all_entries(home, function, inputs, output, entries):
            return False
            
    return True

def flatten(xs):
    arr = []
    for x in xs:
        if isinstance(x, (list, tuple)):
            arr.extend(flatten(x))
        else:
            arr.append(x)
    return arr

def nested_map(f, xs):
    """Lists and tuples are converted to lists.
    """
    arr = []
    for x in xs:
        if isinstance(x, (list, tuple)):
            arr.append(nested_map(f, x))
        else:
            arr.append(f(x))
    return arr

def run(home, function, inputs, outputs, group_id):
    """Run a function, then make an entry to cache
    filesystem output. The function must be stateless except for files 
    listed in inputs and outputs.
    """
    
    function()
    
    inputs = flatten(inputs)
    for output in outputs:   
        entry = make_entry(function, inputs, output, group_id)
        add_entry(home, entry)

def make_entry(function, inputs, output, group_id):
    """
    function gets hashed
    inputs is a list of filenames
    output is a filename
    """
    
    return {'flash':  make_flash(function), 
            'inputs': [(f, os.stat(f)) for f in inputs],
            'output': (output, os.stat(output)),
            'group_id': group_id
           }

def check_entry(function, inputs, output, entry, flash=None):
    """Can we skip running this function? Before running this, check
    whether the inputs exist.
    - function hasn't changed
    - inputs haven't changed
    - output hasn't changed
    """
    if flash is None:
        flash = make_flash(function)
    # function changed?
    if flash != entry['flash']:
        return 'function changed'
    
    # inputs changed?
    old_inputs = dict(entry['inputs'])
    # same inputs, assume the files exist
    for f in flatten(inputs):
        if not os.path.exists(f):
            return "file doesn't exist"
        if not f in old_inputs:
            return 'file not in old inputs'
        if not same_stat(os.stat(f), old_inputs[f]):
            return 'input stat changed', f, old_inputs[f]

    
    # output changed?
    old_output, old_stat = entry['output']
    # not the same output
    if output != old_output:
        return 'not the same output', output, old_output

    # the file itself changed
    if not os.path.exists(output):
        return 'the file is gone'

    if not same_stat(os.stat(output), old_stat):
        return 'output stat changed', old_stat, output
                      
    return 'ok'

def check_all_entries(home, function, inputs, output, entries):
    flash = make_flash(function)
    for entry in entries:
        check = check_entry(function, inputs, output, entry, flash=flash)
        if check == 'ok':
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


def make_flash(f):
    """Don't even bother hashing. Can look at function code later if 
    we really want to?
    """
    return hashlib.md5(cloudpickle.dumps(f)).hexdigest()

def complete_garbage2(f):
    g = types.FunctionType(f.func_code, f.func_globals, name=f.func_name,
                           argdefs=f.func_defaults,
                           closure=f.func_closure)
    return f.func_globals

def complete_garbage(f):
    import types
    codes = {}
    arr = list(f.func_closure)
    while arr:
        c = arr.pop().cell_contents
        if not isinstance(c, types.FunctionType):
            continue
        if c.func_name in codes:
            continue
        codes[c.func_name] = c.func_code.co_code
        try:
            arr.extend(c.func_closure)
        except:
            pass
    return codes

def partial(f, *args, **kwargs):
    g = functools.partial(f, *args, **kwargs)
    return functools.update_wrapper(g, f)

# DOMAIN-SPECIFIC

def same_stat(a, b):
    return a.st_mtime == b.st_mtime

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

def seed_files(home):
    for i, key in enumerate('ABC'):
        name = os.path.join(home, demo_tagger(key, 'cells'))
        dump_data(name, i)
        if key != 'A':
            name = os.path.join(home, demo_tagger(key, 'nuclei'))
            dump_data(name, -1 * i)

def find_jobs(home, task):
    filenames = find_files(home)
    return task(filenames)

def find_files(home):
    from glob import glob
    # doesn't grab .tasker
    files = glob(os.path.join(home, '*.*'))
    return files

# SCHEDULER

def loop_schedule(home, file_finder, tasks, n):
    for i in range(n):
        files = file_finder()
        jobs = [job for task in tasks for job in task(files)]
        available_jobs = len(jobs)
        entries = load_entries(home)
        jobs = [job for job in jobs if not is_cached(home, *job, entries=entries)]
        if len(jobs) == 0:
            break
        print  'run %d with %d/%d jobs, checked %d entries' % (i, len(jobs), available_jobs, len(entries))
        job = jobs[0]
        print '=>', job[2]
        run(home, *job)