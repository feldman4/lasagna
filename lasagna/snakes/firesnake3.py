#python3
import uuid
import json
import os
import shutil

WIN_PYTHON2 = ['C:\ProgramData\Miniconda2\python.exe', 'C:\ProgramData\Anaconda2\python.exe']
FIRESNAKE2 = __file__.replace('firesnake3', 'firesnake')

def stitch_input(wildcards):
    # doesn't respect wildcard constraints
    format_ = 'MAX/{cycle}/20X_{cycle}_{well}-Site_{{site}}.max.tif'.format(**wildcards)
    sites = firesnake.stitch_input_sites(wildcards['tile'], site_shape=SITE_SHAPE, tile_shape=TILE_SHAPE)
    inputs = []
    for site in sites:
        inputs.append(format_.format(site=site))
    return inputs

def dump_json(**info):
    if not os.path.isdir('json'):
        print('creating json directory...')
        os.mkdir('json')
    name = 'json/input_%s.json' % uuid.uuid4()
    with open(name, 'w') as fh:
        json.dump(info, fh)
    return name

def call_firesnake(method, output, **info):
    json_name = dump_json(**info)
    from subprocess import call
    cmd = [find_python2(), FIRESNAKE2, method, 
            '--input_json', json_name,
            '--output', str(output)]
    call(cmd)

def find_python2():
    python2 = shutil.which('python2')
    if python2 is not None:
        return python2

    for path in WIN_PYTHON2:
        if os.path.exists(path):
            return path

    raise ValueError('python2.exe not found')
    