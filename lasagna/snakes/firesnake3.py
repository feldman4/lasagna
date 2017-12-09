#python3
import uuid
import json
import shutil

WIN_PYTHON2 = 'C:\ProgramData\Anaconda2\python.exe'
FIRESNAKE2 = __file__.replace('firesnake3', 'firesnake')

def stitch_input(wildcards):
    # doesn't respect wildcard constraints
    format_ = 'MAX/{cycle}/20X_{cycle}_{well}-Site_{{site}}.max.tif'.format(**wildcards)
    sites = firesnake.stitch_input_sites(wildcards['tile'], site_shape=SITE_SHAPE, tile_shape=TILE_SHAPE)
    inputs = []
    for site in sites:
        inputs.append(format_.format(site=site))
    return inputs

def dump_json(input, **info):
    name = 'json/input_%s.json' % uuid.uuid4()
    info = info.copy()
    info.update({'input': input})
    with open(name, 'w') as fh:
        json.dump(info, fh)
    return name

def call_firesnake(input, output, method, cmds=None, **info):
    json_name = dump_json(input, **info)
    from subprocess import call
    cmd = [find_python2(), FIRESNAKE2, method, 
            '--input_json', json_name,
            '--output', str(output)]
    if cmds:
        cmd += cmds
    call(cmd)

def find_python2():
    python2 = shutil.which('python2')
    if python2 is None:
        return WIN_PYTHON2
    return python2