from collections import defaultdict
from lasagna import conditions_, io, process

worksheet = '20150716 96W-G005'
index = 4

def test_conditions():

    wells, conditions, cube = conditions_.load_sheet('20150716 96W-G005')
    comparisons = conditions_.find_comparisons(cube)
    c = conditions_.find_comparisons_second_order(cube)


def test_nuclei():
    io.initialize_paths('20150717', subset='20X_hyb_and_washes/',
                        lasagna_dir='/Volumes/blainey_lab/David/lasagna/')
    master = io.DIR['stacks'][0]

    test_file = io.DIR['stacks'][3]
    data = io.read_stack(test_file, master=master, memmap=False)

    process.DOWNSAMPLE = 2
    n = process.get_nuclei(data[0, 0, :, :])

    return n


def test_nuclei_stitch(well=(('A2',),0)):
    io.initialize_paths('20150716', subset='stitched/',
                        lasagna_dir='/Volumes/blainey_lab/David/lasagna/')

    dict_20X = defaultdict(list)
    [dict_20X[io.get_well_site(f)].append(f) for f in io.DIR['stacks'] if '20X' in f]

    test_files = dict_20X[well]
    print '(well, site) found', dict_20X.keys()
    print 'from', well, 'loading', test_files
    data = {f: io.read_stack(f, memmap=False) for f in test_files}

    offsets = process.register_images(data.values())

    C = io.compose_stacks([io.offset(d, offset)
                           for d, offset in zip(data.values(), offsets)])
    return data, offsets, C


def test_nuclei_to_table():
    io.initialize_paths(dataset='20150629/40X_scan',
                        lasagna_dir='/Volumes/blainey_lab/David/lasagna')
    df = process.table_from_nuclei(io.DIR['stacks'][0])
    return df

if __name__ == '__main__':
    # x = test_nuclei_stitch()
    # io.save_hyperstack('/Users/feldman/Desktop/test.tif', x[2])
    df = test_nuclei_to_table()
    df.to_pickle(io.DIR['analysis'] + '/test.pkl')

# test_conditions()


