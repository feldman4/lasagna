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
    data = io.read_stack(test_file, master, memmap=False)

    process.DOWNSAMPLE = 2
    n = process.get_nuclei(data[0, 0, :, :])

    return n

if __name__ == '__main__':
    test_nuclei()

# test_conditions()


