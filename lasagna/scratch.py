from lasagna import conditions_

worksheet = '20150716 96W-G005'
index = 4

def setup():

    wells, conditions, cube = conditions_.load_sheet('20150716 96W-G005')
    comparisons = conditions_.find_comparisons(cube)
    c = conditions_.find_comparisons_second_order(cube)
setup()
