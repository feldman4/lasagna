#!/usr/bin/env python

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(
        description='Extract relevant info from lif series files.')
    parser.add_argument('files', help='The files to be converted.', nargs='+')
    # parser.add_argument('-v', '--verbose', action='store_true',
    #                     help='Print out runtime information.')
    # parser.add_argument('-n', '--nseries', type=int, metavar='INT', default=18,
    #                     help='Number of series in each file.')
    # parser.add_argument('-c', '--channel', type=int, metavar='INT', default=0,
    #                     help='Channel of interest.')

    args = parser.parse_args()
    # print args.files


    import regex as re

    patterns = {'sgRNA': "ACCG(.{19,20})GTTT"}

    for name, pattern in patterns.items():
        flag = False
        for f in args.files:
            if f.endswith('.seq'):
                with open(f, 'r') as fh:
                    seq = fh.read().strip()
                    seq = seq.replace('\n', '')
                match = re.findall(pattern, seq)
                if match:
                    if flag:
                        print name
                        flag = False
                    print f, ':', match[0]
