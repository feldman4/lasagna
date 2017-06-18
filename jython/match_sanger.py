#!/usr/bin/env python

def nearest(query, database, threshold):
    """
    """
    tmp = []
    for name, seq in database:
        tmp += [(distance(query, seq), name, seq)]
    tmp = [t for t in sorted(tmp)]
    if len(tmp) == 1:
        tmp += [(-1, -1, -1)]
        
    dist, name, _ = tmp[0]
    dist2 = tmp[1][0]
    if dist > threshold:
        name = 'unmatched'
    return name, dist, dist2
    
    return tmp[0][1], tmp[0][0], tmp[1][0]

    
def read_database(f):
    with open(f, 'r') as fh:
        txt = fh.read()
    database = []
    for line in txt.split('\n'):
        entries = tuple(line.split('\t'))
        if len(entries) == 2 and all(entries):
            database += [entries]
    return sorted(set(database))
    


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(
        description='Extract relevant info from lif series files.')
    parser.add_argument('files', help='The files to be converted.', nargs='+')
    parser.add_argument('-d', '--database', nargs=1,
                        help='Database of (name, sequence) pairs. Expects tab-delimited text.')
    parser.add_argument('-t', '--threshold', type=int, help='Maximum edit distance to display match name.')

    args = parser.parse_args()

    import regex as re

    patterns = {'sgRNA': "ACCG(.{19,20})GTTT",
                'sgRNA26': "ACCG(.{25,27})GTTT",
                's1_UMI': 'CCGGT(.{19,20})TTCCCA',
                's1_UMI_rc': 'TGGGAA(.{19,20})ACCGG',
                'TM10': 'AGAAAT(.{32,90})GTACA'
                }

    if args.database:
        from Levenshtein import distance
        database_name = args.database[0]
        database = read_database(database_name)
        print 'read %d unique pairs from %s' % (len(database), database_name)

        threshold = 2
        if args.threshold is not None:
            threshold = args.threshold


    for name, pattern in patterns.items():
        flag = name == patterns.items()[0][0]
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
                    line = name, match[0], f
                    if args.database:

                        line += nearest(match[0], database, threshold)

                    print '\t'.join(str(x) for x in line)
