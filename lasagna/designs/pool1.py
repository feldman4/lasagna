import pool0
from pool0 import *
from collections import defaultdict
  
more_layouts = \
		  { 'pL42-BbsI':
	          ('dialout_5', 'BsmBI', 'C', 
	          'sticky_U6', 'G', 'sgRNA', 
	          'sticky_scaffold', 'NN', 'BbsI_rc', 'spacer', 'BbsI', 'NN',
	          'sticky_Pd42_5', 'barcode', 
	          'sticky_Pd42_3', 'C', 'BsmBI_rc', 'dialout_3_rc')
          , 'pL42-gibson':
	          ('dialout_5', 
	           'gibson_U6_5',
	           'sgRNA',
	           'sticky_scaffold', 'N', 'BsmBI_rc', 'spacer', 'BsmBI', 'N', 'sticky_Pd42_5', 
	           'barcode',
	           'gibson_P42_3',
	           'dialout_3_rc')
          }

more_parts = { 'BbsI': 'GAAGAC'
			 , 'BbsI_rc': 'GTCTTC'
			 , 'gibson_U6_5': 'TGGAAAGGACGAAACACCG'
			 , 'gibson_P42_3': 'ACTGGCTATTCATTCGCCC'
			 }

default_layouts.update(more_layouts)
default_parts.update(more_parts)



def prepare_barcodes():
    """Top level
    """
    df_barcodes = load_8bp_barcodes()
    full = degenerate_and_fix()
    df_barcodes['12bp'] = [full[x] for x in df_barcodes['8bp']]
    df_barcodes['12bp_layout'] = df_barcodes['12bp'].apply(layout_12bp_barcode)
    df_barcodes = filter_barcodes_pL42(df_barcodes)
    df_barcodes = pool0.color(df_barcodes, orange=500, red=0, green=500)

    return df_barcodes

def pair_sgRNAs_barcodes(df_sgRNAs, df_barcodes, df_layout):
    df_barcodes = df_barcodes.sort_values('score', ascending=False)
    tagged_barcodes = defaultdict(list)

    barcode_it =  {'green':  iter(df_barcodes.query('green')['12bp_layout'])
                  ,'orange': iter(df_barcodes.query('orange')['12bp_layout'])
                  ,'white':  iter(df_barcodes.query('~green & ~orange')['12bp_layout'])
                  }

    def get_barcode(tag, i):
    	try:
    		return tagged_barcodes[tag][i]
    	except (IndexError, KeyError) as e:
    		# add a new barcode to the list
    		if 'orange' in tag:
    			color = 'orange'
    		elif 'green' in tag:
    			color = 'green'
    		else:
    			color = 'white'
    		tagged_barcodes[tag] += [barcode_it[color].next()]
    		withdrawn[tag] += 1
    		return get_barcode(tag, i)

    def endo_go_ranked_sgRNAs(df_sgRNAs, n=3):
        return (df_sgRNAs.query('endo_go_count > 0')
                         .sort_values(['endo_go_count', 'source'], ascending=False)
                         .groupby('gene_id').head(n))

    design_names = {
      'FR_GFP_TM':             lambda x: x.query('tag=="GFP_TM"')
    , 'endocytosis GO top 50': lambda x: endo_go_ranked_sgRNAs(x, n=3)[:150]
    , 'endocytosis GO all':    lambda x: endo_go_ranked_sgRNAs(x)
    , 'nontargeting controls': lambda x: x.query('tag=="nontargeting"')
    }

    withdrawn = Counter()
    oligos = []
    for _, row in df_layout.iterrows():
    	# subpools are 1-indexed in layout sheet
    	dialout_ix = int(row['subpool'] - 1)
    	dialout_5, dialout_3 = dialout_primers[dialout_ix]
    	print dialout_ix, dialout_5, dialout_3
        tag = row['tag']
        design = row['sgRNA design']
        layout = row['barcode design']
        barcode_coverage = int(row['barcodes/sgRNA'])
        spot_coverage = int(row['spots/oligo'])
        num_sgRNAs = int(row['# of sgRNAs'])

        df_sgRNAs_subset = design_names[design](df_sgRNAs)

        print '-'*10, '%s | %s | dialout %d' % (layout, design, dialout_ix), '-'*10 
        print 'layout uses %d / %d available sgRNAs' % (num_sgRNAs, len(df_sgRNAs_subset))
        print
        
        it = zip(df_sgRNAs_subset['sgRNA'], df_sgRNAs_subset['gene_id'])
        assert len(it) >= num_sgRNAs
        barcode_ix = 0
        for sgRNA, gene_id in it[:num_sgRNAs]:
            for _ in range(barcode_coverage):
            	barcode = get_barcode(tag, barcode_ix)
            	barcode_ix += 1
                parts = {'barcode': barcode
                        ,'sgRNA':   sgRNA
                        ,'dialout_5': dialout_5
                        ,'dialout_3_rc': rc(dialout_3)
                        }
                
                oligo = build_oligo(default_layouts[layout], override_parts=parts)
                oligo = complete_oligo(oligo)
                [oligos.append([gene_id, sgRNA, barcode, dialout_ix, tag, design, layout, oligo]) for _ in range(spot_coverage)]

    columns = 'gene_id', 'sgRNA', 'barcode', 'dialout', 'tag', 'design', 'layout', 'oligo'
    oligos = pd.DataFrame(oligos, columns=columns)
    oligos['subpool'] = (oligos['dialout'] + 1).astype(int)

    print 'used %d / %d available barcodes' % (sum(withdrawn.values()), len(df_barcodes))
    print sorted(withdrawn.items())
    
    return oligos

def pool_stats(df_oligos):
    c = ['subpool', 'layout']
    x = df_oligos.drop_duplicates(['subpool', 'sgRNA']).groupby(c).size()
    x.name = 'num_sgRNAs'
    y = df_oligos.drop_duplicates(['subpool', 'barcode']).groupby(c).size()
    y.name = 'num_barcodes'
    z = df_oligos.groupby(c).size()
    z.name = 'num_spots'
    return pd.concat([x,y,z], axis=1)

def get_nontargeting_wang():
    x = pd.read_excel('Wang_2015_supplement/S1.xlsx')
    y = pd.read_excel('Wang_2015_supplement/S2.xlsx')
    x.loc[x['Symbol'].isnull(), 'Chromosome'] = 'CTRL'
    x['G'] = x['sgRNA sequence'].apply(lambda x: x[0]=='G')

    a = y.filter(regex='initial').as_matrix()
    b = y.filter(regex='final').as_matrix()
    scores = (np.log2(1 + b) - np.log2(1 + a)).mean(axis=1)

    filt = scores > -0.6
    filt &= y['sgRNA'].apply(lambda x: x.startswith('CTRL'))

    nontargeting = y.loc[filt, 'sgRNA']
    nontargeting_sgRNAs = x.set_index('sgRNA ID').loc[nontargeting, 'sgRNA sequence']
    return nontargeting_sgRNAs
