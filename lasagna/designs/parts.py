from lasagna.design import reverse_complement as rc
import re


lasagna_enzymes = {
 'AgeI': 'ACCGGT',
 'BamHI': 'GGATCC',
 'BbsI': 'GAAGAC',
 'BsaI': 'GGTCTC',
 'BsiWI': 'CGTACG',
 'BsmBI': 'CGTCTC',
 'Bsp1407I': 'TGTACA',
 'BsrGI': 'TGTACA',
 'EcoRI': 'GAATTC',
 'KflI': 'GGGWCCC',
 'KpnI': 'GGTACC',
 'MluI': 'ACGCGT',
 'NheI': 'GCTAGC',
 'PspXI': 'VCTCGAGB',
 'SbfI': 'CCTGCAGG',
 'XhoI': 'CTCGAG',
 'XmaI': 'CCCGGG'}

# from Kosuri 2010
# minimum edit distance 6, look OK on thermo MPA
dialout_primers = \
[('CTTAAACCGGCCAACATACC', 'ATGCTACTCGTTCCTTTCGA'),
 ('TGCTCTTTATTCGTTGCGTC', 'TCTTATCGGTGCTTCGTTCT'),
 ('TGAGCCTTATGATTTCCCGT', 'GTCCGTTTTCCTGAATGAGC'),
 ('CGTTCTAAACGGCTAGATGC', 'AGTCTGTCTTTCCCCTTTCC'),
 ('GTATCCGAAGCGTGGAGTAT', 'CAGGTATGCGTAGGAGTCAA'),
 ('CTTGTTATGGACGAGTTGCC', 'TTAATGGCGCGTTCATACTG'),
 ('CCAAAGATTCAACCGTCCTG', 'ATTAGCCATTTCAGGACGGA'),
 ('TATTCATGCTTGGACGGACT', 'ACTATGTACCGCTTGTTGGA'),
 ('ATCGACAATGGTATGGCTGA', 'TATGTCTCCTAGCCACTCCT'),
 ('GTCCTAGTGAGGAATACCGG', 'CCGAAGAATCGCAGATCCTA'),
 ('TTAGATAGGTGTGTAGGCGC', 'TAAGGTGCGTACTAGCTGAC'),
 ('TTCCGTTTATGCTTTCCAGC', 'TCCTTGGAGTTTAGAGCGAG'),
 ('GTATAGTTTGTGCGGTGGTC', 'ATCAATCCCCTACACCTTCG'),
 ('TCAGCCTTTCATTGATTGCG', 'TTCCTTGATACCGTAGCTCG'),
 ('AGGGTCGTGGTTAAAGGTAC', 'CGTTTCTTTCCGGTCGTTAG'),
 ('TGCAAGTGTACAAATCCAGC', 'GAACGGTGATCCCTTTCCTA'),
 ('CTTAAGGTTTGCCCATTCCC', 'TGTTATAGCTTCCACGGTGT'),
 ('TGGTTCGTTAGTCGATCTCC', 'AGACGGGATTTTACTGGGTC'),
 ('TATTTTGTAGAGCGTTCGCG', 'TCTTTGCTTCGCAAGTCTTG'),
 ('TTCTGTAAGTTTCGTCGGGA', 'CTAAACACCGCACCTCACTA'),
 ('TTGACGTACGTAGGTTCTCC', 'GAACACAACTACACTGACGC'),
 ('GAGATGAGTAGACGAGTGGG', 'ATGGTCACTGACTCGCATTA'),
 ('CTTTGGGCTTTCAGATGAGC', 'CAAAGATTTCTGTCGGTCGG'),
 ('TGTCATATGCTAACGTCCGT', 'TGGCTACTTTCTTAGCGGAA'),
 ('TTGCGACATCACAATTCTCG', 'TACTTCGAGACTTCATGCGT'),
 ('TCAGTATGGCGTCTTGAAGT', 'ATGGCCCGACCTCTATTATG'),
 ('TCATGTCGTGACCAGTAGAC', 'TGGGTCTAGTGAACTTCGTC'),
 ('AACTAACGGATTTAAGCGCG', 'AACATATGTTGCTTCGTCCG'),
 ('CATTTTCTGTTCCCCAGTGG', 'TCGAGTTAGATTGTCACCCC'),
 ('ATTTGCCTAACCACTCCACT', 'TCAGAGCTTTTCGGTACAGT'),
 ('TGACTTATGAACCTTTGCGC', 'GCCCAGGAGTAGTCGTTAAT'),
 ('ATAGGATTAGCTGATGGGCC', 'TCTGTGTTCCGACTAAGGTC'),
 ('TGAGATTCGGGACTATTCGG', 'TCTGTTGTTAGACTCCGACC'),
 ('TTGGTTAGTACACGGGACTC', 'GTACGTCTGAACTTGGGACT'),
 ('ATTTGTGTATCGAGGCTCGT', 'AGACACGCGATTGTTTAACC'),
 ('ATCGTTCCCCATCACATTCT', 'CCGTTCGTTTTGAGCACTTA'),
 ('ATTACCATGTTATCGGGCGA', 'AGGTTAGGGAACGCAAGATT'),
 ('TCGGTGGATATGACGTAACC', 'CCAGACTGTGCTCGTTATCT'),
 ('GGTCAGATGGTTTACATGCG', 'AGTTGTTCTCTATCCGCGAT'),
 ('TCTCGTTCGAAAATCATCGC', 'GATTAAATCTCGCCGGTGAC'),
 ('TGCAAATGTGAGGTAGCAAC', 'TTGTAGTTTTCGCTTGCGTT'),
 ('AAAGTCAAAGTGCGTTTCGT', 'TGTGTTGCTCTCTCATAGCC')]


default_parts = \
    {'BsmBI' :   'CGTCTC'
    ,'BsmBI_rc': 'GAGACG'
    ,'BsaI' : 'GGTCTC'
    ,'BsaI_rc' : 'GAGACC'
    ,'A': 'A'
    ,'T': 'T'
    ,'C' : 'C'
    ,'G' : 'G'
    ,'sticky_U6': 'CACC'
    ,'sticky_scaffold': 'GTTT'
    ,'spacer': 'nnnnnn'# 'GGATAC'
    ,'dialout_5': dialout_primers[0][0]
    ,'dialout_3_rc': rc(dialout_primers[0][1])
    ,'N': 'N'
    ,'NN': 'NN'
    ,'sticky_Pd42_5': 'TTCC'
    ,'sticky_Pd42_3': 'ACTG'
     
    ,'sgRNA': 'N'*20
    ,'barcode': 'N'*12
    ,'padding': 'CGCTACAAACTTCTCTCTGCTGAAACAAGCCGGTGACGTCGAAGAGAA'

    ,'BbsI': 'GAAGAC'
	  ,'BbsI_rc': 'GTCTTC'
	  ,'gibson_U6_5': 'TGGAAAGGACGAAACACCG'
	  ,'gibson_P42_3': 'ACTGGCTATTCATTCGCCC'

    ,'pL30_5': 'TGTTCAATCAACATTCC' 
    ,'pL30_3': 'ACTGGCTATTCATTCGC'
    ,'pL30_3_extended': 'actggctattcattcgcCTCCTGTTCG' 
}


oligos = {'Pd_pL30_short': 'actggctattcattcgcCTCCTGTTCGACAGTCAGCCGCATCTGCGTCTATTTAGTGGAGCCCTTGtgttcaatcaacattcc'
}


default_layouts = \
          { 'pL42': 
          ('dialout_5', 'BsaI', 'C', 
          'sticky_U6', 'sgRNA', 
          'sticky_scaffold', 'N', 'BsmBI_rc', 'spacer', 'BsmBI', 'N',
          'sticky_Pd42_5', 'barcode', 
          'sticky_Pd42_3', 'C', 'BsaI_rc', 'dialout_3_rc')
          
          , 'gecko':
          ('dialout_5', 'BsmBI', 'N', 'sticky_U6', 'sgRNA', 'sticky_scaffold', 'N', 'BsmBI_rc', 'dialout_3_rc', 'padding')
          
          , 'CROP-BsmBI':
          ('dialout_5', 'BsmBI', 'G', 'sticky_U6', 'sgRNA', 'sticky_scaffold', 'C', 'BsmBI_rc', 'dialout_3_rc')

          , 'pL42-BbsI':
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


iupac = {'R': 'AG'
        ,'Y': 'CT'
        ,'S': 'GC'
        ,'W': 'AT'
        ,'K': 'GT'
        ,'M': 'AC'
        ,'B': 'CGT'
        ,'D': 'AGT'
        ,'H': 'ACT'
        ,'V': 'ACG'
        ,'N': 'ACGT'
        }

iupac_re = re.compile('(%s)' % '|'.join(iupac.keys()))