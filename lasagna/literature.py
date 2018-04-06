import urllib2
import requests
import warnings
import pandas as pd
from bs4 import BeautifulSoup as bs
from lasagna.utils import Memoized
import lasagna.config



def mitocheck(gene, screens=('Mitocheck primary screen', 
                             'Mitocheck validation screen'),
             limit=10, substitute='(),'):
    """Search Mitocheck database for given gene name (or Ensembl id)
    and return DataFrame containing download links.
    """
    bsurl = lambda x: bs(urllib2.urlopen(x).read())
    
    request = 'http://mitocheck.org/cgi-bin/mtc?query=%s' % gene
    x = bsurl(request)
    y = x.find(title='List all movies/images associated with this gene')
    if y is None:
        print 'zero or multiple entries for', gene
        return None
    z = bsurl('http://mitocheck.org' + y['href'])
    df = pd.read_html(str(z.find('table')), header=0)[2].dropna(how='all')
    df = df[df['Source'].isin(screens)]
    df = df.groupby('Source').head(10)

    for ix, movie_id in df['Movie/Image ID'].iteritems():
        request = 'http://mitocheck.org/cgi-bin/mtc?action=show_movie;query=%s' % movie_id 
        x = bs(urllib2.urlopen(request).read())
        df.loc[ix, 'link'] = x.find_all('a', text=u'Download this movie')[0]['href']
        movie_id = int(movie_id)
        tmp = (df.loc[ix, 'link'].split('/')[-1]
                             .replace('.avi', '.%d.avi' % movie_id))
        df.loc[ix, 'avi'] = ''.join([c if c not in substitute else '_' for c in tmp])

        
    return df.drop(df.columns[0], axis=1)


def mitocheck_to_tiff(link, avi):
    """Generate shell script to download mitocheck .avi and convert
    to 8-bit tiff stack. Requires ffmpeg and libtiff to run.
    """
    avi_short = avi.replace('.avi', '')
    cmd1 = 'wget "%s" -O %s' % (link, avi)
    cmd2 = 'ffmpeg -i %s -compression_algo raw -pix_fmt gray %s.%%3d.tiff'
    cmd2 = cmd2 % (avi, avi_short)
    cmd3 = 'tiffcp %s.???.tiff %s.tif' % (avi_short, avi_short)
    cmd4 = 'rm %s.???.tiff' % avi_short

    return '\n'.join([cmd1, cmd2, cmd3, cmd4])

def primerbank(query, kind='NCBI Gene Symbol', species='Human'):
    """Retrieve qPCR primer sequences from PrimerBank.
    """
    url = 'https://pga.mgh.harvard.edu/cgi-bin/primerbank/new_search2.cgi'
    data = {'selectBox': kind,
            'species':   species,
            'searchBox': query}
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*InsecureRequestWarning.*')
        r = requests.post(url, verify=False, data=data)
        
    if 'No primer pair is found' in r.text:
        print 'No primer pairs found for %s.' % query
        return None
    tables = pd.read_html(r.text, match='Primer Pair', index_col=0)[1:]
    results = []
    for table in tables:
        result = {}
        result['Amplicon Size'] = table.loc['Amplicon Size', 1]
        result['PrimerBank ID'] = table.loc['PrimerBank ID', 1]
        result['Forward Primer'] = tuple(table.loc['Forward Primer'])
        result['Reverse Primer'] = tuple(table.loc['Reverse Primer'])
        results += [result]
    return results

wolfram_client = None

def _wolfram_setup():
    import wolframalpha
    with open(lasagna.config.wolfram, 'r') as fh:
        app_id = fh.read().splitlines()[0]
    
    client = wolframalpha.Client(app_id)
    return client

def wolfram(query, show=True):
    from IPython import display
    global wolfram_client
    if wolfram_client is None:
        wolfram_client = _wolfram_setup()
    result = _wolfram(query, wolfram_client)
    if len(result.pods)<2:
        return 'no result'
    pod = result.pods[1]
    if show:
        display.display(display.Image(url=pod.img, format='.gif'))
    return pod.text

@Memoized
def _wolfram(query, client):
    return client.query(query)


