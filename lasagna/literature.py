import urllib2
import pandas as pd
from bs4 import BeautifulSoup as bs



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