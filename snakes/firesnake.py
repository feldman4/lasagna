import sys
sys.path.append('C:/Users/LabAdmin/Documents/GitHub/lasagna')
import json
import re

import fire
from collections import defaultdict

if sys.version_info.major == 2:
    # python 2
    import numpy as np
    import pandas as pd
    import lasagna.bayer
    import lasagna.process
    import lasagna.io
    read = lasagna.io.read_stack
    save = lasagna.io.save_stack 

class Snake():
    @staticmethod
    def stitch(input_json=None, output=None):
        from lasagna.pipelines._20171031 import load_tile_configuration, parse_filename

        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges, tile_config = [inputs[k] for k in ('input', 'display_ranges', 'tile_config')]

        _, ij = load_tile_configuration(tile_config)
        positions = []
        for file in files:
            site = re.findall('Site_(\d+)', file)[0]
            site = int(site)
            positions += [ij[site]]
    
        data = np.array([read(file) for file in files])
        # lasagna.io._imread._reset()

        arr = []
        for c in range(data.shape[1]):
            result = lasagna.process.alpha_blend(data[:, c], positions)
            arr += [result]
        stitched = np.array(arr)

        save(output, stitched, display_ranges=display_ranges)

    @staticmethod
    def align(input_json=None, output=None):
        """Align data using DAPI.
        """
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges = inputs['input'], inputs['display_ranges']

        data = np.array([read(file) for file in files])

        arr = []
        for d in data:
            # DO
            if d.shape[0] == 3:
                d = np.insert(d, [1, 2], 0, axis=0)
            arr.append(d)
        data = np.array(arr)
        dapi = data[:,0]
        aligned = lasagna.bayer.register_and_offset(data, registration_images=data[:, 0])

        save(output, aligned, display_ranges=display_ranges)
    
    @staticmethod
    def segment_nuclei(input_json=None, output=None, nuclei_threshold=5000, nuclei_area_max=1000):
        """Find nuclei from DAPI. Find cell foreground from aligned but unfiltered 
        data. Expects data to have shape C x I x J.
        """
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files = inputs['input']

        data = read(files[0])

        nuclei = lasagna.process.find_nuclei(data[0], 
                                area_max=nuclei_area_max, 
                                threshold=lambda x: nuclei_threshold)
        nuclei = nuclei.astype(np.uint16)
        save(output, nuclei)
    
    @staticmethod
    def segment_cells(input_json=None, output=None, threshold=750):
        """Segment cells from aligned data. To use less than full cycles for 
        segmentation, filter the input files.
        """
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files = inputs['input']

        data = read(files[0])
        nuclei = read(files[1])

        # no DAPI, min over cycles, mean over channels
        mask = data[:, 1:].min(axis=0).mean(axis=0)
        mask = mask > threshold

        cells = lasagna.process.find_cells(nuclei, mask)

        save(output, cells)
    
    @staticmethod
    def transform_LoG(input_json=None, output=None):
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges = inputs['input'], inputs['display_ranges']

        data = read(files[0])
        loged = lasagna.bayer.log_ndi(data)
        loged[:,0] = data[:,0]

        save(output, loged, display_ranges=display_ranges)
    
    @staticmethod
    def find_peaks(input_json=None, output=None, cutoff=50):
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges = inputs['input'], inputs['display_ranges']

        data = read(files[0])
        peaks = [lasagna.process.find_peaks(x) 
                    if x.max() > 0 else x 
                    for x in data]
        peaks = np.array(peaks)
        peaks[peaks < cutoff] = 0 # for compression

        save(output, peaks, display_ranges=display_ranges)
    
    @staticmethod
    def max_filter(input_json=None, output=None, width=5):
        import scipy.ndimage.filters
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges = inputs['input'], inputs['display_ranges']

        data = read(files[0])
        maxed = np.zeros_like(data)
        maxed[:, 1:] = scipy.ndimage.filters.maximum_filter(data[:,1:], size=(1, 1, width, width))
        maxed[:, 0] = data[:, 0] # DAPI

        save(output, maxed, display_ranges=display_ranges)

    @staticmethod
    def extract_barcodes(input_json=None, output=None):

        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges = inputs['input'], inputs['display_ranges']
        threshold_DO, index_DO, cycles = [inputs[k] for k in ('threshold_DO', 'index_DO', 'cycles')]


        peaks, data_max, cells = [read(f) for f in files]
        
        data_max = data_max[:, 1:] # no DAPI
        blob_mask = (peaks[index_DO] > threshold_DO) & (cells > 0)
        values = data_max[:, :, blob_mask].transpose([2, 0, 1])
        labels = cells[blob_mask]
        positions = np.array(np.where(blob_mask)).T

        index = ('cycle', cycles), ('channel', list('TGCA'))
        df = lasagna.utils.ndarray_to_dataframe(values, index)

        df_positions = pd.DataFrame(positions, columns=['position_i', 'position_j'])
        df = (df.stack(['cycle', 'channel'])
           .reset_index()
           .rename(columns={0:'intensity', 'level_0': 'blob'})
           .join(pd.Series(labels, name='cell'), on='blob')
           .join(df_positions, on='blob')
           )
        for k,v in inputs['wildcards'].items():
            df[k] = v
        df.to_pickle(output)


    @staticmethod
    def extract_phenotypes():
        def correlate_dapi_myc(region):
            dapi, fitc, myc = region.intensity_image_full

            filt = dapi > 0
            if filt.sum() == 0:
                assert False
                return np.nan

            dapi = dapi[filt]
            myc  = myc[filt]
            corr = (dapi - dapi.mean()) * (myc - myc.mean()) / (dapi.std() * myc.std())

            return corr.mean()

        features = {
            'corr'       : correlate_dapi_myc,
            'dapi_median': lambda r: np.median(r.intensity_image_full[0]),
            'fitc_median': lambda r: np.median(r.intensity_image_full[1]),
            'myc_median' : lambda r: np.median(r.intensity_image_full[2]),
            'cell'       : lambda r: r.label
        }

        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges = inputs['input'], inputs['display_ranges']

        data_DO, data_phenotype, nuclei = [read(f) for f in files]

        from lasagna.pipelines._20170914_endo import feature_table_stack
        dapi = data_DO[0]
        data = np.array([dapi] + list(data_phenotype[1:]))
        df = feature_table_stack(data, nuclei, features)

        for k,v in inputs['wildcards'].items():
            df[k] = v
        df.to_pickle(output)


###

def stitch_input(wildcards, format_, site_shape, tile_shape):
    """Map tile ID onto site IDs. Fill in wildcards ourselves.
    """

    d = site_to_tile(site_shape, tile_shape)
    d2 = defaultdict(list)
    [d2[v].append(k) for k, v in d.items()]

    sites = d2[int(wildcards['tile'])]
    
    arr = []
    for site in sites:
        arr.append(format_.format(site=site, **wildcards))

    return arr

def site_to_tile(site_shape, tile_shape):
        """Create dictionary from site number to tile number.
        """
        result = {}
        rows_s, cols_s = site_shape
        rows_t, cols_t = tile_shape
        for i_s in range(rows_s):
            for j_s in range(cols_s):
                i_t = int(i_s * (float(rows_t) / rows_s))
                j_t = int(j_s * (float(cols_t) / cols_s))

                site = i_s * cols_s + j_s
                tile = i_t * cols_t + j_t
                result[site] = tile
        return result


if __name__ == '__main__':

    fire.Fire(Snake)