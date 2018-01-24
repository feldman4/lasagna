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
            site = re.findall('Site[-_](\d+)', file)[0]
            site = int(site)
            positions += [ij[site]]
    
        data = np.array([read(file) for file in files])
        if data.ndim == 3:
            data = data[:, None]
        # GOT FUCKED UP IN NOTEBOOK MAX PROJECT
        if 'c0-DO' in file:
            data[:, 2] = data[:, 3]
            data[:, [1, 3]] = 0
        # lasagna.io._imread._reset()

        arr = []
        for c in range(data.shape[1]):
            result = lasagna.process.alpha_blend(data[:, c], positions)
            arr += [result]
        stitched = np.array(arr)

        save(output, stitched, display_ranges=display_ranges)

    @staticmethod
    def align(input_json=None, output=None):
        """Align data using DAPI. Optional channel offset.
        """
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges = inputs['input'], inputs['display_ranges']

        # might be stitched with different configs
        # keep shape consistent with DO
        data = [read(file) for file in files]
        shape = data[0].shape
        data = lasagna.io.pile(data)
        data = data[..., :shape[-2], :shape[-1]]

        arr = []
        for d in data:
            # DO
            if d.shape[0] == 3:
                d = np.insert(d, [1, 2], 0, axis=0)
            arr.append(d)
        data = np.array(arr)
        dapi = data[:,0]
        aligned = lasagna.bayer.register_and_offset(data, registration_images=data[:, 0])

        try:
            aligned = fix_channel_offsets(aligned, inputs['channel_offsets'])
        except KeyError:
            pass

        save(output, aligned, display_ranges=display_ranges)

    @staticmethod
    def align2(input_json=None, output=None):
        """Align data using 2nd channel. Align internal channels to 2nd channel.
        """
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges = inputs['input'], inputs['display_ranges']

        # might be stitched with different configs
        # keep shape consistent with DO
        data = [read(file) for file in files]
        shape = data[0].shape
        data = lasagna.io.pile(data)
        # TODO: crop zeros at boundaries
        data = data[..., :shape[-2], :shape[-1]]

        # align within rounds
        fwd = [1, 0, 2, 3, 4]
        rev = [1, 0, 2, 3, 4]
        arr = []
        for data_ in data:
            # stupid DO hack.
            if data_[1].sum() == 0:
                x = data_[2].copy()
                data_[1] = x

            data_ = data_[fwd]
            data_ = lasagna.bayer.register_and_offset(data_)
            data_ = data_[rev]
            arr += [data_]
        data = np.array(arr)

        aligned = lasagna.bayer.register_and_offset(data, registration_images=data[:, 1])

        save(output, aligned, display_ranges=display_ranges)

    @staticmethod
    def align_DAPI_H2B(input_json=None, output=None):
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges = inputs['input'], inputs['display_ranges']

        # align DAPI and rearrange channels
        # DAPI, CH1-4, CH0 (H2B)
        dapi = read(files[0])
        data = read(files[1])
        data_reg = data.mean(axis=0).astype(np.uint16)

        xs = lasagna.bayer.register_and_offset([data_reg, dapi])
        dapi_ = xs[1]
        sl = lasagna.process.trim(xs, return_slice=True)

        aligned = np.array([dapi_] + list(data[[1, 2, 3, 4, 0]]))
        aligned = aligned[sl]

        save(output, aligned, display_ranges=display_ranges)

    
    @staticmethod
    def consensus_DO(input_json=None, output=None):
        """Use variance to estimate DO.
        """
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files = inputs['input']

        data = read(files[0])
        if data.ndim == 4:
            consensus = np.std(data[:, 1:], axis=(0, 1))
        elif data.ndim == 3:
            consensus = np.std(data[1:], axis=0)

        save(output, consensus)
    
    @staticmethod
    def segment_nuclei(input_json=None, output=None):
        """Find nuclei from DAPI. Find cell foreground from aligned but unfiltered 
        data. Expects data to have shape C x I x J.
        """
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files = inputs.pop('input')
        threshold = 5000
        if 'threshold' in inputs:
            threshold = inputs.pop('threshold')

        data = read(files[0])
        nuclei = lasagna.process.find_nuclei(data[0], 
                                threshold=lambda x: threshold, **inputs)
        nuclei = nuclei.astype(np.uint16)
        save(output, nuclei)
    
    @staticmethod
    def segment_cells(input_json=None, output=None):
        """Segment cells from aligned data. To use less than full cycles for 
        segmentation, filter the input files.
        """
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files = inputs['input']
        threshold = 750
        if 'threshold' in inputs:
            threshold = inputs['threshold']

        data = read(files[0])
        nuclei = read(files[1])

        if data.ndim == 4:
            # no DAPI, min over cycles, mean over channels
            mask = data[:, 1:].min(axis=0).mean(axis=0)
        else:
            mask = np.median(data[1:], axis=0)

        mask = mask > threshold
        cells = lasagna.process.find_cells(nuclei, mask)

        save(output, cells)
    
    @staticmethod
    def transform_LoG(input_json=None, bsub=False, output=None):
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges = inputs['input'], inputs['display_ranges']

        data = read(files[0])
        if data.ndim == 3:
            data = data[None]
        loged = lasagna.bayer.log_ndi(data)
        loged[..., 0, :, :] = data[..., 0, :, :] # DAPI

        if bsub:
            loged = loged - np.sort(loged, axis=0)[-2].astype(float)
            loged[loged < 0] = 0

        save(output, loged, display_ranges=display_ranges)
    
    @staticmethod
    def find_peaks(input_json=None, output=None, cutoff=50):
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges = inputs['input'], inputs['display_ranges']

        data = read(files[0])
        if data.ndim == 2:
            data = [data]
        peaks = [lasagna.process.find_peaks(x) 
                    if x.max() > 0 else x 
                    for x in data]
        peaks = np.array(peaks)
        peaks[peaks < cutoff] = 0 # for compression

        save(output, peaks, display_ranges=display_ranges)
    
    @staticmethod
    def max_filter(input_json=None, output=None):
        import scipy.ndimage.filters
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files, display_ranges = inputs['input'], inputs['display_ranges']
        if 'width' in inputs:
            width = inputs['width']
        else:
            width = 5

        data = read(files[0])
        if data.ndim == 3:
            data = data[None]
        maxed = np.zeros_like(data)
        maxed[:, 1:] = scipy.ndimage.filters.maximum_filter(data[:,1:], size=(1, 1, width, width))
        maxed[:, 0] = data[:, 0] # DAPI

        save(output, maxed, display_ranges=display_ranges)

    @staticmethod
    def extract_barcodes(input_json=None, output=None):

        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files = inputs['input']
        threshold_DO, index_DO, cycles = [inputs[k] for k in ('threshold_DO', 'index_DO', 'cycles')]
        if index_DO is None:
            index_DO = Ellipsis

        peaks, data_max, cells = [read(f) for f in files]
        
        if data_max.ndim == 3:
            data_max = data_max[None]

        data_max = data_max[:, 1:] # no DAPI
        blob_mask = (peaks[index_DO] > threshold_DO) & (cells > 0)
        values = data_max[:, :, blob_mask].transpose([2, 0, 1])
        labels = cells[blob_mask]
        positions = np.array(np.where(blob_mask)).T

        index = ('cycle', cycles), ('channel', list('GTAC'))
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
    def extract_phenotype(input_json=None, output=None):
        def correlate_dapi_myc(region):
            dapi, myc, bkgd = region.intensity_image_full

            filt = dapi > 0
            if filt.sum() == 0:
                # assert False
                return np.nan

            dapi = dapi[filt]
            myc  = myc[filt]
            corr = (dapi - dapi.mean()) * (myc - myc.mean()) / (dapi.std() * myc.std())

            return corr.mean()

        features = {
            'corr'       : correlate_dapi_myc,
            'dapi_median': lambda r: np.median(r.intensity_image_full[0]),
            'bkgd_median': lambda r: np.median(r.intensity_image_full[1]),
            'myc_median' : lambda r: np.median(r.intensity_image_full[2]),
            'cell'       : lambda r: r.label
        }

        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files = inputs['input']

        data_phenotype, nuclei = [read(f) for f in files]

        from lasagna.pipelines._20170914_endo import feature_table_stack
        df = feature_table_stack(data_phenotype, nuclei, features)

        from lasagna.process import feature_table, default_object_features

        features = default_object_features.copy()
        features['cell'] = features.pop('label')
        df2 = feature_table(nuclei, nuclei, features)
        df = df.join(df2.set_index('cell'), on='cell')

        for k,v in inputs['wildcards'].items():
            df[k] = v
        df.to_pickle(output)

    @staticmethod
    def align_phenotype(input_json=None, output=None):
        """Align using DAPI.
        """
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)
        files = inputs['input']

        data_DO, data_phenotype = [read(f) for f in files]

        _, offset = lasagna.process.register_images([data_DO[0], data_phenotype[0]])
        aligned = lasagna.io.offset(data_phenotype, offset)

        save(output, aligned)



###

def fix_channel_offsets(data, channel_offsets):
    d = data.transpose([1, 0, 2, 3])
    x = [lasagna.io.offset(a, b) for a,b in zip(d, channel_offsets)]
    x = np.array(x).transpose([1, 0, 2, 3])
    return x

def stitch_input_sites(tile, site_shape, tile_shape):
    """Map tile ID onto site IDs. Fill in wildcards ourselves.
    """

    d = site_to_tile(site_shape, tile_shape)
    d2 = defaultdict(list)
    [d2[v].append(k) for k, v in d.items()]

    sites = d2[int(tile)]
    
    return sites

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