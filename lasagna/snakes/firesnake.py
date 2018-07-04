import sys
sys.path.append('C:/Users/LabAdmin/Documents/GitHub/lasagna')
import json
import re
import inspect
import fire
from collections import defaultdict
import functools
import os

if sys.version_info.major == 2:
    # python 2
    import numpy as np
    import pandas as pd
    import skimage
    import lasagna.bayer
    import lasagna.features
    import lasagna.process
    import lasagna.io

def load_csv(f):
    with open(f, 'r') as fh:
        txt = fh.readline()
    sep = ',' if ',' in txt else '\s+'
    return pd.read_csv(f, sep=sep)


def load_pkl(f):
    return pd.read_pickle(f)


def load_tif(f):
    return lasagna.io.read_stack(f)


def save_csv(f, df):
    df.to_csv(f, index=None)


def save_pkl(f, df):
    df.to_pickle(f)


def save_tif(f, data_, **kwargs):
    kwargs = restrict_kwargs(kwargs, save)
    # make sure `data` doesn't come from the Snake method since it's an
    # argument name for the save function, too
    kwargs['data'] = data_
    lasagna.io.save_stack(f, **kwargs)


def restrict_kwargs(kwargs, f):
    f_kwargs = set(get_kwarg_defaults(f).keys()) | set(get_arg_names(f))
    keys = f_kwargs & set(kwargs.keys())
    return {k: kwargs[k] for k in keys}


def load_file(f):
    # python 2, unicode coming from python 3
    if not isinstance(f, (str, unicode)):
        raise TypeError
    if not os.path.isfile(f):
        raise ValueError
    if f.endswith('.tif'):
        return load_tif(f)
    elif f.endswith('.pkl'):
        return load_pkl(f)
    elif f.endswith('.csv'):
        return load_csv(f)
    else:
        raise ValueError(f)


def load_arg(x):
    one_file = load_file
    many_files = lambda x: map(load_file, x)
    
    for f in one_file, many_files:
        try:
            return f(x)
        except (ValueError, TypeError) as e:
            pass
    else:
        return x


def save_output(f, x, inputs):
    """Saves a single output file. Can extend to list if needed.
    Saving .tif might use kwargs (luts, ...) from input.
    """
    if f.endswith('.tif'):
        return save_tif(f, x, **inputs)
    elif f.endswith('.pkl'):
        return save_pkl(f, x)
    elif f.endswith('.csv'):
        return save_csv(f, x)
    else:
        raise ValueError('not a recognized filetype: ' + f)


def get_arg_names(f):
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        return argspec.args
    n = len(argspec.defaults)
    return argspec.args[:-n]


def get_kwarg_defaults(f):
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        return {}
    defaults = {k: v for k,v in zip(argspec.args[::-1], argspec.defaults[::-1])}
    return defaults


def call_from_fire(f):
    """Turn a function that acts on a mix of image data, table data and other 
    arguments and may return image or table data into a function that acts on 
    filenames for image and table data, and json-encoded values for other arguments.

    If output filename is provided, saves return value of function.

    Supported filetypes are .pkl, .csv, and .tif.
    """
    def g(input_json=None, output=None):
        
        with open(input_json, 'r') as fh:
            inputs = json.load(fh)

        # remove unused keyword arguments
        # would be better to remove only the output arguments so 
        # incorrectly named arguments raise a sensible error
        inputs = restrict_kwargs(inputs, f)

        # provide all arguments as keyword arguments
        kwargs = {x: load_arg(inputs[x]) for x in inputs}
        try:
            kwargs['wildcards']['tile'] = int(kwargs['wildcards']['tile'])
        except KeyError:
            pass
        result = f(**kwargs)

        if output:
            save_output(output, result, inputs)

    return functools.update_wrapper(g, f)


class Snake():
    @staticmethod
    def add_method(class_, name, f):
        f = staticmethod(f)
        exec('%s.%s = f' % (class_, name))

    @staticmethod
    def load_methods():
        methods = inspect.getmembers(Snake)
        for name, f in methods:
            if name not in ('__doc__', '__module__') and name.startswith('_'):
                Snake.add_method('Snake', name[1:], call_from_fire(f))

    @staticmethod
    def _stitch(data, tile_config):
        from lasagna.pipelines._20171031 import load_tile_configuration, parse_filename

        _, ij = load_tile_configuration(tile_config)
        positions = []
        for file in files:
            site = re.findall('Site[-_](\d+)', file)[0]
            site = int(site)
            positions += [ij[site]]
    
        data = np.array(data)
        if data.ndim == 3:
            data = data[:, None]

        arr = []
        for c in range(data.shape[1]):
            result = lasagna.process.alpha_blend(data[:, c], positions)
            arr += [result]
        stitched = np.array(arr)

        return stitched

    @staticmethod
    def _align(data, index_align=0, channel_offsets=None):
        """Align data using first channel. If data is a list of stacks with different 
        IJ dimensions, the data will be piled first. Optional channel offset.
        Images are aligned to the image at `index_align`.

        """

        # shapes might be different if stitched with different configs
        # keep shape consistent with DO

        shape = data[0].shape
        data = lasagna.utils.pile(data)
        data = data[..., :shape[-2], :shape[-1]]

        indices = range(len(data))
        indices.pop(index_align)
        indices_fwd = [index_align] + indices
        indices_rev = np.argsort(indices_fwd)
        aligned = lasagna.process.register_and_offset(data[indices_fwd], registration_images=data[indices_fwd,0])
        aligned = aligned[indices_rev]
        if channel_offsets:
            aligned = fix_channel_offsets(aligned, channel_offsets)

        return aligned

    @staticmethod
    def _align_log_SBS_stack(data):
        from lasagna.process import Align
        data = np.array([x[-4:] for x in data])
        aligned = np.array(map(Align.align_within_cycle, data))
        aligned = Align.align_between_cycles(aligned)
        loged = lasagna.bayer.log_ndi(aligned)
        return loged

    @staticmethod
    def _align_one_stack(data, index_align=0, channel_offsets=None, reverse=False):
        """Align data using first channel. If data is a list of stacks with different 
        IJ dimensions, the data will be piled first. Optional channel offset.
        Images are aligned to the image at `index_align`.

        """

        # shapes might be different if stitched with different configs
        # keep shape consistent with DO

        data = data[0]
        data = data[::-1]
        indices = range(len(data))
        indices.pop(index_align)
        indices_fwd = [index_align] + indices
        indices_rev = np.argsort(indices_fwd)
        aligned = lasagna.process.register_and_offset(data[indices_fwd], registration_images=data[indices_fwd,0])
        aligned = aligned[indices_rev]
        if channel_offsets:
            aligned = fix_channel_offsets(aligned, channel_offsets)

        return aligned[::-1]

    @staticmethod
    def _align_no_DAPI(data, index_align=0, channel_offsets=None):
        aligned = Snake._align(data, index_align=index_align, channel_offsets=channel_offsets)

        shape = list(aligned.shape)
        shape[1] += 1 # channels
        aligned_ = np.zeros(shape, dtype=aligned.dtype)
        aligned_[:, 1:] = aligned
        return aligned_

    @staticmethod
    def _align_DAPI_first(data, index_align=0, channels=4, channel_offsets=None):
        """Aligns trailing channels, copies in DAPI from first image stack.
        """
        dapi = data[0][0]
        data = [x[-channels:] for x in data]
        aligned = Snake._align(data, index_align=index_align, channel_offsets=channel_offsets)

        shape = list(aligned.shape)
        shape[1] += 1 # channels
        aligned_ = np.zeros(shape, dtype=aligned.dtype)
        aligned_[:, 1:] = aligned
        aligned_[:, 0] = dapi
        return aligned_

    @staticmethod
    def _consensus_DO(data):
        """Use variance to estimate DO.
        """
        if data.ndim == 4:
            consensus = np.std(data[:, -4:], axis=(0, 1))
        elif data.ndim == 3:
            consensus = np.std(data[-4:], axis=0)

        return consensus
    
    @staticmethod
    def _segment_nuclei(data, threshold=5000, **kwargs):
        """Find nuclei from DAPI. Find cell foreground from aligned but unfiltered 
        data. Expects data to have shape C x I x J.
        """
        dapi = data[0]
        if dapi.dtype in (np.float32, np.float64):
            dapi[dapi < 0] = 0
            dapi[dapi > 1] = 1

        nuclei = lasagna.process.find_nuclei(dapi, 
                                threshold=lambda x: threshold, **kwargs)
     
        return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_nuclei_bsub(data, width=50, threshold=200, **kwargs):
        """Segment cells from aligned data. To use less than full cycles for 
        segmentation, filter the input files.

        !!! matches cell labels to nuclei labels !!!
        """
        
        from scipy.ndimage.filters import minimum_filter
        data[0] = data[0] - minimum_filter(data[1], size=width)

        return Snake._segment_nuclei(data, threshold=threshold, **kwargs)
    
    @staticmethod
    def _segment_nuclei_stack(data, dapi_index, **kwargs):
        arr = []
        for frame in data[:, [dapi_index]]:
            arr += [Snake._segment_nuclei(frame, **kwargs)]

        return np.array(arr) 

    @staticmethod
    def _segment_cells(data, nuclei, threshold=750):
        """Segment cells from aligned data. To use less than full cycles for 
        segmentation, filter the input files.

        !!! matches cell labels to nuclei labels !!!
        """
        if data.ndim == 4:
            # no DAPI, min over cycles, mean over channels
            mask = data[:, 1:].min(axis=0).mean(axis=0)
        else:
            mask = np.median(data[1:], axis=0)

        mask = mask > threshold
        try:
            cells = lasagna.process.find_cells(nuclei, mask)
        except ValueError:
            print('segment_cells error -- no cells')
            cells = nuclei

        return cells

    @staticmethod
    def _segment_cells_bsub(data, nuclei, threshold=200):
        """Segment cells from aligned data. To use less than full cycles for 
        segmentation, filter the input files.

        !!! matches cell labels to nuclei labels !!!
        """
        def bsub(x, diameter=20):
            from scipy.ndimage.filters import minimum_filter
            return x - minimum_filter(x, size=diameter)
        data[1] = bsub(data[1])

        return Snake._segment_cells(data, nuclei, threshold=threshold)

    @staticmethod
    def _transform_LoG(data, bsub=False):
        if data.ndim == 3:
            data = data[None]
        loged = lasagna.bayer.log_ndi(data)
        loged[..., 0, :, :] = data[..., 0, :, :] # DAPI

        if bsub:
            loged = loged - np.sort(loged, axis=0)[-2].astype(float)
            loged[loged < 0] = 0

        return loged

    @staticmethod
    def _median_filter(data, index, width):
        from scipy.ndimage import filters
        data[index] = filters.median_filter(data[index], size=(width, width))
        return data

    @staticmethod
    def _find_peaks(data, cutoff=50):
        if data.ndim == 2:
            data = [data]
        peaks = [lasagna.process.find_peaks(x) 
                    if x.max() > 0 else x 
                    for x in data]
        peaks = np.array(peaks)
        peaks[peaks < cutoff] = 0 # for compression

        return peaks

    @staticmethod
    def _max_filter(data, width=5):
        import scipy.ndimage.filters

        if data.ndim == 3:
            data = data[None]
        
        maxed = np.zeros_like(data)
        maxed[:, 1:] = scipy.ndimage.filters.maximum_filter(data[:,1:], size=(1, 1, width, width))
        maxed[:, 0] = data[:, 0] # DAPI
    
        return maxed

    @staticmethod
    def _extract_barcodes(peaks, data_max, cells, 
        threshold_DO, cycles, wildcards, channels=4, index_DO=None):
        """Assumes sequencing covers 'GTAC'[:channels].
        """

        if data_max.ndim == 3:
            data_max = data_max[None]
        if index_DO is None:
            index_DO = Ellipsis

        data_max = data_max[:, -channels:]

        blob_mask = (peaks[index_DO] > threshold_DO) & (cells > 0)
        values = data_max[:, :, blob_mask].transpose([2, 0, 1])
        labels = cells[blob_mask]
        positions = np.array(np.where(blob_mask)).T

        bases = list('GTAC')[:channels]
        index = ('cycle', cycles), ('channel', bases)
        try:
            df = lasagna.utils.ndarray_to_dataframe(values, index)
        except ValueError:
            print('extract_barcodes failed to reshape, writing dummy')
            return pd.DataFrame()

        get_cycle = lambda x: int(re.findall('c(\d+)-', x)[0])
        df_positions = pd.DataFrame(positions, columns=['i', 'j'])
        df = (df.stack(['cycle', 'channel'])
           .reset_index()
           .rename(columns={0:'intensity', 'level_0': 'blob'})
           .join(pd.Series(labels, name='cell'), on='blob')
           .join(df_positions, on='blob')
           .assign(cycle=lambda x: x['cycle'].apply(get_cycle))
           .sort_values(['cell', 'blob', 'cycle'])
           )
        for k,v in wildcards.items():
            df[k] = v

        return df

    @staticmethod
    def _align_phenotype(data_DO, data_phenotype):
        """Align using DAPI.
        """
        _, offset = lasagna.process.register_images([data_DO[0], data_phenotype[0]])
        aligned = lasagna.utils.offset(data_phenotype, offset)
        return aligned

    @staticmethod
    def _align_phenotype_2159(data_DO, data_phenotype):
        from lasagna.process import Align
        arr = np.array([data_DO[0]] + list(data_phenotype))
        return Align.align_within_cycle(arr)[1:]

    @staticmethod
    def _segment_perimeter(data_nuclei, width=5):
        """Expand mask to generate perimeter (e.g., area around nuclei).
        """
        from lasagna.pipelines._20180302 import get_nuclear_perimeter
        return get_nuclear_perimeter(data_nuclei, width=width)

    @staticmethod
    def _extract_phenotype_FR(data_phenotype, nuclei, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA channels.
        """
        from lasagna.features import features_frameshift
        return Snake._extract_features(data_phenotype, nuclei, wildcards, features_frameshift)       

    @staticmethod
    def _extract_phenotype_FR_myc(data_phenotype, nuclei, data_sbs_1, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA, myc channels.
        """
        from lasagna.features import features_frameshift_myc
        return Snake._extract_features(data_phenotype, nuclei, wildcards, features)     

    @staticmethod
    def _extract_phenotype_translocation_ring(data_phenotype, nuclei, wildcards, width=3):
        selem = np.ones((width, width))
        perimeter = skimage.morphology.dilation(nuclei, selem)
        perimeter[nuclei > 0] = 0

        inside = skimage.morphology.erosion(nuclei, selem)
        inner_ring = nuclei.copy()
        inner_ring[inside > 0] = 0

        return Snake._extract_phenotype_translocation(data_phenotype, inner_ring, perimeter, wildcards)

    @staticmethod
    def _extract_phenotype_translocation(data_phenotype, nuclei, cells, wildcards):
        import lasagna.features

        features_n = lasagna.features.translocation_cell
        features_c = lasagna.features.translocation_nuclear

        features_n.update(lasagna.features.cell_features)
        features_c['cell'] = features_n['cell']

        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_n)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_c) 
        
        # inner join discards nuclei without corresponding cells
        df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell')], axis=1, join='inner')
                .reset_index())
        
        return df

    @staticmethod
    def _extract_features(data, nuclei, wildcards, features):
        """Extracts features in dictionary and combines with generic region
        features.
        """
        from lasagna.process import feature_table

        df = feature_table(data, nuclei, features)

        for k,v in wildcards.items():
            df[k] = v
        
        return df

    @staticmethod
    def _extract_minimal_phenotype(data_phenotype, nuclei, wildcards):
        from lasagna.pipelines._20170914_endo import feature_table_stack
        from lasagna.process import feature_table, default_object_features

        features = default_object_features.copy()
        features['cell'] = features.pop('label')
        df = feature_table(nuclei, nuclei, features)

        for k,v in wildcards.items():
            df[k] = v
        
        return df

    @staticmethod
    def _extract_phenotype_live_translocation(data_phenotype, nuclei, cells, wildcards):
        
        extract = functools.partial(Snake._extract_phenotype_translocation, 
            nuclei=nuclei, cells=cells, wildcards=wildcards)


        arr = []
        for frame, d in enumerate(data_phenotype):
            arr.append(extract(d).assign(frame=frame))
        
        return pd.concat(arr)

    @staticmethod
    def _check_cy3_quality(data, wildcards, dapi_threshold=1500, cy3_threshold=5000):
        from lasagna.process import feature_table

        labeled_dapi = skimage.measure.label(data[0] > dapi_threshold)
        labeled_cy3  = skimage.measure.label(data[1] > cy3_threshold)

        features = {'area': lambda r: r.area, 'intensity': lambda r: r.mean_intensity}  
        df_dapi = feature_table(data[0], labeled_dapi, features).assign(channel='DAPI')
        df_cy3  = feature_table(data[1], labeled_cy3,  features).assign(channel='Cy3')

        ds = 32
        arr = []
        channels = 'DAPI', 'Cy3', 'A594', 'Cy5', 'Cy7'
        for channel, frame in zip(channels, data):
            counts = np.bincount(frame.flatten()/ds, minlength=2**16/ds)
            (pd.DataFrame({
            'pixel_count': counts,
            'pixel_intensity': ds * np.arange(len(counts))})
            .assign(channel=channel)
            .pipe(arr.append))

        df = pd.concat([df_dapi, df_cy3] + arr)

        for k,v in wildcards.items():
            df[k] = v

        return df


###

def fix_channel_offsets(data, channel_offsets):
    d = data.transpose([1, 0, 2, 3])
    x = [lasagna.utils.offset(a, b) for a,b in zip(d, channel_offsets)]
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

    Snake.load_methods()
    fire.Fire(Snake)