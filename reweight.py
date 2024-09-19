import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_dd
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from ftag.hdf5 import H5Writer, H5Reader
import dataclasses
from pathlib import Path
import sys
from typing import Callable, Union
import tqdm
import argparse

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_file", '-i', type=str, required=True)
#     parser.add_argument("--out_file", '-o', type=str, required=False)
#     parser.add_argument("--hists_dir", '-h', type=str, required=False)
#     parser.add_argument("--config", '-c', type=str, required=False)
#     return parser.parse_args()


# import tqdm
# Try on this file first
input_file="/share/lustre/npond/datasets/xbb/reweight_vs_resample/no_resample/output/pp_output_train.h5"
out_file="/share/lustre/npond/datasets/xbb/reweight_vs_resample/no_resample/output/pp_output_train_weighted.h5"
hists_dir="/share/lustre/npond/datasets/xbb/reweight_vs_resample/no_resample/output/histograms.h5"

def join_structured_arrays(arrays: list):
    """Join a list of structured numpy arrays.

    See https://github.com/numpy/numpy/issues/7811

    Parameters
    ----------
    arrays : list
        List of structured numpy arrays to join

    Returns
    -------
    np.array
        A merged structured array
    """
    dtype: list = sum((a.dtype.descr for a in arrays), [])
    newrecarray = np.empty(arrays[0].shape, dtype=dtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]

    return newrecarray



file = h5py.File(input_file, "r")
# file = {'jets' : file['jets'][:100]}
print(file['jets'].dtype.names)

def plot_hist(data, var, bins, weights, fname):
    # fig, axs = plt.subplots(1, 1, sharex=True)
    plt.figure()
    for cls in np.unique(data['flavour_label']):
        mask = data['flavour_label'] == cls
        plt.hist(data[mask][var], bins=bins, histtype='step', label=f"Class {cls}", density=True)
    plt.yscale('log')
    plt.legend()
    path = Path(input_file).parent / f"{fname}_unweighted_{var}.png"
    print('\t', path)
    plt.savefig(path)
    plt.close()

    plt.figure()
    for cls in np.unique(data['flavour_label']):
        mask = data['flavour_label'] == cls
        plt.hist(data[mask][var], bins=bins, histtype='step', label=f"Class {cls}", density=True, weights=weights[mask])
    plt.yscale('log')
    plt.legend()
    path = Path(input_file).parent / f"{fname}_reweighted_{var}.png"
    print('\t', path)
    plt.savefig(path)
    plt.close()

    plt.figure()
    # define log width bins
    # w_bins = np.linspace(np.min(weights), np.max(weights), 100)
    print(weights.shape, weights[weights > 0].shape)
    w_bins = np.logspace(np.log10(np.min(weights[weights > 0])), np.log10(np.max(weights)), 100)
    # print(len(weights))
    # print(len(weights[weights < 0]), len(weights[weights==0]))
    # print(np.min(weights), np.max(weights))
    # print(w_bins)
    for cls in np.unique(data['flavour_label']):
        plt.hist(weights[data['flavour_label'] == cls], bins=w_bins, histtype='step', label=f"Class {cls}")
    print("Sum of weights", np.sum(weights), " for number of jets ", len(weights))
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig(Path(input_file).parent / f"{fname}_weights.png")
    plt.close()
def bin_jets(array: dict, bins: list) -> np.ndarray:
    """Create the histogram and bins for the given resampling variables.

    Parameters
    ----------
    array : dict
        Dict with the loaded jets and the resampling
        variables.
    bins : list
        Flat list with the bins which are to be used.

    Returns
    -------
    hist : np.ndarray, shape(nx1, nx2, nx3,...)
        The values of the selected statistic in each two-dimensional bin.
    out_bins : (N,) array of ints or (D,N) ndarray of ints
        This assigns to each element of `sample` an integer that represents the
        bin in which this observation falls.  The representation depends on the
        `expand_binnumbers` argument.  See `Notes` for details.
    """
    hist, _, out_bins = binned_statistic_dd(
        sample=s2u(array),
        values=None,
        statistic="count",
        bins=bins,
        expand_binnumbers=True,
    )
    out_bins -= 1
    return hist, out_bins



class Distribution:

    def __init__(self, name : str, bins: list[np.ndarray], values: np.ndarray):
        self.name
        self.bins = bins
        self.values = values

        assert len(bins) == len(values) + 1
        # more checks?
    def __str__(self) -> str:
        return self.name

@dataclasses.dataclass
class Reweight:
    group : str # The group our variables in the h5 file are in
    reweight_vars : list[str] # The variables we want to reweight
    bins : list[np.ndarray] # The bins we want to use for the reweighting
    class_var : str | None = None
    class_target : int | tuple | None = None
    distribution_target : Distribution | None = None # TODO implement this?
    target_hist_func: Union[Callable, None] = None
    target_hist_func_name: str | None = None


    def __post_init__(self):

        if self.class_var is None and self.class_target is not None:
            raise ValueError("Cannot set class_target without setting class_var")
        if self.class_target is not None and self.distribution_target is not None:
            raise ValueError("Only one of class_target or distribution_target can be set")
        if self.target_hist_func is not None:    
            if self.target_hist_func_name is None:
                self.target_hist_func_name = self.target_hist_func.__name__
    
    def __repr__(self):
        target_str = 'target_' 
        if self.target_hist_func_name is not None:
            target_str += f"{self.target_hist_func_name}_"
        if self.class_target is not None:
            if isinstance(self.class_target, (list, tuple)):
                target_str += '_'.join(map(str, self.class_target))
            else:
                target_str += f"{self.class_target}_{self.class_var}"
        elif self.distribution_target is not None:
            target_str += f"{self.distribution_target}"
        else:
            target_str += 'none'
        return f"weight_{self.group}_{'_'.join(self.reweight_vars)}_{self.class_var}_{target_str}"


def calculate_weights(
    input_file : str,
    reweights : list[Reweight],
):
    '''
    Generates all the calculate_weights for the reweighting and returns them in a dict
    of the form:
    {
        'group_name' : {
            'repr(reweight)' : {
                'bins': np.ndarray, # The bins used for the histogram
                'histograms' : {
                        label_0 : hists_for_label_0, # np.ndarray
                        label_1 : hists_for_label_1,
                        ...
                    }
                }
        
    }
    
    '''

    print(f"Calculating weights for {len(reweights)} reweights")

    reader = H5Reader(input_file)
    all_vars = {}

    # Get the variables we need to reweight
    for rw in reweights:
        rw_group = rw.group
        if rw_group not in all_vars:
            all_vars[rw_group] = []
        if rw.class_var is not None:
            all_vars[rw_group].append(rw.class_var)
        all_vars[rw_group].extend(rw.reweight_vars)
    all_vars = {k: list(set(v)) for k,v in all_vars.items()}
    num_in_hists = {}
    all_histograms = {}
    for batch in tqdm.tqdm(reader.stream(all_vars), total=reader.num_jets / reader.batch_size):

        # Keep track of how many items we've used to generate our histograms
        for k, v in batch.items():
            if k not in num_in_hists:
                num_in_hists[k] = v.shape[0]
            else:
                num_in_hists[k] += v.shape[0]

        for rw in reweights:
            rw_group = rw.group
            if rw_group not in batch:
                continue
            data = batch[rw_group]

            if len(data.shape) != 1:
                assert 'valid' in data.dtype.names
                data = data[data['valid']]
            
            if rw.class_var is not None:
                classes = np.unique(data[rw.class_var])
            else:
                classes = [None]
            for cls in classes:
                mask = data[rw.class_var] == cls
                hist, outbins = bin_jets(data[mask][rw.reweight_vars], rw.bins)
                if rw.class_var is not None:
                    cls = str(cls)
                if rw_group not in all_histograms:
                    all_histograms[rw_group] = {}
                if repr(rw) not in all_histograms[rw_group]:
                    all_histograms[rw_group][repr(rw)] = {
                        'bins' : rw.bins,
                        'histograms' : {}
                    }
                if cls not in all_histograms[rw_group][repr(rw)]['histograms']:
                    all_histograms[rw_group][repr(rw)]['histograms'][cls] = hist
                else:
                    all_histograms[rw_group][repr(rw)]['histograms'][cls] += hist

        # break
    
    all_targets = {}
    for rw in reweights:
        rw_group = rw.group
        if rw_group not in all_histograms:
            raise ValueError(f"Group {rw_group} not found in histograms... What?")
            
        if rw_group not in all_targets:
            all_targets[rw_group] = {}
        
        rw_rep = repr(rw)

        target = None

        if isinstance(rw.class_target, int): 
            target = all_histograms[rw_group][rw_rep]['histograms'][str(rw.class_target)]
        elif isinstance(rw.class_target, str) and rw.class_target == 'mean':
            
            for cls, hist in all_histograms[rw_group][rw_rep]['histograms'].items():
                
                if target is None:
                    target = hist.copy()
                else:
                    target += hist
            target /= len(all_histograms[rw_group][rw_rep]['histograms'])
        elif isinstance(rw.class_target, (list, tuple)):
            for cls, hist in all_histograms[rw_group][rw_rep]['histograms'].items():
                cast_cls_target = tuple(map(str, rw.class_target))
                if cls in cast_cls_target:
                    if target is None:
                        target = hist.copy()
                    else:
                        target += hist
            target /= len(rw.class_target)
        else:
            raise ValueError("Unknown class_target type")
        
        if np.any(target == 0):
            num_zeros = np.sum(target == 0)
            print(f"Target histogram has {num_zeros} bins with zero entries out of total {target.shape} : {repr(rw)}")
        if np.any(target < 0):
            raise ValueError(f"Target histogram has bins with negative entries : {repr(rw)}")
        if np.any(np.isnan(target)):
            raise ValueError(f"Target histogram has bins with NaN entries : {repr(rw)}")

        # Apply the target histogram function
        if rw.target_hist_func is not None:
            target = rw.target_hist_func(target)
        
        

        all_targets[rw_group][rw_rep] = target

    output_weights = {}
    for rw in reweights:
        rw_group = rw.group
        rw_rep = repr(rw)
        if rw_group not in output_weights:
            output_weights[rw_group] = {}
        if rw_rep not in output_weights[rw_group]:
            output_weights[rw_group][rw_rep] = {}
        output_weights[rw_group][rw_rep] = {
            'weights' : {},
            'bins' : all_histograms[rw_group][rw_rep]['bins'],
            'rw_vars' : rw.reweight_vars,
            'class_var' : rw.class_var,
        }
        for cls, hist in all_histograms[rw_group][rw_rep]['histograms'].items():
            
            output_weights[rw_group][rw_rep]['weights'][cls] = np.where(hist > 0, all_targets[rw_group][rw_rep] / hist, 0)
            
    return output_weights

def save_weights_hdf5(weights_dict, filename):
    with h5py.File(filename, 'w') as f:
        for group, data in weights_dict.items():
            group_obj = f.create_group(group)
            for reweight_name, reweight_data in data.items():
                reweight_group = group_obj.create_group(reweight_name)
                
                # Create a group for bins, as it's a list of arrays
                bins_group = reweight_group.create_group('bins')
                for i, bin_array in enumerate(reweight_data['bins']):
                    bins_group.create_dataset(f'bin_{i}', data=bin_array)

                reweight_group.create_dataset('rw_vars', data=np.array(reweight_data['rw_vars'], dtype=h5py.special_dtype(vlen=str)))
                reweight_group.create_dataset('class_var', data=np.array([reweight_data['class_var']], dtype=h5py.special_dtype(vlen=str)))

                # Save histograms
                hist_group = reweight_group.create_group('weights')
                for label, hist in reweight_data['weights'].items():
                    hist_group.create_dataset(f'{label}', data=hist)
                


def load_weights_hdf5(filename):
    weights_dict = {}
    with h5py.File(filename, 'r') as f:
        # Iterate through the groups in the file (top-level groups represent 'group_name')
        for group in f.keys():
            weights_dict[group] = {}
            group_obj = f[group]
            # For each group, iterate through the reweight names
            for reweight_name in group_obj.keys():
                reweight_group = group_obj[reweight_name]
                
                # Load the bins, which is now a list of arrays
                bins_group = reweight_group['bins']
                bins = [bins_group[f'bin_{i}'][:] for i in range(len(bins_group))]
                
                reweight_vars = [var.decode('utf-8') for var in reweight_group['rw_vars'][:]]
                class_var = [var.decode('utf-8') for var in reweight_group['class_var'][:]][0]
                # Load the histograms
                histograms = {}
                hist_group = reweight_group['weights']
                for label in hist_group.keys():
                    histograms[label] = hist_group[label][:]
                
                # Reconstruct the structure
                weights_dict[group][reweight_name] = {
                    'bins': bins,
                    'weights': histograms,
                    'rw_vars': reweight_vars,
                    'class_var': class_var,
                }
    return weights_dict


pt_bins = np.linspace(250_000, 1_300_000, 50)

# pt_bins = np.array([20_000, 100_000, 250_000, 6_000_000])
abs_eta_bins = np.linspace(0, 2, 20)
mass_bins = np.linspace(50_000, 300_000, 50)
# print(pt_bins)

reweight_pt_eta_cjets = Reweight(
    group='jets',
    reweight_vars=['pt', 'eta', 'mass'],
    bins=[pt_bins, abs_eta_bins, mass_bins],
    class_var='flavour_label',
    class_target=1, # target c-jets
)
reweight_pt_eta_bjets = Reweight(
    group='jets',
    reweight_vars=['pt', 'eta', 'mass'],
    bins=[pt_bins, abs_eta_bins, mass_bins],
    class_var='flavour_label',
    class_target=0, # target c-jets
)
reweight_pt_eta_hfjets = Reweight(
    group='jets',
    reweight_vars=['pt', 'eta', 'mass'],
    bins=[pt_bins, abs_eta_bins, mass_bins],
    class_var='flavour_label',
    class_target=(0,1), # target HF-jets
)

reweight_pt_eta_meanjets = Reweight(
    group='jets',
    reweight_vars=['pt', 'eta', 'mass'],
    bins=[pt_bins, abs_eta_bins, mass_bins],
    class_var='flavour_label',
    class_target='mean', # target mean
)

reweight_pt_eta_logb = Reweight(
    group='jets',
    reweight_vars=['pt', 'eta', 'mass'],
    bins=[pt_bins, abs_eta_bins, mass_bins],
    class_var='flavour_label',
    class_target=0, # target mean
    target_hist_func=np.log
)

all_reweights = [
    reweight_pt_eta_cjets, reweight_pt_eta_bjets, 
    
    reweight_pt_eta_hfjets, 
    reweight_pt_eta_meanjets, 
    
    reweight_pt_eta_logb
    ]

if not Path(hists_dir).exists():
    calculated_weights = calculate_weights(input_file, all_reweights)
    save_weights_hdf5(calculated_weights, hists_dir)

# sys.exit(0)
calculated_weights = load_weights_hdf5(hists_dir)

# print(calculated_weights)

def get_sample_weights(batch, calculated_weights, scale : dict):
    '''
    Parameters
    ----------
    batch : dict
        A dictionary of numpy arrays, where the keys are the group names
        and the values are the structured arrays of the data
    calculated_weights : dict
        A dictionary of the calculated weights, as returned by `calculate_weights`
    scale : dict
        A dictionary of the scaling factors for the weights, to ensure that the 
        sum of all weights is equal to the number of jets. If this is empty,
        it will be populated with the scaling factors
    '''
    sample_weights = {}
    for group, reweights in calculated_weights.items():
        if group not in sample_weights:
            sample_weights[group] = {}
        if group not in scale:
            scale[group] = {}
        # print(reweights)
        for rwkey, rw in reweights.items():
            
            rw_vars = rw['rw_vars']
            class_var = rw['class_var']
            _, bins = bin_jets(batch[group][rw_vars], rw['bins'])
            this_weights = np.zeros(batch[group][class_var].shape, dtype=float)
            for i in range(this_weights.shape[0]):
                bin_idx = bins[:, i]
                cls = batch[group][class_var][i]
                thishist = rw['weights'][str(cls)][tuple(bin_idx)]
                this_weights[i] = np.where(thishist > 0, thishist, 0)
            # print(this_weights)
            if rwkey not in scale[group]:
                # We return the scale such that
                # final_weight = weight * scale
                # we can therefore just divide by the mean of the weights to 
                # get the scale
                scale[group][rwkey] = 1/np.mean(this_weights)
            sample_weights[group][rwkey] = this_weights * scale[group][rwkey]
        
    # print(sample_weights)

    sample_w_as_struct_arr = {}

    for group, reweights in sample_weights.items():
        dtype = [(key, arr.dtype) for key, arr in reweights.items()]
        structured_array = np.zeros(len(next(iter(reweights.values()))), dtype=dtype)
        for key in reweights:
            structured_array[key] = reweights[key]
        sample_w_as_struct_arr[group] = structured_array

    # print(sample_w_as_struct_arr)
        
    return sample_w_as_struct_arr, scale

def write_sample_with_weights(
    input_file : str,
    output_file : str,
    weights : dict,
):
    print("Writing weights to ", output_file)
    all_groups = {}
    with h5py.File(input_file, 'r') as f:
        for group in f.keys():
            all_groups[group] = None
    reader = H5Reader(input_file)
    writer = None
    additional_vars = {}

    for group, reweight in weights.items():
        
        additional_vars[group] = list(reweight.keys())
        
    dtypes = { k : v.descr for k, v in reader.dtypes().items() }
    # new_dtypes = {}
    # for group, g_dtypes in dtypes.items():
    #     new_dtypes[group] = []

    # print(dtypes)
    for group, rw_output_names in additional_vars.items():
        for rw_name in rw_output_names:
            dtypes[group] += [(rw_name, 'f4')]
    
    # The amount we scale all final weights by, to ensure that the sum of all weights 
    # is (approximatly) equal to the number of jets
    scale = {}
    for batch in tqdm.tqdm(reader.stream(all_groups), total=reader.num_jets / reader.batch_size):
        all_sample_weights, scale = get_sample_weights(batch, weights, scale)
        to_write = {}
        for key in batch.keys():
            if key in all_sample_weights:
                to_write[key] = join_structured_arrays([batch[key], all_sample_weights[key]])
            else:
                to_write[key] = batch[key]
        # print(batch.keys())
        if writer is None:
            shapes = {k: (reader.num_jets,) + v.shape[1:] for k, v in to_write.items()}
            # print(shapes)
            writer = H5Writer(output_file, dtypes, shapes, shuffle=False)
        writer.write(to_write)
        # to_write = {}
        # for group, data in batch.items():

    # print(dtypes)

    # writer = H5Writer(output_file)
    # for batch in reader.stream():
    #     for group, data in batch.items():
    #         if group not in weights:
    #             writer.write({group : data})
    #             continue
    #         for reweight_name, reweight_data in weights[group].items():
    #             bins = reweight_data['bins']
    #             weights = reweight_data['weights']
    #             for cls, hist in weights.items():
    #                 mask = data['flavour_label'] == cls
    #                 hist, outbins = bin_jets(data[mask][reweight.reweight_vars], bins)
    #                 weights = np.where(hist > 0, hist / weights, 0)
    #                 data[mask]['weights'] = weights
    #             writer.write({group : data})
    # writer.close()
# print('LOL')
# sys.exit(0)
write_sample_with_weights(
    input_file,
    out_file,
    calculated_weights
)

def make_plots(all_rw, fpath):

    h5 = h5py.File(fpath, "r")

    for rw in all_rw:
        data = h5[rw.group][:]

        for v, b in zip(rw.reweight_vars, rw.bins):
            weights = h5[rw.group][f'{repr(rw)}']
            # print(data, v, rw.bins, weights)
            plot_hist(data, v, b, weights, rw)

make_plots(all_reweights, out_file)

# reweight_pt_eta_meanjets = Reweight(
#     group='jets',
#     reweight_vars=['pt_btagJes', 'absEta_btagJes'],
#     bins=[pt_bins, np.linspace(0, 2.5, 20)],
#     class_var='flavour_label',
#     class_target='mean', # target mean
# )
# reweight_pt_eta_logmeanjets = Reweight(
#     group='jets',
#     reweight_vars=['pt_btagJes', 'absEta_btagJes'],
#     bins=[pt_bins, np.linspace(0, 2.5, 20)],
#     class_var='flavour_label',
#     class_target='mean', # target mean
#     target_hist_func=np.log
# )

# reweight_pt_eta_logbcjets = Reweight(
#     group='jets',
#     reweight_vars=['pt_btagJes', 'absEta_btagJes'],
#     bins=[pt_bins, np.linspace(0, 2.5, 20)],
#     class_var='flavour_label',
#     class_target=(0,1), # target mean
#     target_hist_func=np.log
# )

# # reweight_truth_hadrons = Reweight(
# #     group='truth_hadrons',
# #     reweight_vars=['pt', 'Lxy'],
# #     bins=[np.linspace(20_000, 200_000, 50)],
# #     class_var='flavour',
# #     class_target=5, # target b-hadrons
# # )

# def generate_weights(h5, reweight: Reweight, hist_path=None):

#     data = h5[reweight.group][:]

#     # if Path(hist_path).exists():
#     original_shape = data.shape

#     is_1d = len(data.shape) == 1
#     full_mask = None
#     if not is_1d:
#         assert 'valid' in data.dtype.names
#     print(reweight.reweight_vars)
#     print(data.shape)
#     # Ensure all data is in the correct range
#     for v, b in zip(reweight.reweight_vars, reweight.bins):
#         up,low = b[-1], b[0]

#         mask = (data[v] < up) & (data[v] > low)
#         if full_mask is None:
#             full_mask = mask
#         else:
#             full_mask &= mask
#     if is_1d:
#         data = data[full_mask]
#     else:
#         # For multi-dimension, just set all outside the mask to False
#         data['valid'][~full_mask] = False
#     print(full_mask.shape)
#     if not is_1d:

#         valid_indices = np.nonzero(data['valid'])
#         print(valid_indices)
#         # valid_indices = tuple(np.sort(idx) for idx in valid_indices)
#         data = data[valid_indices]
#     else:
#         # full_mask=None
#         valid_indices = None


#     # print('uhh', data.shape)
#     weights = np.zeros(data[reweight.class_var].shape, dtype=float)
    
#     if reweight.class_var is None:
#         class_var = np.ones(data[reweight.reweight_vars[0]].shape, dtype=int)
#     else:
#         class_var = data[reweight.class_var]
    
#     classes = np.unique(class_var)

#     histograms = {}
#     all_outbins = np.zeros((len(reweight.reweight_vars), class_var.shape[0]), dtype=int)
#     target = None

#     for cls in classes:
#         mask = class_var == cls
#         hist, outbins = bin_jets(data[mask][reweight.reweight_vars], reweight.bins)
#         histograms[cls] = hist
#         # all_outbins[cls] = outbins
#         all_outbins[:, mask] = outbins
#         if isinstance(reweight.class_target, int) and reweight.class_target == cls:
#             target = hist
#         elif isinstance(reweight.class_target, str) and reweight.class_target == 'mean':
#             if target is None:
#                 target = hist
#             else:
#                 target += hist
#         elif isinstance(reweight.class_target, (list, tuple)) and cls in reweight.class_target:
#             if target is None:
#                 target = hist
#             else:
#                 target += hist
    
#     if reweight.class_target == 'mean':
#         target /= len(classes)
#     elif isinstance(reweight.class_target, (list, tuple)):
#         target /= len(reweight.class_target)
    
#     # Apply the target histogram function
#     if reweight.target_hist_func is not None:
#         target = reweight.target_hist_func(target)
    



#     # doing this as a for loop is ugly, but for *some* reason its actually faster than
#     # the vectorized version commented out below
#     for i in range(weights.shape[0]):
#         bin_idx = all_outbins[:, i]
#         cls = class_var[i]
#         thishist = histograms[cls][tuple(bin_idx)]
#         weights[i] = np.where(thishist > 0, target[tuple(bin_idx)] / thishist, 0) 
#     # bin_indices = tuple(all_outbins)
#     # cls_histogram_values = np.array([histograms[cls][bin_indices] for cls in class_var])
#     # target_histogram_values = target[bin_indices]
#     # weights = np.where(cls_histogram_values > 0, target_histogram_values / cls_histogram_values, 0)
#     if valid_indices is not None:
#         # weights = weights[valid_indices]
#         data_out = np.zeros(original_shape, dtype=data.dtype)
#         weights_out = np.full(original_shape, -1.0, dtype=float)
#         data_out[valid_indices] = data
#         weights_out[valid_indices] = weights
#         # data = data[valid_indices]
#         return data_out, weights_out, full_mask
#     # Avoid division by zero and calculate weights
#     return data, weights, full_mask


# def run_full_reweight(h5path, reweights : list[Reweight]):

#     h5 = h5py.File(h5path, "r")
#     rw_vars = []
#     rw_bins = []
#     rw_names = []
#     rw_groups = []
#     rw_weights = []

#     all_weights = []
#     all_groups = []

#     existing_dtype_by_group = {
#         group: h5[group].dtype.descr for group in h5.keys()
#     }
#     new_groups = {}
#     for rew in reweights:
#         rw_vars += rew.reweight_vars
#         rw_bins += rew.bins
#         nt = len(rew.reweight_vars)
#         # print(rw_vars)
#         # print(rw_bins)

#         rw_names += [repr(rew)]*nt
#         rw_groups += [rew.group]*nt
#         all_groups += [rew.group]
#         data, weights, mask = generate_weights(h5, rew)
#         rw_weights += [weights]*nt
        
#         all_weights.append((mask, weights))
#         # print(rew.group)
#         if rew.group not in new_groups:
#             new_groups[rew.group] = [(str(rew), weights.dtype)]
#         else:
#             new_groups[rew.group].append((str(rew), weights.dtype))
#     merged_groups = {}
#     # print("all rw vars", len(rw_vars), rw_vars)
#     # print("all rw bins", len(rw_bins), rw_bins)
#     print(len(all_groups), len(all_weights), len(rw_names), len(rw_vars), len(rw_bins))
#     with h5py.File(out_file, "w") as f:
#         for group in existing_dtype_by_group.keys():
#             merged_groups[group] = existing_dtype_by_group[group].copy()
#             if group in new_groups:
#                 print('merging', merged_groups[group])
#                 merged_groups[group].extend(new_groups[group])
#             newdata = np.zeros(h5[group].shape, dtype=merged_groups[group])
#             for name, dtype in existing_dtype_by_group[group]:
#                 newdata[name] = h5[group][name]
#             if group in new_groups:
                
#                 for name, dtype in new_groups[group]:
#                     for w, g in zip(all_weights, all_groups):
#                         if g == group:
#                             print(newdata[name].shape, w[0].shape, w[1].shape, )
#                             if len(w[1].shape) == 1:
#                                 newdata[name][w[0]] = w[1]
#                             else:
#                                 newdata[name] = w[1]

#             f.create_dataset(group, data=newdata)
#             # Copying over attributes from the original dataset
#             for attr_name, attr_value in h5[group].attrs.items():
#                 f[group].attrs[attr_name] = attr_value
#         for group, gbins, weights, v, name in zip(rw_groups, rw_bins, rw_weights, rw_vars,rw_names):
#             print("Making plots?", group)
#             print(group, name, v )
#             if group=='jets':
                
                
#                 print('\t', v)
#                 plot_hist(f['jets'], v, gbins, weights, name)
# run_full_reweight(input_file, 
#                   [
#                     reweight_pt_eta_cjets, 
#                     reweight_pt_eta_bjets,
#                     reweight_pt_eta_meanjets,
#                     reweight_pt_eta_logmeanjets,
#                     reweight_pt_eta_logbcjets
                    

#                     ])
