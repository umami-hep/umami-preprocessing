# Reweighting

A different approach to balancing classes to resampling is to instead reweight them instead.
In resampling, we bin our jets over their kinematics. We look in each bin, and if we have more jets of a given flavour compared to the target flavour, we throw those jets away. If we have less of a given flavour than the target flavour, we copy-and-paste (upsample) the jet until the bins are equal.
This ensures that the flavour ratio of any two bins is approximately constant.

This does however have some disadvantages:
- results in throwing away some jets
- upsampling introduces copies of the same jet, meaning we are passing the same data through the model more than once per epoch
- bins always fave some non-zero finite size, such that there can still be a residual distibution shape, see [this mention](https://indico.cern.ch/event/1510815/contributions/6361217/attachments/3014489/5316063/Effects%20of%20Trackless%20Jets%20and%20UPP%20on%20NN%20Based%20Preselection.pdf) of shark-toothing

We can instead reweight, where instead of throwing away or copying jets, we assign each jet in a bin a weight such that we end up with a constant weighted flavour ratio per bin.

## Running the reweighting

The reweighting consists of 4 main stages:
- component splitting
- histogram/weights calculation
- merging
- normalisation

The normalisation step utilises the same commands as in resampling, and so is not discussed here further

### Component Splitting

The first stage takes each sample and splits it into its split and flavour components. For example, if we have a ttbar sample, and we have 4 jet flavours (b,c,light,tau) then the splitting will produce 12 files: `train_bjets, val_bjets, test_bjets,...`.

To run the splitting, 
```
python upp/main.py --config {config} --split-components [--container {single container} --files {files}]
```

which will split the components. If `--container` is defined, then only that container from the config will be split. If this is not included, then all contains in the config will be split.
Similarly, if `--files` is defined then only the specified files in the specified container will be split.
Once the splitting is complete, a meta-data file is automatically created which points towards each file needed for the next stages of reweighting

#### Split on the grid
The splitting stage can be quite time consuming, and is often limited by system IO. To speed the processing up, a grid-submission script is provided.
First, you must setup a virtual environment to be packaged up and sent to the grid. This can be done by

```
setupATLAS
source upp/grid/setup_env.sh
```

Once this has setup, you can run the grid dump by:

```
python upp/grid/grid_split.py --config {config} --rucio_user {your rucio username} --tag {something to identify this dump} [--dryrun]
```

Where the dryrun flag will prepare the submission directory without submitting to the grid.

Once all your jobs are complete, you can use the download_and_prepare script:

```
python upp/grid/download_and_prepare.py -config {config} --rucio_user {your rucio username} --tag {same tag as before}
```

which will then automatically download, package up, and generate the meta data ready for the next stage.

## Generate weights

Once all the samples are prepared, we can calculate the weights. An example config can be found in `configs/GN3v01/GN3V01-RW.yaml`. We can look at the reweighting section:


```yaml

reweighting:
  num_jets_estimate: 1_500_000
  merge_num_proc: 20
  reweights:
    - group: jets
      reweight_vars: [pt_btagJes, eta_btagJes]
      bins:
        pt_btagJes:
          [
            [20_000, 250_000, 50],
            [250_000, 1_000_000, 50],
            [1_000_000, 6_000_000, 50],
          ]
        eta_btagJes: [[-2.5, 2.5, 40]]
      class_var: flavour_label
      class_target: mean

```

`num_jets_estimate` represents the number of each jet flavour used to generate the reweighting histograms. The `merge_num_proc` variable will be relevent in the next section of these docs.
Then, you have the `reweights` section, which includes a list of reweight configurations. In this example, we have the first reweight calculated over the jets group. It reweights based on the flavour-label, over the pt and eta distributions. The bins follow the same logic as in resampling.
The class target can then either be chosen as a single label (e.g, if 0 then the reweighting would target the distribution for `flavour_label==0`), or one of `mean, min, max` which will instead target either the mean distribution, or always take the maximum/minumum bin counts as the target.
The reweighting can also be performed over track variables, for example

```yaml

group: jets
reweight_vars: [pt_frac]
bins:
    pt_frac:
        [
        [0, 0.5, 50],
        [0.5, 1.0, 20]
        ]
class_var: ftagTruthOriginLabel
class_target: mean
```

Would calculate weights such that we have equivilent pt_frac distributions across the track labels.

To run the reweighting simply do

```
python upp/main.py --config {config} --rw
```

## Merging

Finally, we can merge all the relevent jets with their weights. This is done by

```
python upp/main.py --rwm --split {train/test/val}
```

This can either work in series to create 1 single large file, or we can produce multiple files with multi-processing. To do this, ensure the `global` section of the pre-processing config includes `num_jets_per_output_file` and the `reweighting` section has `merge_num_proc>1`.
This will then launch `merge_num_proc` processes, with approximately `num_jets_per_output_file` per file*.

* Due to the nature of the H5Reader, the actual number of jets per file will be slightly smaller than what is requested, on the order of 0.1%.

