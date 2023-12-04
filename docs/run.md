# Run

Before running UPP, make sure you have modified the configuration file according to the [configuration instructions](configuration.md)


### Basic Usage 

To run all preprocessing stages for the `train` split use:

```bash
preprocess --config configs/test.yaml
```

For a comprehensive list of available flags, refer to `preprocess --help`.

### Splits 

The data is divided into three splits: training (`train`), validation (`val`), and testing (`test`).
These splits are defined in [configuration files](configuration.md#global-cuts), typically based on the `eventNumber` variable.
By default, the train split contains 80% of the jets, while val and test contain 10% each.

If you want to preprocess the `val` or `test` split, use the `--split` argument:

```bash
preprocess --config configs/config.yaml --split val
```

You can also process `train`, `val`, and `test` with a single command using `--split=all`.

### Stages 

The preprocessing is broken up into several stages.

To run with only specific stages enabled, include the flag for the required stages:

```bash
preprocess --config configs/config.yaml --prep --resample
```

To run the whole chain excluding certain stages, include the corresponding negative flag (`--no-*`).
For example to run without plotting

```bash
preprocess --config configs/config.yaml --no-plot
```

The stages are described below.

#### 1. Prepare
The prepare stage (`--prep`) reads a specified number of jets (`num_jets_estimate`) for each flavor and constructs histograms of the resampling variables.
These histograms are stored in `<base_dir>/hists`.

#### 2. Resample 
The resample stage (`--resample`) resamples jets to achieve similar $p_T$ and $\eta$ distributions across flavours.
After execution, resampled samples for each flavor, sample, and split are saved separately in `<base_dir>/components/<split>/`.
You need to run the resampling stage even if you don't apply any resampling (e.g. you configured with `method: none`).

#### 3. Merge 
The merge stage (`--merge`) combines the resampled samples into a single file named `<tbase_dir>/<out_dir>/pp_output_<split>.h5`.
It also handles shuffling.

#### 4. Normalise 
The normalise stage (`--norm`) calculates scaling and shifting values for all variables intended for training. The results are stored in` <tbase_dir>/<out_dir>/norm_dict.yaml`.

#### 5. Plotting 

The plotting stage (`--plot`) produces histograms of resampled variables to verify the resampling quality.
You can find these plots in `<tbase_dir>/plots/`.