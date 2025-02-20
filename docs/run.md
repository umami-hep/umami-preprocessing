# Run

Before running UPP, make sure you have modified the configuration file according to the [configuration instructions](configuration.md)


### Basic Usage 

To run all preprocessing stages for the `train` split use:

```bash
preprocess --config configs/test.yaml
```

For a comprehensive list of available flags, refer to `preprocess --help`.

!!!info "If you are running on lxplus you may need to use `python3 upp/main.py` instead of `preprocess`"

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
The prepare stage (`--prep`) reads a specified number of jets (`num_jets_estimate_hist`) for each flavor and constructs histograms of the resampling variables.
These histograms are stored in `<base_dir>/hists`.

???info "Paralellisation"
    This step can be paralellised to speed up the histogram creation. To do so, you need to provide the additional `--component` flag. The argument for the flag is the name of the component, which is to be processed. The argument can be constructed when looking closer at the different blocks in the `components` part of the config file. As an example, we take the `ghost-highstat.yaml` config file from the `gn3` folder in `configs/`:

    ```yaml
    - region:
        <<: *lowpt
        sample:
        <<: *ttbar
        flavours: [ghostsplitbjets]
        num_jets: 22_000_000
        num_jets_test: 2_000_000
    ```

    The argument for the component flag can be constructed by taking the name of the region (this is defined in the definition of `lowpt`)

    ```yaml
    lowpt: &lowpt
    name: lowpt
    cuts:
        - [pt_btagJes, ">", 20_000]
        - [pt_btagJes, "<", 250_000]
    ```

    plus the name of the sample which is used (this is defined in the definition of `ttbar`)

    ```yaml
    ttbar: &ttbar
    name: ttbar
    equal_jets: False
    pattern:
        - "user.svanstro.601589.e8547_s3797_r13144_p6368.tdd.GN3_dev.25_2_27.24-09-17_v00_output.h5/*.h5" # mc20d
        - "user.svanstro.601589.e8549_s4159_r14799_p6368.tdd.GN3_dev.25_2_27.24-09-17_v00_output.h5/*.h5" # mc23a
    ```

    and finally the flavour that is used. In this case, `ghostsplitbjets`. The full name of the component is therefore: `lowpt_ttbar_ghostsplitbjets`. The full command would look like this:

    ```bash
    preprocess --config configs/config.yaml --prep --component lowpt_ttbar_bjets
    ```

    It is hardly discouraged to run multiple steps with this option enabled. This option is mainly to paralellise the processing on HPCs. 

#### 2. Resample 
The resample stage (`--resample`) resamples jets to achieve similar $p_T$ and $\eta$ distributions across flavours.
After execution, resampled samples for each flavor, sample, and split are saved separately in `<base_dir>/components/<split>/`.
You need to run the resampling stage even if you don't apply any resampling (e.g. you configured with `method: none`).

???info "Paralellisation"
    Similar to the `--prep` step, the resampling step is also able to run in parallel, but only for the different region (e.g. `lowpt` & `highpt`). To do so, you need to run with the command line argument `--region` which takes as input the region on which to run. Please ensure that all components for this region were prepared in the `--prep` step before running this!

    The command to run the specific region would look like this:

    ```bash
    preprocess --config configs/config.yaml --resample --region lowpt
    ```

    Similar to the `--prep` step, it is discouraged to run this multiple steps with this option enabled. This option is mainly to paralellise the processing on HPCs. Once all regions are resampled, you can continue with the following steps.

#### 3. Merge 
The merge stage (`--merge`) combines the resampled samples into a single file named `<tbase_dir>/<out_dir>/pp_output_<split>.h5`.
It also handles shuffling.

#### 4. Normalise 
The normalise stage (`--norm`) calculates scaling and shifting values for all variables intended for training based on (`num_jets_estimate_norm`). The results are stored in` <tbase_dir>/<out_dir>/norm_dict.yaml`.

#### 5. Plotting 

The plotting stage (`--plot`) produces histograms of resampled variables to verify the resampling quality.
You can find these plots in `<tbase_dir>/plots/`.