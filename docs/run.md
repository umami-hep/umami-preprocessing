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
The prepare stage (`--prep`) checks first the number of inital jets that are available per group/sample. For each of the entries in the `pattern` of the group, it checks how many jets are in total available. If this differs too much between the entries in `pattern`, an error is thrown because it indicates that you will might introduce biases in the training. For example, usually entries in `pattern` are different MC campaigns and by using drastically different numbers of initial jets, a campaign dependency can be introduced. If you manually checked it and you expect large differences, you can skip this by adding the command line argument `--skip-sample-check`. If you run the script the first time and you want to run the prepare stage in parallel, please let this script run first! It creates virtual datasets for each entry in `pattern` which could become corrupted if you do run this script in parallel multiple times! Instructions on how to run this check stand-alone can be found in [here](#additional-scripts-initial-sample-check).

Afterwards, the prepare stage reads a specified number of jets (`num_jets_estimate_hist`) for each flavor and constructs histograms of the resampling variables. These histograms are stored in `<base_dir>/hists`.

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
    preprocess --config configs/config.yaml --prep --component lowpt_ttbar_ghostsplitbjets
    ```

    It is hardly discouraged to run multiple steps with this option enabled. This option is mainly to paralellise the processing on HPCs. In addition, do not run this in the same job with multiple threads! h5py has access issues when the same file is read by multiple threads in the same job. Use multiple instances/jobs to run this.

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

    Similar to the `--prep` step, it is hardly discouraged to run multiple steps with this option enabled. This option is mainly to paralellise the processing on HPCs. Once all regions are resampled, you can continue with the following steps.

    If you want to go one step further, you can also tell the resampling to resample each component in the region in it's own process. To do so, you need to provide the `--region` command line argument together with the `--component` command line argument. Very important is here the full name of the component, which is constructed in the same way as already explained in the paralellisation chapter of the Prepare stage.

    The command to run the specific region would look like this:

    ```bash
    preprocess --config configs/config.yaml --resample --region lowpt --component lowpt_ttbar_ghostsplitbjets
    ```

    Similar to the `--prep` step and the previous `--region` explanation, it is hardly discouraged to run multiple steps with this option enabled. This option is mainly to paralellise the processing on HPCs. Once all components from all regions are resampled, you can continue with the following steps. Furthermore, do not run this in the same job with multiple threads! h5py has access issues when the same file is read by multiple threads in the same job. Use multiple instances/jobs to run this.
    Also, please do NOT use this functionality if you don't have fast I/O (Harddrives). This is very heavy in terms of I/O load and ends up to be slower if you are using "default" HDD drives.

#### 3. Merge 
The merge stage (`--merge`) combines the resampled samples into a single file named `<tbase_dir>/<out_dir>/pp_output_<split>.h5`.
It also handles shuffling.

#### 4. Normalise 
The normalise stage (`--norm`) calculates scaling and shifting values for all variables intended for training based on (`num_jets_estimate_norm`). The results are stored in` <tbase_dir>/<out_dir>/norm_dict.yaml`.

#### 5. Plotting 

The plotting stage (`--plot`) produces histograms of resampled variables to verify the resampling quality.
You can find these plots in `<tbase_dir>/plots/`.

### Additional Scripts: Initial Sample Check

The check for the inital samples from the prepare stage can also be run stand-alone. This is important if you plan to run in parallel mode. To do so, you can simply use the following command:

```bash
check_input_samples --config_path <path/to/your/config>
```

You can also add the `--deviation-factor`, which is by default `10.0` and the `--verbose` flags. The latter will print the number of inital jets to your terminal.

