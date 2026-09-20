# Run

Before running UPP, make sure you have modified the configuration file according to the [configuration instructions](configuration.md)


### Basic Usage 

To run all preprocessing stages for the `train` split use:

```bash
preprocess --config upp/configs/test.yaml
```

For a comprehensive list of available flags, refer to `preprocess --help`.

!!!info "If you are running on lxplus you may need to use `python3 upp/main.py` instead of `preprocess`"

### Step by step

A complete run, from a config to the files you train on, looks like this:

```bash
# 1. check the input samples and create the virtual datasets
check_input_samples --config_path path/to/config.yaml --verbose

# 2. work out how many objects you can ask for
estimate_object_counts --config path/to/config.yaml

# 3. run the stages, one after the other
preprocess --config path/to/config.yaml --prep
preprocess --config path/to/config.yaml --resample
preprocess --config path/to/config.yaml --merge
preprocess --config path/to/config.yaml --norm
preprocess --config path/to/config.yaml --plot
```

Step 2 is only needed the first time and whenever your samples or cuts change. It tells you
the largest object counts your samples support, which you either copy into your config or
pick up automatically with `num_global_objects: auto`
(see [Automatic object counts](configuration.md#automatic-object-counts)).

Steps 3 onwards produce the `train` split. Add `--split val` and `--split test` to produce
the other two, or use `--split all` to run all three in one go.

!!!warning "Finish one stage before starting the next"

    Each stage reads what the previous one wrote, so let all jobs of a stage finish before
    submitting the next one. The one exception is the normalisation, which is computed from
    the merged training file and therefore needs the merge of the `train` split only.

???info "Running everything with a single command"

    Leaving out the stage flags runs the whole chain in order:

    ```bash
    preprocess --config path/to/config.yaml --split all
    ```

    This is the simplest way to run small setups. For anything with real statistics the
    stages are better submitted as separate batch jobs, see [Running on HPC](hpc.md).

### Logging

By default UPP logs at the `INFO` level. You can change this with the `UPP_LOG_LEVEL`
environment variable or with the `--log-level` flag, which wins over the environment variable:

```bash
export UPP_LOG_LEVEL=DEBUG
preprocess --config path/to/config.yaml --log-level WARNING
```

The available levels are `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`.

Every log message is prefixed with the date and time. Debug and info messages are written to
stdout while warnings, errors and critical messages are written to stderr, so in a batch job
the `.out` file holds the progress of the run and the `.err` file only the problems.

### Splits 

The data is divided into three splits: training (`train`), validation (`val`), and testing (`test`).
Which objects end up in which split is decided by the cuts you define in the
[configuration file](configuration.md#global-cuts), typically using the modulo of the
`eventNumber`. The split configs shipped with UPP in `upp/configs/splits/` use 80% for
training and 10% each for validation and testing, and there are k-folded variants as well.

If you want to preprocess the `val` or `test` split, use the `--split` argument:

```bash
preprocess --config path/to/config.yaml --split val
```

You can also process `train`, `val`, and `test` with a single command using `--split=all`.

!!!info "The splits are not the same data with a different name"

    Cuts and object counts differ between splits. The `test` split drops the region cuts
    entirely and applies no resampling, and if you do not set `num_global_objects_val` or
    `num_global_objects_test`, each defaults to a tenth of the training count.

### Stages 

The preprocessing is broken up into several stages.

To run with only specific stages enabled, include the flag for the required stages:

```bash
preprocess --config path/to/config.yaml --prep --resample
```

To run the whole chain excluding certain stages, include the corresponding negative flag (`--no-*`).
For example to run without plotting

```bash
preprocess --config path/to/config.yaml --no-plot
```

The stages are described below.

#### 1. Prepare
The prepare stage (`--prep`) checks first the number of initial objects that are available per group/sample. For each of the entries in the `pattern` of the group, it checks how many objects are in total available. If this differs too much between the entries in `pattern`, an error is thrown because it indicates that you will might introduce biases in the training. For example, usually entries in `pattern` are different MC campaigns and by using drastically different numbers of initial objects, a campaign dependency can be introduced. If you manually checked it and you expect large differences, you can skip this by adding the command line argument `--skip-sample-check`. Instructions on how to run this check stand-alone can be found [here](#additional-scripts-initial-sample-check).

Keep in mind that this check counts all objects of each entry in `pattern`, before any cuts are applied. It tells you whether your input samples are balanced, not how many objects are left in a component once its cuts are applied.

Afterwards, the prepare stage reads a specified number of objects (`num_global_objects_estimate_hist`) for each class and constructs histograms of the resampling variables. These histograms are stored in `<base_dir>/hists/`. This part only runs for the `train` split, and only when resampling is enabled.

!!!warning "Run the sample check once before starting parallel prepare jobs"

    The check creates the virtual datasets which wrap the files of each `pattern`. These can
    get corrupted when several jobs create them at the same time, so let a single run finish
    before submitting the prepare jobs in parallel.

???info "The available objects reported here are for the full sample"

    For each component the stage reports a line like

    ```
    Estimated 220,586,493 lowpt_ttbar_bjets objects available - 5,000,000 requested
    ```

    The requested number here is `num_global_objects_estimate_hist`, not the
    `num_global_objects` of your component, and the estimate leaves the train/val/test split
    cut out, so it is the number for the full sample rather than for one split. The stage
    only reports it and never stops because of it. For per-split numbers that take the class
    ratios into account, use [`estimate_object_counts`](#additional-scripts-object-count-estimate).

!!!info "`num_global_objects` is not used in this stage"

    Setting the counts of your components very high here to take all the statistics has no
    effect, they are only used from the resample stage onwards.

???info "Paralellisation"
    This step can be parallelized to speed up the histogram creation. To do so, you need to provide the additional `--component` flag. The argument for the flag is the name of the component, which is to be processed. The argument can be constructed when looking closer at the different blocks in the `components` part of the config file. As an example, we take the `ghost-highstat.yaml` config file from the `GN3V00` folder in `upp/configs/`:

    ```yaml
    - region:
        <<: *lowpt
        sample:
        <<: *ttbar
        classes: [ghostsplitbjets]
        num_global_objects: 22_000_000
        num_global_objects_test: 2_000_000
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
    equal_global_objects: False
    pattern:
        - "user.svanstro.601589.e8547_s3797_r13144_p6368.tdd.GN3_dev.25_2_27.24-09-17_v00_output.h5/*.h5" # mc20d
        - "user.svanstro.601589.e8549_s4159_r14799_p6368.tdd.GN3_dev.25_2_27.24-09-17_v00_output.h5/*.h5" # mc23a
    ```

    and finally the class that is used. In this case, `ghostsplitbjets`. The full name of the component is therefore: `lowpt_ttbar_ghostsplitbjets`. The full command would look like this:

    ```bash
    preprocess --config path/to/config.yaml --prep --component lowpt_ttbar_ghostsplitbjets
    ```

    Instead of building the names by hand you can also let UPP print them for you:

    ```bash
    list_components --config path/to/config.yaml
    ```

    It is hardly discouraged to run multiple steps with this option enabled. This option is mainly to parallelize the processing on HPCs. In addition, do not run this in the same job with multiple threads! h5py has access issues when the same file is read by multiple threads in the same job. Use multiple instances/jobs to run this.

#### 2. Resample 
The resample stage (`--resample`) resamples objects to achieve similar $p_T$ and $\eta$ distributions across classes.
After execution, resampled samples for each class, sample, and split are saved separately in `<base_dir>/components/<split>/`.
You need to run the resampling stage even if you don't apply any resampling (e.g. you configured with `method: none`).

Before anything is written, the stage checks for every component that the requested
`num_global_objects` can be delivered with the configured sampling fraction.

!!!warning "Requesting more objects than are available"

    If a component cannot deliver what you asked for, the run stops right away with an error
    naming that component. Raising the number further does not give you more statistics, it
    only moves the error. Run
    [`estimate_object_counts`](#additional-scripts-object-count-estimate) to get the numbers
    your samples support, or set `num_global_objects: auto` and let UPP fill them in.

    The check is based on an estimate taken from a subsample, so it can also be too
    optimistic. In that case the run starts and stops later with
    `Ran out of <component> objects after writing N`. Increasing
    `num_global_objects_estimate_available` makes the estimate more precise, and `-1` counts
    the objects exactly at the cost of a full pass over the inputs.

!!!warning "`sampling_fraction: auto` with the countup method"

    With `auto`, the sampling fraction of a component is derived from the objects it has
    available and the number you requested. If that lands above one, the `countup` method
    stops with an error, because it cannot select more objects than it reads. Ask for fewer
    objects, or switch to `pdf`, which only warns.

???info "Parallelization"
    Similar to the `--prep` step, the resampling step is also able to run in parallel, but only for the different region (e.g. `lowpt` & `highpt`). To do so, you need to run with the command line argument `--region` which takes as input the region on which to run. Please ensure that all components for this region were prepared in the `--prep` step before running this!

    The command to run the specific region would look like this:

    ```bash
    preprocess --config path/to/config.yaml --resample --region lowpt
    ```

    Similar to the `--prep` step, it is hardly discouraged to run multiple steps with this option enabled. This option is mainly to parallelize the processing on HPCs. Once all regions are resampled, you can continue with the following steps.

    If you want to go one step further, you can also tell the resampling to resample each component in the region in it's own process. To do so, you need to provide the `--region` command line argument together with the `--component` command line argument. Very important is here the full name of the component, which is constructed in the same way as already explained in the parallelization chapter of the Prepare stage.

    The command to run the specific region would look like this:

    ```bash
    preprocess --config path/to/config.yaml --resample --region lowpt --component lowpt_ttbar_ghostsplitbjets
    ```

    Similar to the `--prep` step and the previous `--region` explanation, it is hardly discouraged to run multiple steps with this option enabled. This option is mainly to parallelize the processing on HPCs. Once all components from all regions are resampled, you can continue with the following steps. Furthermore, do not run this in the same job with multiple threads! h5py has access issues when the same file is read by multiple threads in the same job. Use multiple instances/jobs to run this.
    Also, please do NOT use this functionality if you don't have fast I/O (hard drives). This is very heavy in terms of I/O load and ends up to be slower if you are using "default" HDD drives.

#### 3. Merge 
The merge stage (`--merge`) combines the resampled components into a single file named `<out_dir>/pp_output_<split>.h5`, where the file name comes from `out_fname`.
It also handles shuffling.

???info "Several output files instead of one"

    With `num_global_objects_per_output_file` set in the global config, the output is split
    into several files of that size, written to a split-specific subdirectory as
    `<out_dir>/<split>/pp_output_<split>_split_000.h5`, `_001`, and so on. Each file is
    shuffled across the components, so no file ends in a block of a single class.

    For the `test` split the components are merged per sample, giving one file per sample
    (`pp_output_test_ttbar.h5`), unless you set `merge_test_samples: true` in the global
    config.

#### 4. Normalise 
The normalise stage (`--norm`) calculates scaling and shifting values for all variables intended for training, based on `num_global_objects_estimate_norm` objects. The results are stored in `<out_dir>/norm_dict.yaml`, which can be renamed with the top-level `norm_fname` key.

!!!info "The normalisation only runs for the training split"

    With `--split val` or `--split test` the stage is skipped, and with `--split all` it runs
    during the training split only. This is what you want: validation and test data have to be
    normalised with the values from the training data.

#### 5. Plotting 

The plotting stage (`--plot`) produces histograms of the resampling variables before and after resampling, so you can verify the resampling quality.
You can find these plots in `<out_dir>/plots/`, which can be changed with `output_directory` in the [plotting config](configuration.md#plotting).

### Additional Scripts: Initial Sample Check

The check for the initial samples from the prepare stage can also be run stand-alone. This is important if you plan to run in parallel mode. To do so, you can simply use the following command:

```bash
check_input_samples --config_path <path/to/your/config>
```

You can also add the `--deviation-factor`, which is by default `10.0` and the `--verbose` flags. The latter will print the number of initial objects to your terminal.

!!!info "This script takes `--config_path`, not `--config`"

    The other UPP scripts use `--config`. This one is spelled differently for historical
    reasons.

### Additional Scripts: Object Count Estimate

Picking the `num_global_objects` values by hand is tedious: each component can only provide as
many objects as it has after the cuts, and the numbers have to keep the class ratios the same
in every region, otherwise the preprocessing stops with an error. The following command works
these numbers out for you:

```bash
estimate_object_counts --config <path/to/your/config>
```

It measures how many objects each component has available for the train, validation and test
splits, and then prints the largest object counts you can request, together with a snippet you
can copy into your config. Since the resampling only uses a fraction of the objects it reads,
that fraction is already taken into account, so the numbers it suggests will not fail the
availability check later on.

If you would rather not copy anything, set `num_global_objects: auto` in every component
instead and the preprocessing picks the numbers up on its own:

```yaml
components:
  - region:
      <<: *lowpt
    sample:
      <<: *ttbar
    classes: [bjets, cjets, ujets]
    num_global_objects: auto
```

Each split gets its own count this way, so `num_global_objects_val` and
`num_global_objects_test` are not needed either. The counts that were used end up in the config
copy in your output directory. Either all components use `auto` or none of them do, since the
counts of all components are solved together.

By default every class is made as large as its available objects allow. If you want a specific
composition instead, add an [`auto_counts`](configuration.md#automatic-object-counts) block to
your config.

How much of the sample each region gets is set with `sample_weight` on the sample
(see [Input H5 Samples](configuration.md#input-h5-samples)). With a weight of `2` on
$t\bar{t}$ and `1` on $Z'$, twice as many objects are taken from $t\bar{t}$ as from $Z'$ for
every class.

!!!warning "This reads a large part of your input samples"

    The measurement is as expensive as one pass over `num_global_objects_estimate_available`
    objects per component and belongs in a batch job, see [Running on HPC](hpc.md). Its
    results are cached in `<out_dir>/availability.yaml` and reused whenever your samples and
    cuts have not changed, so a second call is cheap. Use `--force` to measure again anyway,
    and `--splits` to restrict the work to certain splits.
