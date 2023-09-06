# Run

Modify the configuretion file accoriding to the instructions on the next page.
To run all preprocessing stages for the `"train"` split use:

```bash
preprocess --config configs/test.yaml
```

Several steps are done in the preprocessing:

1. `--prep` Preparation step : Extract the different flavours from the .h5 files from the training-dataset-dumper and separate them into flavour-specific files. Also the the split in training/validation/evaluation is done at this step.
2. `--resample` Resampling step : Combine and resample the different processes/flavours to achieve similar $p_T$ and $\eta$ distributions for all used flavours.
4. `--merge` Merging step : Combine 
3. `--norm` Scaling/Shifting step: Calculate scaling/shifting values for all variables that are about to be used in the training.
4. `--plot` plotting: Plots histograms of the resampled variables to validate the resmapling

To run with only specific steps enabled, include the flag for the required steps.
For example

```bash
preprocess --config configs/config.yaml --prep --resample
```

will run the first two steps.

To run the whole preprocessing excluding certain steps, include the corresponding negative flag (`--no-*`).
For example to run without plotting

```bash
preprocess --config configs/config.yaml --no-plot
```

The data is split into training (`train`), validation (`val`) and testing (`test`) splits. The fiters for those are defined in configs and usually are based on the `eventNumber` variable in such a way that val gets 10% of jets the same as test 

If you want to preprocess the validation or test split, use the `--split` argument:

```bash
preprocess --config configs/config.yaml --split val
```

You can also use `split=all` to run each of the train/val/test splits in a single command.

See `preprocess --help` for the full list of flags.
