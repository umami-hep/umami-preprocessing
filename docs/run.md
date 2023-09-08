# Run

Modify the configuretion file accoriding to the instructions on the next page.
To run all preprocessing stages for the `"train"` split use:

```bash
preprocess --config configs/test.yaml
```

Other flags are described below and you can also find them in `preprocess --help` for the full list of flags.

### Splits 

The data is split into training (`train`), validation (`val`) and testing (`test`) splits. The fiters for those are defined in configs and usually are based on the `eventNumber` variable in such a way that train gets 80% of jets while val and test correspond to 10% each. 

If you want to preprocess the validation or test split, use the `--split` argument:

```bash
preprocess --config configs/config.yaml --split val
```

You can also use `split=all` to run each of the train/val/test splits in a single command.

### Stages 

Several steps are done during the preprocessing:

1. `--prep` Read some jets of each flavour to construct the histograms of the jet distribution in the space of resampled variables. After running this step you will find histograms stored in `<base_dir>/hists`.
2. `--resample` Resampling step : Combine and resample the different processes/flavours to achieve similar $p_T$ and $\eta$ distributions for all used flavours. After running this step you will find resampled samples for each flavour, component and split separately in `<base_dir>/components/<split>/`.
4. `--merge` Merging step : combines the samples into a single file `<tbase_dir>/&ltout_dir>/pp_output_<split>.h5`
3. `--norm` Scaling/Shifting step: Calculate scaling/shifting values for all variables that are about to be used in the training. Produces `<tbase_dir>/&ltout_dir>/norm_dict.yaml`
4. `--plot` plotting: Plots histograms (with other binning than used for resmpling) of the resampled variables to validate the resmapling quality. Find the plots in `<tbase_dir>/plots/`

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

