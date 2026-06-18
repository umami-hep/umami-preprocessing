# Umami Integration

UPP can be used from within the Umami preprocessing workflow when it is installed in
the same environment as Umami. The UPP config keeps the normal UPP sections and adds
an `umami` block with the settings Umami needs for scaling, writing, and optional
TFRecord conversion.

## Umami-Specific Config

Add an `umami` block to the UPP preprocessing config. A minimal example looks like:

```yaml
umami:
  general:
    plot_name: PFlow_ext-hybrid
    plot_type: pdf
    use_atlas_tag: true
    atlas_first_tag: "Simulation Internal"
    atlas_second_tag: "$\\sqrt{s}=13$ TeV, PFlow jets"
    legend_sample_category: true
    var_file: umami/user/upp_prep_small/config/Dips_Variables_R22.yaml
    dict_file: <base_dir>/<out_dir>/PFlow-scale_dict.json
    compression: lzf
    precision: float16
    concat_jet_tracks: false
  sampling:
    use_validation_samples: false
    options:
      n_jets_to_plot: 3e4
      save_tracks: true
      save_track_labels: true
      bool_attach_sample_weights: false
      tracks_names: ["tracks"]
      n_jets_scaling: 0
```

The `general` keys mirror Umami's top-level preprocessing settings. The `sampling`,
`sampling.options`, `parameters`, and `convert_to_tfrecord` keys mirror the
corresponding Umami config sections.

The `parameters` block is optional. If it is not provided, UPP writes to its configured
`out_dir`. Add `convert_to_tfrecord` only if the Umami workflow should also convert
the preprocessed dataset to TFRecord format.

For Umami-specific option details, refer to the
[Umami preprocessing documentation](https://umami.docs.cern.ch/preprocessing/Overview/).

## Running Through Umami

Run Umami's preprocessing entry point with the UPP config, as in the old Umami
preprocessing workflow:

```bash
cd umami
preprocessing.py --config_file path/to/my_upp_config.yaml --resampling
preprocessing.py --config_file path/to/my_upp_config.yaml --scaling
preprocessing.py --config_file path/to/my_upp_config.yaml --write
```

Umami first tries to read the config as an old-style Umami preprocessing config. If
that fails, it reads it as a UPP preprocessing config.

- `--resampling` runs UPP preprocessing with `--split all`.
- `--scaling` runs Umami's scaling step and writes the JSON scale dictionary to
  `dict_file`.
- `--write` runs Umami's scaling/writing step and produces the default Umami
  preprocessing plots.
- `--to_records` runs Umami's TFRecord conversion step when configured.

The old Umami `--prepare` step is not used for UPP configs. The
`--resampling --hybrid_validation` combination is also not available for UPP because
UPP handles all splits during the resampling step.

## Training Outputs

After preprocessing, the structured UPP outputs can be used for Umami training. For
example:

```yaml
model_name: user_DL1r-PFlow_new-taggers-stats-22M-tdd-upp
preprocess_config: /path/to/upp_prepr.yaml

model_file:

train_file: <base_dir>/<out_dir>/pp_output_train.h5

validation_files:
  r22_hybrid_val:
    path: <base_dir>/<out_dir>/pp_output_val.h5
    label: "Hybrid Validation"

test_files:
  ttbar_r22:
    path: <base_dir>/<out_dir>/pp_output_test_ttbar.h5
    <<: *variable_cuts_ttbar

  zpext_r22:
    path: <base_dir>/<out_dir>/pp_output_test_zprime.h5
    <<: *variable_cuts_zpext
```

Alternatively, run the Umami `--write` step and train from the unstructured scaled
output:

```yaml
train_file: <base_dir>/<out_dir>/pp_output_train_resampled_scaled_shuffled.h5
```
