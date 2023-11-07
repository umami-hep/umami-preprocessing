UPP is istalled alongside Umami and can be used from within the umami framework in a very simple manner

### Umami-specific configs 
First you need to add umami-specific configs to the config file.
Here is an example config:
```yaml
umami:
  general:
    plot_name: PFlow_ext-hybrid
    plot_type: "pdf"
    use_atlas_tag: True
    atlas_first_tag: "Simulation Internal"
    atlas_second_tag: "$\\sqrt{s}=13$ TeV, PFlow jets"
    legend_sample_category: True
    var_file: umami/user/upp_prep_small/config/Dips_Variables_R22.yaml
    dict_file: <base_dir><out_dir>PFlow-scale_dict.json
    compression: lzf
    precision: float16
    concat_jet_tracks: False
  sampling:
    use_validation_samples: false
    options:
      n_jets_to_plot: 3e4
      save_tracks: true
      save_track_labels: true
      bool_attach_sample_weights: false
      tracks_names: ["tracks"]
      n_jets_scaling: 0

This config part mimics the umami config structure. Parameters in `general` mimic ones that are in the root of umami config. Parameters in `sampling`, `sampling.options`, `parameters` and `convert_to_tfrecord` mimic the corresponding structures in umami config. All the parameters given in the example should be given in order for UPP integration in umami to work except `parameters` and `convert_to_tfrecord`. You need to provide `convert_to_tfrecord` if you need to convert dataset to TFrecord and `parameters` oonly if you does not want to saveinto `<base_dir><out_dir>` by default.
Please refer to umami documentation [https://umami.docs.cern.ch/preprocessing/Overview/] for up-to-date explanation.

### Running preprocessing

After you make the necessary changes to the config file you can perform preprocessing in umami by running the umami/preprocessing.py script
the same way as you would do with the old umami preprocessing 

```bash
cd umami 
preprocessing.py --config_file path/to/my_upp_config.yaml --resampling
preprocessing.py --config_file path/to/my_upp_config.yaml --scaling
preprocessing.py --config_file path/to/my_upp_config.yaml --write
```

Umami will first ```try``` to read the config file as an old umami preprocessing configuration. When that fails it will read the config as a UPP preprocessing config.

* ```--resampling``` step will perform Upp preprocessing with this config file and ```split==all```
* ```--scaling``` step will execute umami version of rescaling code that will prepare a json scaling dictionary at ```dict_file``` location
* ```--write``` step will execute umamii code for scaling the variables and writing them in an unstructured scaling array it will also produce default umami preprocessing plots
* ```--to_records``` step will execute umamii code for converting dataset to a TFrecords format 

Note: ```--prepare``` step will do nothing and will only trow an arror as UPP does not require (same) preparation as old umami preprocession. 
```--resampling --hybrid_validation``` is also not available for upp as it does both splits at the resampling step

After that one can use the results of the preprocessing for umami trainig for example for DL1 or DIPS.
One can either only run ```--resampling``` and ```--scaling``` and train on the structured array data using TDDgenerator by setting your training configs similar to this:
```
# Set modelname and path to Pflow preprocessing config file
model_name: user_DL1r-PFlow_new-taggers-stats-22M-tdd-upp
preprocess_config: /home/users/o/oleksiyu/WORK/umami/user/upp_prep_small/config/upp_prepr.yaml

# Add here a pretrained model to start with.
# Leave empty for a fresh start
model_file: 

# Add training file
train_file: <base_dir><out_dir>pp_output_train.h5

# Defining templates for the variable cuts
...

#Add validation files
validation_files:
    r22_hybrid_val:
        path: <base_dir><out_dir>pp_output_val.h5
        label: "Hybrid Validation"

test_files:
    ttbar_r22:
        path: <base_dir><out_dir>pp_output_test_ttbar.h5
        <<: *variable_cuts_ttbar

    zpext_r22:
        path: <base_dir><out_dir>pp_output_test_zprime.h5
        <<: *variable_cuts_zpext

```
or you can perform all three steps to train using unstructured array data. This way one looses time to write the dataset but the training may be somewhat faster. 
To do this just chnge your training_file in the example above to 
```
train_file: <base_dir><out_dir>pp_output_train_resampled_scaled_shifted.h5
```



