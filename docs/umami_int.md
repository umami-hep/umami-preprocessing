UPP is istalled alongside Umami and can be used from within the umami framework in a very simple manner


### Umami-specific configs 
First you need to add umami-specific configs to the config file

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

Note: ```--prepare``` step will do nothing and will only trow an arror as UPP does not require (same) preparation as old umami preprocession. 
```--resampling --hybrid_validation``` is also not available for upp as it does both splits at the resampling step

After that one can use the results of the preprocessing for umami trainig for example for DL1 or DIPS.
One can either only run ```--resampling``` and ```--scaling``` and train on the structured array data using TDDgenerator by setting your training configs similar to this:
```
```
or you can perform all three steps to train using unstructured array data. This way one looses time to write the dataset but the training may be somewhat faster. 
To do this just chnge your training_file in the example above to 
```
```



