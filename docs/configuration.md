# Configuration

The configuration of the preprocessing is done with a [`.yaml`](https://en.wikipedia.org/wiki/YAML) file which steers the whole preprocessing. A general example of such a file can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/preprocessing/PFlow-Preprocessing.yaml).

Several example config files for UPP can be found in `upp/config`.

There are 4 main parts of the configs:
1. TDD Samples
2. Global cuts
3. Components
4. Variables
5. Resampling
6. General 

### TDD samples

First of all one has to define the input TDD **samples** to be preprocessed. These are needed to construct the components later on in configs thus one should define them as [anchors](https://support.atlassian.com/bitbucket-cloud/docs/yaml-anchors/) `<sample name>: &<sample name>.
Below we provide an example and a table explaining each setting 

```yaml
ttbar: &ttbar
  name: ttbar
  equal_jets: False
  pattern:
    - name1.*.410470.*/*.h5
    - name2.*.410470.*/*.h5
```

| Setting | Type | Explanation |
| ------- | ---- | ----------- |
|`name`|`str`| The name of the sample to be used in intermediate resampled files|
|`pattern`|`list` of `str`| one single pattern sring or a list of pattern strings that give all the files from which jets for this components should be read out. |
|`equal_jets`|`bool`| **Optional (default True)** ?????????????????????????????????????????|

### Global cuts

The **cuts** that should be applied to all the data should be listed under `common:`. For example these could be the outlier cuts.
To do this one first provides the variable name (`str`), then the comparison operator (`str`) and a number to compare to (`int`, `float` or list).
Possible operators are `"=="`, `"!="`, `"<="`, `">="`, `">"`, `"<"` which work the same as in python, `"in"` and `"notin"` to chek if the value is in the list, and
`"%{i}=="`, `"%{i}=="`, `"%{i}=="` operators to compare the modulo w.r.t. `i` of an integer number. 

Further one has to specify the cuts that separate `train`, `val` and `test` splits using modulo of `eventNumber`. For example:

```yaml
global_cuts:
  common:
    - [JetFitterSecondaryVertex_mass, "<", 25000]
    - [JetFitter_deltaR, "<", 0.6]

  train:
    - [eventNumber, "%10<=", 7]

  val:
    - [eventNumber, "%10==", 8]

  test:
    - [eventNumber, "%10==", 9]
```


### Components
It may be helpful to define the anchors fot the cuts of different resampling **regions** where one plans to use events from different samples. 
For each region one has to provede the string name and a list of **cuts** (see above).
Here is an example:

```yaml
lowpt: &lowpt
  name: lowpt
  cuts:
    - [pt_btagJes, ">", 20_000]
    - [pt_btagJes, "<", 250_000]
highpt: &highpt
  name: highpt
  cuts:
    - [pt_btagJes, ">", 250_000]
    - [pt_btagJes, "<", 6_000_000]
```

Thereafter one provides settings for a separate `components:` for each **region**, **sample** and flavour (or combination of flavours), and provides the number of jets that should be sampled from each component. Below you can see an example with several components and the description for the settings that one has to define for each component. Notice that we use `<<*` aliasing tool to insert the **region**, **sample** configs that we have defined above.

```yaml
components:
  - region:
      <<: *lowpt
    sample:
      <<: *ttbar
    flavours: [bjets, cjets]
    num_jets: 52_500_000

  - region:
      <<: *lowpt
    sample:
      <<: *ttbar
    flavours: [ujets]
    num_jets: 105_000_000

  - region:
      <<: *lowpt
    sample:
      <<: *ttbar
    flavours: [taujets]
    num_jets: 6_250_000

  - region:
      <<: *highpt
    sample:
      <<: *zprime
    flavours: [bjets, cjets]
    num_jets: 21_000_000

  - region:
      <<: *highpt
    sample:
      <<: *zprime
    flavours: [ujets]
    num_jets: 42_000_000
```

| Setting | Type | Explanation |
| ------- | ---- | ----------- |
|`region`|region conf| `name` of the region and the list of `cuts` to be applied to get the samples from a specific region|
|`sample`|sample conf| TDD sample configs (see above)|
|`flavours`|`list` of `str`| list of one or more flavours. If more then one is provided the saparate component will be created foer each flavour in the list but with the same other settings.|
|`num_jets`|`int`| The number of jets to be sampled from this component in the training split|
|`num_jets_val`|`int`| **Optional (default `num_jets//10`)** number of jets of this component in validation set.|
|`num_jets_test`|`int`| **Optional (default `num_jets//10`)** number of jets of this component in a test set.|

### Variables

One needs to provide the variables that would be taken from the TDD samples and written in the resampled dataset. One can siply define them under `variables:` like:

```yaml
variables:
  jets:
    inputs:
      - pt_btagJes
      - absEta_btagJes
    labels:
      - HadronConeExclTruthLabelID
      - eta_btagJes
      - pt
      - eta

  tracks:
    inputs:
      - dphi
      - deta
      - qOverP
      - IP3D_signed_d0_significance
      - IP3D_signed_z0_significance
    labels:
      - truthOriginLabel
      - truthVertexIndex
```
Where one fist specifies the dataset the same way it is called in TDD h5 file (e.g. `jets`, `tracks`, `hits`). All the variables that should be saved in resampled files should be provided in one of two lists `inputs` or in `labels`, the output file however doent have this structure and the variables from `inputs` and `labels` are mixed in each dataset. 

Alternatively include the variables from your custom variable config by providing the full path to the file after an include statement.
The file you provide should have the same structure as shown above but without `variable:` level. Examle:
```yaml
variables: !include xbb-variables.yaml
```

One can also import vaiables configs already provided in this package `upp/config/
` yaml files using just the yaml file name e.g.:

```yaml
variables: !include /<full path to your file>.yaml
```

### Resampling

There are currently two ewsampling methods implemented in the package `pdf` and `countup` and they share most of setting. Below is the example of setting up the `pdf` resampling method and a table describitng all the parameters.

```
resampling:
  target: cjets
  method: pdf
  upscale_pdf: 2
  sampling_fraction: auto
  variables:
    pt_btagJes:
      bins: [[20_000, 250_000, 50], [250_000, 1_000_000, 50], [1_000_000, 6_000_000, 50]]
    absEta_btagJes:
      bins: [[0, 2.5, 20]]
```

| Setting | Type | Explanation |
| ------- | ---- | ----------- |
|`target`|`str`| The resampling is done in such a way that the distribution of the kinematic variables matches the distribution of those in one particular flavour given in here. Usually it is the leat populated flavour, as this flavour will not be resampled instead all jets of this flavour are taken.|
|`method`|`str`| Either  `"pdf"` and `"countup"` depending on the method you would like to use|
|`upscale_pdf`|`int`| **Optional** only availabe for `"pdf"` preprocessing. The coarse approximation of the pdf functions based on histograms are interpolated and to bins that are upscale_pdf**dimensions times smaller than original|
|`sampling_fraction`|`auto`, Null or `float`| The number of the jets sampled from each batch is equal to the sampling fraction time number of the jets in input batch (after the curs and flavour selection). The large is this variable, the more are jets upsampled i.e. repeated, thus smaller values are prefered. On the other hand eith smaller sampling fractions lead to longer preprocesing times. `auto` option gives the smallest resampling fraction for each component depending on the number of available jets and number of jets that is asked for but caps it from below at 0.1 to prevent long preprocessing times when enough statistic is present. |
|`variables`|`dict`| The jets will be resampled according to the distribution of the kinematic variables you provide here. The variable names must correspond to the ones in TDD. For each variable prlease provide a `bins` setting with a list of lists of 2 floats and a an integer each. Each of the sub lists represent a binning region and is described by lower bound upper bound and the number of bins of equal width in this regions. The bins from each region will be combined to provide one (heterogenous width) binning. When upscaling the pdf each bin region is upscaled separately. THerefore is not necessary but advisable to have a split in binnings at the same place where the cut betwenn **regions** takes place to better handle the discontinuities.|

### Global 

```
global:
  batch_size: 1_000_000
  num_jets_estimate: 5_000_000
  base_dir: /home/users/o/oleksiyu/WORK/umami-preprocessing/user/user6/replicate3_pdf_sfauto/
  out_dir: test_out/
  ntuple_dir: /srv/beegfs/scratch/groups/dpnc/atlas/FTag/samples/r22/
```

| Setting | Type | Explanation |
| ------- | ---- | ----------- |
|`batch_size`|`int`| The data is pereprocessed in batches. From each batch `sampling_fraction*batch_size_after_cuts`. It is recommended to choose high batch sizes especially to the `countup` method to achive best agreement of target and resampled distributions|
|`num_jets_estimate`|`int`| Number of jets of each flavour that are used to construct histograms for probability density function estimation. Larger numbers give a better quality estmate of the pdfs|
|`base_dir`|`str`| Directory for sabving all the intermedate and final steps  |
|`out_dir`|`str`| The subdirectory of `base_dir` where all the final results are saved |
|`ntuple_dir`|`ntuple_dir`| Directory where TDD ntuples are searched for using patterns defined before|
