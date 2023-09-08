# Configuration

The configuration of the preprocessing is done with a [`.yaml`](https://en.wikipedia.org/wiki/YAML) file which steers the whole preprocessing.
Available example config files for UPP can be found in [`upp/configs`]({{repo_url}}tree/main/upp/configs).

Each aspect of the configuration is described in detail below.

### Global Cuts

The selections that should be applied to all the data should be listed under `common:`.
For example these could be outlier removal cuts, or a common kinematic selection.
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


### Input H5 Samples

Here we define the input h5 samples which are to be preprocessed.
The samples are used to define components later on in configs and so one should define them with [anchors](https://support.atlassian.com/bitbucket-cloud/docs/yaml-anchors/).
Each sample is defined using one or more DSIDs, which generally come from the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper).
If a list of DSIDs is provided, jets from each DSID will be merged according to the `equal_jets` flag.


Below we provide an example and a table explaining each setting:

```yaml
ttbar: &ttbar
  name: ttbar
  equal_jets: False
  pattern:
    - name1.*.410470.*/*.h5
    - name2.*.410470.*/*.h5
```

| Setting | Type | Explanation | Default |
| ------- | ---- | ----------- | ------- |
|`name`   |`str`| The name of the sample, used in output filenames.| *Required* |
|`pattern`|`str` or `list[str]`| A single pattern or a list of pattern that match h5 files in a downloaded dataset. H5 files matching each pattern will be transparently merged using virtual datasets. | *Required* |
|`equal_jets`|`bool`| Only relevant when providing a list of patterns. If `True`, the same number of jets from each DSID are selected. If `False` this is not enforced, allowing for larger numbers of available jets. | `True` |


### Resampling Regions

Next we define any kinematic regions which need to be resampled separately, again using anchors as these will also be used in the definition of our components.
For each region you need to provide a name and a list of cuts (see above).
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

Aliasing these just helps to reduce duplication of information when defining the components as can be seen below.

### Components

A component is a combination of a region, a sample and a flavour.

They allow for full flexibility when defining different preprocessing pipelines
(e.g. single-b versus Xbb).
An example is provided below.
Notice that we use `<<*` aliasing tool to insert already defined regions and samples.

```yaml
components:
- region:
    <<: *lowpt
    sample:
    <<: *ttbar
    flavours: [bjets, cjets, ujets]
    num_jets: 10_000_000

- region:
    <<: *highpt
    sample:
    <<: *zprime
    flavours: [bjets, cjets, ujets]
    num_jets: 5_000_000
```


| Setting | Type | Explanation |
| ------- | ---- | ----------- |
| `region`| anchor | The pre-defined kinematic region anchor, e.g. `lowpt` or `highpt`, or `inclusive` if not splitting in $p_T$ |
| `sample`| anchor | The pre-defined sample anchor, e.g. $t\bar{t}$ or $Z'$ |
| `flavours` | `list[str]` | One or more jet flavours, e.g. `[bjets]` or `[ujets]`. The list syntax is pure syntactic sugar. If more then one is provided, separate components are created for each flavour.|
|`num_jets`|`int`| The number of jets to be sampled from this component in the training split|
|`num_jets_val`|`int`| **Optional** (default: `num_jets//10`) number of jets of this component in validation set.|
|`num_jets_test`|`int`| **Optional** (default: `num_jets//10`) number of jets of this component in a test set.|



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

There are currently two resampling methods implemented in the package `pdf` and `countup` and they share most of setting.
Below is the example of setting up the `pdf` resampling method and a table describitng all the parameters.

```yaml
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

### Global Config 

#### ::: upp.classes.preprocessing_config.PreprocessingConfig
