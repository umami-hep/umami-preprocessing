# Configuration

The configuration of the preprocessing is done with a [`.yaml`](https://en.wikipedia.org/wiki/YAML) file which steers the whole preprocessing.
Available example config files for UPP can be found in [`upp/configs`](https://github.com/umami-hep/umami-preprocessing/tree/main/upp/configs).

Each aspect of the configuration is described in detail below.

The config is built from these top-level keys:

| Key | Required | Section |
| --- | -------- | ------- |
| `global` | yes | [Global Config](#global-config) |
| `variables` | yes | [Variables](#variables) |
| `global_cuts` | yes | [Global Cuts](#global-cuts) |
| `components` | yes | [Components](#components) |
| `resampling` | no (omit to skip resampling) | [Resampling](#resampling) |
| `auto_counts` | no | [Automatic object counts](#automatic-object-counts) |
| `transform` | no | [Variable transformations](#variable-transformations) |
| `plotting` | no | [Plotting](#plotting) |
| `norm_fname` | no | name of the normalisation file, `norm_dict.yaml` by default |
| `reweighting` | no | [Reweighting](reweighting.md) |
| `umami` | no | [Umami integration](umami_int.md) |

Sample, region and class blocks (`ttbar`, `lowpt`, ...) are defined at the top level as well
and pulled into `components` with yaml anchors, as shown below.


### Input H5 Samples

Here we define the input h5 samples which are to be preprocessed.
Each sample is defined using one or more DSIDs, which generally come from the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper).
If a list of DSIDs is provided, the objects from each DSID will be merged according to the `equal_global_objects` flag (see below).
The samples are used to define components later on in configs and so one should define them with [anchors](https://support.atlassian.com/bitbucket-cloud/docs/yaml-anchors/).

Below is an example and a table explaining each setting.

=== "Single DSID"

    ```yaml
    ttbar: &ttbar
      name: ttbar
      pattern: name1.*.410470.*/*.h5
    ```

=== "Multiple DSIDs"

    ```yaml
    ttbar: &ttbar
      name: ttbar
      equal_global_objects: False
      pattern:
        - name1.*.410470.*/*.h5
        - name2.*.410470.*/*.h5
    ```

| Setting | Type | Explanation | Default |
| ------- | ---- | ----------- | ------- |
|`name`   |`str`| The name of the sample, used in output filenames.| *Required* |
|`pattern`|`str` or `list[str]`| A single pattern or a list of pattern that match h5 files in a downloaded dataset. H5 files matching each pattern will be transparently merged using virtual datasets. | *Required* |
|`sample_weight`|`int`| The relative number of objects taken from this sample, used by `estimate_object_counts` (see [Object Count Estimate](run.md#additional-scripts-object-count-estimate)). A sample with weight `2` contributes twice as many objects as one with weight `1`. | `1` |
|`equal_global_objects`|`bool`| Only relevant when providing a list of patterns. If `True`, the same number of objects from each DSID are selected. This is required for e.g. in Xbb QCD where each DSID belongs to a different slice, and the resampling would break if you tried to resample with one or more slices missing. If `False` this is not enforced, allowing for larger numbers of available objects. | `True` |

The virtual dataset files created from wildcard patterns are by default stored alongside the input ntuples.
If you have no write access to the input ntuples directory and would like to collect all VDS files in an accessible directory instead, set `vds_dir` in the global config (see [Global Config](#global-config)).
Each pattern gets its own VDS file named after its DSID directory (e.g. `vds_dir/user.wlai.601589.e8547_..._output_vds.h5`).



### Global Cuts

The selections that should be applied to all the data should be listed under `common:`.
For example these could be outlier removal cuts, or a global kinematic selection.
To do this one first provides the variable name (`str`), then the comparison operator (`str`) and a number to compare to (`int`, `float` or `list`).
Possible operators are:

-  `"=="`, `"!="`, `"<="`, `">="`, `">"`, `"<"` which work the same as in python.
- `"in"` and `"notin"` to check if the value is in the list.
- `"%{i}=="`, `"%{i}!="`, `"%{i}<="`, `"%{i}>="` operators to compare the modulo w.r.t. `i` of an integer, e.g. `"%10<="`. `i` can be any integer from 2 to 100.

Along with the common selection cuts, you should also specify the cuts that separate `train`, `val` and `test` splits using modulo of `eventNumber`.
For example:

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

???info "More info about cuts"

    The `Cuts` class is defined in the [`atlas-ftag-tools`](https://github.com/umami-hep/atlas-ftag-tools/blob/main/ftag/cuts.py) package.


???info "k-fold training selection"

    If you are training a model that will be used in production, you may need to worry about overtraining.
    A variable `jetFoldHash` is included in newer h5 dumps which allows you to independent models on different
    folds of the data.
    If you are just performing studies, then don't worry about applying any selections on the `jetFoldHash`, 
    since the train/val/test split will suffice.


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

Again, aliasing these just helps to reduce duplication of information when defining the components as can be seen below.

### Components

The `components` section is where all the configuration comes together. 
A component is a combination of a region, a sample and a flavour.
They allow for full flexibility when defining different preprocessing pipelines
(e.g. single-b versus Xbb).

An example `components` block is provided below.

```yaml
components:
- region:
    <<: *lowpt
    sample:
    <<: *ttbar
    classes: [bjets, cjets, ujets]
    num_global_objects: 10_000_000

- region:
    <<: *highpt
    sample:
    <<: *zprime
    classes: [bjets, cjets, ujets]
    num_global_objects: 5_000_000
```

Notice that we use `<<*` insertion tool to insert already defined regions and samples.

| Setting | Type | Explanation |
| ------- | ---- | ----------- |
| `region`| anchor | The pre-defined kinematic region anchor, e.g. `lowpt` or `highpt`, or `inclusive` if not splitting in $p_T$ |
| `sample`| anchor | The pre-defined sample anchor, e.g. $t\bar{t}$ or $Z'$ |
| `classes` | `list[str]` | One or more object classes (flavours), e.g. `[bjets]` or `[ujets]`. Each name must exist in the active class container (the atlas-ftag-tools bundled flavours by default, or your own file via `class_config` — see [Custom classes](#custom-classes)). The list syntax is pure syntactic sugar. If more then one is provided, separate components are created for each class.|
|`num_global_objects`|`int` or `auto`| The number of objects to be sampled from this component in the training split. When resampling is skipped, `-1` writes all objects of this component passing the cuts. Set it to `auto` in every component to let UPP work the numbers out for you (see [Object Count Estimate](run.md#additional-scripts-object-count-estimate)).|
|`num_global_objects_val`|`int`| **Optional** (default: `num_global_objects//10`, or the automatic count of the validation split) number of objects of this component in the validation set.|
|`num_global_objects_test`|`int`| **Optional** (default: `num_global_objects//10`, or the automatic count of the test split) number of objects of this component in the test set.|

!!!warning "The class ratios have to match in every region"

    UPP checks that each class makes up the same fraction of the objects in every region and
    stops with an error if not. With two regions and three classes at `10M/10M/10M` in `lowpt`,
    the `highpt` region has to use the same ratios (e.g. `5M/5M/5M`), not different ones.
    Letting UPP work the numbers out for you (see below) takes care of this.

### Automatic object counts

Instead of writing the counts yourself, set `num_global_objects: auto` in **every** component
and let UPP solve them from a measurement of how many objects are actually available. The
measurement is done by [`estimate_object_counts`](run.md#additional-scripts-object-count-estimate),
which has to be run once before the preprocessing.

```yaml
components:
  - region:
      <<: *lowpt
    sample:
      <<: *ttbar
    classes: [bjets, cjets, ujets]
    num_global_objects: auto
```

Each split gets its own count, so `num_global_objects_val` and `num_global_objects_test` are
not needed. How many objects a region gets relative to the others is set with `sample_weight`
on the sample (see [Input H5 Samples](#input-h5-samples)).

The optional `auto_counts` block tunes what is solved for:

```yaml
auto_counts:
  class_ratios:
    bjets: 1
    cjets: 1
    ujets: 2
```

| Setting | Type | Explanation | Default |
| ------- | ---- | ----------- | ------- |
|`class_ratios`|`dict` or `auto`| Ratios the classes should be mixed in. The largest counts matching these ratios are used. With `auto`, every class is made as large as its own available objects allow, so the ratios follow the input samples. | `auto` |

!!!info "Where the solved numbers end up"

    The counts are solved when the config is loaded, and the result is written into the config
    copy in your output directory under `auto_counts.solved_<split>`. The config you maintain
    keeps saying `auto`, while the copy next to the output records what was actually used.

### Custom classes

By default the class definitions come from the flavour labels bundled with
`atlas-ftag-tools`, selected with `class_category` (`standard` or `extended`).
These are jet flavours, but the framework itself is object-agnostic: to classify
any other object type, point `class_config` at your own classes yaml. It is a
list of class definitions, each with a `name`, plotting `label`, selection
`cuts`, a `colour`, a `category`, and an optional `_px` probability name:

```yaml
- name: heavy
  label: Heavy objects
  cuts: ["HadronConeExclTruthLabelID == 5"]
  colour: tab:red
  category: custom
- name: light
  label: Light objects
  cuts: ["HadronConeExclTruthLabelID == 0"]
  colour: tab:blue
  category: custom
```

Reference it from the global config; a relative path is resolved against
`base_dir`, and `class_config` takes precedence over `class_category`:

```yaml
global:
  global_name: objects
  class_config: custom_flavours.yaml
```

The `classes` listed for each component then refer to the `name` entries in this
file (e.g. `classes: [heavy, light]`).

### Variables

The next thing you need is to provide the variables that are taken from the TDD files and written in the resampled dataset.
Selecting only a subset of variables keeps the output files lightweight, and ensures the dataloading does not become a bottleneck during training.

One can simply define them under `variables:` like:

```yaml
variables:
  jets:
    inputs:
      - pt_btagJes
      - absEta_btagJes
    labels:
      - HadronConeExclTruthLabelID
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
      - ftagTruthOriginLabel
      - ftagTruthVertexIndex
```
Each key under `variables:` corresponds to a dataset name in the TDD h5 file (e.g. `jets`, `tracks`, `hits`).
The combined set of variables in `inputs` and `labels` are carried over to the output files to a dataset with the same name as the input dataset.
Internally, UPP will compute normalisation parameters for variables in the `inputs`, and compute class weightings (for categorical labels) for variables in the `labels` block.

Alternatively include the variables from your custom variable config by providing the full path to the file after an include statement.
The file you provide should have the same structure as shown above but without `variable:` level.
For example:
```yaml
variables: !include xbb-variables.yaml
```

One can also import variable configs already provided in this package under `upp/configs/` by using the yaml file name, e.g.:

```yaml
variables: !include variables.yaml
```

???info "You can choose later which variables in your output files are used for training"

    When it comes to defining your training config, you will be required to [define the variables used for training](https://ftag-salt.docs.cern.ch/configuration/#selecting-training-variables).
    So it's okay to include here input variables you are not sure whether you will need, for example when testing the importance of different inputs.
    This is straightforward since we always store data using structured arrays (in the same format as the TDD outputs).
    
    
### Track selections

You can apply on the fly selections to tracks in the preprocessing stage (specifically the merging step).

To do this, include a `selection` key in the variable config block under the tracks, for example:

```
  tracks:
    inputs:
      - d0
    labels:
      - ftagTruthOriginLabel
    selection:
      - [d0, ">", 0.1]
```

### Variable transformations

The optional `transform:` block renames variables and remaps values while the input files are
read. This is useful when the naming in your ntuples does not match what the training expects.
It is handed to the `Transform` class of
[`atlas-ftag-tools`](https://github.com/umami-hep/atlas-ftag-tools/blob/main/ftag/transform.py),
which supports three maps:

```yaml
transform:
  variable_map:
    tracks:
      truthOriginLabel: ftagTruthOriginLabel
      truthVertexIndex: ftagTruthVertexIndex
  ints_map:
    tracks:
      ftagTruthOriginLabel:
        0: 1
        -2: 0
  floats_map:
    jets:
      pt: log
```

| Setting | Type | Explanation |
| ------- | ---- | ----------- |
|`variable_map`|`dict`| Renames variables, `variable_map[dataset][old_name] = new_name` |
|`ints_map`|`dict`| Replaces integer values, `ints_map[dataset][variable][old_value] = new_value` |
|`floats_map`|`dict`| Applies a function to a float variable, `floats_map[dataset][variable] = func`, where `func` is the name of a numpy function such as `log` |

!!!warning "Every map starts with the dataset name"

    The first level of each map is the dataset (`jets`, `tracks`, ...), and only then comes the
    variable. A map that omits this level is silently ignored, because the dataset name is
    looked up in the batch before the variable is touched. Nothing fails, the values simply
    stay as they are.

### Resampling

There are currently two resampling methods implemented in the package `pdf` and `countup` and they share most of setting.
Below is the example of setting up the `pdf` resampling method and a table describing all the parameters.

In order to run UPP without any kinematic resampling, just set `method: none`. 
Note you will still need to run the resampling stage of the preprocessing pipeline.

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
|`target`|`str`| The resampling is done in such a way that the distribution of the kinematic variables matches the distribution of those in one particular flavour given in here. Usually it is the least populated class, as this class will not be resampled and instead all of its objects are taken.|
|`method`|`str`| Either  `pdf`, `countup` or `none`, depending on the method you would like to use|
|`upscale_pdf`|`int`| **Optional** only available for `pdf` preprocessing. The coarse approximation of the pdf functions based on histograms are interpolated and to bins that are upscale_pdf**dimensions times smaller than original|
|`sampling_fraction`|`None`, `float` or `auto`| The number of objects sampled from each batch is equal to the sampling fraction times the number of objects in the input batch (after the cuts and the class selection). The larger this variable, the more objects are upsampled i.e. repeated, thus smaller values are preferred. On the other hand, smaller sampling fractions lead to longer preprocessing times. The `auto` option gives the smallest sampling fraction for each component depending on the number of available objects and the number of objects that is asked for, but caps it from below at 0.1 to prevent long preprocessing times when enough statistics are present. |
|`variables`|`dict`| The objects will be resampled according to the distribution of the kinematic variables you provide here. The variable names must correspond to the ones in TDD. For each variable please provide a `bins` setting with a list of lists of 2 floats and a an integer each. Each of the sub lists represent a binning region and is described by lower bound upper bound and the number of bins of equal width in this regions. The bins from each region will be combined to provide one (heterogenous width) binning. When upscaling the pdf each bin region is upscaled separately. THerefore is not necessary but advisable to have a split in binnings at the same place where the cut between **regions** takes place to better handle the discontinuities.|

### Plotting

Plot labels and styles can be configured under the optional `plotting:` key. Any omitted setting uses the default defined by the `PlottingConfig` dataclass (documented below). Variable-label keys are matched case-insensitively against the plotted variable name, so a single `pt` entry applies to variables such as `pt_btagJes`.

```yaml
plotting:
  num_global_objects_plotting: 10_000_000
  variable_labels:
    pt: "Jet $p_\\mathrm{T}$ [GeV]"
    eta: "Jet $|\\eta|$"
    mass: "Jet Mass [GeV]"
  sample_labels:
    ttbar: "$t\\bar{t}"
    zprime: "$Z'$"
  atlas_first_tag: Simulation Internal
  atlas_second_tag: "$\\sqrt{s} = 13/13.6\\,\\mathrm{TeV}$"
  output_formats: [pdf, png]
  bins: 50
  y_scale: 1.5
  figsize: [6, 4]
  logy: true
  linestyles: ["-", "--", "-.", ":"]
  legend_location: upper right
  linestyle_legend_location: upper center
  linestyle_legend_anchor: [0.55, 1]
  output_directory: plots
```

The `ylabel` setting supports a `{global_name}` placeholder. Histogram normalisation and overflow handling can be controlled with `norm` and `underoverflow`.

::: upp.classes.plotting_config.PlottingConfig

### Global Config 

::: upp.classes.preprocessing_config.PreprocessingConfig
