# UPP: Umami PreProcessing

This is a modular preprocessing pipeline for jet tagging.
It addresses [several issues](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/issues/?label_name%5B%5D=Preprocessing) with the current umami preprocessing workflow, and uses the [`atlas-ftag-tools`](https://github.com/umami-hep/atlas-ftag-tools/) package extensively.

This package is work in progress and still needs

- [ ] documentation
- [ ] integration and unit tests
- [ ] integration with umami



### Setup

Get the code:

```bash
git clone https://github.com/umami-hep/umami-preprocessing.git
cd umami-preprocessing
```

Conda environents are a great way to keep organised, but are not essential for installing the code. 
With a fresh [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) or [mamba](https://github.com/conda-forge/miniforge#install) environment:

```bash
mamba create -n upp python=3.11
mamba activate upp
python -m pip install -e .[dev]
```

See [here](https://abpcomputing.web.cern.ch/guides/python_inst/) for instructions on installing conda on lxplus.
If you don't want to use conda environments on lxplus, you could probably just install with

```bash
setupATLAS
lsetup "python 3.9.11-x86_64-centos7"
python -m pip install -e .[dev]
```

You could also consider using Python's [venv](https://docs.python.org/3/library/venv.html).

### Run

Take a look inside the config and modify as necessary.
To run all preprocessing stages for the train split:

```bash
preprocess --config configs/test.yaml
```

To run with only specific steps enabled, include the flag for the required steps.
For example

```bash
preprocess --config configs/config.yaml --prep --resample
```

will run the first two steps.

To run without certain steps, include the corresponding negative flag (`--no-*`).
For example to run without plotting

```bash
preprocess --config configs/config.yaml --no-plot
```

If you want to preprocess the validation or test split, use the `--split` argument:

```bash
preprocess --config configs/config.yaml --split val
```

You can also use `split=all` to run each of the train/val/test splits in a single command.

See `preprocess --help` for the full list of flags.


### Comparisons with umami

#### Main changes

- modular, class-based design
- h5 virtual datasets to wrap the source files
- 2 main stages: resample -> merge -> done!
- parallelised processing of flavours within a sample
- support for different resampling "regions", which is usefull for [Xbb preprocessing](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/issues/225)
- ndim sampling support, which is also useful for Xbb
- "new" improved training file format (which is actually just the tdd output format)
    - structured arrays are smaller on disk and therefore faster to read
    - only one dataloader is needed and can be reused for training and testing
    - other plotting scripts can support a single file format
    - normalisation/concatenation is applied on the fly during training
    - training files can contain supersets of variables used for training
- new "countup" samping which is more efficient than pdf (it uses more the available statistics and reduces duplication of jets)
- the code estimates the number of unique jets for you and saves this number as an attribute in the output file


#### Performance and LOC

Compared with a comparable preprocessing config from umami:

1. train file size decreased by 30%
2. train read speed improved by 30% (separate from file size reduction, by using `read_direct`)
3. only one command is needed to generate all preprocessing outputs (running with `--split=all` will produce train/val/test files)
4. lines of code are reduced vs umami by 4x
5. 10x faster than default umami preprocessing (0.06 vs 0.825 hours/million jets)


