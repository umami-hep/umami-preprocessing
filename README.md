[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/umami-hep/umami-preprocessing/graph/badge.svg?token=K8MJI20UZO)](https://codecov.io/gh/umami-hep/umami-preprocessing)
[![PyPI version](https://badge.fury.io/py/umami-preprocessing.svg)](https://badge.fury.io/py/umami-preprocessing)
[![docs](https://img.shields.io/badge/info-documentation-informational)](https://umami-hep.github.io/umami-preprocessing//)


# UPP: Umami PreProcessing

This is a modular preprocessing pipeline for jet tagging.
It addresses [several issues](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/issues/?label_name%5B%5D=Preprocessing) with the current umami preprocessing workflow, and uses the [`atlas-ftag-tools`](https://github.com/umami-hep/atlas-ftag-tools/) package extensively.

Documentation is under construction [here](https://umami-hep.github.io/umami-preprocessing/)


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


