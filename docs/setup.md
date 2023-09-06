

### Get the code:

clone and enter the repository using:

```bash
git clone https://github.com/umami-hep/umami-preprocessing.git
cd umami-preprocessing
```
### Setup a virtual environment or load python>=3.8

Conda environents are a great way to keep organised, but are not essential for installing the code. 
With a fresh [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) or [mamba](https://github.com/conda-forge/miniforge#install) environment:

```bash
mamba create -n upp python=3.8.10
mamba activate upp
```

See [here](https://abpcomputing.web.cern.ch/guides/python_inst/) for instructions on installing conda on lxplus.
If you don't want to use conda environments on lxplus, you could probably just install with

```bash
setupATLAS
lsetup "python 3.9.11-x86_64-centos7"
```

You could also consider using Python's [venv](https://docs.python.org/3/library/venv.html).

### Install package 
Install the package as in editable mode if you would like to develop the code:

```bash
python -m pip install -e .[dev]
```

Or do a simple installation if you only plan to use the provided functionality

```bash
python -m pip install .
```

### Run the tests (Oprional)

Use ```pytest``` to run the tests to make sure the package works

```bash
pytest tests 
```

Or install ```coverage``` to also get a coverage report:

```bash
coverage run --source ftag -m pytest --show-capture=stdout
coverage report 
```