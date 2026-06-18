This guide will walk you through the process of setting up the Umami-Preprocessing Python package on your system.

### Environment Setup

Install UPP in a virtual environment to avoid conflicts with other software libraries.
UPP currently supports Python 3.11, 3.12, 3.13 and 3.14.

=== "uv"

    [uv](https://docs.astral.sh/uv/) is a fast Python package and project manager and is
    the recommended way to develop UPP. Install it following the
    [official instructions](https://docs.astral.sh/uv/getting-started/installation/), then
    let `uv` create and manage the environment for you:

    ```bash
    uv venv
    source .venv/bin/activate
    ```

    When working from a clone of the repository you can skip the manual environment setup
    entirely and use `uv sync` (see below).

=== "conda"

    Set up a fresh [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) or [mamba](https://github.com/conda-forge/miniforge#install) environment:

    ```bash
    conda create -n upp python=3.13
    conda activate upp
    ```

=== "venv"

    [venv](https://docs.python.org/3/library/venv.html) is a lightweight solution for creating virtual python environments, however it is not as fully featured as a fully fledged package manager such as conda.
    Create a fresh virtual environment and activate it using

    ```bash
    python3 -m venv upp
    source upp/bin/activate
    ```

=== "lxplus"

    If you don't want to use conda environments on lxplus, you can setup python via

    ```bash
    setupATLAS
    lsetup "python 3.11.9-x86_64-el9"
    ```

    !!!info "You can also [set up conda on lxplus](https://abpcomputing.web.cern.ch/guides/python_inst/)"


### PyPi installation 

If you don't plan on editing the source code, the simplest way to install UPP directly from the Python Package Index (PyPI):

```bash
python -m pip install umami-preprocessing
```

!!!info "On lxplus you may have to use ```python3``` instead of just ```python```"

### Download source code
The following instructions are only relevant for those that wish to modify UPP source code.
Start by cloning the Umami-Preprocessing repository. If you want to contribute to the development of UPP, you should fork the repository and make sure you do all your edits in a development branch.

```bash
git clone https://github.com/umami-hep/umami-preprocessing.git
```


### Install package from code 

Navigate to the newly downloaded repository and install the package in editable mode.

=== "uv"

    `uv sync` creates a virtual environment in `.venv` and installs UPP together with the
    development dependency group in editable mode:

    ```bash
    cd umami-preprocessing
    uv sync
    ```

    Prefix subsequent commands with `uv run` (e.g. `uv run preprocess ...`) or activate the
    environment with `source .venv/bin/activate`.

=== "pip"

    Install UPP in editable mode, then add the development dependency group (requires
    `pip >= 25.1`, which understands PEP 735 dependency groups):

    ```bash
    cd umami-preprocessing
    python -m pip install -e .
    python -m pip install --group dev
    ```

If you don't plan on editing the code you can do a regular install instead

```bash
python -m pip install .
```

!!!info "Note for running on lxplus"

    Again, you may have to use ```python3``` here.

    You may also see a warning like this:

    ```
    WARNING: The script preprocess is installed in '/afs/cern.ch/user/X/Y/.local/bin' which is not on PATH.
    Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
    ```

    if you do, you can add the directory to your path using

    ```
    export PATH=$PATH:/afs/cern.ch/user/X/Y/.local/bin
    ```

    Alternatively, you can just run the scripts by pointing to the `main.py`

    ```bash
    python3 upp/main.py
    ```


### Run the tests (Optional)

To ensure that the package is working correctly, you can run the tests using the pytest framework. 

Use ```pytest``` to run the tests to make sure the package works

```bash
pytest tests 
```

If you want to measure test coverage you can use the commands:
```bash
coverage run --source upp -m pytest tests --show-capture=stdout
coverage report 
```
