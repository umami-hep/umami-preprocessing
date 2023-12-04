This guide will walk you through the process of setting up the Umami-Preprocessing Python package on your system.

### Environment Setup

Creating a virtual environment helps keep your Python environment isolated and organized.
While it's not essential, we highly recommend using one.
You can set up a virtual environment using either Conda or Python's venv.
UPP requires Python 3.8 or later.

=== "conda"

    Set up a fresh [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) or [mamba](https://github.com/conda-forge/miniforge#install) environment:

    ```bash
    mamba create -n upp python
    mamba activate upp
    ```

=== "venv"

    [venv](https://docs.python.org/3/library/venv.html) is a lightweight solution for creating virtual python environments, however it is not as fully featured as a fully fledged package manager such as conda.
    Create a fresh virtual environment and activate it using

    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

=== "lxplus"

    If you don't want to use conda environments on lxplus, you can setup python via

    ```bash
    setupATLAS
    lsetup "python 3.9.18-x86_64-centos7"
    ```

    !!!info "You can also [set up conda on lxplus](https://abpcomputing.web.cern.ch/guides/python_inst/)"


### PyPi installation 

A simple installation can be done via `pip` from Python Packade Index:

```bash
pip install umami-preprocessing
```

!!!info "On lxplus you may have to use ```pip3 install umami-preprocessing``` due to pip otherwise defaulting to an older version of python"

After this installation all the fuctionality for the user is available. The further steps are only useful for the development of the package.

### Get the code

Start by cloning the Umami-Preprocessing repository and navigating into the project directory using the following commands in your terminal:

```bash
git clone https://github.com/umami-hep/umami-preprocessing.git
cd umami-preprocessing
```

### Install package from code 

Install the package as in editable mode if you would like to develop the code:

```bash
python -m pip install -e .[dev]
```

Or do a simple installation if you only plan to use the provided functionality as is

```bash
python -m pip install .
```

### Run the tests (Optional)

To ensure that the package is working correctly, you can run tests using the pytest framework. 
Additionally, you can install coverage to generate a coverage report.

Use ```pytest``` to run the tests to make sure the package works

```bash
pytest tests 
```

If you want to measure test coverage, first install coverage, and then run tests as follows:

```bash
coverage run --source upp -m pytest tests --show-capture=stdout
coverage report 
```
