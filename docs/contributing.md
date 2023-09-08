If you find a bug, have a feature request or similar, feel free to submit an issue.

### Contributing guidelines

If you want to contribute to the development of the UPP, you should create a [fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html) of the repository.
You can read about forking workflows [here](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow), or take a look at the contributing guidelines in the [training dataset dumper documentation](https://training-dataset-dumper.docs.cern.ch/development/#contributing-guidelines).

You should make changes inside a [feature branch](https://docs.gitlab.com/ee/gitlab-basics/feature_branch_workflow.html) in your fork. It is generally a good idea not to work directly on the the `main` branch in your fork. Then, when your feature is ready, open a [merge request](https://docs.gitlab.com/ee/user/project/merge_requests/) to the target branch on upstream (which will usually be `main`). Once this is merged to upstream, you can `git pull upstream main` from your fork to bring back in the changes, and then fork again off `main` to start the development of a new feature. If your feature branch becomes outdated with its target, you may have to rebase or merge in the changes from the target branch, and resolve any conflicts, before you can merge.

Remember to keep you fork [up to date](https://about.gitlab.com/blog/2016/12/01/how-to-keep-your-fork-up-to-date-with-its-origin/) with upstream.

### Code Formatting

It's good practice to document your code with module and function docstrings, and inline comments.
Consider also providing type hints for the function in/outputs.
It's also recommended to use [black](https://github.com/psf/black) to format your contributions.
You can take a look at the [umami docs](https://umami-docs.web.cern.ch/setup/development/) for more guidelines on code style.

For the formatting in this project use pre-commit hooks:

```bash
pre-commit run --all-files
```

### Testing

It is highly encoraged to adhere to provide unit and/or integration tests for every new added feature.
You can test your code and check the coverage using.
```bash
coverage run --source ftag -m pytest --show-capture=stdout
coverage report 
```
You may also find the [codecov](https://about.codecov.io/) tool helpful
