# squonk2-skl

Data Manager Jobs built on [scikit-learn](https://scikit-learn.org/). The
repository provides one Job, `hyperparameters`, which tunes a
`RandomForestRegressor` with a grid search.

# Setting up the environment
The development environment requires [Poetry](https://python-poetry.org/docs/main#installing-with-pipx).
From the project root run:


    poetry shell
    poetry install --no-root

When done, set up pre-commit:


    pre-commit install -t commit-msg -t pre-commit

## The `research` dependency group

`jaqpotpy` - and through it `torch` - lives in an optional `research` group,
so the command above does not install it and it is not in the published image.
Nothing the Jobs run needs it: its only consumer is
`src/helpers/cross_train.py`, which nothing imports.

Install it if you are picking that work back up:


    poetry install --no-root --with research

Expect it to take some time. It is around 8.7 GB installed, nearly all of
it the CUDA runtime that ships with the default `torch` wheel.

`PyTDC` is not in the group, because it cannot be installed here. 0.3.x needs
`rdkit-pypi` 2022.9.5, whose last wheel is cp311 and which publishes no sdist,
and 1.x pins `numpy < 2`. `src/helpers/get_data.py` imports nothing, so it
still loads; add `PyTDC` back if that work resumes, on a Python and a numpy
it supports.
