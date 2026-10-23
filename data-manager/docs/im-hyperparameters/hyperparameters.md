# Random forest hyperparameters

Runs a [grid search] over the hyperparameters of a scikit-learn
[`RandomForestRegressor`] and writes out the best model it finds.

## Input

A `.smi` or `.csv` file with one molecule per row: an ID column, a molecule
(SMILES) column, and a column holding the Y variable being regressed against.
The remaining columns are taken as the descriptors to train on, so the file is
normally the output of a descriptor Job such as `mordred-descriptor-generator`.

Column positions are given as zero-based indices - `idColumn` (default 0),
`molColumn` (default 1) and `yColumn` (default 2) - and all three must differ.
Set `readHeader` if the first line names the fields.

## Options

Each tunable `RandomForestRegressor` parameter takes a **comma-separated list**
of values, and the search runs over every combination of the lists you give.
`bootstrap: True,False` with `ccp_alpha: 0.0,0.1,0.5` is six fits per
cross-validation fold. Anything left unset contributes a single value, the
estimator's own default.

The grid grows multiplicatively, so a handful of short lists is already a long
run. The `n_jobs_search` and `n_jobs_estimator` options control parallelism -
`-1` uses every available processor.

Cross-validation and scoring are configured separately, through `cv`,
`scoring`, `pre_dispatch`, `error_score` and `return_train_score`, which are
passed through to [`GridSearchCV`].

## Output

Two files, named from `outputBaseName` (`best_model` by default), which may
include a subdirectory:

- `<outputBaseName>.json` - the search results: for each of the top
  `nTopResults` parameter sets, its rank, its mean and standard deviation of
  test score across the cross-validation folds, and the full parameter set.
- `<outputBaseName>.jmodel` - the best parameter set, refitted on all the data
  and pickled. Not written when `refitBestModels` is false.

[grid search]: https://scikit-learn.org/stable/modules/grid_search.html
[`GridSearchCV`]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
[`RandomForestRegressor`]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
