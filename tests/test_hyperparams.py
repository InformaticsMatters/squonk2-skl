import json
import pickle

import numpy as np
import pandas as pd

from hyperparams import run


def test_saved_model_has_best_params(tmp_path, monkeypatch):
    # The pickled model must be the one the search ranked first, not a
    # default estimator. Neither n_estimators value is the default (100),
    # so an untuned model cannot pass by accident.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=60)
    df = pd.DataFrame(X, columns=[f"d{i}" for i in range(4)])
    df.insert(0, "Y", y)
    df.insert(0, "ID", [f"m{i}" for i in range(60)])
    df.insert(0, "SMILES", ["C"] * 60)
    infile = tmp_path / "train.csv"
    df.to_csv(infile, index=False)

    monkeypatch.chdir(tmp_path)
    run(
        str(infile),
        "best_model.jmodel",
        delimiter="comma",
        mol_column=0,
        id_column=1,
        y_column=2,
        read_header=True,
        n_top=1,
        hyper_params={"n_estimators": [3, 7], "max_depth": [1, 4]},
        search_params={"cv": 3},
        refit_models=True,
    )

    with open(tmp_path / "best_model.json", encoding="utf-8") as f:
        best = [r["params"] for r in json.load(f) if r["rank"] == 1]
    with open(tmp_path / "best_model.jmodel", "rb") as f:
        model = pickle.load(f)

    params = model.get_params()
    assert {k: params[k] for k in best[0]} in best
    assert hasattr(model, "estimators_")
