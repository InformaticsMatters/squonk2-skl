import json
import pickle

import numpy as np
import pandas as pd
import pytest

from hyperparams import run


@pytest.fixture(name="infile")
def fixture_infile(tmp_path):
    # ID, SMILES and Y first, as in the Job's default column layout,
    # then the descriptors
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=60)
    df = pd.DataFrame(X, columns=[f"d{i}" for i in range(4)])
    df.insert(0, "Y", y)
    df.insert(0, "SMILES", ["C"] * 60)
    df.insert(0, "ID", [f"m{i}" for i in range(60)])
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return str(path)


def _run(infile, outbase, refit_models=True):
    run(
        infile,
        outbase,
        delimiter="comma",
        read_header=True,
        n_top=1,
        hyper_params={"n_estimators": [3, 7], "max_depth": [1, 4]},
        search_params={"cv": 3},
        refit_models=refit_models,
    )


def test_saved_model_has_best_params(infile, tmp_path, monkeypatch):
    # The pickled model must be the one the search ranked first, not a
    # default estimator. Neither n_estimators value is the default (100),
    # so an untuned model cannot pass by accident.
    monkeypatch.chdir(tmp_path)
    _run(infile, "best_model")

    with open(tmp_path / "best_model.json", encoding="utf-8") as f:
        best = [r["params"] for r in json.load(f) if r["rank"] == 1]
    with open(tmp_path / "best_model.jmodel", "rb") as f:
        model = pickle.load(f)

    params = model.get_params()
    assert {k: params[k] for k in best[0]} in best
    assert hasattr(model, "estimators_")


def test_outputs_go_beside_each_other(infile, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run(infile, "results/tuned")

    assert (tmp_path / "results" / "tuned.json").exists()
    assert (tmp_path / "results" / "tuned.jmodel").exists()
    assert not list(tmp_path.glob("tuned.*"))


def test_no_model_without_refit(infile, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run(infile, "best_model", refit_models=False)

    assert (tmp_path / "best_model.json").exists()
    assert not (tmp_path / "best_model.jmodel").exists()
