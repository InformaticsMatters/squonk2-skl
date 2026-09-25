import argparse

# import time
from collections import OrderedDict

import pandas as pd
from dm_job_utilities.dm_log import DmLog
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    make_scorer,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

# from tdc.benchmark_group import admet_group

# jaqpot has stopped working
# from jaqpotpy.datasets import SmilesDataset
# from jaqpotpy.descriptors.molecular import (
#     MACCSKeysFingerprint,
#     MordredDescriptors,
#     RDKitDescriptors,
#     TopologicalFingerprint,
# )
# from jaqpotpy.doa.doa import Leverage
# from jaqpotpy.models import MolecularSKLearn
# from jaqpotpy.models.evaluator import Evaluator


# from helpers.cross_train import cross_train_sklearn
# from helpers.get_data import get_dataset

default_num_chars = 2
default_num_levels = 2

delimiter_choices = {
    "space": None,
    "tab": "\t",
    "comma": ",",
    "pipe": "|",
}


def create_common_args():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--data", required=True, help="Training data")
    argParser.add_argument(
        "-f",
        "--featurizers",
        nargs="+",
        choices=["mordred", "maccs", "topo", "rdkit"],
        help="Molecular feature generators to use",
    )
    argParser.add_argument(
        "-s",
        "--scoring-functions",
        nargs="+",
        choices=["MAE", "ACC", "AUC", "AUPRC", "SPM"],
        help="Scoring functions to use in the Evaluator",
    )
    argParser.add_argument("--doa", help="DOA algorithm (leverage or None)")
    return argParser


def create_featurizer(name: str):
    if name == "mordred":
        #     return MordredDescriptors()
        # elif name == "maccs":
        #     return MACCSKeysFingerprint()
        # elif name == "topo":
        #     return TopologicalFingerprint()
        # elif name == "rdkit":
        #     return RDKitDescriptors()
        # else:
        raise ValueError("Invalid featurizer name: " + name)


def create_featurizers(names):
    return [create_featurizer(n) for n in names]


# def create_evaluator(scoring_function_names):
#     val = Evaluator()
#     for name in scoring_function_names:
#         if name == "MAE":
#             val.register_scoring_function(name, mean_absolute_error)
#         elif name == "ACC":
#             val.register_scoring_function(name, accuracy_score)
#         elif name == "AUC":
#             val.register_scoring_function(name, roc_auc_score)
#         elif name == "AUPRC":
#             val.register_scoring_function(name, average_precision_score)
#         elif name == "SPM":
#             val.register_scoring_function(name, spearmanr)
#         else:
#             raise ValueError("Invalid scoring function name: " + name)
#     return val


def create_doa(s_doa):
    """
    Domain of applicability will be Leverage() or None
    :param s_doa:
    :return:
    """
    if s_doa is None:
        return None
    # elif s_doa.lower() == "leverage":
    #     return Leverage()
    else:
        print(
            "invalid value for doa. only leverage is supported. {} was specified".format(
                s_doa
            )
        )
        exit(1)


# class Runner:
#     def __init__(
#         self, dataset_name: str, models: dict, doa, evaluator, featurizers, task
#     ):
#         self.group = admet_group(path="data/")
#         self.benchmark, self.name = get_dataset(dataset_name, self.group)

#         self.train_val = self.benchmark["train_val"]
#         self.test = self.benchmark["test"]

#         self.models = models
#         self.doa = doa
#         self.evaluator = evaluator
#         self.featurizers = featurizers
#         self.task = task

#     def run_cross_validation(self):
#         print("Evaluating {} models".format(len(self.models)))

#         t0 = time.time()
#         results = []
#         for featurizer in self.featurizers:
#             dummy_train = SmilesDataset(
#                 smiles=self.train_val["Drug"],
#                 y=self.train_val["Y"],
#                 featurizer=featurizer,
#                 task=self.task,
#             )
#             for key in self.models:
#                 model = self.models[key]
#                 skl_model = MolecularSKLearn(
#                     dummy_train, doa=self.doa, model=model, eval=self.evaluator
#                 )

#                 # Cross Validate and check robustness
#                 evaluation = cross_train_sklearn(
#                     self.group, skl_model, self.name, self.test, task=self.task
#                 )
#                 results.append((key + ", " + str(featurizer), evaluation))
#         t1 = time.time()

#         print("\n\n")
#         for result in results:
#             print("Evaluation of the model:", result[0], result[1])
#         print("Execution took {} seconds".format(round(t1 - t0)))
#         return results


def validate_args(id_column, mol_column, y_column):
    if id_column == mol_column:
        DmLog.emit_event("ERROR: mol_column and id_column must be different")
        exit(1)
    if y_column == id_column:
        DmLog.emit_event("ERROR: y_column and id_column must be different")
        exit(1)
    if y_column == mol_column:
        DmLog.emit_event("ERROR: y_column and mol_column must be different")
        exit(1)


def create_core_columns(df, id_column, mol_column, y_column):
    print("y col in core cols", y_column)
    print(df.columns)
    print(df)

    core_cols = [
        ("SMILES", mol_column, df.columns.to_list()[mol_column]),
        ("ID", id_column, df.columns.to_list()[id_column]),
    ]

    print(core_cols)

    if y_column is not None:
        core_cols.append(("Y", y_column, df.columns[y_column]))

    core_cols.sort(key=lambda x: x[1])
    return core_cols


def create_scorers(scoring_function_names):
    scorers = OrderedDict()
    for name in scoring_function_names:
        if name == "MAE":
            scorers[name] = make_scorer(mean_absolute_error)
        elif name == "ACC":
            scorers[name] = make_scorer(accuracy_score)
        elif name == "AUC":
            scorers[name] = make_scorer(roc_auc_score)
        elif name == "AUPRC":
            scorers[name] = make_scorer(average_precision_score)
        elif name == "SPM":
            scorers[name] = make_scorer(spearmanr)
        elif name == "R2":
            scorers[name] = make_scorer(r2_score)
        else:
            raise ValueError("Unsupported scoring function name: " + name)
    return scorers


def create_common_options(parser):
    parser.add_argument(
        "-i", "--infile", required=True, help="Input file (.smi, .tab, .txt)"
    )
    parser.add_argument("-d", "--delimiter", help="Delimiter when using SMILES")
    parser.add_argument(
        "--read-header",
        action="store_true",
        help="Read a header line with the field names when reading .smi or .txt",
    )
    parser.add_argument(
        "--id-column", type=int, default="0", help="Column index for molecule ID"
    )
    parser.add_argument(
        "--mol-column",
        type=int,
        default="1",
        help="Column index for molecule when using .smi",
    )
    parser.add_argument(
        "--y-column",
        type=int,
        default="2",
        help="Column index for the Y variable when using .smi",
    )
    parser.add_argument(
        "--fp-column",
        type=int,
        help="Column index for a bit string that contains the fingerprint"
        + " (if not provided, then all other columns are assumed to have the data items)",
    )


def read_csv(
    filename: str,
    read_header: bool,
    delimiter: str,
    id_column: int,
    mol_column: int,
    y_column: int,
):
    header = 0 if read_header else None

    df = pd.read_csv(
        filename, sep=delimiter, header=header, index_col=None, low_memory=False
    )

    core_cols = create_core_columns(df, id_column, mol_column, y_column)
    # print(core_cols)

    y = df.iloc[:, y_column]
    # print(y.shape)
    df.drop(df.columns[[id_column, mol_column, y_column]], axis=1, inplace=True)
    X = df

    return X, y, core_cols


def run_grid_search(method, param_grid, scorers, X, y):
    scoring = create_scorers(scorers)

    gs = GridSearchCV(
        method,
        param_grid=param_grid,
        scoring=scoring,
        refit=scorers[0],
        n_jobs=-1,
        return_train_score=True,
        # verbose=2
    )

    gs.fit(X, y)
    results = gs.cv_results_
    best_estimator = gs.best_estimator_
    best_params = best_estimator.get_params()
    print("Best model")
    for k in param_grid:
        print(k, best_params[k])
    print("best_score:", gs.best_score_)
    print("best_index:", gs.best_index_)
    for scorer in scorers:
        print(scorer, results["mean_test_" + scorer][gs.best_index_])

    return best_estimator
