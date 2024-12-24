import shutil
from collections import namedtuple, defaultdict
from copy import deepcopy
from typing import List, Dict, Any, Union, Optional, Tuple, NamedTuple

import numpy as np
import json
import os
import neurotorch as nt
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import pythonbasictools as pbt
import pathlib
import psutil
from scipy import stats

from dataset import get_dataloader
from models import get_model, ModelType
from train_script import test_model


def eval_all_model_objects(
        model_objects_list: List[Dict[str, Any]],
        **kwargs
) -> List[Dict[str, Any]]:
    n_test = kwargs.get("n_test", 3)
    verbose = kwargs.get("verbose", True)

    test_model_out_list = pbt.multiprocessing_tools.apply_func_multiprocess(
        test_model,
        iterable_of_args=[
            (m_objects["model"], m_objects["dataloader"].dataset)
            for i, m_objects in enumerate(model_objects_list)
        ],
        iterable_of_kwargs=[
            dict(n_test=n_test, verbose=False, load=False)
            for _ in model_objects_list
        ],
        desc=f"Testing models [{n_test=}]",
        verbose=verbose,
        nb_workers=kwargs.get("nb_workers", -2),
    )
    for i, (m_objects, test_model_out) in enumerate(zip(model_objects_list, test_model_out_list)):
        y_pred, y_target, pvar, pvar_mean, pvar_std = test_model_out
        pvar, pvar_mean, pvar_std = nt.to_numpy(pvar), nt.to_numpy(pvar_mean), nt.to_numpy(pvar_std)
        model_objects_list[i]["pvar"] = pvar
        model_objects_list[i]["pvar_mean"] = pvar_mean
        model_objects_list[i]["pvar_std"] = pvar_std

    return model_objects_list


def _filter_models_(
        model_objects_list: List[Dict[str, Any]],
        layers: List[str] = None,
        **kwargs
) -> List[Dict[str, Any]]:
    if not isinstance(layers, (list, tuple)):
        layers = [layers for _ in range(len(model_objects_list))]

    verbose = kwargs.get("verbose", True)
    print_filtering = kwargs.get("print_filtering", verbose)
    b_test_models = kwargs.get("test_models", True)

    if b_test_models:
        model_objects_list = eval_all_model_objects(model_objects_list, **kwargs)
    min_perf = kwargs.get("min_perf", -np.inf)
    min_perf_key = kwargs.get("min_perf_key", "pvar")
    rejected_models_idx, rejected_models, rejected_layers = [], [], []

    for i, m_objects in enumerate(model_objects_list):
        if np.nan_to_num(model_objects_list[i][min_perf_key], nan=-np.inf) < min_perf:
            # reject the model if it's not good enough
            rejected_models_idx.append(i)

    if len(rejected_models_idx) > 0:
        rejected_models_idx = np.flip(np.sort(rejected_models_idx))
        rejected_models = [model_objects_list.pop(i) for i in rejected_models_idx]
        rejected_layers = [layers.pop(i) for i in rejected_models_idx]
        if verbose and print_filtering:
            print(f"{len(rejected_models)} models has been rejected because of {min_perf_key} < {min_perf}.")
            print(f"Rejected models:\n{json.dumps([{k: str(v) for k, v in m.items()} for m in rejected_models], indent=4)}")
    if verbose and print_filtering:
        print(f"{len(model_objects_list)} models has pass the test.")
    return model_objects_list


def _make_model_copy(model_dict: Dict[str, Any], location: str):
    new_path = os.path.join(location, os.path.basename(model_dict["rootpath"]))
    shutil.copytree(model_dict["rootpath"], new_path)
    model_dict["rootpath"] = new_path
    return model_dict
 

def copy_models_in_folder(
    model_objects_list: List[Dict[str, Any]],
    location: str,
    **kwargs
):
    model_objects_list = pbt.multiprocessing_tools.apply_func_multiprocess(
        _make_model_copy,
        iterable_of_args=[(m, location) for m in model_objects_list],
        desc=f"Copying models",
        verbose=kwargs.get("verbose", True),
        nb_workers=kwargs.get("nb_workers", -2),
    )
    return model_objects_list


if __name__ == '__main__':
    import sys
    from resilience import gather_models_from_folders

    is_running_in_terminal = sys.stdout.isatty()
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = "./data/tr_eprop"
    try:
        _models = gather_models_from_folders([
            str(p)
            for p in pathlib.Path(data_root).glob("*")
            if p.is_dir() and any([f.suffix == ".pth" for f in p.glob("*")]) and "eprop" in str(p)
        ], nb_workers=max(0, psutil.cpu_count(logical=False)))
        _m_list = _filter_models_(
            _models,
            n_test=32,
            min_perf=0.8,
            nb_workers=max(0, psutil.cpu_count(logical=False)),
        )
        _m_list = copy_models_in_folder(
            _m_list,
            location=f"{data_root}_filtered",
            nb_workers=max(0, psutil.cpu_count(logical=False)),
        )
    except Exception as e:
        if is_running_in_terminal:
            print(e)
        else:
            raise e
    if is_running_in_terminal:
        _ = input("Press Enter to close.")
