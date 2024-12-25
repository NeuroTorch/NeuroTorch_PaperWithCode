import json
import os
import shutil
from copy import deepcopy

import neurotorch as nt
import numpy as np
import pandas as pd
import torch
from neurotorch.callbacks.convergence import ConvergenceTimeGetter
from neurotorch.regularization.connectome import ExcRatioTargetRegularization
from pythonbasictools import logs_file_setup, log_device_setup, DeepLib
import matplotlib
import pickle

from torch.utils.data import DataLoader

from dataset import get_dataloader, TimeSeriesDataset
from models import get_model, ModelType
from neuronal_time_series_reproduction.reports.data_report_analysis import get_bests_by_key
from utils import load_model, gather_model_from_path, return_default_on_except, maybe_to_numpy
from train_script import train, set_trainer_default_params, test_model, get_exc_ratios
from figures_script import complete_report, plot_model_comparison, plot_simple_report


def load_or_compute_test_prediction(model: nt.SequentialRNN, dataset: TimeSeriesDataset, **kwargs):
    pred_file = os.path.join(model.checkpoint_folder, "objs", "test_pred.pkl")
    os.makedirs(os.path.dirname(pred_file), exist_ok=True)
    encoder = model.get_layer()
    if os.path.exists(pred_file):
        pred_data = pickle.load(open(pred_file, "rb"))
        y_pred, y_target, pvar, pvar_mean, pvar_std = (
            pred_data["pred"], pred_data["target"], pred_data["pvar"], pred_data["pvar_mean"], pred_data["pvar_std"]
        )
        print(f"[{model.name}] pVar: {pvar:.2f}, E[pVar]: {pvar_mean:.2f} +/- {pvar_std:.2f}")
    else:
        y_pred, y_target, pvar, pvar_mean, pvar_std, hh = test_model(
            model, dataset, n_test=kwargs.get("n_test", 1), load=False, verbose=True, return_hidden_states=True,
        )
        spikes = nt.to_numpy(hh[encoder.name][-1]) if isinstance(encoder, (nt.LIFLayer, nt.SpyLIFLayer)) else None
        pred_data = dict(
            pred=nt.to_numpy(y_pred), target=nt.to_numpy(y_target), pvar=nt.to_numpy(pvar),
            pvar_mean=nt.to_numpy(pvar_mean), pvar_std=nt.to_numpy(pvar_std),
            hh=hh, spikes=nt.to_numpy(spikes),
            note="The shapes are (n_test, n_time_steps, n_units) where n_test is the batch_size."
        )
        pred_data.update({
            f"{layer.name}.{weights_attr}": return_default_on_except(getattr, None, layer, weights_attr, None)
            for layer in model.get_all_layers()
            for weights_attr in ["forward_weights", "recurrent_weights"]
        })
        pickle.dump(pred_data, open(pred_file, "wb"))
    return y_pred, y_target, pvar, pvar_mean, pvar_std


def model_test_from_path(model_path: str, **kwargs):
    infos_dirname = "test_infos"

    model_dict = gather_model_from_path(model_path, raise_exception=True)
    params = model_dict['params']
    # model.load_checkpoint(load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR, verbose=True)
    y_pred, y_target, pvar, pvar_mean, pvar_std = load_or_compute_test_prediction(
        model_dict["model"], model_dict["dataloader"].dataset, **kwargs
    )
    viz_pred, viz_target = complete_report(
        model_dict["model"], y_pred=y_pred, y_target=y_target, show=False
    )
    plot_simple_report(
        model_dict["model"], y_pred=y_pred, y_target=y_target, show=False,
    )
    os.makedirs(f"{model_dict['model'].checkpoint_folder}/{infos_dirname}", exist_ok=True)
    json.dump(
        params,
        open(f"{model_dict['model'].checkpoint_folder}/{infos_dirname}/params.json", "w+"),
        indent=4,
    )
    results = dict(
        pvar=float(nt.to_numpy(pvar)),
        pvar_mean=float(nt.to_numpy(pvar_mean)),
        pvar_std=float(nt.to_numpy(pvar_std)),
        n_units=float(nt.to_numpy(params["n_units"])),
        n_time_steps=float(nt.to_numpy(params["n_time_steps"])),
        n_iterations=float(nt.to_numpy(params["n_iterations"])),
        # iteration=float(nt.to_numpy(wc_trainer.state.iteration)),
        batch_size=float(nt.to_numpy(params["batch_size"])),
        exc_ratio=get_exc_ratios(model_dict['model']),
        model_name=model_dict['model'].name,
    )
    # results.update(get_convergence_infos(snn_trainer))
    json.dump(
        results, open(f"{model_dict['model'].checkpoint_folder}/{infos_dirname}/results.json", "w+"), indent=4
    )
    return model_dict, viz_pred, viz_target, y_pred


def main(snn_path: str, wc_path: str, **kwargs):
    matplotlib.use('Agg')
    # Wilson-Cowan
    wc_model_dict, wc_viz_pred, wc_viz_target, wc_y_pred = model_test_from_path(wc_path, **kwargs)
    # SNN-LPF
    snn_model_dict, snn_viz_pred, snn_viz_target, snn_y_pred = model_test_from_path(snn_path, **kwargs)

    # Plot results
    hdf5_params = deepcopy(wc_model_dict['params'])
    hdf5_params["filename"] = wc_model_dict['params']["filename"].replace(".npy", ".hdf5")
    hdf5_file = get_dataloader(verbose=False, **hdf5_params).dataset.hdf5_file
    save_folder = f"{wc_model_dict['model'].checkpoint_folder}/comparison_figures"
    saved_files = plot_model_comparison(
        model0=wc_model_dict['model'], model1=snn_model_dict['model'],
        y_pred0=wc_y_pred, y_pred1=snn_y_pred,
        visualise0=wc_viz_pred, visualise1=snn_viz_pred,
        visualise_target=wc_viz_target,
        save_folder=save_folder,
        timesteps=np.asarray(hdf5_file["timesteps"]),
        show=False,
    )
    snn_dest_folder = os.path.join(snn_model_dict['model'].checkpoint_folder, "comparison_figures")
    os.makedirs(snn_dest_folder, exist_ok=True)
    for file in saved_files:
        shutil.copyfile(file, os.path.join(snn_dest_folder, os.path.basename(file)))
    print(f"Saved figures to: {save_folder} & {snn_dest_folder}.")
    return 0


if __name__ == '__main__':
    import sys
    logs_file_setup(sys.argv[0].replace('.', '_'), add_stdout=False)
    log_device_setup(deepLib=DeepLib.Pytorch)
    # model_test_from_path(
    #     model_path="./data/tr_eprop_dff_N1644_bs1_075pvar/ckpt_snn_lpf_dff_matrix_1644_npy_eprop_dale_0"
    # )
    # model_test_from_path(
    #     model_path="./data/tr_eprop_dff_N1644_T1000_bs1_075pvar_714sigma/ckpt_snn_lpf_dff_matrix_1000timesteps_npy_eprop_dale_0"
    # )
    _data_root = "./data/tr_eprop_Stimulus_data_2022_02_23_fish3_1"
    main(
        snn_path=os.path.join(_data_root, "ckpt_snn_lpf_Stimulus_data_2022_02_23_fish3_1_npy_eprop_dale_0"),
        wc_path=os.path.join(_data_root, "ckpt_wilson_cowan_Stimulus_data_2022_02_23_fish3_1_npy_eprop_dale_0"),
    )

    # model_dict = gather_model_from_path(
    #     model_path="./data/tr_data/sigma10/eprop_dale/ckpt_snn_lpf_Stimulus_data_2022_02_23_fish3_1_npy",
    #     raise_exception=True
    # )
    # y_pred, y_target, pvar, pvar_mean, pvar_std = load_or_compute_test_prediction(
    #     model_dict["model"], model_dict["dataloader"].dataset
    # )
    # plot_simple_report(model_dict["model"], y_pred=y_pred, y_target=y_target, show=True)



