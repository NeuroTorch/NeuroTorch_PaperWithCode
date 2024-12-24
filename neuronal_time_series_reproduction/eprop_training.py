import json
import os
import pathlib
from copy import deepcopy
from typing import Optional

import neurotorch as nt
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
from neurotorch.callbacks.convergence import ConvergenceTimeGetter
from pythonbasictools import logs_file_setup, log_device_setup, DeepLib
import matplotlib
from tqdm import tqdm

from dataset import get_dataloader
from models import get_model, ModelType
from utils import get_cmd_kwargs, gather_models_from_folders
from _filter_models import _filter_models_
from train_script import train, set_trainer_default_params, test_model, get_exc_ratios, run_model


def _get_la_name_from_model_object_dict(model_object_dict: dict, learning_algorithms_2_args: dict):
	out_la = None
	for la_name, la_params in learning_algorithms_2_args.items():
		all_match = True
		for k, v in la_params.items():
			all_match = all_match and model_object_dict["params"][k] == v
		if all_match:
			out_la = la_name
			break
	return out_la


def _get_la_model_counts(model_objects_list, learning_algorithms_2_args, model_types):
	return {
		(la, mt): len([
			m
			for m in model_objects_list
			if _get_la_name_from_model_object_dict(m, learning_algorithms_2_args) == la and m["model_type"] == mt
		])
		for la in learning_algorithms_2_args.keys()
		for mt in model_types
	}


def main(
		output_results_file: str = "results.csv",
		output_tr_data_folder: str = "data/tr_eprop",
		n_pts: int = 10,
		**kwargs
):
	learning_algorithms_2_args = {
		"eprop_dale": dict(
			learning_algorithm="eprop",
			force_dale_law=True,
		),
		"eprop": dict(
			learning_algorithm="eprop",
			force_dale_law=False,
		),
	}
	args = dict(
		output_tr_data_folder=output_tr_data_folder,
		filename=kwargs.get("filename", "Stimulus_data_2022_02_23_fish3_1.npy"),
		n_units=-1,
		smoothing_sigma=kwargs.get("smoothing_sigma", 3.57),
		n_iterations=kwargs.get("n_iterations", 5_000),
		early_stopping_threshold=kwargs.get("early_stopping_threshold", 0.98),
		convergence_threshold=0.9,
		force_dale_law=False,
		force_overwrite=False,
		use_time_steps_scheduler=kwargs.get("use_time_steps_scheduler", False),
		batch_size=kwargs.get("batch_size", 32),
		time_steps_scheduler_target=kwargs.get("time_steps_scheduler_target", 0.8),
		time_steps_scheduler_n_stages=kwargs.get("time_steps_scheduler_n_stages", 500),
		normalize_by_unit=kwargs.get("normalize_by_unit", False),
		use_out_layer=kwargs.get("use_out_layer", True),
	)
	output_results_filepath = os.path.join(args["output_tr_data_folder"], output_results_file)
	if os.path.exists(output_results_filepath):
		df = pd.read_csv(output_results_filepath)
	else:
		df = pd.DataFrame(columns=["info", "info_idx", "model_type"])
	model_types = kwargs.get("model_types", [m for m in ModelType])

	filter_kwargs = deepcopy(kwargs)
	filter_kwargs["verbose"] = False

	def _gather_models_from_given_folders():
		return gather_models_from_folders([
				str(p)
				for p in pathlib.Path(output_tr_data_folder).glob("*")
				if p.is_dir() and any([f.suffix == ".pth" for f in p.glob("*")])
			],
			**filter_kwargs
		)

	model_objects_list = _gather_models_from_given_folders()
	model_objects_list_filtered = _filter_models_(deepcopy(model_objects_list), **filter_kwargs)
	lm_counts = _get_la_model_counts(model_objects_list, learning_algorithms_2_args, model_types)
	lm_counts_filtered = _get_la_model_counts(model_objects_list_filtered, learning_algorithms_2_args, model_types)

	def _sum_counts(__counts):
		return sum([v for k, v in __counts.items()])

	total_p_bar = n_pts * len(learning_algorithms_2_args) * len(model_types)
	p_bar = tqdm(
		range(total_p_bar),
		desc="Running experiments", unit="exp",
		initial=_sum_counts(lm_counts_filtered),
		disable=total_p_bar < 2,
	)
	for la, la_args in learning_algorithms_2_args.items():
		for model_type in model_types:
			while lm_counts_filtered[(la, model_type)] < n_pts:
				counts_repr = f"{lm_counts[(la, model_type)]}-{lm_counts_filtered[(la, model_type)]}/{n_pts}"
				p_bar.set_postfix(LA=la, model=f"{model_type.value}", counts=counts_repr)
				main_args = deepcopy(args)
				main_args.update(la_args)
				main_args["seed"] = lm_counts[(la, model_type)]
				try:
					results = run_model(
						model_type=model_type, post_name=f"_{la}_{lm_counts[(la, model_type)]}",
						verbose=p_bar.disable, p_bar=p_bar, **main_args
					)
					results.update(la_args)
					results["smoothing_sigma"] = main_args["smoothing_sigma"]
					results["info"] = la
					results["info_idx"] = lm_counts[(la, model_type)]
					results["model_type"] = model_type.name
					results = {k: [v] for k, v in results.items()}
					df = pd.concat([df, pd.DataFrame(results, index=[0])], ignore_index=True)
				except Exception as e:
					print(f"Error: {e}")
					if kwargs.get("raise_exception", False):
						raise e
				finally:
					df.to_csv(output_results_filepath, index=False)
					model_objects_list = _gather_models_from_given_folders()
					model_objects_list_filtered = _filter_models_(deepcopy(model_objects_list), **filter_kwargs)
					lm_counts = _get_la_model_counts(model_objects_list, learning_algorithms_2_args, model_types)
					lm_counts_filtered = _get_la_model_counts(
						model_objects_list_filtered, learning_algorithms_2_args, model_types
					)
					p_bar.update(max(0, _sum_counts(lm_counts_filtered) - p_bar.n))
	df.to_csv(output_results_filepath, index=False)


if __name__ == '__main__':
	import sys

	logs_file_setup("tr_eprop", add_stdout=False)
	log_device_setup(deepLib=DeepLib.Pytorch)
	sys_kwgs = get_cmd_kwargs({
		1: "./data/tr_eprop_Stimulus_data_2022_02_23_fish3_1",
		"filename": "Stimulus_data_2022_02_23_fish3_1.npy",
		"n_pts": 1,
		"n_test": 32,
		"nb_workers": 0,
		"min_perf": 0.8,
		"use_time_steps_scheduler": 0,
		"n_iterations": 5_000,
		"batch_size": 1,
		"time_steps_scheduler_target": 0.75,
		"time_steps_scheduler_n_stages": 100,
		"smoothing_sigma": 10,
		"normalize_by_unit": 0,
		"use_out_layer": 1,
		"model_types": "SNN_LPF,WILSON_COWAN",
	})
	is_running_in_terminal = sys.stdout.isatty()
	if not is_running_in_terminal:
		sys_bs = sys_kwgs["batch_size"]
		sys_pvar = str(sys_kwgs["time_steps_scheduler_target"]).replace(".", "")
		sys_sigma = str(sys_kwgs["smoothing_sigma"]).replace(".", "")
		sys_nrm = str(sys_kwgs["normalize_by_unit"])
		sys_uol = str(sys_kwgs["use_out_layer"])
	nt.set_seed(42)
	main(
		output_tr_data_folder=sys_kwgs[1],
		n_pts=sys_kwgs["n_pts"],
		model_types=[
			ModelType.from_repr(model_type.strip())
			for model_type in sys_kwgs["model_types"].split(",")
		],
		n_test=sys_kwgs["n_test"],
		min_perf=sys_kwgs["min_perf"],
		nb_workers=sys_kwgs["nb_workers"],
		filename=sys_kwgs["filename"],
		use_time_steps_scheduler=bool(sys_kwgs["use_time_steps_scheduler"]),
		n_iterations=sys_kwgs["n_iterations"],
		batch_size=sys_kwgs["batch_size"],
		time_steps_scheduler_target=sys_kwgs["time_steps_scheduler_target"],
		time_steps_scheduler_n_stages=sys_kwgs["time_steps_scheduler_n_stages"],
		smoothing_sigma=sys_kwgs["smoothing_sigma"],
		normalize_by_unit=bool(sys_kwgs["normalize_by_unit"]),
		use_out_layer=bool(sys_kwgs["use_out_layer"]),
		raise_exception=not is_running_in_terminal,
	)


