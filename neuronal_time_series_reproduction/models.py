import enum
import json
import os
import pickle
from typing import Optional

import neurotorch as nt
import numpy as np
import torch

from neurotorch_fixes import SequentialRNN
from utils import ModelType, return_default_on_except, maybe_to_numpy, SigmoidTransform, str_to_activation, \
	FiringRateLayer, LinearJtoI


def init_weights(params: dict):
	w_init = str(params.get("w_init", None)).lower()
	if w_init == "dale":
		params["forward_weights"] = nt.init.dale_(
			torch.empty(params["n_units"], params["n_aux_units"]), inh_ratio=0.5, rho=0.2
		)
		if params["use_recurrent_connection"]:
			params["recurrent_weights"] = nt.init.dale_(
				torch.empty(params["n_units"], params["n_units"]), inh_ratio=0.5, rho=0.2
			)
	elif w_init == "randn":
		# params["forward_weights"] = torch.randn(params["n_units"], params["n_aux_units"])
		params["forward_weights"] = torch.nn.init.normal_(
			torch.empty(params["n_units"], params["n_aux_units"]), mean=0.0, std=1.0
		)
		if params["use_recurrent_connection"]:
			# params["recurrent_weights"] = torch.randn(params["n_units"], params["n_units"])
			params["recurrent_weights"] = torch.nn.init.normal_(
				torch.empty(params["n_units"], params["n_units"]), mean=0.0, std=1.0
			)
	elif w_init == "xavier":
		params["forward_weights"] = torch.nn.init.xavier_normal_(
			torch.empty(params["n_units"], params["n_aux_units"])
		)
		if params["use_recurrent_connection"]:
			params["recurrent_weights"] = torch.nn.init.xavier_normal_(
				torch.empty(params["n_units"], params["n_units"])
			)
	else:
		params["forward_weights"] = None
		params["recurrent_weights"] = None
	
	if not params["use_recurrent_connection"]:
		params["recurrent_weights"] = None
	return params


def set_wc_default_param(params: Optional[dict] = None) -> dict:
	if params is None:
		params = {}
	params.setdefault("n_time_steps", -1)
	params.setdefault("std_weights", 1)
	params.setdefault("dt", 0.02)
	params.setdefault("mu", 0.0)
	params.setdefault("mean_mu", 0.0)
	params.setdefault("std_mu", 1.0)
	params.setdefault("r", 0.1)
	params.setdefault("mean_r", 0.5)
	params.setdefault("std_r", 0.4)
	params.setdefault("tau", 0.1)
	params.setdefault("learn_mu", True)
	params.setdefault("learn_r", True)
	params.setdefault("learn_tau", True)
	params.setdefault("force_dale_law", False)
	params.setdefault("n_units", 200)
	params.setdefault("n_aux_units", params["n_units"])
	params.setdefault("use_recurrent_connection", False)
	params.setdefault("forward_sign", 0.5)
	params.setdefault("activation", "sigmoid")
	params.setdefault("hh_init", "inputs" if params["n_aux_units"] == params["n_units"] else "random")
	
	if params["force_dale_law"]:
		params["alpha"] = 0.0
		params["gamma"] = 0.0
		# params["params_lr"] = 1e-4
		# params["output_params_lr"] = 2e-4
	
	params = init_weights(params)
	return params


def get_wc_layer(params, device):
	return nt.WilsonCowanLayer(
		params["n_units"], params["n_aux_units"],
		std_weights=params["std_weights"],
		forward_sign=params["forward_sign"],
		dt=params["dt"],
		r=params["r"],
		mean_r=params["mean_r"],
		std_r=params["std_r"],
		mu=params["mu"],
		mean_mu=params["mean_mu"],
		std_mu=params["std_mu"],
		tau=params["tau"],
		learn_r=params["learn_r"],
		learn_mu=params["learn_mu"],
		learn_tau=params["learn_tau"],
		hh_init=params["hh_init"],
		device=device,
		name="Encoder",
		force_dale_law=params["force_dale_law"],
		activation=params["activation"],
		use_recurrent_connection=params["use_recurrent_connection"],
		forward_weights=params["forward_weights"],
		recurrent_weights=params["recurrent_weights"],
	).build()


def get_wc_curbd_layer(params, device):
	return nt.WilsonCowanCURBDLayer(
		params["n_units"], params["n_aux_units"],
		std_weights=params["std_weights"],
		forward_sign=params["forward_sign"],
		dt=params["dt"],
		r=params["r"],
		mean_r=params["mean_r"],
		std_r=params["std_r"],
		mu=params["mu"],
		mean_mu=params["mean_mu"],
		std_mu=params["std_mu"],
		tau=params["tau"],
		learn_r=params["learn_r"],
		learn_mu=params["learn_mu"],
		learn_tau=params["learn_tau"],
		hh_init=params["hh_init"],
		device=device,
		name="Encoder",
		force_dale_law=params["force_dale_law"],
		activation=params["activation"],
		use_recurrent_connection=params["use_recurrent_connection"],
		forward_weights=params["forward_weights"],
		recurrent_weights=params["recurrent_weights"],
	).build()


def get_fr_layer(params, device):
	return FiringRateLayer(
		params["n_units"], params["n_aux_units"],
		std_weights=params["std_weights"],
		forward_sign=params["forward_sign"],
		dt=params["dt"],
		r=params["r"],
		mean_r=params["mean_r"],
		std_r=params["std_r"],
		mu=params["mu"],
		mean_mu=params["mean_mu"],
		std_mu=params["std_mu"],
		tau=params["tau"],
		learn_r=params["learn_r"],
		learn_mu=params["learn_mu"],
		learn_tau=params["learn_tau"],
		hh_init=params["hh_init"],
		device=device,
		name="Encoder",
		force_dale_law=params["force_dale_law"],
		activation=params["activation"],
		use_recurrent_connection=params["use_recurrent_connection"],
		forward_weights=params["forward_weights"],
		recurrent_weights=params["recurrent_weights"],
	).build()


def set_snn_lpf_default_param(params: Optional[dict] = None) -> dict:
	if params is None:
		params = {}
	params.setdefault("n_time_steps", -1)
	params.setdefault("dt", 0.02)
	params.setdefault("force_dale_law", False)
	params.setdefault("n_units", 200)
	params.setdefault("n_aux_units", params["n_units"])
	params.setdefault("use_recurrent_connection", False)
	params.setdefault("forward_sign", 0.5)
	params.setdefault("activation", "sigmoid")
	params.setdefault("hh_init", "inputs" if params["n_aux_units"] == params["n_units"] else "random")
	params = init_weights(params)
	return params


def get_snn_lpf_layer(params, device):
	return nt.SpyLIFLayerLPF(
		params["n_units"], params["n_aux_units"],
		dt=params["dt"],
		hh_init=params["hh_init"],
		force_dale_law=params["force_dale_law"],
		use_recurrent_connection=params["use_recurrent_connection"],
		forward_weights=params["forward_weights"],
		recurrent_weights=params["recurrent_weights"],
		name="Encoder",
		device=device,
	).build()


def set_lif_default_param(params: Optional[dict] = None) -> dict:
	if params is None:
		params = {}
	params.setdefault("n_time_steps", -1)
	params.setdefault("dt", 0.02)
	params.setdefault("force_dale_law", False)
	params.setdefault("n_units", 200)
	params.setdefault("n_aux_units", params["n_units"])
	params.setdefault("use_recurrent_connection", False)
	params.setdefault("forward_sign", 0.5)
	params.setdefault("activation", "sigmoid")
	params.setdefault("hh_init", "inputs" if params["n_aux_units"] == params["n_units"] else "random")
	params = init_weights(params)
	return params


def get_lif_layer(params, device):
	return nt.LIFLayer(
		params["n_units"], params["n_aux_units"],
		dt=params["dt"],
		hh_init=params["hh_init"],
		force_dale_law=params["force_dale_law"],
		use_recurrent_connection=params["use_recurrent_connection"],
		forward_weights=params["forward_weights"],
		recurrent_weights=params["recurrent_weights"],
		name="Encoder",
		device=device,
	).build()


def get_spy_lif_layer(params, device):
	return nt.SpyLIFLayer(
		params["n_units"], params["n_aux_units"],
		dt=params["dt"],
		hh_init=params["hh_init"],
		force_dale_law=params["force_dale_law"],
		use_recurrent_connection=params["use_recurrent_connection"],
		forward_weights=params["forward_weights"],
		recurrent_weights=params["recurrent_weights"],
		name="Encoder",
		device=device,
	).build()


def get_linear_layer(params, device):
	return nt.Linear(
		params["n_units"], params["n_aux_units"],
		dt=params["dt"],
		force_dale_law=params["force_dale_law"],
		# forward_weights=params["forward_weights"],
		forward_weights=nt.to_tensor(np.random.randn(params["n_units"], params["n_aux_units"]) / np.sqrt(params["n_units"])),
		use_bias=True,
		bias_weights=torch.tensor([0.01 for _ in range(params["n_aux_units"])]),
		activation=params["activation"],
		name="Encoder",
		device=device,
	).build()


def get_output_transform(params):
	output_transform = params.get("out_transform", params.get("output_transform", None))
	if isinstance(output_transform, str):
		output_transform = [str_to_activation(output_transform)]
	elif callable(output_transform):
		output_transform = [output_transform]
	elif output_transform is None:
		pass
	else:
		raise ValueError(f"Unknown output transform: {output_transform}")
	return output_transform


def get_model(
		model_type: ModelType,
		params: Optional[dict] = None,
		device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
		output_tr_data_folder: str = "data/tr_data",
		post_name: str = "",
) -> nt.SequentialRNN:
	if params is None:
		params = {}
	params["model_type"] = model_type.name
	if model_type == ModelType.WILSON_COWAN:
		params = set_wc_default_param(params)
		layer = get_wc_layer(params, device)
	elif model_type == ModelType.WilsonCowanCURBD:
		params = set_wc_default_param(params)
		layer = get_wc_curbd_layer(params, device)
	elif model_type == ModelType.SNN_LPF:
		params = set_snn_lpf_default_param(params)
		layer = get_snn_lpf_layer(params, device)
	elif model_type == ModelType.LIF:
		params = set_lif_default_param(params)
		layer = get_lif_layer(params, device)
	elif model_type == ModelType.SpyLIF:
		params = set_lif_default_param(params)
		layer = get_spy_lif_layer(params, device)
	elif model_type == ModelType.Linear:
		params = set_lif_default_param(params)
		layer = get_linear_layer(params, device)
	elif model_type == ModelType.FiringRate:
		params = set_wc_default_param(params)
		layer = get_fr_layer(params, device)
	elif model_type == ModelType.LinearJtoI:
		params = set_lif_default_param(params)
		layer = LinearJtoI(
			params["n_units"], params["n_aux_units"],
			dt=params["dt"],
			force_dale_law=params["force_dale_law"],
			# forward_weights=params["forward_weights"],
			forward_weights=nt.to_tensor(np.random.randn(params["n_units"], params["n_aux_units"]) / np.sqrt(params["n_units"])),
			use_bias=True,
			bias_weights=torch.tensor([0.01 for _ in range(params["n_aux_units"])]),
			activation=params["activation"],
			name="Encoder",
			device=device,
		).build()
	else:
		raise ValueError(f"Unknown model type: {model_type}")
	
	layers = [layer, ]
	if params.get("use_out_layer", True):
		out_layer = nt.Linear(
			params["n_aux_units"], params["n_units"],
			device=device,
			use_bias=False,
			activation=params.get("out_activation", params["activation"]),
			# name="Decoder",
		)
		layers.append(out_layer)
	extra_name = params.get("filename", "").replace(".", "_") + str(post_name)
	# hash_key = nt.utils.hash_params(params)
	# extra_name += f"_{hash_key}" if extra_name else hash_key

	model = SequentialRNN(
		layers=layers,
		output_transform=get_output_transform(params),
		device=device,
		foresight_time_steps=params["n_time_steps"] - 1,
		out_memory_size=params["n_time_steps"] - 1,
		hh_memory_size=params["n_time_steps"] - 1,
		checkpoint_folder=f"{output_tr_data_folder}/ckpt_{model_type.name.lower()}_{extra_name}",
		name=f"{model_type.value}",
	).build()
	
	os.makedirs(f"{model.checkpoint_folder}/infos", exist_ok=True)
	with open(f"{model.checkpoint_folder}/infos/model_repr.txt", "w+") as f:
		f.write(repr(model))
	initial_weights = {
		f"{layer.name}.{weights_attr}": maybe_to_numpy(
			return_default_on_except(getattr, None, layer, weights_attr, None)
		)
		for layer in model.get_all_layers()
		for weights_attr in ["forward_weights", "recurrent_weights"]
	}
	pickle.dump(initial_weights, open(f"{model.checkpoint_folder}/infos/initial_weights.pkl", "wb"))
	return model








