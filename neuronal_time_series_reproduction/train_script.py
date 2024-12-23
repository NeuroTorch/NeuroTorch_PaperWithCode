import json
import os
from copy import deepcopy
from typing import Optional, List

import matplotlib
import neurotorch as nt
import numpy as np
import torch
from neurotorch.callbacks.base_callback import BaseCallback
from neurotorch.callbacks.convergence import ConvergenceTimeGetter
from neurotorch.callbacks.lr_schedulers import LRSchedulerOnMetric
from neurotorch.regularization.connectome import ExcRatioTargetRegularization
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataloader, TimeSeriesDataset
from models import get_model
from neurotorch_fixes import TBPTT, RLS
from utils import get_convergence_infos, ModelType, TSTimeStepsTrainingScheduler, SyncDatasetLengthToBatchSizeCallback, \
	LS


class SMSEloss(torch.nn.Module):
	def __init__(self, reduction: str = "mean", **kwargs):
		super().__init__()
		assert reduction in ['mean', 'feature', 'none'], 'Reduction must be one of "mean", "feature", or "none".'
		self.reduction = reduction
		self.epsilon = kwargs.get("epsilon", 1e-5)

	def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		x, y = nt.to_tensor(inputs), nt.to_tensor(target)
		mse_loss = torch.nn.functional.mse_loss(x, y, reduction=self.reduction)
		return mse_loss / (y.var() + self.epsilon)


class EarlyStoppingOnStagnation(nt.callbacks.early_stopping.EarlyStoppingOnStagnation):
	def on_pbar_update(self, trainer, **kwargs) -> dict:
		return {"es_value": f"{self.get_value():.2e}"}


class ShowMetricOnPBarCallback(nt.callbacks.BaseCallback):
	UNPICKEABLE_ATTRIBUTES = ["trainer", "p_bar"]
	
	def __init__(self, metric: str, p_bar: tqdm, **kwargs):
		kwargs["save_state"] = False
		kwargs["load_state"] = False
		super().__init__(**kwargs)
		self.metric = metric
		self.p_bar = p_bar
	
	def on_pbar_update(self, trainer, **kwargs):
		postfix = {k: v for k, v in [s.split("=") for s in self.p_bar.postfix.split(", ")]}
		curr_value = trainer.current_training_state.itr_metrics[self.metric]
		postfix[self.metric] = f"{curr_value:.3f}"
		postfix["progress"] = f"{100 * (trainer.state.iteration + 1) / trainer.state.n_iterations:.2f}%"
		self.p_bar.set_postfix(postfix)


def set_trainer_default_params(params: Optional[dict] = None) -> dict:
	if params is None:
		params = {}
	params.setdefault("filename", None)
	params.setdefault("dataset_length", -1)
	params.setdefault("n_time_steps", -1)
	params.setdefault("shuffle", False)
	params.setdefault("target_skip_first", True)
	params.setdefault("dataset_randomize_indexes", False)
	params.setdefault("rm_dead_units", True)
	params.setdefault("normalize", True)
	params.setdefault("normalize_by_unit", True)
	params.setdefault("smoothing_sigma", 0.0)
	params.setdefault("batch_size", 1)
	params.setdefault("n_workers", 0)
	params.setdefault("n_iterations", 2_000)
	params.setdefault("force_overwrite", False)
	params.setdefault("early_stopping_threshold", 1.0)
	return params


def set_learning_algorithm_default_params(params: Optional[dict] = None) -> dict:
	if params is None:
		params = {}
	params.setdefault("weight_decay", 0.01)
	params.setdefault("lr_schedule_start", 0.4)
	params.setdefault("params_lr", 1e-5)
	params.setdefault("output_params_lr", 2e-5)
	params.setdefault("alpha", 1e-3)
	params.setdefault("gamma", 1e-3)
	params.setdefault("learning_algorithm", "eprop")
	return params


def make_learning_algorithm(model=None, **params) -> List[BaseCallback]:
	params = set_learning_algorithm_default_params(params)
	lr_scheduler = None
	if params["learning_algorithm"].lower() == "eprop":
		la = nt.Eprop(
			params_lr=params["params_lr"],
			output_params_lr=params["output_params_lr"],
			default_optim_kwargs={"weight_decay": params["weight_decay"], "lr": params["params_lr"]},
			grad_norm_clip_value=1.0,
			# learning_signal_norm_clip_value=1.0,
			# eligibility_traces_norm_clip_value=1.0,
			alpha=params["alpha"],
			gamma=params["gamma"],
		)
		lr_scheduler = LRSchedulerOnMetric(
			'val_p_var',
			metric_schedule=np.linspace(params["lr_schedule_start"], 1.0, 100),
			# min_lr=[1e-7, 2e-7],
			min_lr=[1e-6, 2e-6],
			retain_progress=True,
			priority=la.priority + 1,
		)
	elif params["learning_algorithm"].lower() == "bptt":
		la = nt.BPTT(
			params_lr=params["params_lr"],
			weight_decay=params["weight_decay"],
			criterion=SMSEloss(),
			default_optim_kwargs={"weight_decay": params["weight_decay"], "lr": params["params_lr"], "maximize": False},
		)
		lr_scheduler = LRSchedulerOnMetric(
			'val_p_var',
			metric_schedule=np.linspace(params["lr_schedule_start"], 1.0, 100),
			min_lr=[params.get("min_lr", 2e-5)],
			retain_progress=True,
			priority=la.priority + 1,
		)
	elif params["learning_algorithm"].lower() == "tbptt":
		la = TBPTT(
			params_lr=params["params_lr"],
			# learning_rate=params["output_params_lr"],
			weight_decay=params["weight_decay"],
			# criterion=torch.nn.MSELoss(),
			criterion=SMSEloss(),
			default_optim_kwargs={"weight_decay": params["weight_decay"], "lr": params["params_lr"], "maximize": False},
			backward_time_steps=params.get("backward_time_steps", 1),
			optim_time_steps=params.get("optim_time_steps", 1),
			grad_norm_clip_value=1.0,
			alpha=params["alpha"],
		)
		lr_scheduler = LRSchedulerOnMetric(
			'val_p_var',
			metric_schedule=np.linspace(params["lr_schedule_start"], 1.0, 100),
			min_lr=[params.get("min_lr", 2e-7)],
			retain_progress=True,
			priority=la.priority + 1,
		)
	elif params["learning_algorithm"].lower() == "rls":
		# TODO: fix RLS
		la = RLS(
			params=[p for p in model.get_layer().get_weights_parameters() if p.ndim == 2],
			params_lr=params["params_lr"],
			criterion=SMSEloss(),
			strategy=params.get("rls_strategy", "outputs"),
			is_recurrent=True,
			backward_time_steps=params.get("backward_time_steps", 1),
			# default_optimizer_cls=torch.optim.AdamW,
			default_optimizer_cls=params.get("optimizer_cls", torch.optim.AdamW),
			device=params.get("rls_device", getattr(model, "device", torch.device("cpu"))),
		)
		lr_scheduler = LRSchedulerOnMetric(
			'val_p_var',
			metric_schedule=np.linspace(params["lr_schedule_start"], 1.0, 100),
			min_lr=[params.get("min_lr", 2e-5)],
			retain_progress=True,
			priority=la.priority + 1,
		)
	elif params["learning_algorithm"].lower() == "ls":
		la = LS(
			learning_rate=params["params_lr"],
			weight_decay=params["weight_decay"],
			criterion=SMSEloss(),
			default_optim_kwargs={"weight_decay": params["weight_decay"], "lr": params["params_lr"], "maximize": False},
		)
	else:
		raise ValueError(f"Unknown learning algorithm: {params['learning_algorithm']}")
	timer = nt.callbacks.early_stopping.EarlyStoppingOnTimeLimit(delta_seconds=params.get("time_limit", np.inf))
	callbacks = [la, timer]
	if lr_scheduler is not None:
		callbacks.append(lr_scheduler)
	return callbacks


def train(
	model: nt.SequentialRNN,
	dataloader: Optional[DataLoader] = None,
	params: Optional[dict] = None,
	verbose: bool = True,
	*,
	val_dataloader: Optional[DataLoader] = None,
	**kwargs,
) -> nt.trainers.Trainer:
	os.makedirs(f"{model.checkpoint_folder}/infos", exist_ok=True)
	params = set_trainer_default_params(params)
	if dataloader is None:
		dataloader = get_dataloader(verbose=True, **params)
	if val_dataloader is None:
		val_dataloader = dataloader
	dataset: TimeSeriesDataset = dataloader.dataset
	model.foresight_time_steps = dataset.n_time_steps - 1
	model.out_memory_size = model.foresight_time_steps
	model.hh_memory_size = 1
	
	checkpoint_manager = nt.CheckpointManager(
		checkpoint_folder=model.checkpoint_folder,
		checkpoints_meta_path=model.checkpoints_meta_path,
		metric="val_p_var",
		minimise_metric=False,
		save_freq=max(1, int(params["n_iterations"] / 10)),
		start_save_at=1,
		save_best_only=False,
	)
	es_threshold = nt.callbacks.early_stopping.EarlyStoppingThreshold(
		metric="val_p_var", threshold=params["early_stopping_threshold"], minimize_metric=False
	)
	es_nan = nt.callbacks.early_stopping.EarlyStoppingOnNaN(metric="val_p_var")
	es_stagnation = EarlyStoppingOnStagnation(metric="val_p_var", patience=10, tol=1e-4)
	convergence_timer = ConvergenceTimeGetter(
		metric="val_p_var", threshold=params.get("convergence_threshold", params.get("early_stopping_threshold", 0.9)),
		minimize_metric=False,
	)
	callbacks = [
		checkpoint_manager, *make_learning_algorithm(model=model, **params),
		es_threshold, es_nan,
		# es_stagnation,
		convergence_timer,
	]
	if params.get("use_EI_regularisation", False):
		encoder_layer: nt.modules.layers.BaseNeuronsLayer = model.get_layer("Encoder")
		callbacks.append(
			ExcRatioTargetRegularization(
				params=encoder_layer.get_sign_parameters(), Lambda=1.0, exc_target_ratio=0.8,
				priority=BaseCallback.DEFAULT_HIGH_PRIORITY + 1,
			)
		)
	if kwargs.get("p_bar", None) is not None:
		callbacks.append(ShowMetricOnPBarCallback(metric=f"val_p_var", p_bar=kwargs["p_bar"]))
	if params.get("use_time_steps_scheduler", False):
		callbacks.append(
			TSTimeStepsTrainingScheduler(
				time_steps_list=list(np.linspace(
					int(0.01 * dataset.total_n_time_steps), dataset.total_n_time_steps,
					num=params.get("time_steps_scheduler_n_stages", int(0.01 * params["n_iterations"])),
					dtype=int,
				)),
				metric="val_p_var",
				minimize_metric=False,
				metric_target=params.get("time_steps_scheduler_target", 0.8),
				n_iterations_per_stage=params.get("time_steps_scheduler_n_iterations_per_stage", 1_000),
				priority=BaseCallback.DEFAULT_HIGH_PRIORITY + 1,
				p_bar=kwargs["p_bar"],
			)
		)
		callbacks.append(SyncDatasetLengthToBatchSizeCallback())
	trainer = nt.trainers.Trainer(
		model,
		predict_method="get_prediction_trace",
		callbacks=callbacks,
		metrics=[nt.metrics.RegressionMetrics(model, "p_var")],
	)
	for callback in callbacks:
		callback.start(trainer)
	trainer.verbose = verbose
	if verbose:
		print(f"{trainer}")
	with open(f"{model.checkpoint_folder}/infos/trainer_repr.txt", "w+") as f:
		f.write(repr(trainer))
	history = trainer.train(
		dataloader,
		val_dataloader,
		n_iterations=params["n_iterations"],
		exec_metrics_on_train=False,
		load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
		force_overwrite=params["force_overwrite"],
		verbose=verbose,
	)
	history.plot(save_path=f"{model.checkpoint_folder}/figures/tr_history.png", show=False)
	with open(f"{model.checkpoint_folder}/infos/trainer_repr.txt", "w+") as f:
		f.write(repr(trainer))
	return trainer


@torch.no_grad()
def test_model(
	model: nt.SequentialRNN,
	dataset: TimeSeriesDataset,
	n_test: int = 1,
	verbose: bool = True,
	device: Optional[torch.device] = None,
	load: bool = True,
	use_batch: bool = True,
	return_hidden_states: bool = False,
):
	if device is not None:
		model.to(device)
	model.eval()
	if load:
		try:
			model.load_checkpoint(load_checkpoint_mode=nt.LoadCheckpointMode.BEST_ITR, verbose=verbose)
		except:
			model.load_checkpoint(load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR, verbose=verbose)
	y_target = dataset.full_time_series
	model.foresight_time_steps = dataset.n_time_steps - 1
	model.out_memory_size = model.foresight_time_steps
	model.hh_memory_size = model.foresight_time_steps if return_hidden_states else 1
	if len(y_target.shape) == 2:
		y_target = torch.unsqueeze(y_target, dim=0)
	y_target_stack = torch.concat([y_target.to("cpu") for _ in range(n_test)], dim=0)

	if use_batch:
		yy, hh_stack = model.get_prediction_trace(
			torch.unsqueeze(y_target_stack[:, 0].clone(), dim=1), return_hidden_states=True
		)
		y_pred_stack = torch.concat([torch.unsqueeze(y_target_stack[:, 0].clone(), dim=1).to(model.device), yy], dim=1)
	else:
		y_pred_stack, hh_stack = [], []
		for _ in range(n_test):
			yy, hh = model.get_prediction_trace(
				torch.unsqueeze(y_target[:, 0].clone(), dim=1), return_hidden_states=True
			)
			y_pred = torch.concat([torch.unsqueeze(y_target[:, 0].clone(), dim=1).to(model.device), yy], dim=1)
			y_pred_stack.append(y_pred.to("cpu"))
			hh_stack.append(hh)
		y_pred_stack = torch.concat(y_pred_stack, dim=0)

	pvar = nt.losses.PVarianceLoss()(y_pred_stack.to(y_target.device), y_target_stack)
	pvar_mean, std = pvar_mean_std(y_pred_stack.to(y_target.device), y_target_stack)
	if verbose:
		print(f"pVar: {pvar}, E[pVar]: {pvar_mean} +/- {std}")
	if return_hidden_states:
		return y_pred_stack, y_target, pvar, pvar_mean, std, hh_stack
	return y_pred_stack, y_target, pvar, pvar_mean, std


@torch.no_grad()
def pvar_mean_std(x, y, epsilon: float = 1e-8, negative: bool = False):
	"""
	Calculate the mean and standard deviation of the P-Variance loss over the batch.

	:param x: The first input.
	:param y: The second input.
	:param epsilon: A small value to avoid division by zero.
	:param negative: If True, the negative of the P-Variance loss is returned.

	:return: The mean and standard deviation of the P-Variance loss over the batch.
	"""
	x, y = nt.to_tensor(x), nt.to_tensor(y)
	x_reshape, y_reshape = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
	mse_loss = torch.mean((x_reshape - y_reshape)**2, dim=-1)
	var = y_reshape.var(dim=-1)
	loss = 1 - (mse_loss / (var + epsilon))
	if negative:
		loss = -loss
	return loss.mean(), loss.std()


@torch.no_grad()
def get_exc_ratios(model: nt.SequentialRNN) -> List[float]:
	encoder_layer: nt.modules.layers.BaseNeuronsLayer = model.get_layer("Encoder")
	regul = ExcRatioTargetRegularization(params=encoder_layer.get_sign_parameters(), Lambda=1.0, exc_target_ratio=0.8)
	return regul.get_params_exc_ratio()


def run_model(
		*,
		model_type: ModelType,
		output_tr_data_folder,
		post_name: str = "",
		verbose: bool = True,
		p_bar: Optional[tqdm] = None,
		**params
):
	matplotlib.use('Agg')
	params = deepcopy(params)
	device = params.pop("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
	params.setdefault("seed", 0)
	nt.set_seed(params["seed"])

	
	params = set_trainer_default_params(params)
	dataloader = get_dataloader(verbose=verbose, **params)
	dataloader.dataset.set_params_from_self(params)
	val_params = deepcopy(params)
	val_params["dataset_length"] = params.get("batch_size", 1)
	val_dataloader = get_dataloader(verbose=verbose, **val_params)
	
	model = get_model(
		model_type, params=params, device=device, output_tr_data_folder=output_tr_data_folder,
		post_name=post_name,
	)
	os.makedirs(f"{model.checkpoint_folder}/infos", exist_ok=True)
	json.dump(params, open(f"{model.checkpoint_folder}/infos/params.json", "w+"), indent=4)
	trainer = train(model, dataloader, params, val_dataloader=val_dataloader, verbose=verbose, p_bar=p_bar)
	y_pred, y_target, pvar, pvar_mean, pvar_std = test_model(
		model, dataloader.dataset, n_test=params.get("n_test", 1), verbose=verbose,
	)
	json.dump(params, open(f"{model.checkpoint_folder}/infos/params.json", "w+"), indent=4)
	results = dict(
		pvar=float(nt.to_numpy(pvar)),
		pvar_mean=float(nt.to_numpy(pvar_mean)),
		pvar_std=float(nt.to_numpy(pvar_std)),
		n_units=float(nt.to_numpy(params["n_units"])),
		n_time_steps=float(nt.to_numpy(params["n_time_steps"])),
		n_iterations=float(nt.to_numpy(params["n_iterations"])),
		iteration=float(nt.to_numpy(trainer.state.iteration)),
		batch_size=float(nt.to_numpy(params["batch_size"])),
		exc_ratio=get_exc_ratios(model),
		model_name=model.name,
		model_type=model_type.name,
	)
	results.update(get_convergence_infos(trainer))
	json.dump(results, open(f"{model.checkpoint_folder}/infos/results.json", "w+"), indent=4)
	return results

