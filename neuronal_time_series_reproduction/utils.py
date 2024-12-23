import enum
import json
import os
import shutil
from copy import deepcopy
from typing import List, Dict, Union, Optional, Any, Tuple

import neurotorch as nt
import numpy as np
import psutil
import pythonbasictools as pbt
import torch
from neurotorch import to_tensor, to_numpy, Linear, WilsonCowanLayer, WilsonCowanCURBDLayer
from neurotorch.callbacks import BaseCallback
from neurotorch.callbacks.convergence import ConvergenceTimeGetter
from neurotorch.dimension import SizeTypes
from neurotorch.modules.layers import BaseNeuronsLayer
from neurotorch.utils import format_pred_batch
from torch import nn

from dataset import get_dataloader
from neurotorch_fixes import BPTT


class ModelType(enum.Enum):
    __order__ = "WILSON_COWAN SNN_LPF LIF SpyLIF Linear FiringRate LinearJtoI WilsonCowanCURBD"
    WILSON_COWAN = "Wilson-Cowan"
    SNN_LPF = "SNN-LPF"
    LIF = "LIF"
    SpyLIF = "SpyLIF"
    Linear = "Linear"
    FiringRate = "FiringRate"
    LinearJtoI = "LinearJtoI"
    WilsonCowanCURBD = "Wilson-Cowan-CURBD"

    @staticmethod
    def from_name(name: str):
        return ModelType[name.upper().replace("-", "_")]

    @staticmethod
    def from_value(value: str):
        return ModelType[value.upper().replace("-", "_")]

    @staticmethod
    def from_repr(value: str):
        return ModelType[value.upper().replace("-", "_")]


def get_convergence_infos(trainer: nt.Trainer):
    infos = {}
    convergence_timer = [ct for ct in trainer.callbacks if isinstance(ct, ConvergenceTimeGetter)][0]
    infos["itr_convergence"] = convergence_timer.itr_convergence
    infos["time_convergence"] = convergence_timer.time_convergence
    infos["training_time"] = convergence_timer.training_time
    return infos


def try_main_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}")
            return None

    wrapper.__name__ = func.__name__
    return wrapper


def gather_model_from_path(
        model_path: str,
        idx: int = None,
        verbose: bool = False,
        raise_exception: bool = False,
):
    from models import get_model

    try:
        params_filepath = os.path.join(model_path, "infos", "params.json")
        params = json.load(open(params_filepath, "r"))
        results_filepath = os.path.join(model_path, "infos", "results.json")
        results = json.load(open(results_filepath, "r")) if os.path.exists(results_filepath) else {}
        model_type = ModelType.from_name(results.get("model_name", params.get("model_type")))
        post_name = os.path.basename(model_path).split("npy")[-1]
        dataloader = get_dataloader(verbose=False, **params)
        model = get_model(
            model_type, params=params,
            output_tr_data_folder=os.path.dirname(model_path),
            post_name=post_name,
            device=torch.device("cpu")
        )
        initial_model = deepcopy(model)
    except Exception as e:
        if raise_exception:
            raise e
        return e
    try:
        model.load_checkpoint(load_checkpoint_mode=nt.LoadCheckpointMode.BEST_ITR, verbose=verbose)
    except:
        model.load_checkpoint(load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR, verbose=verbose)
    dale_name = " + Dale" if params["force_dale_law"] else ""
    m_name = f"{model.name}{dale_name}"
    return {
        "model": model,
        "initial_model": initial_model,
        "params": params,
        "results": results,
        "dataloader": dataloader,
        "model_type": model_type,
        "name": m_name,
        "rootpath": model_path,
        "id": idx,
    }


def gather_models_from_folders(
        model_paths: List[str],
        **kwargs
):
    model_objects = []
    errors = []
    nb_workers = kwargs.get("nb_workers", -2)
    raise_exception = kwargs.get("raise_exception", False)
    verbose = kwargs.get("verbose", True)
    if nb_workers == 1 or nb_workers == 0:
        nb_workers = 0
    if verbose:
        print(f"{psutil.cpu_count(False)} workers available -> using {nb_workers} workers.")
    objects_list = pbt.multiprocessing_tools.apply_func_multiprocess(
        gather_model_from_path,
        iterable_of_args=[(model_path, idx) for idx, model_path in enumerate(model_paths)],
        iterable_of_kwargs=[{"verbose": False, "raise_exception": raise_exception} for _ in model_paths],
        desc="Gathering models",
        unit="model",
        verbose=verbose,
        nb_workers=nb_workers
    )
    for i, obj in enumerate(objects_list):
        if isinstance(obj, Exception):
            errors.append((model_paths[i], obj))
        else:
            model_objects.append(obj)
    if len(errors) > 0 and verbose:
        print(f"Exceptions during loading: {errors}")
    return model_objects


def get_cmd_kwargs(defaults: Dict = None):
    import sys
    import argparse

    if defaults is None:
        defaults = {}
    cmd_kwargs = {i: v for i, v in enumerate(sys.argv)}
    parser = argparse.ArgumentParser()
    for pos_arg in sorted([k for k in defaults.keys() if isinstance(k, int)]):
        if pos_arg in cmd_kwargs:
            parser.add_argument(f"argv_{pos_arg}", type=str, default=defaults[pos_arg])
    for k, v in defaults.items():
        if isinstance(k, int):
            cmd_kwargs[k] = cmd_kwargs.get(k, v)
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    cmd_kwargs.update(vars(args))
    return cmd_kwargs


def sparsify_matrix_smallest(M: np.ndarray, sparsity: float = 0.0, **kwargs) -> np.ndarray:
    r"""
    Update the matrix M by setting a fraction of its elements to zero in a way that the sparsity :eq:`sparsity` of
    the matrix is approximately equal to the specified value. The elements are chosen by removing the smallest
    elements in absolute value.

    .. math::
        :label: sparsity

        \text{sparsity} = \frac{\sum_{i,j} \delta(M_{i,j} = 0)}{\sum_{i,j} 1}

    :param M: the matrix to be sparsified.
    :type M: np.ndarray
    :param sparsity: the desired sparsity of the matrix. Defaults to 0.0.
    :type sparsity: float, optional

    :return: the sparsified matrix.
    """
    n_zeros = int(sparsity * np.prod(M.shape))
    indices = np.argsort(np.abs(M), axis=None)
    smallest_values = indices[:n_zeros]
    mask = np.ones_like(M, dtype=bool)
    mask.ravel()[smallest_values] = False
    return M * mask.astype(M.dtype)


def sparsify_matrix_smallest_rm(M: np.ndarray, sparsity: float = 0.0, **kwargs) -> np.ndarray:
    r"""
    Update the matrix M by setting a fraction of its elements to zero in a way that the added sparsity :eq:`sparsity` of
    the matrix is approximately equal to the specified value. The elements are chosen by removing the smallest
    elements in absolute value.

    .. math::
        :label: sparsity

        \text{sparsity} = \frac{\sum_{i,j} \delta(M_{i,j} = 0)}{\sum_{i,j} 1}

    :param M: the matrix to be sparsified.
    :type M: np.ndarray
    :param sparsity: the desired sparsity of the matrix. Defaults to 0.0.
    :type sparsity: float, optional

    :return: the sparsified matrix.
    """
    init_n_zeros = int(np.sum(np.isclose(M, 0.0)))
    n_zeros = int(sparsity * (np.prod(M.shape) - init_n_zeros))
    indices = np.argsort(np.abs(M), axis=None)
    smallest_values = indices[:n_zeros]
    mask = np.ones_like(M, dtype=bool)
    mask.ravel()[smallest_values] = False
    return M * mask.astype(M.dtype)


def sparsify_matrix_random(M: np.ndarray, sparsity: float = 0.0, **kwargs) -> np.ndarray:
    r"""
    Update the matrix M by setting a fraction of its elements to zero in a way that the sparsity :eq:`sparsity` of
    the matrix is approximately equal to the specified value. The elements are chosen by removing random
    elements.

    .. math::
        :label: sparsity

        \text{sparsity} = \frac{\sum_{i,j} \delta(M_{i,j} = 0)}{\sum_{i,j} 1}

    :param M: the matrix to be sparsified.
    :type M: np.ndarray
    :param sparsity: the desired sparsity of the matrix. Defaults to 0.0.
    :type sparsity: float, optional

    :return: the sparsified matrix.
    """
    seed = kwargs.get("seed", None)
    rn_gen = np.random.RandomState(seed)
    n_zeros = int(sparsity * np.prod(M.shape))
    mask = rn_gen.choice(
        [0, 1], size=np.prod(M.shape), p=[n_zeros / np.prod(M.shape), 1 - n_zeros / np.prod(M.shape)]
    ).astype(bool).reshape(M.shape)
    return M * mask.astype(M.dtype)


def sparsify_matrix_random_rm(M: np.ndarray, sparsity: float = 0.0, **kwargs) -> np.ndarray:
    r"""
    Update the matrix M by setting a fraction of its elements to zero in a way that the sparsity :eq:`sparsity` of
    the matrix is approximately equal to the specified value. The elements are chosen by removing random
    elements.

    .. math::
        :label: sparsity

        \text{sparsity} = \frac{\sum_{i,j} \delta(M_{i,j} = 0)}{\sum_{i,j} 1}

    :param M: the matrix to be sparsified.
    :type M: np.ndarray
    :param sparsity: the desired sparsity of the matrix. Defaults to 0.0.
    :type sparsity: float, optional

    :return: the sparsified matrix.
    """
    seed = kwargs.get("seed", None)
    rn_gen = np.random.RandomState(seed)
    init_n_zeros = int(np.sum(np.isclose(M, 0.0)))
    n_zeros = int(sparsity * (np.prod(M.shape) - init_n_zeros))
    mask = rn_gen.choice(
        [0, 1], size=np.prod(M.shape), p=[n_zeros / np.prod(M.shape), 1 - n_zeros / np.prod(M.shape)]
    ).astype(bool).reshape(M.shape)
    return M * mask.astype(M.dtype)


def sparsify_matrix(M: np.ndarray, sparsity: float = 0.0, **kwargs) -> np.ndarray:
    r"""
    Update the matrix M by setting a fraction of its elements to zero in a way that the sparsity :eq:`sparsity` of
    the matrix is approximately equal to the specified value. The elements are chosen by removing the smallest
    elements in absolute value.

    .. math::
        :label: sparsity

        \text{sparsity} = \frac{\sum_{i,j} \delta(M_{i,j} = 0)}{\sum_{i,j} 1}

    :param M: the matrix to be sparsified.
    :type M: np.ndarray
    :param sparsity: the desired sparsity of the matrix. Defaults to 0.0.
    :type sparsity: float, optional

    :return: the sparsified matrix.
    """

    strategy = kwargs.get("strategy", "smallest").lower()
    if strategy == "smallest":
        return sparsify_matrix_smallest(M, sparsity=sparsity, **kwargs)
    elif strategy == "random":
        return sparsify_matrix_random(M, sparsity=sparsity, **kwargs)
    elif strategy == "smallest_rm":
        return sparsify_matrix_smallest_rm(M, sparsity=sparsity, **kwargs)
    elif strategy == "random_rm":
        return sparsify_matrix_random_rm(M, sparsity=sparsity, **kwargs)
    else:
        raise ValueError(f"Unknown strategy {strategy}.")


def load_model(
        output_tr_data_folder: str = "./data/tr_init_converg",
        post_name: str = "_eprop_dale_0",
        model_type: ModelType = ModelType.SNN_LPF,
):
    from models import get_model

    folderpath = [
        f for f in os.listdir(output_tr_data_folder)
        if model_type.name.lower() in f.lower() and f.endswith(post_name)
    ][0]
    params_filepath = os.path.join(output_tr_data_folder, folderpath, "infos", "params.json")
    params = json.load(open(params_filepath, "r"))

    model = get_model(
        model_type, params=params, output_tr_data_folder=output_tr_data_folder, post_name=post_name,
        device=torch.device("cpu")
    )
    try:
        model.load_checkpoint(load_checkpoint_mode=nt.LoadCheckpointMode.BEST_ITR, verbose=True)
    except:
        model.load_checkpoint(load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR, verbose=True)
    return model, params


def _save_figures(fig, m_objects, savefig_filenames):
    fig_dir = os.path.join(m_objects["model"].checkpoint_folder, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for filename in savefig_filenames:
        fig.savefig(os.path.join(fig_dir, filename), bbox_inches='tight', pad_inches=0.1, dpi=900)
    return fig_dir


def _copy_figures(m_objects, from_fig_dir, savefig_filenames):
    fig_dir = os.path.join(m_objects["model"].checkpoint_folder, "figures")
    for filename in savefig_filenames:
        from_fig_filepath = os.path.join(from_fig_dir, filename)
        to_fig_filepath = os.path.join(fig_dir, filename)
        if from_fig_filepath != to_fig_filepath:
            shutil.copyfile(from_fig_filepath, to_fig_filepath)
    return fig_dir


class ForesightTimeStepUpdaterOnTarget(BaseCallback):
    DEFAULT_PRIORITY = BaseCallback.DEFAULT_HIGH_PRIORITY

    def __init__(self, **kwargs):
        kwargs["save_state"] = False
        super().__init__(**kwargs)
        self.hh_memory_size = None
        self.out_memory_size = None
        self.foresight_time_steps = None
        self.n_encoder_steps = kwargs.get("n_encoder_steps", 1)
        self.target_skip_first = kwargs.get("target_skip_first", True)
        self.update_val_loss_freq = kwargs.get("update_val_loss_freq", 1_000)
        self.start_intensive_val_at = kwargs.get("start_intensive_val_at", 0.98)
        self.val_dataloader = None
        self.kwargs = kwargs

    def start(self, trainer, **kwargs):
        self.val_dataloader = trainer.state.objects["val_dataloader"]

    def on_batch_begin(self, trainer, **kwargs):
        self.hh_memory_size = trainer.model.hh_memory_size
        self.out_memory_size = trainer.model.out_memory_size
        self.foresight_time_steps = trainer.model.foresight_time_steps
        if not trainer.model.training:
            n_times_steps = trainer.state.y_batch.shape[-2]
            trainer.model.foresight_time_step = trainer.state.y_batch.shape[-2]
            trainer.model.hh_memory_size = np.inf
            trainer.model.out_memory_size = n_times_steps * self.n_encoder_steps
            trainer.model.foresight_time_steps = trainer.model.out_memory_size

    def on_validation_end(self, trainer, **kwargs):
        trainer.model.hh_memory_size = self.hh_memory_size
        trainer.model.out_memory_size = self.out_memory_size
        trainer.model.foresight_time_steps = self.foresight_time_steps

    def on_train_end(self, trainer, **kwargs):
        if isinstance(self.start_intensive_val_at, float):
            at = int(self.start_intensive_val_at * trainer.state.n_iterations)
        else:
            at = self.start_intensive_val_at
        if (trainer.state.iteration >= at) or ((trainer.state.iteration + 1) % self.update_val_loss_freq == 0):
            trainer.state.objects["val_dataloader"] = self.val_dataloader
        else:
            trainer.state.objects["val_dataloader"] = None


class TSTimeStepsTrainingScheduler(BaseCallback):
    DEFAULT_PRIORITY = BaseCallback.DEFAULT_HIGH_PRIORITY

    def __init__(
            self,
            time_steps_list: List[int],
            metric: str,
            metric_target: float,
            minimize_metric: bool,
            *,
            n_iterations_per_stage: Optional[int] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.time_steps_list = time_steps_list
        self.metric = metric
        self.metric_target = metric_target
        self.minimize_metric = minimize_metric
        self.current_stage = 0
        self.stages_start_iteration: Dict[int, int] = {0: 0}
        self.p_bar = kwargs.pop("p_bar", None)
        self.UNPICKEABLE_ATTRIBUTES.append("p_bar")
        self.UNPICKEABLE_ATTRIBUTES.append("kwargs")
        self._initial_out_memory_size = None
        self._initial_hh_memory_size = None
        self._initial_foresight_time_steps = None
        self._initial_train_time_steps = None
        self._initial_val_time_steps = None
        self._completed = False
        self._completed_at = None
        self.trigger_save_on_stage_end = kwargs.get("trigger_save_on_stage_end", True)
        self.enable_checkpoint_managers_after_completed = kwargs.get("enable_checkpoint_managers_after_completed", True)
        self.n_iterations_per_stage = n_iterations_per_stage

    def start(self, trainer, **kwargs):
        if "train_dataloader" in trainer.state.objects:
            self._initial_train_time_steps = trainer.state.objects["train_dataloader"].dataset.n_time_steps
        if "val_dataloader" in trainer.state.objects:
            self._initial_val_time_steps = trainer.state.objects["val_dataloader"].dataset.n_time_steps
        self._initial_out_memory_size = trainer.model.out_memory_size
        self._initial_hh_memory_size = trainer.model.hh_memory_size
        self._initial_foresight_time_steps = trainer.model.foresight_time_steps

        if self.enable_checkpoint_managers_after_completed:
            self.disable_checkpoint_managers(trainer, **kwargs)

        self.update_n_iterations(trainer, **kwargs)

    def close(self, trainer, **kwargs):
        self.reset_time_steps(trainer, **kwargs)

    def check_if_stage_completed(self, trainer, **kwargs):
        if self.minimize_metric:
            return trainer.state.itr_metrics[self.metric] < self.metric_target
        else:
            return trainer.state.itr_metrics[self.metric] >= self.metric_target

    def update_time_steps(self, trainer, **kwargs):
        trainer.model.hh_memory_size = 1
        trainer.model.out_memory_size = self.time_steps_list[self.current_stage] - 1
        trainer.model.foresight_time_steps = trainer.model.out_memory_size
        if "train_dataloader" in trainer.state.objects:
            trainer.state.objects["train_dataloader"].dataset.n_time_steps = self.time_steps_list[self.current_stage]
        if "val_dataloader" in trainer.state.objects:
            trainer.state.objects["val_dataloader"].dataset.n_time_steps = self.time_steps_list[self.current_stage]

    def reset_time_steps(self, trainer, **kwargs):
        trainer.model.out_memory_size = self._initial_out_memory_size
        trainer.model.hh_memory_size = self._initial_hh_memory_size
        trainer.model.foresight_time_steps = self._initial_foresight_time_steps
        trainer.state.objects["train_dataloader"].dataset.n_time_steps = self._initial_train_time_steps
        trainer.state.objects["val_dataloader"].dataset.n_time_steps = self._initial_val_time_steps

    def on_iteration_begin(self, trainer, **kwargs):
        self.update_time_steps(trainer, **kwargs)

    def on_iteration_end(self, trainer, **kwargs):
        if self._completed:
            return
        if self.check_if_stage_completed(trainer, **kwargs):
            is_last_stage = self.current_stage == len(self.time_steps_list) - 1
            if is_last_stage:
                self._completed = True
                self._completed_at = trainer.state.iteration
                self.reset_time_steps(trainer, **kwargs)
                if self.enable_checkpoint_managers_after_completed:
                    self.enable_checkpoint_managers(trainer, **kwargs)
            else:
                self.current_stage += 1
                self.stages_start_iteration[self.current_stage] = trainer.state.iteration
                self.update_time_steps(trainer, **kwargs)
            if self.trigger_save_on_stage_end:
                self.save_on(trainer, **kwargs)

        self.update_n_iterations(trainer, **kwargs)

    def save_on(self, trainer, **kwargs):
        for cm in trainer.checkpoint_managers:
            cm.save_checkpoint(
                trainer.state.iteration,
                trainer.state.itr_metrics,
                best=False,
                state_dict=trainer.model.state_dict(),
                optimizer_state_dict=trainer.optimizer.state_dict() if trainer.optimizer else None,
                training_history=trainer.training_history.get_checkpoint_state(trainer),
                **trainer.callbacks.get_checkpoint_state(trainer)
            )
            if trainer.training_history:
                trainer.training_history.plot(
                    save_path=os.path.join(cm.checkpoint_folder, "training_history.png"),
                    show=False
                )

    def enable_checkpoint_managers(self, trainer, **kwargs):
        for cm in trainer.checkpoint_managers:
            cm.start_save_at = trainer.state.iteration

    def disable_checkpoint_managers(self, trainer, **kwargs):
        for cm in trainer.checkpoint_managers:
            cm.start_save_at = np.inf

    def update_n_iterations(self, trainer: nt.Trainer, **kwargs):
        if self.n_iterations_per_stage is not None:
            if self._completed:
                trainer.update_state_(n_iterations=self._completed_at + self.n_iterations_per_stage)
            else:
                trainer.update_state_(
                    n_iterations=self.stages_start_iteration[self.current_stage] + self.n_iterations_per_stage
                )

    def on_pbar_update(self, trainer, **kwargs) -> dict:
        to_show = {
            "stage": f"{self.current_stage + 1}/{len(self.time_steps_list)}",
            "n_time_steps": f"{trainer.model.foresight_time_steps + 1}/{self.time_steps_list[-1]}"
        }
        if self.p_bar is not None:
            postfix = {k: v for k, v in [s.split("=") for s in self.p_bar.postfix.split(", ")]}
            postfix.update(to_show)
            self.p_bar.set_postfix(postfix)
        return to_show

    def extra_repr(self) -> str:
        repr_str = f"metric={self.metric}"
        repr_str += f", target={self.metric_target}"
        repr_str += f", minimize={self.minimize_metric}"
        repr_str += f", stage={self.current_stage + 1}/{len(self.time_steps_list)}"
        repr_str += f", n_time_steps={self.time_steps_list[self.current_stage]}"
        repr_str += f", completed={self._completed}"
        return repr_str


class SyncDatasetLengthToBatchSizeCallback(BaseCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.UNPICKEABLE_ATTRIBUTES.append("kwargs")

    def _update_dataset_length(self, trainer, **kwargs):
        for dataloader_name in ["train_dataloader", "val_dataloader"]:
            if dataloader_name in trainer.state.objects:
                dataloader = trainer.state.objects[dataloader_name]
                dataset = dataloader.dataset
                dataset.dataset_length = min(
                    max(1, dataset.total_n_time_steps - dataset.n_time_steps), dataloader.batch_size
                )

    def on_iteration_begin(self, trainer, **kwargs):
        self._update_dataset_length(trainer, **kwargs)

    def on_iteration_end(self, trainer, **kwargs):
        self._update_dataset_length(trainer, **kwargs)


class DecreaseSmoothingSigmaCallback(BaseCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.UNPICKEABLE_ATTRIBUTES.append("kwargs")
        self.enable = kwargs.get("enable", True)

    def _update_smoothing_sigma(self, trainer, **kwargs):
        for dataloader_name in ["train_dataloader", "val_dataloader"]:
            if dataloader_name in trainer.state.objects:
                dataloader = trainer.state.objects[dataloader_name]
                dataset = dataloader.dataset
                dataset.smoothing_sigma = max(0.0, dataset.sigma - 0.01)

    def on_iteration_begin(self, trainer, **kwargs):
        self._update_smoothing_sigma(trainer, **kwargs)

    def on_iteration_end(self, trainer, **kwargs):
        self._update_smoothing_sigma(trainer, **kwargs)


def return_default_on_except(func, default, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except:
        return default


def maybe_to_numpy(x, *args, **kwargs):
    if x is None:
        return None
    return nt.to_numpy(x, *args, **kwargs)


class SigmoidTransform(torch.nn.Module):

    def forward(self, x: Any, **kwargs):
        x = nt.to_tensor(x)
        if isinstance(x, torch.Tensor):
            x_view = x.view(-1, x.shape[-1])
            return torch.sigmoid(x_view).view(x.shape)
        elif isinstance(x, dict):
            return {k: torch.sigmoid(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [torch.sigmoid(v) for v in x]
        elif isinstance(x, tuple):
            return tuple(torch.sigmoid(v) for v in x)
        else:
            raise ValueError(f"Unknown type {type(x)}.")


def json_serialize_dict(__dict: dict):
    serialized_dict = {}
    for k, v in __dict.items():
        if isinstance(v, (np.ndarray, np.generic)):
            serialized_dict[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            serialized_dict[k] = v.cpu().numpy().tolist()
        elif isinstance(v, (list, tuple)):
            serialized_dict[k] = [json_serialize_dict(v_) if isinstance(v_, dict) else v_ for v_ in v]
        elif isinstance(v, dict):
            serialized_dict[k] = json_serialize_dict(v)
        else:
            try:
                json.dumps(v)
                serialized_dict[k] = v
            except:
                serialized_dict[k] = str(v)
    return serialized_dict


def str_to_activation(activation: Union[torch.nn.Module, str] = "identity"):
    """
    Return the activation function.

    :param activation: Activation function.
    :type activation: Union[torch.nn.Module, str]
    """
    str_to_activation = {
        "identity": torch.nn.Identity(),
        "relu": torch.nn.ReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "softmax": torch.nn.Softmax(dim=-1),
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "prelu": torch.nn.PReLU(),
        "leakyrelu": torch.nn.LeakyReLU(),
        "leaky_relu": torch.nn.LeakyReLU(),
        "logsigmoid": torch.nn.LogSigmoid(),
        "log_sigmoid": torch.nn.LogSigmoid(),
        "logsoftmax": torch.nn.LogSoftmax(dim=-1),
        "log_softmax": torch.nn.LogSoftmax(dim=-1),
    }
    if isinstance(activation, str):
        if activation.lower() not in str_to_activation.keys():
            raise ValueError(
                f"Activation {activation} is not implemented. Please use one of the following: "
                f"{str_to_activation.keys()} or provide a torch.nn.Module."
            )
        activation = str_to_activation[activation.lower()]
    return activation


class FiringRateLayer(BaseNeuronsLayer):
    """
    This layer is use for Wilson-Cowan neuronal dynamics.
    This dynamic is also referred to as firing rate model.
    Wilson-Cowan dynamic is great for neuronal calcium activity.
    This layer use recurrent neural network (RNN).
    The number of parameters that are trained is N^2 (+2N if mu and r is train)
    where N is the number of neurons.

    For references, please read:

        - Excitatory and Inhibitory Interactions in Localized Populations of Model Neurons :cite:t:`wilson1972excitatory`
        - Beyond Wilson-Cowan dynamics: oscillations and chaos without inhibitions :cite:t:`PainchaudDoyonDesrosiers2022`
        - Neural Network dynamic :cite:t:`VogelsTimRajanAbbott2005NeuralNetworkDynamics`.

    The Wilson-Cowan dynamic is one of many dynamical models that can be used
    to model neuronal activity. To explore more continuous and Non-linear dynamics,
    please read Nonlinear Neural Network: Principles, Mechanisms, and Architecture :cite:t:`GROSSBERG198817`.


    """

    def __init__(
            self,
            input_size: Optional[SizeTypes] = None,
            output_size: Optional[SizeTypes] = None,
            dt: float = 1e-3,
            use_recurrent_connection: bool = False,
            device=None,
            **kwargs
    ):
        """
        :param input_size: size of the input
        :type input_size: Optional[SizeTypes]
        :param output_size: size of the output
            If we are predicting time series -> input_size = output_size
        :type output_size: Optional[SizeTypes]
        :param learning_type: Type of learning for the gradient descent
        :type learning_type: LearningType
        :param dt: Time step (Euler's discretisation)
        :type dt: float
        :param device: device for computation
        :type device: torch.device
        :param kwargs: Additional parameters for the Wilson-Cowan dynamic.

        :keyword Union[torch.Tensor, np.ndarray] forward_weights: Forward weights of the layer.
        :keyword float std_weight: Instability of the initial random matrix.
        :keyword Union[float, torch.Tensor] mu: Activation threshold. If torch.Tensor -> shape (1, number of neurons).
        :keyword float mean_mu: Mean of the activation threshold (if learn_mu is True).
        :keyword float std_mu: Standard deviation of the activation threshold (if learn_mu is True).
        :keyword bool learn_mu: Whether to train the activation threshold.
        :keyword float tau: Decay constant of RNN unit.
        :keyword bool learn_tau: Whether to train the decay constant.
        :keyword float r: Transition rate of the RNN unit. If torch.Tensor -> shape (1, number of neurons).
        :keyword float mean_r: Mean of the transition rate (if learn_r is True).
        :keyword float std_r: Standard deviation of the transition rate (if learn_r is True).
        :keyword bool learn_r: Whether to train the transition rate.

        Remarks: Parameter mu and r can only be a parameter as a vector.
        """
        super(FiringRateLayer, self).__init__(
            input_size=input_size,
            output_size=output_size,
            use_recurrent_connection=use_recurrent_connection,
            dt=dt,
            device=device,
            **kwargs
        )
        self.std_weight = self.kwargs["std_weight"]
        self.mu = torch.nn.Parameter(to_tensor(self.kwargs["mu"]).to(self.device), requires_grad=False)
        self.mean_mu = self.kwargs["mean_mu"]
        self.std_mu = self.kwargs["std_mu"]
        self.learn_mu = self.kwargs["learn_mu"]
        self.tau_sqrt = torch.nn.Parameter(
            torch.sqrt(to_tensor(self.kwargs["tau"])).to(self.device), requires_grad=False
        )
        self.learn_tau = self.kwargs["learn_tau"]
        self.r_sqrt = torch.nn.Parameter(
            torch.sqrt(to_tensor(self.kwargs["r"], dtype=torch.float32)).to(self.device), requires_grad=False
        )
        self.mean_r = self.kwargs["mean_r"]
        self.std_r = self.kwargs["std_r"]
        self.learn_r = self.kwargs["learn_r"]
        self.activation = self._init_activation(self.kwargs["activation"])

    def _set_default_kwargs(self):
        self.kwargs.setdefault("std_weight", 1.0)
        self.kwargs.setdefault("mu", 0.0)
        self.kwargs.setdefault("tau", 1.0)
        self.kwargs.setdefault("learn_tau", False)
        self.kwargs.setdefault("learn_mu", False)
        self.kwargs.setdefault("mean_mu", 2.0)
        self.kwargs.setdefault("std_mu", 0.0)
        self.kwargs.setdefault("r", 0.0)
        self.kwargs.setdefault("learn_r", False)
        self.kwargs.setdefault("mean_r", 2.0)
        self.kwargs.setdefault("std_r", 0.0)
        self.kwargs.setdefault("hh_init", "inputs")
        self.kwargs.setdefault("activation", "sigmoid")

    def _assert_kwargs(self):
        assert self.std_weight >= 0.0, "std_weight must be greater or equal to 0.0"
        assert self.std_mu >= 0.0, "std_mu must be greater or equal to 0.0"
        assert self.tau > self.dt, "tau must be greater than dt"

    @property
    def r(self):
        """
        This property is used to ensure that the transition rate will never be negative if trained.
        """
        return torch.pow(self.r_sqrt, 2)

    @r.setter
    def r(self, value):
        self.r_sqrt.data = torch.sqrt(torch.abs(to_tensor(value, dtype=torch.float32))).to(self.device)

    @property
    def tau(self):
        """
        This property is used to ensure that the decay constant will never be negative if trained.
        """
        return torch.pow(self.tau_sqrt, 2)

    @tau.setter
    def tau(self, value):
        self.tau_sqrt.data = torch.sqrt(torch.abs(to_tensor(value, dtype=torch.float32))).to(self.device)

    def initialize_weights_(self):
        """
        Initialize the parameters (weights) that will be trained.
        """
        super().initialize_weights_()
        if self.kwargs.get("forward_weights", None) is not None:
            self._forward_weights.data = to_tensor(self.kwargs["forward_weights"]).to(self.device)
        else:
            torch.nn.init.normal_(self._forward_weights, mean=0.0, std=self.std_weight)

        # If mu is not a parameter, it takes the value 0.0 unless stated otherwise by user
        # If mu is a parameter, it is initialized as a vector with the correct mean and std
        # unless stated otherwise by user.
        if self.learn_mu:
            if self.mu.dim() == 0:  # if mu is a scalar and a parameter -> convert it to a vector
                self.mu.data = torch.empty((1, int(self.output_size)), dtype=torch.float32, device=self.device)
            self.mu = torch.nn.Parameter(self.mu, requires_grad=self.requires_grad)
            torch.nn.init.normal_(self.mu, mean=self.mean_mu, std=self.std_mu)
        if self.learn_r:
            _r = torch.empty((1, int(self.output_size)), dtype=torch.float32, device=self.device)
            torch.nn.init.normal_(_r, mean=self.mean_r, std=self.std_r)
            self.r_sqrt = torch.nn.Parameter(torch.sqrt(torch.abs(_r)), requires_grad=self.requires_grad)
        if self.learn_tau:
            self.tau_sqrt = torch.nn.Parameter(self.tau_sqrt, requires_grad=self.requires_grad)

    def create_empty_state(self, batch_size: int = 1, **kwargs) -> Tuple[torch.Tensor]:
        if self.kwargs["hh_init"] == "zeros":
            state = [torch.zeros(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            ) for _ in range(1)]
        elif self.kwargs["hh_init"] == "random":
            mu, std = self.kwargs.get("hh_init_mu", 0.0), self.kwargs.get("hh_init_std", 1.0)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(self.kwargs.get("hh_init_seed", 0))
            state = [(torch.rand(
                (batch_size, int(self.output_size)),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
                generator=gen,
            ) * std + mu) for _ in range(1)]
        elif self.kwargs["hh_init"] == "inputs":
            assert "inputs" in kwargs, "inputs must be provided to initialize the state"
            assert kwargs["inputs"].shape == (batch_size, int(self.output_size))
            state = (kwargs["inputs"].clone(),)
        else:
            raise ValueError("Hidden state init method not known. Please use 'zeros', 'inputs' or 'random'")
        return tuple(state)

    def forward(
            self,
            inputs: torch.Tensor,
            state: Optional[Tuple[torch.Tensor, ...]] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass.
        With Euler discretisation, Wilson-Cowan equation becomes:

        output = input * (1 - dt/tau) + dt/tau * (1 - input @ r) * sigmoid(input @ forward_weight - mu)

        :param inputs: time series at a time t of shape (batch_size, number of neurons)
            Remark: if you use to compute a time series, use batch_size = 1.
        :type inputs: torch.Tensor
        :param state: State of the layer (only for SNN -> not use for RNN)
        :type state: Optional[Tuple[torch.Tensor, ...]]

        :return: (time series at a time t+1, State of the layer -> None)
        :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]
        """
        batch_size, nb_features = inputs.shape

        # Model:
        # Linear1(128) -> 256
        # FR(256) -> 512, hh
        # Linear2(512) -> 128

        hh, = self._init_forward_state(state, batch_size, inputs=inputs)
        if self.use_recurrent_connection:
            rec_inputs = torch.matmul(hh, torch.mul(self.recurrent_weights, self.rec_mask))
        else:
            rec_inputs = 0.0

        interaction = rec_inputs + torch.matmul(self.activation(inputs), self.forward_weights)
        output = interaction + self.mu + (1 - self.r) * inputs
        return output, (torch.clone(output),)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", activation: {self.activation.__class__.__name__}"


class LinearJtoI(Linear):
    def build(self):
        """
        Build the layer. This method must be call after the layer is initialized to make sure that the layer is ready
        to be used e.g. the input and output size is set, the weights are initialized, etc.

        In this method the :attr:`forward_weights`, :attr:`recurrent_weights` and :attr: `rec_mask` are created and
        finally the method :meth:`initialize_weights_` is called.

        :return: The layer itself.
        :rtype: BaseLayer
        """
        super().build()
        self._forward_weights = nn.Parameter(
            torch.empty((int(self.output_size), int(self.input_size)), device=self.device, dtype=torch.float32),
            requires_grad=self.requires_grad
        )
        self.initialize_weights_()
        return self

    def forward(
            self,
            inputs: torch.Tensor,
            state: Tuple[torch.Tensor, ...] = None,
            **kwargs
    ):
        out_shape = tuple(inputs.shape[:-1]) + (self.forward_weights.shape[0],)
        inputs_view = inputs.view(-1, inputs.shape[-1])
        inputs_permuted = inputs_view.permute(1, 0)
        pre_activation = torch.matmul(self.forward_weights, inputs_permuted)
        pre_activation = pre_activation.permute(1, 0).view(out_shape)
        return self.activation(pre_activation + self.bias_weights.view(pre_activation.shape))


class WilsonCowanCURBDJtoILayer(WilsonCowanCURBDLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        super().build()
        self._forward_weights = nn.Parameter(
            torch.empty((int(self.output_size), int(self.input_size)), device=self.device, dtype=torch.float32),
            requires_grad=self.requires_grad
        )
        self.initialize_weights_()
        return self

    def forward(
            self,
            inputs: torch.Tensor,
            state: Optional[Tuple[torch.Tensor, ...]] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        # TODO: inverse I->J to J->I
        batch_size, nb_features = inputs.shape
        hh, = self._init_forward_state(state, batch_size, inputs=inputs)

        out_shape = tuple(inputs.shape[:-1]) + (self.forward_weights.shape[0],)
        inputs_view = inputs.view(-1, inputs.shape[-1])
        inputs_permuted = inputs_view.permute(1, 0)

        output = self.activation(hh)

        if self.use_recurrent_connection:
            rec_inputs = torch.matmul(hh, torch.mul(self.recurrent_weights, self.rec_mask))
        else:
            rec_inputs = 0.0

        r = rec_inputs + torch.matmul(output, self.forward_weights)
        hh = hh + self.dt * (r - hh) / self.tau
        return output, (hh,)


class LS(BPTT):
    layer_type_to_mth = {
        LinearJtoI: "infer_affine_model"
    }

    @staticmethod
    def infer_affine_model(time_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes a time series matrix and returns matrices A and b
        defining the affine model
            x_{t+1} = A x_{t} + b
        """
        batch_size, n, T = time_series.shape
        mean_ts = np.mean(time_series, axis=0)
        X = mean_ts[:, 0:T - 1]  # ts left shift
        Y = mean_ts[:, 1:T]  # ts right shift
        K = np.eye(T - 1) - 1 / (T - 1) * np.ones((T - 1, T - 1))
        e = np.ones((T - 1, 1))
        A = Y @ K @ np.linalg.pinv(X @ K, rcond=1e-9, hermitian=False)
        # A = np.linalg.pinv(K @ X, rcond=1e-9, hermitian=False) @ K @ Y
        # b = 1 / N * (Y - X @ A) @ e
        b = 1 / (T - 1) * (Y - A @ X) @ e
        return A, b

    def start(self, trainer, **kwargs):
        super().start(trainer, **kwargs)

    def _make_optim_step(self, pred_batch, y_batch, retain_graph=False):
        # self.optimizer.zero_grad()
        with torch.no_grad():
            batch_loss = self.apply_criterion(pred_batch, y_batch)

        x_batch = self.trainer.current_training_state.x_batch
        y_batch = self.trainer.current_training_state.y_batch
        np_ts = np.concatenate([to_numpy(x_batch), to_numpy(y_batch)], axis=1)
        layer = self.trainer.model.get_layer()

        assert type(layer) in self.layer_type_to_mth.keys(), f"Layer {type(layer)} not supported."
        _mth = self.layer_type_to_mth[type(layer)]
        if isinstance(_mth, str):
            _mth = getattr(self, _mth)
        assert callable(_mth), f"Method {_mth} is not callable."
        _A, _b = _mth(np_ts.transpose((0, 2, 1)))
        _A = to_tensor(_A)
        _b = to_tensor(_b)


        layer.set_forward_weights_data(_A.view(layer.forward_weights.shape))
        layer.bias_weights.data = nt.to_tensor(_b).to(layer.device).view(layer.bias_weights.shape)

        _A_shape = tuple(_A.shape)
        # _A_shape = self._A.shape[1:] if len(self._A.shape) > 1 and self._A.shape[0] == 1 else self._A.shape
        # _A_shape = _A_shape[:-1] if len(_A_shape) > 1 and _A_shape[-1] == 1 else _A_shape
        #
        # _b_shape = tuple(self._b.shape)
        _b_shape = _b.shape[1:] if len(_b.shape) > 1 and _b.shape[0] == 1 else _b.shape
        _b_shape = _b_shape[:-1] if len(_b_shape) > 1 and _b_shape[-1] == 1 else _b_shape

        _A_found, _b_found = False, False
        for p in list(self.trainer.model.parameters()):
            p_shape = tuple(p.shape)
            # p_shape = p.shape[1:] if len(p.shape) > 1 and p.shape[0] == 1 else p.shape
            # p_shape = p_shape[:-1] if len(p_shape) > 1 and p_shape[-1] == 1 else p_shape

            if not _A_found and p_shape == _A_shape:
                # p.data = _A.to(p.device).reshape(p.shape)
                _A_found = True
            if not _b_found and p_shape == _b_shape:
                # p.data = _b.to(p.device).reshape(p.shape)
                _b_found = True
            if _A_found and _b_found:
                break

        # self.optimizer.step()
        return batch_loss.detach_()

    def on_optimization_begin(self, trainer, **kwargs):
        y_batch = trainer.current_training_state.y_batch
        pred_batch = format_pred_batch(trainer.current_training_state.pred_batch, y_batch)
        batch_loss = self._make_optim_step(pred_batch, y_batch)
        trainer.update_state_(batch_loss=batch_loss)



