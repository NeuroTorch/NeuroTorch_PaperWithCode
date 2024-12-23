from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Sequence, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F
from neurotorch import TBPTT as nt_TBPTT, ToDevice
from neurotorch import RLS as nt_RLS
from neurotorch import BPTT as nt_BPTT
from neurotorch import SequentialRNN as nt_SequentialRNN
from neurotorch.learning_algorithms.learning_algorithm import LearningAlgorithm
from neurotorch.utils import unpack_out_hh, list_insert_replace_at, recursive_detach, format_pred_batch


class BPTT(nt_BPTT):
    def start(self, trainer, **kwargs):
        LearningAlgorithm.start(self, trainer, **kwargs)
        if self.params and self.optimizer is None:
            self.optimizer = self.create_default_optimizer()
        elif not self.params and self.optimizer is not None:
            self.param_groups = self.optimizer.param_groups
            self.params.extend([
                param
                for i in range(len(self.optimizer.param_groups))
                for param in self.optimizer.param_groups[i]["params"]
            ])
        elif not self.params and self.optimizer is None:
            self.params = list(trainer.model.parameters())
            self.optimizer = self.create_default_optimizer()

        if self.criterion is None and getattr(trainer, "criterion", None) is not None:
            self.criterion = trainer.criterion


class TBPTT(nt_TBPTT):
    t_list = []

    def start(self, trainer, **kwargs):
        BPTT.start(self, trainer, **kwargs)
        self.initialize_output_layers(trainer)
        self.initialize_layers(trainer)
        self._initialize_original_forwards()

    def decorate_forwards(self):
        # TODO: make sure the decorator is not applied twice. Update e-prop.
        if self._forwards_decorated:
            pass
        if self.trainer.model.training:
            if not self._forwards_decorated:
                self._initialize_original_forwards()
            self._hidden_layer_names.clear()

            for layer in self.layers:
                layer.forward = self._decorate_hidden_forward(layer.forward, layer.name)
                self._hidden_layer_names.append(layer.name)

            for layer in self.output_layers:
                layer.forward = self._decorate_forward(layer.forward, layer.name)
            self._forwards_decorated = True

    def on_batch_begin(self, trainer, **kwargs):
        # super().on_batch_begin(trainer)
        self.trainer = trainer
        if trainer.model.training:
            self._data_n_time_steps = self._get_data_time_steps_from_y_batch(
                trainer.current_training_state.y_batch, trainer.current_training_state.x_batch
            )
            self._maybe_update_time_steps()
            self.optimizer.zero_grad()
            self._grads_zeros_()
            self.decorate_forwards()

    def on_batch_end(self, trainer, **kwargs):
        # super().on_batch_end(trainer)
        if trainer.model.training:
            for layer_name in self._layers_buffer:
                backward_t = len(self._layers_buffer[layer_name])
                if backward_t > 0:
                    self._backward_at_t(self._data_n_time_steps - 1, backward_t, layer_name)
                    self.optimizer.step()
        self.undecorate_forwards()
        self._layers_buffer.clear()
        self.optimizer.zero_grad()
        self._grads_zeros_()

    def _decorate_forward(self, forward, layer_name: str):
        def _forward(*args, **kwargs):
            out = forward(*args, **kwargs)
            t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
            self.t_list.append(t)
            if t is None:
                return out
            out_tensor, hh = unpack_out_hh(out)
            list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
            self._optim_counter += 1
            buff = self._layers_buffer[layer_name]
            if len(self._layers_buffer[layer_name]) == self.backward_time_steps:
                self._backward_at_t(t, self.backward_time_steps, layer_name)
                out = recursive_detach(out)
            if ((t + 1) % self.optim_time_steps) == 0:
                self._make_optim_step()
            return out

        return _forward

    def _backward_at_t(self, t: int, backward_t: int, layer_name: str):
        y_batch = self._get_y_batch_slice_from_trainer((t + 1) - backward_t, t + 1, layer_name)
        pred_batch = self._get_pred_batch_from_buffer(layer_name)
        batch_loss = self.apply_criterion(pred_batch, y_batch)
        if batch_loss.grad_fn is None:
            raise ValueError(
                f"batch_loss.grad_fn is None. This is probably an internal error. Please report this issue on GitHub."
            )

        if np.isclose(self.alpha, 0.0):
            batch_loss.backward()
        else:
            self._compute_decay_grads_(batch_loss)
            self._apply_grads()
        self._clip_grads()
        self._layers_buffer[layer_name].clear()


class SequentialRNN(nt_SequentialRNN):
    t_list = []

    def _integrate_inputs_(
            self,
            inputs: Dict[str, torch.Tensor],
            hidden_states: Dict[str, List[torch.Tensor]],
            outputs_trace: Dict[str, List[torch.Tensor]],
            time_steps: int,
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
        """
        Integration of the inputs or the initial conditions.

        :param inputs: the inputs to integrate.
        :type inputs: Dict[str, torch.Tensor]
        :param hidden_states: the hidden states of the model.
        :type hidden_states: Dict[str, List]
        :param outputs_trace: the outputs trace of the model.
        :type outputs_trace: Dict[str, List[torch.Tensor]]
        :param time_steps: the number of time steps to integrate.
        :type time_steps: int

        :return: the integrated inputs and the hidden states.
        :rtype: Tuple[Dict[str, torch.Tensor], Dict[str, List]]
        """
        for t in range(time_steps):
            self.t_list.append(t)
            forward_tensor = self._inputs_forward_(inputs, hidden_states, idx=t, t=t, forecasting=False)
            forward_tensor = self._hidden_forward_(forward_tensor, hidden_states, t=t, forecasting=False)
            outputs_trace = self._outputs_forward_(forward_tensor, hidden_states, outputs_trace, t=t, forecasting=False)

            outputs_trace = {
                layer_name: self._pop_memory_(trace, self._out_memory_size)
                for layer_name, trace in outputs_trace.items()
            }
            hidden_states = {
                layer_name: self._pop_memory_(trace, self._hh_memory_size)
                for layer_name, trace in hidden_states.items()
            }

        return outputs_trace, hidden_states

    def _forecast_integration_(
            self,
            hidden_states: Dict[str, List],
            outputs_trace: Dict[str, List[torch.Tensor]],
            inputs_time_steps: int,
            foresight_time_steps: int,
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List]]:
        """
        Foresight prediction of the initial conditions.

        :param hidden_states: the hidden states of the model.
        :type hidden_states: Dict[str, List]
        :param outputs_trace: the outputs trace of the model.
        :type outputs_trace: Dict[str, List[torch.Tensor]]
        :param foresight_time_steps: the number of time steps to forecast.
        :type foresight_time_steps: int

        :return: the forecasted outputs and the hidden states.
        :rtype: Tuple[Dict[str, List[torch.Tensor]], Dict[str, List]]
        """
        if self._outputs_to_inputs_names_map is None:
            self._map_outputs_to_inputs()

        for tau in range(foresight_time_steps - 1):
            t = inputs_time_steps + tau
            self.t_list.append(t)
            foresight_inputs_tensor = {
                self._outputs_to_inputs_names_map[layer_name]: torch.unsqueeze(trace[-1], dim=1)
                for layer_name, trace in outputs_trace.items()
            }
            forecast_kwargs = dict(forecasting=True, tau=tau)
            forward_tensor = self._inputs_forward_(
                foresight_inputs_tensor, hidden_states, idx=-1, t=t, **forecast_kwargs
            )
            forward_tensor = self._hidden_forward_(forward_tensor, hidden_states, t=t, **forecast_kwargs)
            outputs_trace = self._outputs_forward_(forward_tensor, hidden_states, outputs_trace, t=t, **forecast_kwargs)

            outputs_trace = {
                layer_name: self._pop_memory_(trace, self._out_memory_size)
                for layer_name, trace in outputs_trace.items()
            }
            hidden_states = {
                layer_name: self._pop_memory_(trace, self._hh_memory_size)
                for layer_name, trace in hidden_states.items()
            }

        return outputs_trace, hidden_states


class RLS(nt_RLS):
    DEFAULT_OPTIMIZER_CLS = torch.optim.SGD

    def __init__(
            self,
            *,
            params: Optional[Sequence[torch.nn.Parameter]] = None,
            layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
            criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]] = None,
            backward_time_steps: Optional[int] = None,
            is_recurrent: bool = False,
            **kwargs
    ):
        """
        Constructor for RLS class.

        :param params: The parameters to optimize. If None, the parameters of the model's trainer will be used.
        :type params: Optional[Sequence[torch.nn.Parameter]]
        :param layers: The layers to optimize. If not None the parameters of the layers will be added to the
            parameters to optimize.
        :type layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]]
        :param criterion: The criterion to use. If not provided, torch.nn.MSELoss is used.
        :type criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]]
        :param backward_time_steps: The frequency of parameter optimisation. If None, the number of
            time steps of the data will be used.
        :type backward_time_steps: Optional[int]
        :param is_recurrent: If True, the model is recurrent. If False, the model is not recurrent.
        :type is_recurrent: bool
        :param kwargs: The keyword arguments to pass to the BaseCallback.

        :keyword bool save_state: Whether to save the state of the optimizer. Defaults to True.
        :keyword bool load_state: Whether to load the state of the optimizer. Defaults to True.
        """
        kwargs.setdefault("auto_backward_time_steps_ratio", 0)
        kwargs.setdefault("save_state", True)
        kwargs.setdefault("load_state", True)
        TBPTT.__init__(
            self,
            params=params,
            layers=layers,
            criterion=criterion,
            backward_time_steps=backward_time_steps,
            optimizer=None,
            optim_time_steps=None,
            **kwargs
        )

        # RLS attributes
        self.P_list = None
        self.delta = kwargs.get("delta", 1.0)
        self.Lambda = kwargs.get("Lambda", 1.0)
        self.kappa = kwargs.get("kappa", 1.0)
        self._device = kwargs.get("device", None)
        self.to_cpu_transform = ToDevice(device=torch.device("cpu"))
        self.to_device_transform = None
        self._other_dims_as_batch = kwargs.get("other_dims_as_batch", False)
        self._is_recurrent = is_recurrent
        self.strategy = kwargs.get("strategy", "inputs").lower()
        self.strategy_to_mth = {
            "inputs": self.inputs_mth_step,
            "outputs": self.outputs_mth_step,
            "grad": self.grad_mth_step,
            "jacobian": self.jacobian_mth_step,
            "scaled_jacobian": self.scaled_jacobian_mth_step,
            "gince": self.gince_mth_step,
        }
        self.kwargs = kwargs
        self._asserts()
        self._last_layers_buffer = defaultdict(list)

    def on_batch_begin(self, trainer, **kwargs):
        LearningAlgorithm.on_batch_begin(self, trainer, **kwargs)
        self.trainer = trainer
        if self._is_recurrent:
            self._data_n_time_steps = self._get_data_time_steps_from_y_batch(
                trainer.current_training_state.y_batch, trainer.current_training_state.x_batch
            )
            self._maybe_update_time_steps()
            self.decorate_forwards()

    def start(self, trainer, **kwargs):
        TBPTT.start(self, trainer, **kwargs)
        if self._device is None:
            self._device = trainer.model.device
        self.to_device_transform = ToDevice(device=self._device)

    def _decorate_forward(self, forward, layer_name: str):
        def _forward(*args, **kwargs):
            out = forward(*args, **kwargs)
            t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
            if t is None:
                return out
            out_tensor, hh = unpack_out_hh(out)
            list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
            if len(self._layers_buffer[layer_name]) == self.backward_time_steps:
                self._backward_at_t(t, self.backward_time_steps, layer_name)
                if self.strategy in ["grad", "jacobian", "scaled_jacobian", "gince"]:
                    out = recursive_detach(out)
            return out
        return _forward

    def outputs_mth_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        This method is inspired by the work of Perich and al. :cite:t:`perich_inferring_2021` with
        the CURBD algorithm.

        x.shape = [B, f_in]
        y.shape = [B, f_out]
        error.shape = [B, f_out]
        epsilon = mean[B](error[B, f_out]) -> [1, f_out]
        phi = mean[B](y[B, f_out]) [1, f_out]

        P.shape = [f_out, f_out]
        K = P[f_out, f_out] @ phi.T[f_out, 1] -> [f_out, 1]
        h = 1 / (labda[1] + kappa[1] * phi[1, f_out] @ K[f_out, 1]) -> [1]
        P = labda[1] * P[f_out, f_out] - h[1] * kappa[1] * K[f_out, 1] @ K.T[1, f_out] -> [f_out, f_out]
        grad = h[1] * K[f_out, 1] @ epsilon[1, f_out] -> [N_in, N_out]

        In this case [N_in, N_out] must be equal to [f_out, f_out].

        :param x_batch: inputs of the layer
        :param pred_batch: outputs of the layer
        :param y_batch: targets of the layer

        """
        model_device = self.trainer.model.device
        assert isinstance(x_batch, torch.Tensor), "x_batch must be a torch.Tensor"
        assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
        assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
        self.optimizer.zero_grad()

        x_batch_view = x_batch.view(-1, x_batch.shape[-1])  # [B, f_in]
        pred_batch_view = pred_batch.view(-1, pred_batch.shape[-1])  # [B, f_out]
        y_batch_view = y_batch.view(-1, y_batch.shape[-1])  # [B, f_out]
        error = self.to_device_transform(pred_batch_view - y_batch_view)  # [B, f_out]
        # error = self.to_device_transform(y_batch_view - pred_batch_view)  # [B, f_out]

        if self.P_list is None:
            self.initialize_P_list(m=pred_batch_view.shape[-1])
            for p in self.params:
                # make sure that p is two-dimensional.
                if p.ndim != 2:
                    raise ValueError(
                        f"The parameters must be two-dimensional, got {p.ndim} instead of 2."
                    )
                # making sur that f_out = N_in.
                if p.shape[0] != pred_batch_view.shape[-1]:
                    raise ValueError(
                        f"For inputs of shape [B, f_in], the first dimension of the parameters must be f_in, "
                        f"got {p.shape[0]} instead of {x_batch_view.shape[-1]}."
                    )
                # making sure that f_out = N_out.
                if p.shape[1] != pred_batch_view.shape[-1]:
                    raise ValueError(
                        f"For targets of shape [B, f_out], the second dimension of the parameters must be f_out, "
                        f"got {p.shape[1]} instead of {y_batch_view.shape[-1]}."
                    )
        self.P_list = self.to_device_transform(self.P_list)
        self.params = self.to_device_transform(self.params)

        epsilon = error.mean(dim=0).view(1, -1)  # [1, f_out]
        phi = pred_batch_view.mean(dim=0).view(-1, 1).detach().clone()  # [f_out, 1]
        K_list = [torch.matmul(P, phi) for P in self.P_list]  # [f_out, f_out] @ [f_out, 1] -> [f_out, 1]
        h_list = [
            1.0 / (self.Lambda + self.kappa * torch.matmul(phi.T, K)).item()
            for K in K_list
        ]  # [1, f_out] @ [f_out, 1] -> [1]

        for p, K, h in zip(self.params, K_list, h_list):
            p.grad = h * torch.outer(K.view(-1), epsilon.view(-1))  # [f_out, 1] @ [1, f_out] -> [N_in, N_out]
            # p.grad = h * torch.outer(epsilon.view(-1), K.view(-1))  # [f_out, 1] @ [1, f_out] -> [N_in, N_out]

        self.optimizer.step()
        self.P_list = [
            self.Lambda * P - h * self.kappa * torch.matmul(K, K.T)
            for P, h, K in zip(self.P_list, h_list, K_list)
        ]  # [f_in, 1] @ [1, f_in] -> [f_in, f_in]

        self._put_on_cpu()
        self.trainer.model.to(model_device, non_blocking=True)

    def inputs_mth_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
        """


        x.shape = [B, f_in]
        y.shape = [B, f_out]
        error.shape = [B, f_out]
        P.shape = [f_in, f_in]

        epsilon = mean[B](error[B, f_out]) -> [1, f_out]
        phi = mean[B](x[B, f_in]) [1, f_in]

        K = P[f_in, f_in] @ phi.T[f_in, 1] -> [f_in, 1]
        h = 1 / (labda[1] + kappa[1] * phi[1, f_in] @ K[f_in, 1]) -> [1]
        P = labda[1] * P[f_in, f_in] - h[1] * kappa[1] * K[f_in, 1] @ K.T[1, f_in] -> [f_in, f_in]
        grad = h[1] * K[f_in, 1] @ epsilon[1, f_out] -> [N_in, N_out]

        In this case [N_in, N_out] must be equal to [f_in, f_out].

        :param x_batch: inputs of the layer
        :param pred_batch: outputs of the layer
        :param y_batch: targets of the layer

        """
        model_device = self.trainer.model.device
        assert isinstance(x_batch, torch.Tensor), "x_batch must be a torch.Tensor"
        assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
        assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
        self.optimizer.zero_grad()

        x_batch_view = x_batch.view(-1, x_batch.shape[-1])  # [B, f_in]
        pred_batch_view = pred_batch.view(-1, pred_batch.shape[-1])  # [B, f_out]
        y_batch_view = y_batch.view(-1, y_batch.shape[-1])  # [B, f_out]
        error = self.to_device_transform(pred_batch_view - y_batch_view)  # [B, f_out]

        if self.P_list is None:
            self.initialize_P_list(m=x_batch_view.shape[-1])
            for p in self.params:
                # make sure that p is two-dimensional.
                if p.ndim != 2:
                    raise ValueError(
                        f"The parameters must be two-dimensional, got {p.ndim} instead of 2."
                    )
                # making sur that f_in = N_in.
                if p.shape[0] != x_batch_view.shape[-1]:
                    raise ValueError(
                        f"For inputs of shape [B, f_in], the first dimension of the parameters must be f_in, "
                        f"got {p.shape[0]} instead of {x_batch_view.shape[-1]}."
                    )
                # making sure that f_out = N_out.
                if p.shape[1] != y_batch_view.shape[-1]:
                    raise ValueError(
                        f"For targets of shape [B, f_out], the second dimension of the parameters must be f_out, "
                        f"got {p.shape[1]} instead of {y_batch_view.shape[-1]}."
                    )
        self.P_list = self.to_device_transform(self.P_list)
        self.params = self.to_device_transform(self.params)

        epsilon = error.mean(dim=0).view(1, -1)  # [1, f_out]
        phi = x_batch_view.mean(dim=0).view(1, -1).detach().clone()  # [1, f_in]
        K_list = [torch.matmul(P, phi.T) for P in self.P_list]  # [f_in, f_in] @ [f_in, 1] -> [f_in, 1]
        h_list = [
            1.0 / (self.Lambda + self.kappa * torch.matmul(phi, K)).item()
            for K in K_list
        ]  # [1, f_in] @ [f_in, 1] -> [1]

        for p, K, h in zip(self.params, K_list, h_list):
            p.grad = h * torch.outer(K.view(-1), epsilon.view(-1))  # [f_in, 1] @ [1, f_out] -> [N_in, N_out]

        self.optimizer.step()
        self.P_list = [
            self.Lambda * P - h * self.kappa * torch.matmul(K, K.T)
            for P, h, K in zip(self.P_list, h_list, K_list)
        ]  # [f_in, 1] @ [1, f_in] -> [f_in, f_in]

        self._put_on_cpu()
        self.trainer.model.to(model_device, non_blocking=True)

    def on_optimization_begin(self, trainer, **kwargs):
        x_batch = trainer.current_training_state.x_batch
        y_batch = trainer.current_training_state.y_batch
        pred_batch = format_pred_batch(trainer.current_training_state.pred_batch, y_batch)

        if self._is_recurrent:
            for layer_name in self._layers_buffer:
                backward_t = len(self._layers_buffer[layer_name])
                if backward_t > 0:
                    self._backward_at_t(self._data_n_time_steps - 1, backward_t, layer_name)
        else:
            self.optimization_step(x_batch, pred_batch, y_batch)

        trainer.update_state_(batch_loss=self.apply_criterion(pred_batch, y_batch).detach_())

    def gince_mth_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        This method is inspired by the work of Zhang and al. :cite:t:`zhang_revisiting_2021`. Unfortunately, this
        method does not seem to work with the current implementation.

        TODO: Make it work.

        x.shape = [B, f_in]
        y.shape = [B, f_out]
        error.shape = [B, f_out]
        P.shape = [f_in, f_in]

        epsilon = mean[B](error[B, f_out]) -> [1, f_out]
        phi = mean[B](x[B, f_in]) [1, f_in]

        K = P[f_in, f_in] @ phi.T[f_in, 1] -> [f_in, 1]
        h = 1 / (labda[1] + kappa[1] * phi[1, f_in] @ K[f_in, 1]) -> [1]
        grad = h[1] * P[f_in, f_in] @ grad[N_in, N_out] -> [N_in, N_out]
        P = labda[1] * P[f_in, f_in] - h[1] * kappa[1] * K[f_in, 1] @ K.T[1, f_in] -> [f_in, f_in]

        In this case f_in must be equal to N_in.

        :param x_batch: inputs of the layer
        :param pred_batch: outputs of the layer
        :param y_batch: targets of the layer

        """
        model_device = self.trainer.model.device
        assert isinstance(x_batch, torch.Tensor), "x_batch must be a torch.Tensor"
        assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
        assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
        self.optimizer.zero_grad()

        mse_loss = F.mse_loss(pred_batch, y_batch)
        mse_loss.backward()

        x_batch_view = x_batch.view(-1, x_batch.shape[-1])  # [B, f_in]
        pred_batch_view = pred_batch.view(-1, pred_batch.shape[-1])  # [B, f_out]
        y_batch_view = y_batch.view(-1, y_batch.shape[-1])  # [B, f_out]
        error = self.to_device_transform(pred_batch_view - y_batch_view)  # [B, f_out]

        if self.P_list is None:
            self.initialize_P_list(m=pred_batch_view.shape[-1])
            for p in self.params:
                # making sur that f_out = N_out.
                if p.shape[-1] != pred_batch_view.shape[-1]:
                    raise ValueError(
                        f"For inputs of shape [B, f_out], the last dimension of the parameters must be f_out, "
                        f"got {p.shape[-1]} instead of {pred_batch_view.shape[-1]}."
                    )
        self.P_list = self.to_device_transform(self.P_list)
        self.params = self.to_device_transform(self.params)

        epsilon = error.mean(dim=0).view(1, -1)  # [1, f_out]
        phi = epsilon.mean(dim=0).view(1, -1).detach().clone()  # [1, f_out]
        K_list = [torch.matmul(P, phi.T) for P in self.P_list]  # [f_out, f_out] @ [f_out, 1] -> [f_out, 1]
        h_list = [
            1.0 / (self.Lambda + self.kappa * torch.matmul(phi, K)).item()
            for K in K_list
        ]  # [1, f_out] @ [f_out, 1] -> [1]

        for p, P, h in zip(self.params, self.P_list, h_list):
            p.grad = h * torch.matmul(P, p.grad.T).T  # ([f_out, f_out] @ [N_out, N_in]).T -> [N_in, N_out]

        self.optimizer.step()
        self.P_list = [
            self.Lambda * P - h * self.kappa * torch.matmul(K, K.T)
            for P, h, K in zip(self.P_list, h_list, K_list)
        ]  # [f_in, 1] @ [1, f_in] -> [f_in, f_in]

        self._put_on_cpu()
        self.trainer.model.to(model_device, non_blocking=True)


