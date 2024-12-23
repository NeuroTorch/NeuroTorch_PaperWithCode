import pprint
from collections import OrderedDict

import torch
from torchvision.transforms import Compose
import json

from dataset import get_dataloaders, DatasetId
import neurotorch as nt
from neurotorch import Dimension, DimensionProperty
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.metrics import ClassificationMetrics
from neurotorch.trainers import ClassificationTrainer
from neurotorch.transforms.spikes_encoders import SpyLIFEncoder


def run(
		*,
		dataset_id: DatasetId,
		batch_size: int,
		n_steps: int,
		n_hidden_neurons: int,
		dt: float,
		learning_rate: float,
		n_iterations: int,
		seed: int = 0,
		force_overwrite: bool = False,
):
	nt.set_seed(seed=seed)
	checkpoint_folder = f"./checkpoints/{dataset_id.name}"
	checkpoint_manager = CheckpointManager(checkpoint_folder, save_best_only=True)
	input_transform = Compose([torch.nn.Flatten(start_dim=2), SpyLIFEncoder(n_steps=n_steps, n_units=28 * 28)])
	
	dataloaders = get_dataloaders(
		dataset_id=dataset_id,
		batch_size=batch_size,
		train_val_split_ratio=0.98,
		nb_workers=0,
		pin_memory=True,
	)
	
	network = nt.SequentialRNN(
		input_transform=input_transform,
		layers=[
			nt.LIFLayer(
				input_size=[Dimension(None, DimensionProperty.TIME), Dimension(28 * 28, DimensionProperty.NONE)],
				use_recurrent_connection=False,
				output_size=n_hidden_neurons,
				dt=dt,
			),
			nt.SpyLILayer(dt=dt, output_size=10),
		],
		name=f"{dataset_id.name}_network",
		checkpoint_folder=checkpoint_folder,
		hh_memory_size=1,
	).build()
	callbacks = [
		checkpoint_manager,
		nt.BPTT(
			optimizer=torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-8),
			criterion=torch.nn.NLLLoss(),
		),
	]
	trainer = ClassificationTrainer(
		model=network,
		callbacks=callbacks,
		verbose=True,
	)
	print(trainer)
	training_history = trainer.train(
		dataloaders["train"],
		dataloaders["val"],
		n_iterations=n_iterations,
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
		force_overwrite=force_overwrite,
	)
	training_history.plot(save_path=f"{dataset_id.name}_results/figures/tr_history.png", show=False)
	network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.BEST_ITR, verbose=True)
	accuracies = {
		k: ClassificationMetrics.accuracy(network, dataloaders[k], verbose=True, desc=f"{k}_accuracy")
		for k in dataloaders
	}
	precisions = {
		k: ClassificationMetrics.precision(network, dataloaders[k], verbose=True, desc=f"{k}_precision")
		for k in dataloaders
	}
	recalls = {
		k: ClassificationMetrics.recall(network, dataloaders[k], verbose=True, desc=f"{k}_recall")
		for k in dataloaders
	}
	f1s = {
		k: ClassificationMetrics.f1(network, dataloaders[k], verbose=True, desc=f"{k}_f1")
		for k in dataloaders
	}
	results = OrderedDict(
		dict(
			network=str(network),
			accuracies=accuracies,
			precisions=precisions,
			recalls=recalls,
			f1s=f1s,
		)
	)
	
	with open(f"{dataset_id.name}_results/trainer_repr.txt", "w+") as f:
		f.write(repr(trainer))
	
	json.dump(results, open(f"{dataset_id.name}_results/results.json", "w+"), indent=4)
	pprint.pprint(results, indent=4)

