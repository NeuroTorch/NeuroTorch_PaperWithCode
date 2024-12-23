import torch
from pythonbasictools.device import log_device_setup, DeepLib
from pythonbasictools.logging_tools import logs_file_setup

from dataset import DatasetId
from images_classification.train_and_test_script import run

if __name__ == '__main__':
	logs_file_setup("fashion_mnist", add_stdout=False)
	log_device_setup(deepLib=DeepLib.Pytorch)
	if torch.cuda.is_available():
		torch.cuda.set_per_process_memory_fraction(0.9)
	
	dataset_id = DatasetId.FASHION_MNIST
	
	run(
		dataset_id=dataset_id,
		
		# Training parameters
		n_iterations=30,
		batch_size=256,
		learning_rate=2e-4,
		
		# Network parameters
		n_steps=100,
		n_hidden_neurons=100,
		dt=1e-3,
		
		# Other parameters
		seed=0,
		force_overwrite=True,
	)
