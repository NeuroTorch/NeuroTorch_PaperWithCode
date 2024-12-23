import torch
import neurotorch as nt


def make_learning_algorithm(**kwargs):
	la_name = kwargs.get("learning_algorithm", kwargs.get("la", "bptt")).lower()
	model = kwargs.get("model", kwargs.get("network", None))
	if model is None:
		raise ValueError("Model must be passed to make_learning_algorithm.")
	if la_name == "bptt":
		optimizer = torch.optim.AdamW(
			model.parameters(), lr=kwargs.get("learning_rate", 2e-4),
			weight_decay=kwargs.get("weight_decay", 0.0)
		)
		learning_algorithm = nt.BPTT(optimizer=optimizer, criterion=torch.nn.NLLLoss())
	elif la_name == "eprop":
		learning_algorithm = nt.Eprop(
			criterion=torch.nn.MSELoss(),
		)
	else:
		raise ValueError(f"Unknown learning algorithm: `{la_name}`.")
	return learning_algorithm
