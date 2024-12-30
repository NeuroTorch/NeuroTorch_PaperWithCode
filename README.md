# NeuroTorch_PaperWithCode
Code associated with the paper : [NeuroTorch: A Python library for neuroscience-oriented machine learning](https://www.biorxiv.org/content/10.1101/2024.12.29.630683v1).


# Citation

If you use NeuroTorch in your research, please cite the following paper:

```
@article {Gince2024.12.29.630683,
	author = {Gince, J{\'e}r{\'e}mie and Drouin, Anthony and Desrosiers, Patrick and Hardy, Simon V},
	title = {NeuroTorch: A Python library for neuroscience-oriented machine learning},
	elocation-id = {2024.12.29.630683},
	year = {2024},
	doi = {10.1101/2024.12.29.630683},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Machine learning (ML) has become a powerful tool for data analysis, leading to significant advances in neuroscience research. While ML algorithms are proficient in general-purpose tasks, their highly technical nature often hinders their compatibility with the observed biological principles and constraints in the brain, thereby limiting their suitability for neuroscience applications. In this work, we introduce NeuroTorch, a comprehensive ML pipeline specifically designed to assist neuroscientists in leveraging ML techniques using biologically inspired neural network models. NeuroTorch enables the training of recurrent neural networks equipped with either spiking or firing-rate dynamics, incorporating additional biological constraints such as Dale{\textquoteright}s law and synaptic excitatory-inhibitory balance. The pipeline offers various learning methods, including backpropagation through time and eligibility trace forward propagation, aiming to allow neuroscientists to effectively employ ML approaches. To evaluate the performance of NeuroTorch, we conducted experiments on well-established public datasets for classification tasks, namely MNIST, Fashion-MNIST, and Heidelberg. Notably, NeuroTorch achieved accuracies that replicated the results obtained using the Norse and SpyTorch packages. Additionally, we tested NeuroTorch on real neuronal activity data obtained through volumetric calcium imaging in larval zebrafish. On training sets representing 9.3 minutes of activity under darkflash stimuli from 522 neurons, the mean proportion of variance explained for the spiking and firing-rate neural network models, subject to Dale{\textquoteright}s law, exceeded 0.97 and 0.96, respectively. Our analysis of networks trained on these datasets indicates that both Dale{\textquoteright}s law and spiking dynamics have a beneficial impact on the resilience of network models when subjected to connection ablations. NeuroTorch provides an accessible and well-performing tool for neuroscientists, granting them access to state-of-the-art ML models used in the field without requiring in-depth expertise in computer science.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/12/30/2024.12.29.630683},
	eprint = {https://www.biorxiv.org/content/early/2024/12/30/2024.12.29.630683.full.pdf},
	journal = {bioRxiv}
}
```