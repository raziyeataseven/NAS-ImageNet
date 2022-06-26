# Neural Architecture Search with Reinforcement Learning

This repository is the official implementation of [NEURAL ARCHITECTURE SEARCH WITH REINFORCEMENT LEARNING](https://arxiv.org/pdf/1611.01578v2.pdf). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

>Some information about the data and where to get...

## Project files

* __model.py__
	* includes a CNN model which applies 2D convolution over the input.
	* returns forward pass output. 
* __controller.py__
	* this file has the Agent which is driven by a neural network architecture.
	* agent has Long Short-Term Memory (LSTM) network. 
* __policy_gradient.py__
	* Aim of this policy build the Convulutional Neural Network model and compare the results with Reinforcement Learning algorithm 
	* CNN model trainded through play_episode method; during the training 
	* play_episode method retuns episode logits from Agent, reward(accuracy) and sum of weighted episode logits.
	* calculate policy loss and entropy with logits and weighted logits probabilities which gathered through batches. 
	* clear gradients
	* backpropagation
	* update the parameters
	* list avarge total rewards and entropy for each epoch
* __train.py__
	* includes parameters # of epochs, learning rate, batch size, # of hiden nodes etc.
	* downloads train and test dataset, and passes it with the config parameters to the model.


## Training

To train the model in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alfa 5 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on MiniImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Results

Our model achieves the following performance on :

### [Neural Architecture Search with Reinforcement Learning](https://paperswithcode.com/paper/neural-architecture-search-with-reinforcement)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Refrence

@misc{https://doi.org/10.48550/arxiv.1611.01578,
			doi = {10.48550/ARXIV.1611.01578},  
			url = {https://arxiv.org/abs/1611.01578},  
			author = {Zoph, Barret and Le, Quoc V.},  
			keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences},  
			title = {Neural Architecture Search with Reinforcement Learning},  
			publisher = {arXiv},  
			year = {2016},  
			copyright = {arXiv.org perpetual, non-exclusive license}
}
