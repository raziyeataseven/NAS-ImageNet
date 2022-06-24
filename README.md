# Neural Architecture Search with Reinforcement Learning

This repository is the official implementation of [My Paper Title](https://arxiv.org/pdf/1611.01578v2.pdf). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

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
