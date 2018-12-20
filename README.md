# hierarchical-attention-model
hierarchical attention model

This repo implemented "Hierarchical Attention Networks for Document Classification" by Zichao Yang et.al. 

It benefited greatly from two resources: the foremost one is Ilya Ivanov's repo on hierarchical attention model: https://github.com/ilivans/tf-rnn-attention .  I followed the way Ilya did in implementing attention and visualization. The difference is that in this implementation it also has sentence-level attention. The other one is r2rt's code in generating batch samples for dynamic rnns: https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html

The code was experimented on imdb data (with only positive and negative labels) 

To prepare the data:

1. pretrain word embeddings and data

```python preprocess```

(By default, the embedding size is 100.)

2. run the model

Train the model and evaluate it on the test set.

```python train```

3. visualization

```python visualize```
