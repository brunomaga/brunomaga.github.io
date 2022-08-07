---
layout: default
title: Publications Bookmark
permalink: /publications/
---

<h1 class="mt-5" itemprop="name headline">{{ page.title | escape }}</h1>

  <div class="mt-3"></div>
A quick summary of some interesting publications I came accross:

|--- ||--- |
|2021||[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, Facebook AI and Gergia Institute of Technology, ICCV 2017](https://arxiv.org/abs/1610.02391)|
||| <img width="90%" height="90%" src="/assets/publications/gradcam.png"/> |
||||
|2021||[Revisiting ResNets: Improved Training and Scaling Strategies, Google and UC Berkelry, NeurIPS 2021](https://arxiv.org/abs/2103.07579)|
||| <img width="70%" height="70%" src="/assets/publications/revisiting_resnets.png"/> |
||||
|2021||[Reduced, Reused and Recycled: The Life of a Dataset in Machine Learning Research, Google and Univ. California, NeurIPS 2021](https://arxiv.org/abs/2112.01716)|
||| winner of the "Datasets and Benchmarks Best Paper Award" at NeurIPS 2021 |
||| <img width="65%" height="65%" src="/assets/publications/reduced_recycled_datasets.png"/> |
||||
|2021||[MLP-Mixer: An all-MLP Architecture for Vision, Google, NeurIPS 2021](https://arxiv.org/abs/2105.01601)|
||| <img width="70%" height="70%" src="/assets/publications/mlp_mixer.png"/> |
||||
|2021||[Pay attention to MLPs, Google, NeurIPS 2021](https://arxiv.org/abs/2105.08050)|
||| <img width="70%" height="70%" src="/assets/publications/pay_attention_to_mlps.png"/> |
||||
|2021||[Long-Short Transformer: Efficient Transformers for Language and Vision, NVIDIA, NeurIPS 2021](https://arxiv.org/abs/2107.02192)|
||| <img width="70%" height="70%" src="/assets/publications/long_short_transformer.png"/> |
||||
|2021||[Dynamic Grained Encoder for Vision Transformers, ..., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/2d969e2cee8cfa07ce7ca0bb13c7a36d-Paper.pdf)|
||| <img width="70%" height="70%" src="/assets/publications/dge_transformer.png"/> |
||||
|2021||[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Google, ICLR 2021](https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1)|
||| An extension of the transformer architecture to images.  Works by passing as input to the transformer a sequence of linear embeddings of image patches. Paper demonstrates better results on classification tasks, compared to CNNs, ResNets and native attention mechanism (that do not scale well as pixels attend to other pixels leading to a quadratic complexity). Transformers lack the inductive bias of CNNs (e.g. translation equivariance and locality), and therefore do not generalize well when training on insufficient amounts of data. Class is added similarly to BERT as the *class* token. VTs use 1D positional encodings, since performance of 2D encoders  did not deliver significant performance gains. Only MLP layers are local and translationally equivariant, yielding an inductive bias much smaller than CNNs. The *hybrid architecture* mode uses feature maps of a CNN instead of raw image patches as input. Similar to the original NLP transformer, it scales well and delivers a reduced training time compared to CNN-based architectures. Performance increases with dataset size. 
||| <img width="70%" height="70%" src="/assets/publications/visual_transformer.png"/> |
||||
|2020||[Language Models are Few-Shot Learners (GPT-3), OpenAI](https://arxiv.org/abs/2005.14165)|
||| <img width="70%" height="70%" src="/assets/publications/gpt3.png"/> |
||||
|2018||[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Google](https://arxiv.org/abs/1810.04805)|
||| <img width="70%" height="70%" src="/assets/publications/bert.png"/> |
||||
|2016||[Attention is all you need (Transformer), Google, NeurIPS 2017](https://arxiv.org/abs/1706.03762)|
||| Contrarily to state-of-art transduction models based on recurrent encoder-decoder architectures (possibly with Attention Mechanisms), the Transformer uses only attention mechanisms, and no recurrence or concolutions. Results show it to be of better performance, more parallelizable (due to non-recurrence in model), and faster to train. Contrarily to recurrent models, the whole source sentence (in the encoder) and target sentence (decoder) are fed at once. Therefore, backpropagation happens on a single step as well. Because the concept of word sequence provided by the recurrence was removed, Transformers use positional encoding of the input embeddings based on the combination of *sin* waves of different frequencies. Each attention mechanism that will relate any word in the target sentence to words in the source sentence is based on a key-value store. In that kv-store, Key-Value pairs are output by the encoder/source sentence, and Queries are output by the decoder/target sentence). The formulation of attention is: $$Attention (Q, K, V) = softmax( QK^T / \sqrt{d_k}) V$$. Here, the dot-product of all queries and the key ($$QK^T$$) gives a value referring to how well aligned the query vectors are for a given key. This is then converted into a distribution wit the $$softmax$$ function and then used to multiply and extract the most meaningfull value $$V$$. This is effectively an indexing mechanism (similar to a dictionary $$value = query\[key\]$$) but in a continuous space.  |
||| <img width="30%" height="30%" src="/assets/publications/transformer.png"/> |
||||
|2015||[Neural Machine Translation by Jointly Learning to Align and Translate, D. Bahdanau, K. Cho, Y. Bengio](https://arxiv.org/abs/1409.0473)|
||| <img width="50%" height="50%" src="/assets/publications/attention_mech.png"/> |
||||
|2014||[Sequence to Sequence Learning with Neural Networks, Google, NeurIPS 2014](https://arxiv.org/abs/1409.3215)|
||| <img width="75%" height="75%" src="/assets/publications/seq2seq.png"/> |
||||
|2014||[Dropout: a simple way to prevent neural networks from overfitting, Univ. Toronto, Journal of ML Research 2014](https://jmlr.org/papers/v15/srivastava14a.html)|
|||A method that drops neurons (in diff. layers) with probability p during train time. For each training mini-batch, a new network is sample. Dropout improved with max-norm regularization, decaying learning rate and high momntum. At test time, all are using neurons, with outgoing weights multiplied by p. Helps reducing overfitting, as the network learns to never rely on any given activations, so it learns "redundant" ways of solving the task with multiple neurons. It also leads to sparse activations, similar to a regularization (L2). Dropping 20% of input units and 50% of hidden units was often found to be optimal. It's computationally less expensive than regular model averaging of multiple trained DNNs. However, it takes 2-3 times longer to train than single fully-connected DNNs because requires way more epochs, as parameter updates are very noisy|
||| <img width="75%" height="75%" src="/assets/publications/dropout.png"/> |


