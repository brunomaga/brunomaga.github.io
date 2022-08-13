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
||| Existing standard language models are unidirectional and that's a major limitation in performance, e.g. attending to previous tokens in the self-attention layers in the Transformer. This is an issue for many problems like question answering, it is crucial to incorporate context from both directions. BERT removes this unidirectionality by using a masked language model instead, that allows it to train a deep bidirectional Transformer. BERT model architecture is a multi-layer bidirectional sequence of Transformer encoder blocks. BERT models are trained in 2 steps: pre-training and fine-tuning. During pre-training, the model is trained on *unlabeled data* on different datasets. During fine-tuned, the pre-trained model is trained for a given specific task. Apart from output layers, the same architectures are used in both pre-training and fine-tuning. During fine-tuning, all parameters are fine-tuned. The input sentence may be a single sentence or a pair of sentences (e.g. question/answer) packed together. Words are embedded with WorkPiece embeddings. [CLS] is the first token of every sentence. [SEP] is a special separator token. To each token (word embedding) it is also added a learned embedding to indicate if  it belongs to sentence A or B. Each input is then the sum of its position embedding, segment embedding and token embedding (Fig. 2). The pre-training happens in two unsupervised tasks: (1) Masked LM, by masking of 15% of input tokens at random and trying to predict them, and (2) and Next Sentence Prediction, by passing sentence pairs and predicting whether the second sentence is a logic follow up from the first, or not. The fine-tuning happens differently for every task: we pass the specific inputs and outputs to the BERT and do a regular training. The input is the sequences A and B and separators. The output is the answer to the task by: replacing [CLS] by the sentence or sentence-pair label when the task is to classify a sentence or pair or sentences; replacing the stard and end tokens to indicate the span of output answer tokens that answers the question passed in the input (when input is a question/answer pair, Fig 1); or the class of each word for Named Entity Recognition tasks. |
||| <img width="70%" height="70%" src="/assets/publications/bert.png"/> |
||| <img width="70%" height="70%" src="/assets/publications/bert2.png"/> |
||||
|2016||[Attention is all you need (Transformer), Google, NeurIPS 2017](https://arxiv.org/abs/1706.03762)|
||| State-of-art transduction models are based on recurrent encoder-decoder architectures (possibly with Attention Mechanisms). The Transformer uses only attention mechanisms, and no recurrence or convolutions. Results show it to be of better performance, more parallelizable (due to non-recurrence in model), and faster to train. Contrarily to recurrent models, the whole source sentence (in the encoder) and target sentence (in the decoder) are fed at once. Therefore, backpropagation happens on a single step as well. Because the concept of word sequence provided by the recurrence was removed, Transformers use positional encoding of the input embeddings based on the combination of sine and coside waves of different frequencies. The encoder and decoder are composed of a stack of 6 layers each. Each encoder layer includes a multi-heard attention module and a feed forward network. The decoder includes also a third module, a *masked* multi-head attention, that ensures that sentence does not learn from subsequent words in sentence. An attention head is a mapping of a query to a set of key-value pairs. Key-Value pairs are output by the encoder, and Queries are output by the decoder. The formulation of this *dot-product attention* is: $$Attention (Q, K, V) = softmax( QK^T / \sqrt{d_k}) V$$. Here, the dot-product of all queries and the key ($$QK^T$$) gives a value referring to how well aligned the query vectors are for a given key. This is then converted into a distribution ($$softmax$$) and then used extract the most meaningfull value $$V$$ (by multiplying). This is effectively an indexing mechanism (similar to a dictionary $$value = query\_dict[key]$$) but in a continuous space. The scaling factor $$\sqrt{d_k}$$ is used to avoid having really small gradients for large values of $$d_k$$ (dimensionality of keys).  The multi-head attention heads allows the model to jointly attend to information from different (8) representation. It is formulation as $$MultiHead(Q,K, V) = Concat(head_1, ..., head_h)W^O$$ where $$head_i = Attention(QW^Q_i ,KW^K_i , VW^V_i)$$, ie it's the linearly-transformed (projected) concatenation of the attention heads with projected Q, K, and V. In terms of performance, self-attention layers have complexity $$O(n^2 d)$$ per layer, compared to $$O(n d^2)$$ in recurrent models (for sequence length $n$ and representation dimension $d$) --- which is typically faster as $$n < d$$ in most use cases. It also requires no recurrence and no attention connectivity between previous words in a sentence. | 
||| <img width="60%" height="60%" src="/assets/publications/transformer.png"/> |
||||
|2015||[Neural Machine Translation by Jointly Learning to Align and Translate (and Attention Mechanism), D. Bahdanau, K. Cho, Y. Bengio](https://arxiv.org/abs/1409.0473)|
||| <img width="50%" height="50%" src="/assets/publications/attention_mech.png"/> |
||| In most encoder-decoder models, encoders encode a sentence into a vector of fixed-length, from which a decoder generates the translation. Thus, neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector. Here authores claim that fixed-length arrays are a bottleneck in performance on encoder-decoder architectures, particularly for long lentences. Therefore, the authors [quote] "propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly [...] The new architecture consists of a bidirectional RNN as an encoder (BiRNN) and an uni-directional RNN decoder that emulates searching through a source sentence during decoding.". A BiRNN consists of a forwards a a backward RNNs, containing the summaries of the preceeding words and the following words. The *annotation* of each word is the concatenation of the forward and backward states. The decoder receives the output of the previous decoded word, a hidden state for time $i$ (e.g. LSTM hidden state) and the context vector from a sequence of annotations --- computed as a *weighted* sum of annotations. In practice, the encoder encodes the input sentence into a sequence of vectors and the decoder chooses a subset of these vectors adaptively while decoding the translation. |
||||
|2014||[Sequence to Sequence Learning with Neural Networks, Google, NeurIPS 2014](https://arxiv.org/abs/1409.3215)|
||| <img width="75%" height="75%" src="/assets/publications/seq2seq.png"/> |
||||
|2014||[Dropout: a simple way to prevent neural networks from overfitting, Univ. Toronto, Journal of ML Research 2014](https://jmlr.org/papers/v15/srivastava14a.html)|
|||A method that drops neurons (in diff. layers) with probability p during train time. For each training mini-batch, a new network is sample. Dropout improved with max-norm regularization, decaying learning rate and high momntum. At test time, all are using neurons, with outgoing weights multiplied by p. Helps reducing overfitting, as the network learns to never rely on any given activations, so it learns "redundant" ways of solving the task with multiple neurons. It also leads to sparse activations, similar to a regularization (L2). Dropping 20% of input units and 50% of hidden units was often found to be optimal. It's computationally less expensive than regular model averaging of multiple trained DNNs. However, it takes 2-3 times longer to train than single fully-connected DNNs because requires way more epochs, as parameter updates are very noisy. |
||| <img width="75%" height="75%" src="/assets/publications/dropout.png"/> |


