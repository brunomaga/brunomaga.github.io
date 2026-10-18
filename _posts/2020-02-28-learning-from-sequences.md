---
layout: post
title:  "Learning from sequences with Encoder-Decoder, Transformers and BERT"
categories: ["ML Architectures"]
tags: ["Transformer", "attention", "encoder-decoder", "BERT"]
---

Translation and many other language tasks map a sequence of words to another sequence. A recurrent encoder-decoder processes the words one at a time, while a transformer processes all the words of a sequence at once, with matrix products that run in parallel. BERT pre-trains the encoder of a transformer once, and fine-tunes it at a low cost for many tasks.

## Encoder-Decoder

The encoder-decoder, also called the sequence-to-sequence model, maps an input sequence to an output sequence of a possibly different length, such as a sentence to its translation in another language. It was introduced with [recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNNs) by [Cho et al. (2014)](https://arxiv.org/abs/1406.1078) and in [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) (Sutskever et al., 2014). As the name says, the model has two parts: an encoder reads the input sequence, and a decoder writes the output sequence, each with a recurrent neural network that runs one iteration per word.

Each word enters the RNN, a small fully connected network shown in the figure, as an [embedding]({{ site.baseurl }}{% post_url 2018-03-27-deep-neural-networks %}#embeddings) of size $$d$$. Unlike a regular neuron, which is stateless, the neurons of the hidden layer keep an internal state, and the states of all the neurons of the layer form its **hidden state**, a vector of size $$d_h$$. At step $$t$$, the input layer holds the concatenation of the hidden state of step $$t-1$$ (green) and the embedding of word $$t$$ (blue), of size $$d_h + d$$. The hidden layer (green) computes the new hidden state, which is fed back to the next step, and the output layer (red) computes the output of the step. On the first step, the previous hidden state is initialized with zeros or with random values.

{: .picture}
<img width="33.8%" height="33.8%" src="{{ site.assets }}/learning-from-sequences/encoder_decoder.svg"/>

{: .caption}
The recurrent neural network trained at each step of the encoder-decoder architecture. Its input layer is the concatenation of the hidden layer of the previous iteration (green) and the embedding of the current input word (blue). The hidden layer (green) holds the states of the GRU or LSTM neurons, and the output layer (red) holds the model output. 

The hidden state carries the information of the previous words from one step to the next. A plain RNN, however, learns long-range dependencies poorly, because the gradients that flow back through many steps shrink until they vanish ([vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)). [Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory) (LSTM) and [gated recurrent unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit) (GRU) cells keep the information over many steps with gates, which learn what to keep in the state and what to forget. An LSTM cell has two states, the hidden state and the cell state, and four weight matrices. Only the hidden state enters the input layer, whose size is thus $$d_h + d$$ for both cells. A GRU, introduced by Cho et al., has a single state and three weight matrices, so a GRU step costs about 3/4 of an LSTM step of the same size, which makes the GRU faster to train.

Training has an encoding and a decoding phase:
- The **encoder** runs the RNN over the words of the input sentence, and passes the hidden state of each step to the next one. Its outputs are discarded: its last hidden state summarizes the whole input sentence, and becomes the initial hidden state of the decoder.
- The **decoder**, a second RNN, starts from a special beginning-of-sentence token (BOS) and predicts one word per step, with a [softmax](https://en.wikipedia.org/wiki/Softmax_function) over the target vocabulary. Each step takes the previous word of the target sentence as input, a method called teacher forcing, and the last step predicts the end-of-sentence token (EOS). The loss is the sum of the [cross-entropies](https://en.wikipedia.org/wiki/Cross-entropy) of the predicted words, and [backpropagation through time](https://en.wikipedia.org/wiki/Backpropagation_through_time) computes its gradient through all the steps of both RNNs, so that the encoder also learns from the errors of the decoder.

In the figure, the bold line carries the last hidden state of the encoder to the decoder, which receives BOS and the words of the target sentence, and predicts at each step the next word, up to EOS.

{: .picture}
<img width="96%" height="96%" src="{{ site.assets }}/learning-from-sequences/encoder_decoder_2.svg"/>

{: .caption}
Training workflow of an encoder-decoder architecture that learns the translation of the English sentence "Hello World." into the French sentence "Bonjour le monde.".

Once the network is trained, a new sentence is translated without a target sentence: the decoder feeds each predicted word back as the input of the next step, until it outputs EOS. This recurrent design has two limits:
- The encoder compresses the whole input sentence into one hidden state of fixed size, and the information of the first words tends to get lost after many steps, so the accuracy drops for long sentences. The [attention mechanism](https://en.wikipedia.org/wiki/Attention_(machine_learning)) of [Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) lets each step of the decoder use the hidden states of all the steps of the encoder, not only the last one.
- Step $$t$$ needs the hidden state of step $$t-1$$, so the steps of a sequence run one after the other. Parallelism then comes only from the batch and from splitting the matrix products of each step across the model, and these products are too small in a small model to give a speedup.

The [PyTorch tutorial "NLP from scratch: translation with a sequence to sequence network and attention"](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) implements the encoder-decoder with and without attention.

## Transformer

The [transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)), introduced by researchers at Google in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), overcomes both limits: it replaces the recurrence with attention, which processes all the positions of a sequence at once, and it reached better translation results than the recurrent models. It keeps the encoder-decoder structure, and its encoder (left) and decoder (right) are each a stack of $$N = 6$$ identical layers, marked $$N\times$$ in the figure.

{: .picture}
<img width="43.8%" height="43.8%" src="{{ site.assets }}/learning-from-sequences/transformer.svg"/>

{: .caption}
The transformer architecture. Grey regions represent one encoder layer (left) and one decoder layer (right), each repeated N times. Adapted from: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, arXiv)</a>

The next sections describe its components, with an emphasis on their computational cost. [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) implements the model in PyTorch, and the [GPT-lite post]({{ site.baseurl }}{% post_url 2023-02-28-gptlite %}) implements a decoder-only transformer.

### Embeddings and Positional Encoding

The first units of the transformer, at the bottom of the figure, map each token, a word or a part of a word, to an embedding vector of size $$d$$ (red boxes), with $$d = 512$$ in the base model. Self-attention, however, ignores the order of the tokens: it is permutation-equivariant, i.e. permuting the input tokens permutes its outputs in the same way. Without positional information, the model would thus see its input as a set of words, and could not tell "dog bites man" from "man bites dog". The transformer therefore adds a **positional encoding** of size $$d$$ to each embedding (the circles). For a token at position $$pos$$, the dimensions $$2i$$ and $$2i+1$$ of the encoding are:

$$
PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i/d}} \right), \qquad PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i/d}} \right)
$$

Each pair of dimensions holds a sine and a cosine of the same frequency, and the wavelengths grow geometrically from $$2\pi$$ to $$10000 \cdot 2\pi$$, so that the fast and the slow waves together give each position its own pattern, much as the bits of a binary counter do. The authors chose this function because, for any fixed offset $$\Delta$$, $$PE_{pos+\Delta}$$ is a linear function of $$PE_{pos}$$, a rotation of each pair of dimensions, which they expected to make relative positions easy to learn. The figure shows the dimensions 4 to 7 for $$d = 20$$, i.e. two pairs with wavelengths of about 40 and 100 positions:

{: .picture}
<img width="70.1%" height="70.1%" src="{{ site.assets }}/learning-from-sequences/transformer-embedding.svg"/>

{: .caption}
Output of the positional encoding of the transformer for dimensions 4 to 7 of an embedding array of size $$d=20$$, for a word whose position in the sentence is given by the x axis. Adapted from: <a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">The Annotated Transformer</a>

### Attention

The main component of the transformer is the attention mechanism, which learns how the words of the input and output sentences relate to each other, i.e. which words hold the relevant information for each word in its context. The transformer runs several attention mechanisms, or heads, in parallel, so that the model can learn several aspects of these relations, and each head answers the question *which part of the input is relevant?*

For a single head and a sentence of $$n$$ words, attention computes how relevant each word is to every word of the sentence, as an $$n \times n$$ matrix of **attention weights** whose rows sum to 1. For the sentence "The big red dog" ($$n = 4$$), the weights may be:

{: .picture}
<img width="33.3%" height="33.3%" src="{{ site.assets }}/learning-from-sequences/transformer_attention.svg"/>

{: .caption}
Attention weights of one attention head for the sentence "The big red dog", of length $$N=4$$. Each row holds the softmax-normalized relevance of that word to every word of the sentence, itself included, and sums to 1.

In this example, each word attends mostly to itself, while the article and the two adjectives also attend to "dog", the noun that they describe.

A head computes these weights by matching **queries** with **keys**. A query is a vector that describes what a word looks for, a key describes what a word offers, and their dot product measures how well the two match. Each key comes with a **value**, and the output of the head for a query is the average of the values, weighted by the normalized matches. The $$N^Q$$ queries, the $$N^{KV}$$ keys and the $$N^{KV}$$ values are the rows of the matrices $$Q \in \mathbb{R}^{N^Q \times D^{QK}}$$, $$K \in \mathbb{R}^{N^{KV} \times D^{QK}}$$ and $$V \in \mathbb{R}^{N^{KV} \times D^V}$$, and the paper computes a head as:

$$
\mathrm{Attention}(Q, K, V) = A \, V, \qquad A = \mathrm{softmax}\left( \frac{Q K^T}{\sqrt{D^{QK}}} \right)
$$

The softmax, also called softargmax, turns each row of the attention **scores** $$Q K^T / \sqrt{D^{QK}}$$ into weights that are positive and sum to 1, so the weight of query $$q$$ on key $$k$$ is

$$
A_{q,k} = \frac{\exp\left( Q_q^T K_k / \sqrt{D^{QK}} \right)}{\sum_l \exp\left( Q_q^T K_l / \sqrt{D^{QK}} \right)}
$$

where $$Q_q$$ and $$K_k$$ are the rows $$q$$ and $$k$$ of $$Q$$ and $$K$$. The output of query $$q$$ is $$Y_q = \sum_k A_{q,k} V_k$$, a vector of size $$D^V$$, so the output $$Y = A V$$ of the head has the size $$N^Q \times D^V$$. The division by $$\sqrt{D^{QK}}$$ keeps the scores in the range where the softmax has useful gradients: for query and key components that are independent, with zero mean and unit variance, the dot product $$Q_q^T K_k$$ has the variance $$D^{QK}$$. Without the division, the scores of a large head would be large, and would push the weights toward 0 and 1, where the gradients of the softmax vanish. In the figure, thicker lines mark larger weights:

{: .picture}
<img width="33.6%" height="33.6%" src="{{ site.assets }}/learning-from-sequences/transformer_attention_lbdl.svg"/>

{: .caption}
The attention operator matches every query $$Q_q$$ with all keys $$K_1, ..., K_{N^{KV}}$$ to obtain normalized attention scores $$A_{q,1},...,A_{q,N^{KV}}$$ (left), and then averages the values $$V_1,...,V_{N^{KV}}$$ with these scores to compute $$Y_q$$ (right). Adapted from: [The Little Book of Deep Learning](https://fleuret.org/public/lbdl.pdf).

In the transformer diagram, each attention module has three inputs, one for each of $$Q$$, $$K$$ and $$V$$. In **self-attention**, the three come from the same sequence, and $$N^Q = N^{KV} = n$$: in this way, the encoder learns the context of the source sentence, and the first attention sublayer of each decoder layer the context of the target sentence. In **cross-attention**, the second attention sublayer of each decoder layer, the queries come from the decoder, and the keys and the values from the output of the encoder: this is where the decoder reads the source sentence and learns the translation.

**Multi-head attention** runs $$h$$ heads in parallel, each on its own learned projections of the queries, the keys and the values, and concatenates and projects their outputs:

$$
\mathrm{MHA}(Q, K, V) = \left[ \mathrm{head}_1, \dots, \mathrm{head}_h \right] W^O, \qquad \mathrm{head}_j = \mathrm{Attention}\left( Q W_j^Q, K W_j^K, V W_j^V \right)
$$

The learned parameters are the projections $$W_j^Q$$, $$W_j^K$$ and $$W_j^V$$ of each head $$j$$, from the size $$d$$ to the $$D^{QK}$$ or $$D^V$$ dimensions of the head, and the output projection $$W^O$$ of the concatenation. Since each head has its own projections, each head can learn a different relation between words. The base model has $$h = 8$$ heads with $$D^{QK} = D^V = d / h = 64$$, so the 8 heads together cost about as much as one head of size $$d$$.

The decoder predicts each word of the target sentence from the words before it, so it must not see the words that follow. Its **masked multi-head attention** therefore sets the scores of the future positions, above the diagonal of $$Q K^T$$, to $$-\infty$$ before the softmax, so that their weights are zero. The decoder input is the target sentence shifted right by one position, with a start token in the first position, so the output at position $$t$$ predicts word $$t$$ from the words before it. In training, the mask lets the decoder process all the positions of the target sentence at once. At inference, the target sentence is unknown, so the decoder generates one word per step and appends it to its input, as the recurrent decoder does. For the decoder input "Le gros chien rouge", the attention weights of the four steps are:

{: .picture}
<img width="30.5%" height="30.5%" src="{{ site.assets }}/learning-from-sequences/transformer_attention_masked.svg"/>

{: .caption}
Attention weights of the masked attention of the decoder for the sentence "Le gros chien rouge". The algorithm performs four iterations, one per word. Attention is computed for every iterated word. Masking sets the attention scores of unseen words to $$-\infty$$ before the softmax, which gives them an attention weight of zero. Each vector sums to 1.

### Feed-Forward, Residual and Output Layers

The other components of the transformer are not specific to it, and were used before in other models:
- Each layer ends with a **feed-forward network** with one hidden layer, which is applied to each position separately and prepares the output of the attention for the next layer: $$\mathrm{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$, with a [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit) activation and a hidden layer of size $$4d = 2048$$.
- Each attention and feed-forward sublayer has a [residual connection]({{ site.baseurl }}{% post_url 2018-03-27-deep-neural-networks %}#skip-connections) followed by [layer normalization](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization), $$\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$$ ("Add & Norm" in the figure), which keep the gradients and the activations of the deep stack in a stable range.
- On top of the decoder, a **linear** layer maps the output at each position to one score per word of the target vocabulary (French in the example), and a **softmax** turns the scores into probabilities. Instead of taking the most probable word at each step, the paper keeps the 4 most likely partial translations, with a [beam search](https://en.wikipedia.org/wiki/Beam_search) of size 4.

### Computational Complexity

Besides the large reduction of the number of sequential steps, the paper compares the cost of one layer of four types, for a sequence of $$n$$ tokens of size $$d$$:

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|:--|:-:|:-:|:-:|
| **Self-Attention** | $$O(n^2 \cdot d)$$ | $$O(1)$$ | $$O(1)$$ |
| Recurrent | $$O(n \cdot d^2)$$ | $$O(n)$$ | $$O(n)$$ |
| Convolutional | $$O(k \cdot n \cdot d^2)$$ | $$O(1)$$ | $$O(\log_k(n))$$ |
| **Self-Attention** (restricted) | $$O(r \cdot n \cdot d)$$ | $$O(1)$$ | $$O(n/r)$$ |

{: .caption}
Computational complexity of four different learning models. Key: $$n$$: sequence length, $$d$$: representation dim., $$k$$: kernel size, $$r$$: size of neighbourhood. Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, arXiv)</a>

At each of its $$n$$ steps, a recurrent layer multiplies a $$d \times d$$ weight matrix by a state vector, so it needs $$O(n d^2)$$ operations in $$n$$ sequential steps. Self-attention instead computes the $$n \times n$$ scores from the queries and the keys of size $$d$$, and multiplies them by the values: $$O(n^2 d)$$ operations, with no sequential dependency. Self-attention thus needs fewer operations than a recurrent layer when $$n < d$$, which seems counter-intuitive, since in most machine learning problems the input is much larger than its embedding. In machine translation, however, $$n$$ is the number of tokens of a sentence, a few tens, while $$d$$ is 512 in the base transformer and 1024 in its big variant.

The projections and the feed-forward network still cost $$O(n d^2)$$ per layer, so the main gain is parallelism. Attention replaces the recurrent iterations with a few matrix-matrix products, which GPUs run efficiently, and training processes all the positions of a sentence at once, in the encoder and, thanks to the mask, in the decoder. The maximum path length, i.e. the number of steps or layers that a signal crosses between two positions, also drops from $$O(n)$$ to $$O(1)$$, which makes long-range dependencies easier to learn. The big model ($$d = 1024$$, $$h = 16$$) reached a new state-of-the-art [BLEU](https://en.wikipedia.org/wiki/BLEU) score of 28.4 on the WMT 2014 English-to-German task, after 3.5 days of training on 8 P100 GPUs. For long sequences, however, the $$n^2$$ term dominates, and the posts on [sequence parallelism]({{ site.baseurl }}{% post_url 2024-08-30-sequence-context-parallelism %}) and [faster attention]({{ site.baseurl }}{% post_url 2025-12-30-inference-attention %}) cover methods that distribute or reduce this cost.

## BERT: Bidirectional Encoder Representations from Transformers

In translation, the two halves of the transformer have different roles: the encoder learns the context of the input language (English in the example above), and the decoder learns the task, i.e. the mapping to the output language. The encoder is efficient to train and learns a context, which raises a question: can this context alone be used to learn other complex tasks? The question led to [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019). BERT keeps only the encoder of the transformer, a stack of encoder layers, as the figure shows. Since the encoder has no mask, each layer computes the representation of a token from the words on both of its sides, hence *bidirectional* in the name of the model.

{: .picture}
<img width="95.3%" height="95.3%" src="{{ site.assets }}/learning-from-sequences/bert.svg"/>

{: .caption}
The BERT model, as a stack of Transformer encoders. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Training has two phases: pre-training learns the contexts of the language from a large amount of unlabeled text, and fine-tuning adapts the pre-trained model to each task, with a much smaller amount of labeled data.

### Pre-training

Pre-training needs no labeled data, because its two tasks, which the model solves at the same time, are [self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning): their labels come from the text itself.
1. **Masked language model**: 15% of the input tokens are selected at random, and the model predicts them from the tokens on both sides, while the loss ignores the other positions. Of the selected tokens, 80% are replaced by a `[MASK]` token, 10% by a random token, and 10% stay unchanged, because the `[MASK]` token never appears in fine-tuning.
2. **Next sentence prediction**: the input holds two sentences, and the model predicts whether the second sentence follows the first in the text (half of the pairs) or is a random sentence (the other half). This task teaches the model the relation between two sentences, which tasks such as question answering need.

Two training examples, with the class of the second task in place of the `[CLS]` token of the output, are:
```
Input = [CLS] the man went to [MASK] store [SEP] He bought a gallon [MASK] milk [SEP]
Output = [IsNext] the man went to the store [SEP] He bought a gallon of milk [SEP]

Input = [CLS] the man [MASK] to the store [SEP] penguins [MASK] to jump [SEP]
Output = [NotNext] the man went to the store [SEP] penguins like to jump [SEP]
```

The output at the `[CLS]` position feeds the classifier of the second task, and the outputs at the masked positions predict the masked words: "the" and "of" in the first example, and "went" and "like" in the second. The input of the model is the sum of three embeddings per token. Its tokens come from a WordPiece vocabulary of 30,000 tokens, which splits rare words into pieces, such as "play" and "##ing", and, unlike the transformer, BERT learns its position embeddings:

{: .picture}
<img width="90.7%" height="90.7%" src="{{ site.assets }}/learning-from-sequences/bert_input.svg"/>

{: .caption}
Input of the BERT model. Position embeddings are learned, one vector per position (up to 512), unlike the fixed sine and cosine encoding of the transformer described above. Segment embeddings flag each word as part of the first or second sentence. Token embeddings are the text embeddings of the input data. The input to the model is the sum of these three embeddings. Adapted from: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

BERT-base has 12 layers, $$d = 768$$ and 12 heads (110M parameters), and BERT-large has 24 layers, $$d = 1024$$ and 16 heads (340M parameters). Both were pre-trained on BooksCorpus (800M words) and English Wikipedia (2,500M words), for 1M steps with batches of 256 sequences, which took 4 days on 16 TPU chips for BERT-base, and on 64 TPU chips for BERT-large. Since the cost of attention grows with $$n^2$$, 90% of the steps use sequences of 128 tokens, and only the last 10% use sequences of 512 tokens, to learn the position embeddings of the later positions.

### Fine-tuning

Fine-tuning adds an output layer for the task at hand, and trains all the parameters on labeled data, starting from the pre-trained weights. This [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) was used before in [convolutional neural networks]({{ site.baseurl }}{% post_url 2018-03-27-deep-neural-networks %}#convolutional-neural-networks) (CNNs), whose first layers, pre-trained on a large dataset, learn general features such as edges and lines, while the layers added later learn to detect the objects of a specific task. Fine-tuning is cheap: every result of the paper takes at most 1 hour on one Cloud TPU. The paper fine-tunes BERT for four families of tasks:
1. **Sentence pair classification** (a): the output at `[CLS]` feeds a classifier, as in pre-training, for example to find whether a sentence entails, contradicts or is neutral to another one (MNLI), or whether two questions have the same meaning (QQP).
2. **Single sentence classification** (b): the same for a single sentence, for example to find the sentiment of a sentence from a movie review (SST-2), or to detect hate speech.
3. **Question answering** (c): the input holds a question and a paragraph that contains the answer. Two learned vectors score each output token of the paragraph as the start or the end of the answer, with a softmax over the positions, and the span with the highest score is the answer. For example:
	- input: "[CLS] How old is John [SEP] His name is John. John is a Swiss carpenter and he is 23 years old";
	- output: the span "23 years old".
4. **Single sentence tagging** (d): the output at each token feeds a classifier over tags, as in [named-entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) (CoNLL-2003), where B-PER marks the first token of the name of a person, and O a token outside any named entity. For example:
	- input: "[CLS] His name is John. John is a Swiss carpenter and he is 23 years old";
	- output: B-PER for both tokens "John", B-MISC (miscellaneous entity, which includes nationalities) for "Swiss", and O for the other tokens.

{: .picture}
<img width="99.7%" height="99.7%" src="{{ site.assets }}/learning-from-sequences/bert_tasks.svg"/>

{: .caption}
Input and output of the fine-tuning phase of a BERT network, applied to four types of tasks: (a) sentence pair classification (MNLI, QQP, QNLI, STS-B, MRPC, RTE, SWAG), (b) single sentence classification (SST-2, CoLA), (c) question answering (SQuAD v1.1) and (d) single sentence tagging (CoNLL-2003 NER). Adapted from: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

BERT has no decoder, so it does not generate text, and the information that it encodes is useful but not sufficient on its own for translation. It can, however, initialize the encoder of a translation model: according to [Clinchant et al. (2019)](https://arxiv.org/abs/1909.12744), "BERT pretraining allows for a better initialization point for [an] NMT model", where NMT stands for neural machine translation. Fine-tuning such an encoder gained 0.6 BLEU points on WMT 2014 English-to-German, while a frozen BERT encoder lost 3.7 points. Text generation is the role of decoder-only transformers, such as GPT, which keep the masked decoder without cross-attention, and generate text one token at a time.
