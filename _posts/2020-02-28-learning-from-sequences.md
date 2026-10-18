---
layout: post
title:  "Learning from sequences with Encoder-Decoder, Transformers and BERT"
categories: ["ML Architectures"]
tags: ["Transformer", "attention", "encoder-decoder", "BERT"]
---

Translation and other language tasks map a sequence of words to another sequence. Recurrent encoder-decoders process the words one at a time. Transformers process all the words of a sequence at once, with matrix products that run in parallel. BERT pre-trains the encoder of a transformer once, and fine-tunes it for many tasks.

## Encoder-Decoder

The encoder-decoder, or sequence-to-sequence model, maps an input sequence to an output sequence of a different length, e.g. a sentence to its translation. It was introduced with [recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNNs) by [Cho et al. (2014)](https://arxiv.org/abs/1406.1078) and in [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) (Sutskever et al., 2014). An RNN processes one word per step. At step $$t$$, it concatenates its hidden state of step $$t-1$$, of size $$d_h$$, with the embedding of word $$t$$, of size $$d$$, and computes a new hidden state and an output:

{: .picture}
<img width="33.8%" height="33.8%" src="{{ site.assets }}/learning-from-sequences/encoder_decoder.svg"/>

{: .caption}
The recurrent neural network trained at each step of the encoder-decoder architecture. Its input layer is the concatenation of the hidden layer of the previous iteration (green) and the embedding of the current input word (blue). The hidden layer (green) holds the states of the GRU or LSTM neurons, and the output layer (red) holds the model output. 

The hidden state carries the information of the previous words. A plain RNN learns long-range dependencies poorly, because the gradients through many steps vanish ([vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)). [Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory) (LSTM) and [gated recurrent unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit) (GRU) cells use gates that keep the information over many steps. An LSTM cell has two states, the hidden state and the cell state, and four weight matrices. A GRU, introduced by Cho et al., has one state and three weight matrices, so a step costs about 3/4 of an LSTM step of the same size.

Training has an encoding and a decoding phase:
- The **encoder** runs the RNN over the input words. Its outputs are discarded. Its last hidden state summarizes the input sentence and becomes the initial hidden state of the decoder.
- The **decoder**, a second RNN, starts from a beginning-of-sentence token (BOS) and predicts one word per step, with a softmax over the target vocabulary. Each step takes the previous word of the target sentence as input (teacher forcing), and the last step predicts the end-of-sentence token (EOS). The loss is the sum of the cross-entropies of the predicted words, and [backpropagation through time](https://en.wikipedia.org/wiki/Backpropagation_through_time) computes its gradient through all the steps of both RNNs.

{: .picture}
<img width="96%" height="96%" src="{{ site.assets }}/learning-from-sequences/encoder_decoder_2.svg"/>

{: .caption}
Training workflow of an encoder-decoder architecture that learns the translation of the English sentence "Hello World." into the French sentence "Bonjour le monde.".

At inference, the decoder feeds each predicted word back as the input of the next step, until it outputs EOS. The recurrence has two limits:
- The encoder compresses the whole input into one hidden state of fixed size, so the accuracy drops for long sentences. The **attention mechanism** of [Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) lets each decoder step use the hidden states of all encoder steps.
- Step $$t$$ needs the hidden state of step $$t-1$$, so the steps of a sequence run one after the other. Parallelism comes only from the batch and from the matrix products within a step, which are small for a small model.

The [PyTorch tutorial "NLP from scratch: translation with a sequence to sequence network and attention"](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) implements the encoder-decoder with and without attention.

## Transformer

The [transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)), introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), replaces the recurrence with attention, so that all the positions of a sequence are processed at once. Its encoder (left) and its decoder (right) are stacks of 6 layers ($$N \times$$ in the figure):

{: .picture}
<img width="43.8%" height="43.8%" src="{{ site.assets }}/learning-from-sequences/transformer.svg"/>

{: .caption}
The transformer architecture. Grey regions represent one encoder layer (left) and one decoder layer (right), each repeated N times. Adapted from: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, arXiv)</a>

[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) implements the model in PyTorch, and the [GPT-lite post]({{ site.baseurl }}{% post_url 2023-02-28-gptlite %}) implements a decoder-only transformer.

### Embeddings and Positional Encoding

Each token is mapped to an embedding vector of size $$d$$, with $$d = 512$$ in the base model. Self-attention is permutation-equivariant: permuting the input tokens permutes the outputs in the same way, so without positional information the model cannot use the order of the words. The transformer adds a **positional encoding** of size $$d$$ to each embedding. For the position $$pos$$ and the dimensions $$2i$$ and $$2i+1$$:

$$
PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i/d}} \right), \qquad PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i/d}} \right)
$$

Each pair of dimensions holds a sine and a cosine of the same frequency, and the wavelengths grow geometrically from $$2\pi$$ to $$10000 \cdot 2\pi$$. For any offset $$k$$, $$PE_{pos+k}$$ is a linear function of $$PE_{pos}$$, a rotation of each pair, which makes relative positions easy to learn. The figure shows the dimensions 4 to 7 for $$d = 20$$:

{: .picture}
<img width="70.1%" height="70.1%" src="{{ site.assets }}/learning-from-sequences/transformer-embedding.svg"/>

{: .caption}
Output of the positional encoding of the transformer for dimensions 4 to 7 of an embedding array of size $$d=20$$, for a word whose position in the sentence is given by the x axis. Adapted from: <a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">The Annotated Transformer</a>

### Attention

Self-attention computes how relevant each word of a sentence is to every other word. For the sentence "The big red dog" ($$n = 4$$ words), it gives a $$4 \times 4$$ matrix of **attention weights**, whose rows sum to 1:

{: .picture}
<img width="33.3%" height="33.3%" src="{{ site.assets }}/learning-from-sequences/transformer_attention.svg"/>

{: .caption}
Attention weights of one attention head for the sentence "The big red dog", of length $$N=4$$. Each row holds the softmax-normalized relevance of that word to every word of the sentence, itself included, and sums to 1.

An attention head computes these weights from queries and keys, and averages values with them. The $$N^Q$$ queries, the $$N^{KV}$$ keys and the $$N^{KV}$$ values are the rows of the matrices $$Q \in \mathbb{R}^{N^Q \times D^{QK}}$$, $$K \in \mathbb{R}^{N^{KV} \times D^{QK}}$$ and $$V \in \mathbb{R}^{N^{KV} \times D^V}$$:

$$
\mathrm{Attention}(Q, K, V) = A \, V, \qquad A = \mathrm{softmax}\left( \frac{Q K^T}{\sqrt{D^{QK}}} \right)
$$

The softmax normalizes each row, so the weight of query $$q$$ on key $$k$$ is

$$
A_{q,k} = \frac{\exp\left( Q_q^T K_k / \sqrt{D^{QK}} \right)}{\sum_l \exp\left( Q_q^T K_l / \sqrt{D^{QK}} \right)}
$$

and the output of query $$q$$ is the average of the values weighted by these weights, $$Y_q = \sum_k A_{q,k} V_k$$, a vector of size $$D^V$$. The output $$AV$$ has the size $$N^Q \times D^V$$. The division by $$\sqrt{D^{QK}}$$ keeps the scores in the range where the softmax has useful gradients: for independent query and key components with zero mean and unit variance, the dot product $$Q_q^T K_k$$ has the variance $$D^{QK}$$.

{: .picture}
<img width="33.6%" height="33.6%" src="{{ site.assets }}/learning-from-sequences/transformer_attention_lbdl.svg"/>

{: .caption}
The attention operator matches every query $$Q_q$$ with all keys $$K_1, ..., K_{N^{KV}}$$ to obtain normalized attention scores $$A_{q,1},...,A_{q,N^{KV}}$$ (left), and then averages the values $$V_1,...,V_{N^{KV}}$$ with these scores to compute $$Y_q$$ (right). Adapted from: [The Little Book of Deep Learning](https://fleuret.org/public/lbdl.pdf).

In self-attention, the queries, the keys and the values come from the same sequence, and $$N^Q = N^{KV} = n$$: the attention sublayers of the encoder read the source sentence, and the first attention sublayer of each decoder layer reads the target sentence. In **cross-attention**, the second attention sublayer of each decoder layer, the queries come from the decoder, and the keys and the values from the output of the encoder: this is where the decoder reads the source sentence.

**Multi-head attention** runs $$h$$ heads in parallel, each on its own learned projections, and concatenates their outputs:

$$
\mathrm{MHA}(Q, K, V) = \left[ \mathrm{head}_1, \dots, \mathrm{head}_h \right] W^O, \qquad \mathrm{head}_i = \mathrm{Attention}\left( Q W_i^Q, K W_i^K, V W_i^V \right)
$$

The learned parameters are the projections $$W_i^Q$$, $$W_i^K$$ and $$W_i^V$$ of each head and the output projection $$W^O$$. Each head can learn a different relation between words. The base model has $$h = 8$$ heads with $$D^{QK} = D^V = d / h = 64$$, so the 8 heads cost about as much as one head of size $$d$$.

**Masked multi-head attention** in the decoder sets the scores of the future positions, above the diagonal of $$Q K^T$$, to $$-\infty$$ before the softmax, so that their weights are zero. The decoder input is the target sentence shifted right by one position, with a start token in the first position, so the output at position $$t$$ predicts word $$t$$ from the words before it. In training, the mask lets the decoder process all the positions of the target sentence at once. At inference, the decoder generates one word per step and appends it to its input. For the output "Le gros chien rouge", the attention weights of the four steps are:

{: .picture}
<img width="30.5%" height="30.5%" src="{{ site.assets }}/learning-from-sequences/transformer_attention_masked.svg"/>

{: .caption}
Attention weights of the masked attention of the decoder for the sentence "Le gros chien rouge". The algorithm performs four iterations, one per word. Attention is computed for every iterated word. Masking sets the attention scores of unseen words to $$-\infty$$ before the softmax, which gives them an attention weight of zero. Each vector sums to 1.

### Feed-Forward, Residual and Output Layers

- Each layer ends with a **feed-forward network** that is applied to each position separately: $$\mathrm{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$, with an inner dimension of $$4d = 2048$$.
- Each attention and feed-forward sublayer has a residual connection followed by [layer normalization](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization), $$\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$$ ("Add & Norm" in the figure).
- A **linear** layer maps the last output of the decoder to one score per word of the target vocabulary, and a **softmax** turns the scores into probabilities. The paper picks the output words with a [beam search](https://en.wikipedia.org/wiki/Beam_search) of size 4.

### Computational Complexity

The paper compares the cost of one layer for a sequence of $$n$$ tokens of size $$d$$:

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|:--|:-:|:-:|:-:|
| **Self-Attention** | $$O(n^2 \cdot d)$$ | $$O(1)$$ | $$O(1)$$ |
| Recurrent | $$O(n \cdot d^2)$$ | $$O(n)$$ | $$O(n)$$ |
| Convolutional | $$O(k \cdot n \cdot d^2)$$ | $$O(1)$$ | $$O(\log_k(n))$$ |
| **Self-Attention** (restricted) | $$O(r \cdot n \cdot d)$$ | $$O(1)$$ | $$O(n/r)$$ |

{: .caption}
Computational complexity of four different learning models. Key: $$n$$: sequence length, $$d$$: representation dim., $$k$$: kernel size, $$r$$: size of neighbourhood. Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, arXiv)</a>

At each of its $$n$$ steps, a recurrent layer multiplies a $$d \times d$$ weight matrix by a state vector: $$O(n d^2)$$ operations in $$n$$ sequential steps. Self-attention multiplies $$n \times d$$ matrices: $$O(n^2 d)$$ operations, with no sequential dependency. Self-attention needs fewer operations than a recurrent layer when $$n < d$$, which holds for the sentences of machine translation. The projections and the feed-forward network still cost $$O(n d^2)$$ per layer, so the main gain is parallelism: training computes all the positions of a sentence with matrix-matrix products, which GPUs run efficiently, instead of $$n$$ dependent steps. The path between two positions also has the length $$O(1)$$ instead of $$O(n)$$, which makes long-range dependencies easier to learn. The big model ($$d = 1024$$, $$h = 16$$) reached a [BLEU](https://en.wikipedia.org/wiki/BLEU) score of 28.4 on the WMT 2014 English-to-German task, after 3.5 days of training on 8 P100 GPUs.

## BERT: Bidirectional Encoder Representations from Transformers

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019) keeps only the encoder of the transformer. Its stack of encoder layers computes a representation of each token from the whole sentence, on both sides of the token (bidirectional):

{: .picture}
<img width="95.3%" height="95.3%" src="{{ site.assets }}/learning-from-sequences/bert.svg"/>

{: .caption}
The BERT model, as a stack of Transformer encoders. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Training has two phases: pre-training on unlabeled text, then fine-tuning on each task.

### Pre-training

Pre-training solves two self-supervised tasks at once:
1. **Masked language model**: 15% of the input tokens are selected at random. Of these, 80% are replaced by a `[MASK]` token, 10% by a random token, and 10% stay unchanged, because the `[MASK]` token never appears in fine-tuning. The model predicts the original tokens at the selected positions, and the loss ignores the other positions.
2. **Next sentence prediction**: the input holds two sentences, and the model predicts whether the second sentence follows the first in the text (half of the pairs) or is a random sentence (the other half).

Two training examples, with the class of the second task in place of the `[CLS]` token of the output:
```
Input = [CLS] the man went to [MASK] store [SEP] He bought a gallon [MASK] milk [SEP]
Output = [IsNext] the man went to the store [SEP] He bought a gallon of milk [SEP]

Input = [CLS] the man [MASK] to the store [SEP] penguins [MASK] to jump [SEP]
Output = [NotNext] the man went to the store [SEP] penguins like to jump [SEP]
```

The output at the `[CLS]` position feeds the classifier of the second task, and the outputs at the masked positions predict the masked words. The input of the model is the sum of three embeddings per token. Unlike the transformer, BERT learns its position embeddings:

{: .picture}
<img width="90.7%" height="90.7%" src="{{ site.assets }}/learning-from-sequences/bert_input.svg"/>

{: .caption}
Input of the BERT model. Position embeddings are learned, one vector per position (up to 512), unlike the fixed sine and cosine encoding of the transformer described above. Segment embeddings flag each word as part of the first or second sentence. Token embeddings are the text embeddings of the input data. The input to the model is the sum of these three embeddings. Adapted from: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

BERT-base has 12 layers, $$d = 768$$ and 12 heads (110M parameters), and BERT-large has 24 layers, $$d = 1024$$ and 16 heads (340M parameters). Both were pre-trained on BooksCorpus (800M words) and English Wikipedia (2,500M words), for 1M steps with batches of 256 sequences: 4 days on 16 TPU chips for BERT-base, and on 64 TPU chips for BERT-large. The cost of attention grows with $$n^2$$, so 90% of the steps use sequences of 128 tokens, and the last 10% use 512 tokens to learn the position embeddings.

### Fine-tuning

Fine-tuning adds an output layer for the task, and trains all the parameters on labeled data, starting from the pre-trained weights, as in the [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) of CNNs. It is cheap: each result of the paper takes at most 1 hour on one Cloud TPU. The paper fine-tunes BERT for four families of tasks:
1. **Sentence pair classification**, e.g. whether a sentence entails, contradicts or is neutral to another one (MNLI): the output at `[CLS]` feeds a classifier.
2. **Single sentence classification**, e.g. the sentiment of a sentence (SST-2): the output at `[CLS]` feeds a classifier.
3. **Question answering** (SQuAD): the input holds a question and a paragraph that contains the answer. Two learned vectors score each output token of the paragraph as the start or the end of the answer, with a softmax over the positions. For example:
	- input: "[CLS] How old is John [SEP] His name is John. John is a Swiss carpenter and he is 23 years old";
	- output: the span "23 years old".
4. **Single sentence tagging**, e.g. [named-entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) (CoNLL-2003): the output at each token feeds a classifier over the tags, such as B-PER for the first token of the name of a person, and O for a token outside any named entity.

{: .picture}
<img width="99.7%" height="99.7%" src="{{ site.assets }}/learning-from-sequences/bert_tasks.svg"/>

{: .caption}
Input and output of the fine-tuning phase of a BERT network, applied to four types of tasks: (a) sentence pair classification (MNLI, QQP, QNLI, STS-B, MRPC, RTE, SWAG), (b) single sentence classification (SST-2, CoLA), (c) question answering (SQuAD v1.1) and (d) single sentence tagging (CoNLL-2003 NER). Adapted from: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

BERT has no decoder, so it does not generate text. It can initialize the encoder of a translation model: [Clinchant et al. (2019)](https://arxiv.org/abs/1909.12744) gain up to 0.6 BLEU points on WMT 2014 English-to-German by fine-tuning such an encoder, while a frozen BERT encoder loses accuracy. Decoder-only transformers, such as GPT, keep the masked decoder without cross-attention, and generate text one token at a time.
