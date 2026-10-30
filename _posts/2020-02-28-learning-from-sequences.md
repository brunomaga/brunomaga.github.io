---
layout: post
title:  "Learning from sequences with Encoder-Decoder, Transformers and BERT"
categories: ["ML Architectures"]
tags: ["Transformer", "attention", "encoder-decoder", "BERT", "RNN", "sequence-to-sequence", "positional encoding", "multi-head attention", "fine-tuning"]
description: "How recurrent encoder-decoders, the Transformer and BERT learn from sequences: the limits of recurrence, positional encoding, scaled dot-product and multi-head attention, their complexity, and BERT's pre-training and fine-tuning."
---

Text, speech and time series are sequences, where the meaning of each element depends on the elements around it. [Recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNNs), proposed in the 1980s, model such data by reading a sequence one element at a time, with a hidden state that keeps the information of the elements read so far. This sequential reading is also their limit: a sentence of 70 words takes 70 steps, and each step waits for the result of the previous one, so the steps cannot run in parallel.

This post looks into alternative methods for ML training on sequential data, with an emphasis on efficiency and parallelism. It covers three models: the Encoder-Decoder, built from RNNs; the Transformer, which replaces the recurrence by attention and processes all the words of a sentence at once; and BERT, a stack of Transformer encoders.

## Encoder-Decoder and Sequence-to-Sequence

The **Encoder-Decoder** was introduced in 2014 in two papers, [Sequence to Sequence Learning with Neural Networks (Google, arXiv)](https://arxiv.org/abs/1409.3215) and [Cho et al., Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, arXiv](https://arxiv.org/abs/1406.1078). It is a model that learns an encoding and a decoding task applied to two sequences, i.e. it trains for a [sequence-to-sequence](https://en.wikipedia.org/wiki/Seq2seq) task such as the translation of a sentence from a source language to a target language. The learning mechanism is a two-phase recurrent algorithm, for the encoder and the decoder respectively, where each phase is a sequence of iterations over an RNN.

The structure of its RNN is as follows:
- Words are passed as part of the input, represented by an [embedding](https://en.wikipedia.org/wiki/Word_embedding) of dimensionality $$d$$;
- The neurons used in the model are not stateless (as is a regular neuron with an activation function), but have an internal state. Two common examples are [Long Short-Term Memory neurons (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) and [Gated Recurrent Units (GRU)](https://en.wikipedia.org/wiki/Gated_recurrent_unit). The set of state variables of the neurons in a single layer is the layer's **hidden state**. In this example, as in many applications, the focus is on GRU, as it includes a single state variable in each neuron (versus two in its LSTM counterpart) and three weight matrices (versus four), making it faster to train;
- The network has three fully connected layers:
	- an input layer of length $$d_h+d$$, holding the concatenation of the hidden state of the previous iteration (*hidden space* $$t-1$$ in the picture below) and the embedding of the current input word. Its length is the same for GRU and LSTM neurons, as the cell state of the LSTM is not part of the input. On the first iteration, the previous hidden state is initialized randomly or with a zero vector;
	- a single hidden layer of size $$d_h$$, holding the hidden state of the current iteration;
	- an output layer of the size of the vocabulary, whose output value, after a [softmax](https://en.wikipedia.org/wiki/Softmax_function), is the probability of each word being the predicted word;

This structure is easily represented by the following picture:

{: .picture}
<img width="33.8%" height="33.8%" src="{{ site.assets }}/learning-from-sequences/encoder_decoder.svg"/>

{: .caption}
The recurrent neural network trained at each step of the encoder-decoder architecture. Its input layer is the concatenation of the hidden layer of the previous iteration (green) and the embedding of the current input word (blue). The hidden layer (green) holds the states of the GRU or LSTM neurons, and the output layer (red) holds the model output. 

The training consists of an encoding and a decoding phase:
- During encoding, the RNN processes all the words of the input sequence iteratively, and reuses (concatenates) the hidden state of an iteration as input to the next one. The output of the RNN is discarded, as only the hidden states are needed. When all words have been processed, the hidden state of the RNN is passed as input to the first iteration of the decoder;
- On its first step, the decoder (a second RNN with the same structure) takes the last hidden state of the encoder and concatenates it with the *Beginning of Sentence (BOS)* flag (a special token added to the vocabulary, with its own embedding). The output is the first predicted word in the target language. All following steps are similar: the output and hidden state of a step are passed as input to the next one, and the model is trained based on the loss on the output (translated word). During training, the ground-truth word is usually passed as input instead of the output, a method called [teacher forcing](https://en.wikipedia.org/wiki/Teacher_forcing). The last iteration's output is the *End of Sentence (EOS)* flag, built similarly to *BOS*. Both RNNs are trained together on the loss of all iterations, as a regular Deep Neural Network unrolled over the iterations ([backpropagation through time](https://en.wikipedia.org/wiki/Backpropagation_through_time)).

This algorithm can be illustrated as follows, where the thick arrow passes the last hidden state of the encoder to the first iteration of the decoder:

{: .picture}
<img width="92.5%" height="92.5%" src="{{ site.assets }}/learning-from-sequences/encoder_decoder_2.svg"/>

{: .caption}
Training workflow of an encoder-decoder architecture that learns the translation of the English sentence "Hello World." into the French sentence "Bonjour le monde.".

Once the network is trained, a new sentence is translated by running decoder iterations until the *EOS* flag is output. As the ground truth is unknown, each iteration takes as input the word predicted by the previous one.

Note that accurate results for long sentences are hard to obtain. The decoder sees the input sentence only through the last hidden state of the encoder, a vector of $$d_h$$ values that is a *bottleneck* for long sentences, and information tends to get *lost* after many iterations. Also, acceleration and scaling are limited, as recurrent iterations cannot be parallelized: each iteration depends on the output of the previous one. Therefore, within a sentence, the only option is to parallelize the model itself across its parameter/activation dimensions, which leads to no speedup on *small* models.

An improvement based on the concept of an **[Attention Mechanism](https://en.wikipedia.org/wiki/Attention_(machine_learning))** addresses the first limit (original paper: [Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)). It delivers better results by using the hidden state of every encoder iteration (not only the last) in the decoding steps, so that each decoding step can look back at the input words it needs. For the sake of brevity, these details are omitted here, and the [PyTorch tutorial "NLP from scratch: translation with a sequence to sequence network and attention"](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) covers the implementation details of regular and attention-based encoder-decoders.

However, attention does not remove the second limit, as the encoder and the decoder still run one iteration per word. The Transformer, described next, removes the recurrence altogether.

## Transformer 

In 2017, researchers at Google introduced the [Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning)) (original paper [Attention is all you need (2017, Google, arXiv)](https://arxiv.org/abs/1706.03762)), overcoming many of the previous issues while demonstrating better results. Its encoder reads all the words of a sentence at once, and its attention relates each word to every other word in a single step. The base model of the paper, trained for 12 hours on one machine with 8 NVIDIA P100 GPUs, translates English to German better than all earlier models, ensembles included, with 27.3 [BLEU](https://en.wikipedia.org/wiki/BLEU) points. Its training cost is a seventh of that of Google's recurrent translation model, GNMT (Table 2 and Section 5.2 in the paper). The transformer architecture is the following:

{: .picture}
<img width="43.9%" height="43.9%" src="{{ site.assets }}/learning-from-sequences/transformer.svg"/>

{: .caption}
The transformer architecture. Grey regions represent one encoder layer (left) and one decoder layer (right), each repeated N times. Adapted from: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, arXiv)</a>

The left- and right-hand side components refer to the encoder and decoder, respectively. Each encoder layer holds a multi-head attention and a feed forward block, and each decoder layer holds a masked multi-head attention, a multi-head attention over the output of the encoder, and a feed forward block. In the paper, both stacks have $$N=6$$ layers, and the embedding has a size of $$d=512$$. The next sections describe these components, and omit the implementation details to focus on computational complexity. The implementation of the Transformer is detailed in [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

### Word and Positional Embedding

The first unit of importance in the transformer is the embedding unit combined with the positional encoding (the red boxes and the circles above them in the previous picture). The transformer model has no recurrence or convolution, so a **positional encoder** is needed to learn the context of a sequence based on the order of its words. Without it, the model sees the input as a set of words, not as a sequence: the attention and feed forward blocks of the encoder are *permutation equivariant*, i.e. swapping two input words only swaps their two outputs. As an example, the word *dog* then gets the same output in "dog bites man" and in "man bites dog". According to the paper, the positional encoding of a word at position $$pos$$ of a sentence, at each dimension of an embedding of size $$d$$, is:

- $$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$ for an even dimension $$2i$$, and
- $$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$ otherwise.

In practice, the positional encoding is given by sine and cosine waves with a different frequency for each pair of dimensions, and a different offset (sine or cosine) for each dimension of a pair. As an example, with an embedding of size $$d=4$$, the four words of "The big red dog", at positions 0 to 3, get the encodings $$[0, 1, 0, 1]$$, $$[0.841, 0.540, 0.010, 1.000]$$, $$[0.909, -0.416, 0.020, 1.000]$$ and $$[0.141, -0.990, 0.030, 1.000]$$. The angle $$pos/10000^{2i/d}$$ of the first pair of dimensions ($$i=0$$) grows by one radian per position, so the first pair repeats every $$2\pi \approx 6.3$$ positions. The angle of the second pair ($$i=1$$) grows by 0.01 radian per position, so the second pair tells apart the positions that the first pair confuses.

For an embedding of size $$d=20$$, the values at dimensions $$4$$, $$5$$, $$6$$ and $$7$$ of a word at position $$pos$$ (x axis) are displayed below, where the second pair of dimensions has a lower frequency than the first:

{: .picture}
<img width="69.2%" height="69.2%" src="{{ site.assets }}/learning-from-sequences/transformer-embedding.svg"/>

{: .caption}
Output of the positional encoding of the transformer for dimensions 4 to 7 of an embedding array of size $$d=20$$, for a word whose position in the sentence is given by the x axis. Adapted from: <a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">The Annotated Transformer</a>

Looking at the picture, dimensions 4 and 5 repeat every 40 positions and dimensions 6 and 7 every 100 positions, as the wavelength of the pair $$i$$ is $$2\pi \cdot 10000^{2i/d}$$. With the $$d=512$$ dimensions of the paper, the wavelengths grow from $$2\pi \approx 6.3$$ positions for the first pair to about 60,600 positions for the last one. Think of each pair as the hand of a clock that turns at its own speed: together, the hands give each position a different pattern.

The authors chose these functions because the encoding of a position shifted by a fixed offset $$k$$ is a linear function of the encoding of the position, so the model may learn to attend by relative positions. In practice, with $$\omega_i = 1/10000^{2i/d}$$, the shift rotates each pair of dimensions by the angle $$\omega_i k$$, as:

$$
\begin{bmatrix} PE_{(pos+k,2i)} \\ PE_{(pos+k,2i+1)} \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} PE_{(pos,2i)} \\ PE_{(pos,2i+1)} \end{bmatrix}
$$

where the [rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix) depends on the offset $$k$$ only, not on the position $$pos$$. Learned positional embeddings gave nearly identical results, but the sinusoids may let the model extrapolate to sentences longer than the ones seen in training (Section 3.5 in the paper).

### Attention Mechanism

The main component of the transformer is the attention mechanism, which determines how the words in the input and output sentences interact. In brief, it is the component that *learns* the relationship between words in such a way that it learns the relevant bits of information in each context (hence the name *Attention* mechanism). The transformer architecture includes not one, but several of these mechanisms, executed in parallel, to allow the model to learn multiple relevant aspects of the input. This **Multi-head Attention Mechanism** answers, for each of its $$h$$ heads, *What part of the input should this head focus on?*

Consider a single attention head for now. Take a sentence of $$n$$ words: how is each word related to every other word in that sentence? A single attention mechanism then computes the $$n \times n$$ matrix storing these inter-word importance metrics. Here is an example:

{: .picture}
<img width="29.0%" height="29.0%" src="{{ site.assets }}/learning-from-sequences/transformer_attention.svg"/>

{: .caption}
Attention weights of one attention head for the sentence "The big red dog", of length $$N=4$$. Each row holds the softmax-normalized relevance of that word to every word of the sentence, itself included, and sums to 1.

The scores of the picture come from three vectors that each word computes from its embedding: a **query**, a **key** and a **value**. The weight matrices that compute them are learned in training (the matrices $$W^Q_i$$, $$W^K_i$$ and $$W^V_i$$ of the multi-head attention below). Think of the query as the question that a word asks of the sentence, the key as the description that each word offers to answer it, and the value as the information that a word passes on. The score of a pair of words is high when the query of the first word matches the key of the second, i.e. when their dot product is large. In the picture, the adjective *red* gives a weight of 0.24 to *dog*, the noun it describes, and only 0.09 and 0.05 to *The* and *big*. The new representation of *red* is then the mix of the values of the four words with the weights of its row: 62% of its own value, 24% of the value of *dog*, 9% of *The* and 5% of *big*.

How does this mechanism work? According to the paper, each attention head computes a **scaled dot-product attention**, formulated as:

$$
Attention(K, V, Q) = A \, V = softmax\left( \frac{QK^T}{\sqrt{D^{QK}}} \right) V
$$

Where:
- $$Q$$ is a tensor of queries of size $$N^Q×D^{QK}$$
- $$K$$ is a tensor of keys of size $$N^{KV}×D^{QK}$$, and
- $$V$$ is a tensor of values of size $$N^{KV}×D^V$$, thus
- $$Attention(K,V,Q)$$ is of dimension $$N^Q×D^V$$.  

The tensor $$A$$ holds the **attention scores**, where $$A_{q,k}$$ is computed for every query index $$q$$ and every key index $$k$$ as the softmax (also called *softargmax*) of the **dot products** between the query $$Q_q$$ and the keys:

$$
A_{q,k} = \frac{     \exp \left( \frac{1}{\sqrt{D^{QK}} } Q^{\intercal}_q K_k  \right)     }{   \sum_l  \exp \left( \frac{1}{\sqrt{D^{QK}}}   Q^{\intercal}_q K_l  \right)  } 
$$

The term $$ \frac{1}{\sqrt{D^{QK}}} $$ helps keep the range of values roughly unchanged even for large $$D^{QK}$$; without it, large dot products would push the softmax into regions where it has extremely small gradients (Section 3.2.1 in the paper). In numbers, if the entries of a query and a key are independent, with mean 0 and variance 1, their dot product has a variance of $$D^{QK}$$. With the $$D^{QK}=64$$ of the paper, its standard deviation is 8. For random queries and keys over the 70 words of a sentence, the largest score then takes 79% of the softmax weight on average, and more than 90% for 43% of the queries. The softmax is then close to a one-hot vector, whose gradients tend to zero as the largest weight tends to 1. Dividing by $$\sqrt{64}=8$$ brings the standard deviation back to 1, and the largest weight down to 10% on average.

For each query, the final value is computed as the sum of the input values weighted by the attention scores: $$Y_q = \sum_k A_{q,k} \, V_k $$. Both steps, the attention scores and the weighted sum, are illustrated in the following picture:

{: .picture}
<img width="33.7%" height="33.7%" src="{{ site.assets }}/learning-from-sequences/transformer_attention_lbdl.svg"/>

{: .caption}
The attention operator matches every query $$Q_q$$ with all keys $$K_1, ..., K_{N^{KV}}$$ to obtain normalized attention scores $$A_{q,1},...,A_{q,N^{KV}}$$ (left), and then averages the values $$V_1,...,V_{N^{KV}}$$ with these scores to compute $$Y_q$$ (right). Adapted from: [The Little Book of Deep Learning](https://fleuret.org/public/lbdl.pdf).

Notice that in the transformer diagram, each attention module has three inputs (arrows) referring to these three variables. The attention module of each encoder layer, and the first one of each decoder layer, retrieve keys, values and queries from their own input (hence the name **self-attention**), learning the context of the source and target languages, respectively, with $$N^Q = N^{KV}$$. However, in the second attention module of each decoder layer, the keys and values are provided by the encoder and the queries by the decoder (the **encoder-decoder attention**, also called **cross-attention**). There, $$N^{KV}$$ is the length of the source sentence and $$N^Q$$ the length of the target sentence. This is the module that is trained for the translation at hand, as it relates the words of the two sentences.

Finally, the multi-head attention mechanism combines all attention modules, and is defined as:

$$
MHA(K, V, Q) = [head_1, ..., head_h]W^{MHA}   \,\text{ and }\,  head_i = Attention(KW^K_i, VW^V_i, QW^Q_i)
$$

I.e. it is a concatenation of all attention heads, and the parameters learned are the weights of the key ($$W^K_i$$), value ($$W^V_i$$) and query ($$W^Q_i$$) spaces of each head $$i$$, and a final transformation of the multi-head concatenation $$W^{MHA}$$. The paper uses $$h=8$$ heads of size $$D^{QK}=D^V=d/h=64$$, so the eight heads together cost about as much as a single head of the full size $$d=512$$ (Section 3.2.2 in the paper). Together, the projections of the eight heads and $$W^{MHA}$$ hold $$4 \times 512 \times 512 \approx 1$$ million parameters per attention module. In terms of computational complexity, the attention mechanism on the encoder is trained on the complete input sequence at once (as illustrated in the "The big red dog" example), instead of looping through all words in the input sequence. Therefore, **the attention mechanism replaces the recurrent (RNN) iterations of an encoder with a set of matrix-matrix multiplications**.

The **Masked Multi-head Attention** component on the decoder is similar to the regular MHA, but replaces the upper triangle of the attention matrix with zeros, to hide the next words from the model (a **causal mask**). To this end, the dot products of the upper triangle are set to $$-\infty$$ before the softmax, as $$\exp(-\infty) = 0$$.

Decoding is performed one word of the output sequence at a time, with previously seen words added to the attention array and the following words set to zero. During training, the whole target sentence is known, so the mask allows the decoder to compute all iterations at once. Applied to "Le gros chien rouge", the French translation of "The big red dog", the four iterations are:

{: .picture}
<img width="30.5%" height="30.5%" src="{{ site.assets }}/learning-from-sequences/transformer_attention_masked.svg"/>

{: .caption}
Attention weights of the masked attention of the decoder for the sentence "Le gros chien rouge". The algorithm performs four iterations, one per word. Attention is computed for every iterated word. Masking sets the attention scores of unseen words to $$-\infty$$ before the softmax, which gives them an attention weight of zero. Each vector sums to 1.

In the picture, *Le* attends only to itself, *gros* to *Le* and to itself (0.1 and 0.9), and only the last word, *rouge*, attends to the whole sentence. Stacked as rows, the four vectors form a $$4 \times 4$$ attention matrix whose upper triangle is zero.

### Feed Forward, Add & Norm, Linear and Softmax

The other components of the transformer are not unique to it, and have been used previously in other machine learning models:
- The *Feed Forward* is a fully connected network with a single hidden layer, $$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$ (Eq. 2 in the paper), where $$x$$ is the vector of one word, $$W_1$$ and $$W_2$$ are matrices of size $$512 \times 2048$$ and $$2048 \times 512$$ with the biases $$b_1$$ and $$b_2$$, and $$\max(0, \cdot)$$ is the [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit) activation. It is applied to each word separately, and transforms the attention vectors into a form that is valid as input to the decoder or to the next computation phase;
- The *Add & Norm* component adds the input of the previous component to its output (a [residual connection](https://en.wikipedia.org/wiki/Residual_neural_network)) and normalizes the result ([layer normalization](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization)), which keeps the training of the stacked layers stable;
- The *Linear* transformation component on the decoder expands the space into an array of size equal to the target vocabulary (French in the example); 
- The *Softmax* operation transforms the output of the previous layer into a probability distribution over the target vocabulary. Picking the word with the highest probability at each step is called *greedy decoding*. The paper uses instead a [beam search](https://en.wikipedia.org/wiki/Beam_search), which keeps the 4 most likely partial translations at each step and outputs the best complete one (Section 6.1 in the paper);

### Computational Complexity

Besides the great reduction in the number of iterations on the encoder side, the authors compare the computational complexity of four different layer types:

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|:--|:-:|:-:|:-:|
| **Self-Attention** | $$O(n^2 \cdot d)$$ | $$O(1)$$ | $$O(1)$$ |
| Recurrent | $$O(n \cdot d^2)$$ | $$O(n)$$ | $$O(n)$$ |
| Convolutional | $$O(k \cdot n \cdot d^2)$$ | $$O(1)$$ | $$O(\log_k(n))$$ |
| **Self-Attention** (restricted) | $$O(r \cdot n \cdot d)$$ | $$O(1)$$ | $$O(n/r)$$ |

{: .caption}
Computational complexity of four different learning models. Key: $$n$$: sequence length, $$d$$: representation dim., $$k$$: kernel size, $$r$$: size of neighbourhood. Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, arXiv)</a>

For the RNN used in the previous Sequence-to-Sequence mechanism, the number of operations is in the order of $$d^2$$ multiplications (multiplication of weights in a fully connected layer) for each of the $$n$$ words in the sentence, therefore $$O(n d^2)$$. In the self-attention mechanism discussed in the Transformer, the attention matrix holds $$n^2$$ dot products of queries and keys of size $$d$$, and the weighted sum of the values takes as many operations, therefore yielding a complexity of $$O(n^2 d)$$.

The claim is that $$O(n^2 d)$$ is better than $$O(n d^2)$$. This sounds illogical in most Machine Learning problems, as the input size is typically much larger than the dimensionality of the embedding. However, $$n$$ here is the number of words in a sentence (in the order of $$n \approx 70$$) and $$d$$ is the size of the embedding ($$d=512$$ in the original Transformer, and $$d=1024$$ in its big variant). Thus, self-attention needs fewer operations than recurrence when the sentence is shorter than the embedding size, as in most translation tasks (Section 4 in the paper).

For $$n=70$$ and $$d=512$$, a self-attention layer takes in the order of $$n^2 d \approx 2.5$$ million operations, and a recurrent layer $$n d^2 \approx 18.4$$ million, 7 times more. The balance flips for long inputs: for a document of $$n=4{,}096$$ words, self-attention needs $$n/d = 8$$ times more operations than recurrence. Note that the table counts the attention only: the projections of the queries, keys and values, and the feed forward block, add $$O(n d^2)$$ operations to each Transformer layer, as the weight matrices do in a recurrent layer. So the parallelism, described next, matters as much as the number of operations.

The two other columns of the table explain the gain in parallelism. A self-attention layer connects all words with a constant number of sequential operations, while a recurrent layer needs one sequential step per word, i.e. 70 steps for a sentence of 70 words. Also, the constant path length between any two words makes long-range dependencies easier to learn (Section 4 in the paper). For very long sequences, the restricted self-attention (last row), where each word only attends to its neighborhood, reduces the complexity at the cost of longer paths between distant words.

To summarize, the encoder-decoder architecture of the transformer allows a faster non-recurrent training of input sequences, and a faster training of output sentences, making it much more efficient than the sequence-to-sequence approaches discussed initially.

## BERT: Bidirectional Encoder Representations from Transformers

Going back to the previous section, the main rationale of the Transformer's Encoder-Decoder is that:
- The encoder learns the context of the input language (English in the previous example);
- The decoder learns the task of mapping the input to the output language (the English-to-French translation);

So, the encoder is trained efficiently and learns a *context*. The main question is then "Can only the Encoder's context be used to learn complex tasks?". This led to the introduction of the BERT model in 2018 (original paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Google AI](https://arxiv.org/abs/1810.04805)).

BERT is a stack of Transformer encoders that learns language contexts and performs interpretation tasks. As the encoder has no mask, the context of each word is learned from the words before and after it (hence *Bidirectional*). The following picture illustrates the model, where each block is one encoder layer of the transformer (the grey region on the left of the transformer picture):

{: .picture}
<img width="83.9%" height="83.9%" src="{{ site.assets }}/learning-from-sequences/bert.svg"/>

{: .caption}
The BERT model, as a stack of Transformer encoders. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

The training is performed in two phases: a pre-training phase learns the contexts of the language, and a [fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) phase adapts the trained model to the task being solved. The pre-training is covered first.

### Pre-training

The pre-training phase is based on the simultaneous resolution of two [self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning) prediction tasks:
1. Masked [language model](https://en.wikipedia.org/wiki/Language_model): given a sentence with words replaced by a flag (masked), the model is trained against the same sentence with those words in place, with a loss computed only on the masked words. The paper picks 15% of the words, replaces 80% of them with the flag and 10% with a random word, and keeps the last 10% unchanged, as the flag never appears in the fine-tuning;
2. Next sentence prediction: given two sentences, the model is trained to guess whether the second sentence follows from the first or not, which is the case for half of the pairs;

Here are two training examples:
```
Input = [CLS] the man went to [MASK] store [SEP] He bought a gallon [MASK] milk [SEP]
Output = [IsNext] the man went to the store [SEP] He bought a gallon of milk [SEP]

Input = [CLS] the man [MASK] to the store [SEP] penguins [MASK] to jump [SEP]
Output = [NotNext] the man went to the store [SEP] penguins like to jump [SEP]
```

The examples use three special tokens, added to the vocabulary: **[CLS]** (classification) starts every input, **[SEP]** (separator) ends each sentence, and **[MASK]** is the flag of a masked word. Note that the IsNext or NotNext label of the second task is predicted from the output at the position of [CLS], which the examples show as the first word of the output.

The layout of the input data is:

{: .picture}
<img width="90.7%" height="90.7%" src="{{ site.assets }}/learning-from-sequences/bert_input.svg"/>

{: .caption}
Input of the BERT model. Position embeddings are learned, one vector per position (up to 512), unlike the fixed sine and cosine encoding of the transformer described above. Segment embeddings flag each word as part of the first or second sentence. Token embeddings are the text embeddings of the input data. The input to the model is the sum of these three embeddings. Adapted from: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

Note that the word *playing* is split into *play* and *##ing*: BERT splits rare words into frequent pieces of a vocabulary of 30,000 *WordPiece* tokens, where ## marks a piece that continues a word.

As a side note, the authors trained this model on BooksCorpus (800M words) and English Wikipedia (2,500M words), with batches of 256 sequences of 512 tokens each, for 1,000,000 steps (about 40 epochs). As attention is quadratic in the sequence length, 90% of the training steps use shorter sequences of 128 tokens, which have 16 times fewer attention scores. The paper trains two sizes: BERT-base, with 12 encoder layers, an embedding size of $$d=768$$ and $$h=12$$ heads, for 110 million parameters, and BERT-large, with 24 layers, $$d=1024$$ and $$h=16$$, for 340 million parameters. Each pre-training took 4 days, on 16 [TPU](https://en.wikipedia.org/wiki/Tensor_Processing_Unit) chips for BERT-base and on 64 for BERT-large (Appendix A.2 in the paper).

### Fine-tuning

The fine-tuning phase adds an extra layer to the pre-trained BERT model, to train the model for the task at hand. This approach, known as [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning), has been applied previously in other models, particularly in [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network). There, the first layers are pre-trained and learn edges and lines, and the remaining layers are added later and trained for the object-specific detection. Fine-tuning is cheap next to the pre-training: every result of the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, from the same pre-trained model (Section 3.2 in the paper).

In the original paper, the authors successfully applied this model to four families of tasks:
1. Sentence Pair Classification Tasks: classification of context from pairs of sentences. Examples: do sentences agree/disagree, does the second sentence follow from the previous one, do they describe a similar context, etc. As in the pre-training, the label is predicted from the output at the position of [CLS] ($$C$$ in the picture below), by the extra layer and a softmax over the classes;
2. Single Sentence Classification Tasks: similar to the previous, but applied to a single sentence. Used for learning contexts like detecting hate speech, or the sentiment of a movie review (the SST-2 task of panel (b) below);
3. Question Answering Tasks: the input is the concatenation of a question and a paragraph that contains the answer to the question. The model predicts, for each word of the paragraph, the probability that the answer starts or ends there. It computes them from the dot product of the output of the word ($$T_i'$$ in the picture below) with a start vector and with an end vector, both learned in the fine-tuning, followed by a softmax over the words of the paragraph. As an example, with the predicted answer between markers:
	- input:  "[CLS] How old is John [SEP] His name is John. John is a Swiss carpenter and he is 23 years old";
	- output: "[CLS] How old is John [SEP] His name is John. John is a Swiss carpenter and he is ###23### years old";
4. Single Sentence Tagging Tasks: retrieves the classes of individual words, e.g. Name, Location, Job, etc. (as in [named-entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)), by predicting the class of each word from its output ($$T_i$$ in the picture below), written here as the class id in place of the word. The picture writes the classes in the [IOB format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)), where B-PER marks the first word of the name of a person and O a word outside any name. As an example:
	- input: "[CLS] His name is John. John is a Swiss carpenter and he is 23 years old";
	- output: "[CLS] His name is [NAME]. [NAME] is a [NATIONALITY] [OCCUPATION] and he is [AGE] years old".

The inputs and outputs of the fine-tuning for these tasks are illustrated in the following picture: 

{: .picture}
<img width="92.2%" height="92.2%" src="{{ site.assets }}/learning-from-sequences/bert_tasks.svg"/>

{: .caption}
Input and output of the fine-tuning phase of a BERT network, applied to four types of tasks: (a) sentence pair classification (MNLI, QQP, QNLI, STS-B, MRPC, RTE, SWAG), (b) single sentence classification (SST-2, CoLA), (c) question answering (SQuAD v1.1) and (d) single sentence tagging (CoNLL-2003 NER). Adapted from: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

As a final note, the information encoded by BERT is useful but, on its own, insufficient to perform a translation task, as BERT has no decoder. However, "BERT pretraining allows for a better initialization point for [a Neural Machine Translation] model" (source: [Clinchant et al., On the use of BERT for Neural Machine Translation, arXiv](https://arxiv.org/abs/1909.12744)).

## Comparison of Encoder-Decoder, Transformer and BERT

The following table sums up the three models, for a sequence of $$n$$ words and an embedding of size $$d$$:

| Model | Architecture and training | Sequential steps per layer | Operations per layer | Context of each word | When to use it |
|---|---|---|---|---|---|
| Encoder-Decoder | Two RNNs, trained on pairs of sentences | $$n$$, one per word | $$O(n d^2)$$ | The words read before it | Inputs read one element at a time: its memory and its cost per word do not grow with the length |
| Transformer | Encoder and decoder stacks of attention, trained on pairs of sentences | 1 in training, one per word in decoding | $$O(n^2 d)$$ in the attention, $$O(n d^2)$$ in the rest | The whole input, and the previous words of the output | Sequence-to-sequence tasks such as translation |
| BERT | Transformer encoders, pre-trained without labels, then fine-tuned per task | 1 | $$O(n^2 d)$$ in the attention, $$O(n d^2)$$ in the rest | The words before and after it | Interpreting a text: classification, question answering and tagging |

To find more about the topic, the [Dive into Deep Learning](https://d2l.ai/) book by Zhang, Lipton, Li and Smola implements the encoder-decoder (chapter 10), the Transformer (chapter 11) and BERT (chapter 15).
