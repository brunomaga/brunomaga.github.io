---
layout: post
title:  "Learning from sequences with Encoder-Decoder, Transformers and BERT"
categories: ["ML Architectures"]
tags: ["Transformer", "attention", "encoder-decoder", "BERT"]
---

This post covers alternative methods for ML training on sequential data, with an emphasis on efficiency and parallelism.

## Encoder-Decoder and Sequence-to-Sequence

The Encoder-Decoder (original paper [Sequence to Sequence Learning with Neural Networks (Google, arXiv)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)) is a model that learns an encoding and a decoding task applied to two sequences. It is trained on a sequence-to-sequence task, such as the translation of a sentence from a source language to a target language. The learning mechanism is a two-phase recurrent algorithm, one phase for the encoder and one for the decoder, where each phase is a sequence of iterations over a recurrent neural network.

Its Recurrent Neural Network (RNN) has the following structure:
- Words are passed as input, represented by an embedding of dimensionality $$d$$;
- The neurons are not stateless (as is a regular neuron with an activation function), but have an internal state. Two common examples are [Long Short-Term Memory neurons (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) and [Gated Recurrent Units (GRU)](https://en.wikipedia.org/wiki/Gated_recurrent_unit). The set of state variables of the neurons in a single layer is the layer's *hidden state*. This example, as most applications, uses GRU, since it has a single state variable per neuron (versus two in an LSTM), which makes it faster to train;
- The network has three fully connected layers:
	- an output layer of size $$d$$, whose output is the embedding of the predicted/ground-truth word;
	- a single hidden layer of size $$h$$, referring to the hidden space of the current iteration;
	- an input layer of length $$h+d$$ (for both GRU and LSTM neurons; the LSTM cell state is not part of the input), holding the concatenation of the previous iteration's hidden space (described next) and the embedding of the current input word. On the first iteration, the previous hidden space is initialized randomly or with a zero vector;

This structure is represented in the following picture:

{: style="text-align:center;"}
<img width="32%" height="32%" src="{{ site.assets }}/learning-from-sequences/encoder_decoder.svg"/>

{: style="text-align:center; font-size: small;"}
The recurrent neural network trained at each step of the encoder-decoder architecture. The blue area represents input neurons. The green area represents a single hidden layer formed by the states of the GRU/LSTM neurons of the RNN. The red area represents the model output. The model input is the concatenation of the hidden layer of the previous iteration and the input of the current iteration. 

Training has an encoding and a decoding phase:
- During encoding, the RNN processes all words of the input sequence iteratively, and reuses (concatenates) the hidden state of each iteration as input to the next one. The output of the RNN is discarded, as only the hidden states are trained. When all words are processed, the hidden state of the RNN is passed as input to the first iteration of the decoder;
- On its first step, the decoder concatenates the last hidden space of the encoder with the *Beginning of Sentence (BOS)* flag (user-defined, usually a vector of only zeros or ones). The output is the first predicted word in the target language. All following steps are similar: the output and hidden state of a step are passed as input to the next one, and the model is trained on the loss of the output (translated word). The output of the last iteration is the *End of Sentence (EOS)* flag, built similarly to *BOS*. The RNN is trained at every iteration, as a regular Deep Neural Network.

This algorithm is illustrated below:

{: style="text-align:center;"}
<img width="90%" height="90%" src="{{ site.assets }}/learning-from-sequences/encoder_decoder_2.svg"/>

{: style="text-align:center; font-size: small;"}
Training workflow of an encoder-decoder architecture that learns the translation of the English sentence "Hello World." into the French sentence "Bonjour le monde.".

Two remarks about the Encoder-Decoder architecture:
- Once the network is trained, a new sentence is translated by running decoder iterations until the *EOS* flag is output;
- The **Attention Mechanism** improves results by using the hidden space of every encoder iteration (not only the last) in the decoding steps, which increases the model's capabilities (original paper: [Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473));

These details are omitted here. The [PyTorch tutorial "NLP from scratch: translation with a sequence to sequence network and attention"](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) covers the implementation of regular and attention-based encoder-decoders.

Accurate results for long sentences are hard to obtain, because recurrent neural networks rely on recurrent steps, and information tends to get *lost* after many iterations. Also, acceleration and scaling are limited, since recurrent iterations cannot be parallelized: each iteration depends on the output of the previous one. The only option is to parallelize the model itself across its parameter/activation dimensions, which gives no speedup on *small* models.

## Transformer 

In 2017, researchers at Google introduced the Transformer (original paper [Attention is all you need (2017, Google, arXiv)](https://arxiv.org/abs/1706.03762)), which overcomes many of these issues and achieves better results. The transformer architecture is:

{: style="text-align:center;"}
<img width="35%" height="35%" src="{{ site.assets }}/learning-from-sequences/transformer.svg"/>

{: style="text-align:center; font-size: small;"}
The transformer architecture. Grey regions represent the Encoder (left) and Decoder (right) architectures.
Adapted from: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, arXiv)</a>


The left- and right-hand side components are the encoder and decoder, respectively.
The next sections describe these components, omitting implementation details to focus on computational complexity. For the implementation, see [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

### Word and Positional Embedding

The first relevant unit of the transformer is the embedding unit combined with the positional encoding (red boxes and circles in the previous picture). The transformer has no recurrence or convolution, so it needs a **positional encoder** to learn the context of a sequence from the order of its words. Without it, the model sees the input as a set of values, not as a sequence, and inputs with swapped tokens yield the same output. According to the paper, the positional encoding of a word at position $$pos$$ of a sentence, at each dimension of an embedding of size $$d$$, is:

- $$PE_{(pos,2i)} = sin\left(\frac{pos}{10000^{2i/d}}\right)$$ for an even dimension $$2i$$, and
- $$PE_{(pos,2i+1)} = cos\left(\frac{pos}{10000^{2i/d}}\right)$$ otherwise.

The positional encoding is thus given by $$sine$$ and $$cosine$$ waves with a different frequency and offset for each dimension. As an example, for a word at position $$pos$$ (x axis), the values at dimensions $$4$$, $$5$$, $$6$$ and $$7$$ are:


{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/learning-from-sequences/transformer-embedding.svg"/>

{: style="text-align:center; font-size: small;"}
Output of the positional encoding of the transformer for dimensions 4 to 7 of the embedding array, for a word whose position in the sentence is given by the x axis. Adapted from: <a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">The Annotated Transformer</a>

### Attention Mechanism

The main component of the transformer is the attention mechanism, which determines how the words in the input and output sentences interact. It *learns* the relationship between words, i.e. the relevant information in each context (hence the name *Attention* mechanism). The transformer includes several of these mechanisms, executed in parallel, so that the model learns multiple relevant aspects of the input. This **Multi-head Attention Mechanism** answers, for each of its $$h$$ heads, the question *which part of the input is relevant?*

Consider a single attention head first. For a sentence of $$N$$ words, how is each word related to every other word in that sentence? The output of a single attention mechanism is the $$N \times N$$ matrix that stores these inter-word importance metrics. An example is shown below:

{: style="text-align:center; font-size: small;"}
<img width="32%" height="32%" src="{{ site.assets }}/learning-from-sequences/transformer_attention.svg"/>

{: style="text-align:center; font-size: small;"}
Output of the attention mechanism. For the sentence "The big red dog", of length $$N=4$$, each row of the attention matrix holds the normalized relevance of that word to every other word in the sentence. Source: unknown. 

According to the paper, each attention head is formulated as:

$$
Attention(K, V, Q) = A \, V = softmax\left( \frac{QK^T}{\sqrt{D^{QK}}} \right) V
$$

Where:
- $$Q$$ is a tensor of queries of size $$N^Q×D^{QK}$$
- $$K$$ is a tensor of keys of size $$N^{KV}×D^{QK}$$, and
- $$V$$ is a tensor of values of size $$N^{KV}×D^V$$, thus
- $$Attention(K,V,Q)$$ is of dimension $$N^Q×D^V$$.  

The tensor $$A$$ holds the **attention scores**, where $$A_{q,k}$$ is computed for every query index $$q$$ and key index $$k$$ as the softmax (also called *softargmax*) of the **dot products** between the query $$Q_q$$ and the keys:

$$
A_{q,k} = \frac{     \exp \left( \frac{1}{\sqrt{D^{QK}} } Q^{\intercal}_q K_k  \right)     }{   \sum_l  \exp \left( \frac{1}{\sqrt{D^{QK}}}   Q^{\intercal}_q K_l  \right)  } 
$$

The term $$ \frac{1}{\sqrt{D^{QK}}} $$ keeps the range of values roughly unchanged, even for large $$D^{QK}$$. For each query, the final value is the sum of the input values weighted by the attention scores: $$Y_q = \sum_k A_{q,k} \, V_k $$. 

{: style="text-align:center;"}
<img width="50%" height="50%" src="{{ site.assets }}/learning-from-sequences/transformer_attention_lbdl.svg"/>

{: style="text-align:center; font-size: small;"}
The attention operator matches every query $$Q_q$$ with all keys $$K_1, ..., K_{N^{KV}}$$ to obtain normalized attention scores $$A_{q,1},...,A_{q,N^{KV}}$$ (left), and then averages the values $$V_1,...,V_{N^{KV}}$$ with these scores to compute $$Y_q$$ (right). Adapted from: [The Little Book of Deep Learning]({{ site.resources_permalink }}). 


In the transformer diagram, each attention module has three inputs (arrows), one for each of these variables. The attention mechanism at the start of the encoder and of the decoder takes keys, values and queries from its own input, and learns the context of the source and target languages, respectively. In one occurrence of the attention mechanism, however, the keys and values come from the encoder and the queries from the decoder: this module is trained for the translation at hand. 


Finally, the multi-head attention mechanism combines all attention modules, and is defined as:

$$
MHA(K, V, Q) = [head_1, ..., head_h]W^{MHA}   \,\text{ and }\,  head_i = Attention(KW^K_i, VW^V_i, QW^Q_i)
$$

It is a concatenation of all attention heads. The learnt parameters are the weights of the key ($$W^K_i$$), value ($$W^V_i$$) and query ($$W^Q_i$$) spaces of each head $$i$$, and a final transformation $$W^{MHA}$$ of the multi-head concatenation. In terms of computational complexity, the attention mechanism of the encoder is trained on the complete input sequence at once (as in the "The big red dog" example), instead of looping through all words of the input sequence. Therefore, **the attention mechanism replaces the recurrent (RNN) iterations of an encoder with a set of matrix-vector multiplications**.


The **Masked Multi-head Attention** component of the decoder is similar to the regular MHA, but sets the upper triangle of the attention score matrix to $$-\infty$$ before the softmax, to hide the next words from the model. After the softmax, these positions have an attention weight of zero. Decoding is performed one word of the output sequence at a time: previously seen words are added to the attention array, and the following words receive zero attention weight. For the previous example, the four iterations are: 

{: style="text-align:center; font-size: small;"}
<img width="27%" height="27%" src="{{ site.assets }}/learning-from-sequences/transformer_attention_masked.svg"/>

{: style="text-align:center; font-size: small;"}
Input of the masked attention mechanism of the decoder for the sentence "Le gros chien rouge". The algorithm performs four iterations, one per word. Attention is computed for every iterated word. Masking sets the attention scores of unseen words to $$-\infty$$ before the softmax, which gives them an attention weight of zero. Source: unknown.


### Other components

The other components of the transformer are not specific to it, and were used before in other machine learning models:
- The *Feed Forward* is a regressor (single hidden-layer DNN) that transforms the attention vectors into a valid input for the decoder or for the next computation phase;
- The *Linear* transformation of the decoder expands the space into an array of size equal to the target vocabulary (French in the example); 
- The *Softmax* transforms the output of the previous layer into a probability distribution. The word with the highest probability is the output;


### Computational Complexity

Besides the large reduction in the number of iterations on the encoder side, the authors compare the computational complexity of four different models:

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|:--|:-:|:-:|:-:|
| **Self-Attention** | $$O(n^2 \cdot d)$$ | $$O(1)$$ | $$O(1)$$ |
| Recurrent | $$O(n \cdot d^2)$$ | $$O(n)$$ | $$O(n)$$ |
| Convolutional | $$O(k \cdot n \cdot d^2)$$ | $$O(1)$$ | $$O(log_k(n))$$ |
| **Self-Attention** (restricted) | $$O(r \cdot n \cdot d)$$ | $$O(1)$$ | $$O(n/r)$$ |

{: style="text-align:center; font-size: small;"}
Computational complexity of four different learning models. Key: $$n$$: sequence length, $$d$$: representation dim., $$k$$: kernel size, $$r$$: size of neighbourhood. Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, arXiv)</a>


For the RNN of the Sequence-to-Sequence model above, the number of operations is in the order of $$d^2$$ multiplications (multiplication of weights in a fully-connected layer of a DNN) for each of the $$n$$ words in the sentence, therefore $$O(n d^2)$$. The self-attention mechanism of the Transformer performs several operations on the attention matrix ($$n^2$$) and on the key and query vectors of embedding size $$d$$, leading to a complexity of $$O(n^2 d)$$. 

The claim is that $$O(n^2 d)$$ is better than $$O(n d^2)$$. This seems counter-intuitive, since in most Machine Learning problems the input size is much larger than the dimensionality of the embedding. However, here $$n$$ is the number of words in a sentence (in the order of $$n \approx 70$$) and $$d$$ is the size of the embedding ($$d=512$$ in the original Transformer, and $$d=1024$$ in its big variant).

In summary, the encoder-decoder architecture of the transformer allows faster non-recurrent training on input sequences and faster training on output sentences, which makes it much more efficient than the sequence-to-sequence approaches described initially.

## BERT: Bidirectional Encoder Representation from Transformers

The main rationale of the Transformer's Encoder-Decoder is that:
- The encoder learns the context of the input language (English in the previous example);
- The decoder learns the task of mapping the input to the output language (the English-to-French translation);

The encoder is trained efficiently and learns a *context*. This raises the question: can the Encoder's context alone be used to learn complex tasks? This led to the BERT model (original paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Google AI](https://arxiv.org/abs/1810.04805)). BERT is a stack of Transformer encoders that learns language contexts and performs interpretation tasks.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="{{ site.assets }}/learning-from-sequences/bert.svg"/>

{: style="text-align:center; font-size: small;"}
The BERT model, as a stack of Transformer encoders.

Training has two phases: a pre-training phase learns the contexts of the language, and a fine-tuning phase adapts the trained model to the task being solved.

### Pre-training

The pre-training phase solves two self-supervised prediction tasks simultaneously:
1. Masked language model: given a sentence with some words replaced by a flag (masked), the model is trained to output the same sentence with those words in place;
2. Next sentence prediction: given two sentences, the model is trained to predict whether the second sentence follows the first;

Two training examples are:
```
Input = [CLS] the man went to [MASK] store [SEP] He bought a gallon [MASK] milk [SEP]
Output = [IsNext] the man went to the store [SEP] He bought a gallon of milk [SEP]

Input = [CLS] the man [MASK] to the store [SEP] penguins [MASK] to jump [SEP]
Output = [NotNext] the man went to the store [SEP] penguins like to jump [SEP]
```

The Yes/No flag of the second task is passed as the first embedded word of the output. The layout of the input data is:

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/learning-from-sequences/bert_input.svg"/>

{: style="text-align:center; font-size: small;"}
Input of the BERT model. Position embeddings are similar to those of the transformer described above. Segment embeddings flag each word as part of the first or second sentence. Token embeddings are the text embeddings of the input data. The input to the model is the sum of these three embeddings. Adapted from: <a href="https://arxiv.org/abs/1810.04805">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018, Google, arXiv)</a>

The authors trained this model on BooksCorpus (800M words) and English Wikipedia (2,500M words), with 12 (BERT-base) or 24 (BERT-large) layers and batches of 256 sequences. Sequences have 128 tokens for 90% of the training steps and 512 tokens for the remaining 10%.

### Fine-tuning

The fine-tuning phase adds an extra layer to the pre-trained BERT model, to train it for the task at hand. This approach was used before in other models, particularly in Convolutional Neural Nets, where the first layers are pre-trained and learn edges and lines, and the remaining layers are added later and trained for object-specific detection.

In the original paper, the authors successfully applied this model to four families of tasks:
1. Sentence Pair Classification Tasks: classification of context from pairs of sentences, e.g. whether the sentences agree or disagree, whether the second sentence follows the first, or whether they describe a similar context. As in the pre-training phase, the label is passed as the first embedded symbol of the output; 
2. Single Sentence Classification Tasks: similar to the previous, but applied to a single sentence, e.g. to detect hate speech;
3. Question Answering Tasks: the input is the concatenation of a question and a paragraph that contains the answer. The output is the input with the embeddings of the start and end words of the answer replaced by a marker. As an example:
	- input:  "[CLS] How old is John [SEP] His name is John. John is a Swiss carpenter and he is 23 years old";
	- output: "[CLS] How old is John [SEP] His name is John. John is a Swiss carpenter and ###he is 23 years old###";
4. Single Sentence Tagging Tasks: retrieve the classes of individual words, e.g. Name, Location, Job, etc., by replacing each relevant word with its class id. As an example:
	- input: "[CLS] His name is John. John is a Swiss carpenter and he is 23 years old";
	- output: "[CLS] His name is [NAME]. [NAME] is a [NATIONALITY] [OCCUPATION] and he is [AGE] years old".

The inputs and outputs of the fine-tuning for these tasks are illustrated below: 

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/learning-from-sequences/bert_tasks.svg"/>

{: style="text-align:center; font-size: small;"}
Input and output of the fine-tuning phase of a BERT network, applied to four different interpretation tasks.
Adapted from: <a href="https://arxiv.org/abs/1810.04805">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018, Google, arXiv)</a>

Finally, the information encoded by BERT is useful but insufficient on its own for a translation task. However, "BERT pre-training allows for a better initialization point for [a] Neural Machine Translation model" (source: [Clinchant et al., On the use of BERT for Neural Machine Translation, arXiv](https://arxiv.org/abs/1909.12744)).
