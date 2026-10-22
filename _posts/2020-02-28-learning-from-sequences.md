---
layout: post
title:  "Learning from sequences with Encoder-Decoder, Transformers and BERT"
categories: ["ML Architectures"]
tags: ["Transformer", "attention", "encoder-decoder", "BERT"]
---

Text, speech and time series are sequences, where the meaning of each element depends on the elements around it. [Recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNNs), proposed in the 1980s, model such data by reading a sequence one element at a time, with a hidden state that keeps the information of the elements read so far.

This post looks into alternative methods for ML training on sequential data, with an emphasis on efficiency and parallelism. It covers three models: the Encoder-Decoder, built from RNNs; the Transformer, based on attention instead of recurrence; and BERT, a stack of Transformer encoders.

## Encoder-Decoder and Sequence-to-Sequence

The **Encoder-Decoder** was introduced in 2014 in two papers, [Sequence to Sequence Learning with Neural Networks (Google, arXiv)](https://arxiv.org/abs/1409.3215) and [Cho et al., Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, arXiv](https://arxiv.org/abs/1406.1078). It is a model that learns an encoding and a decoding task applied to two sequences, i.e. it trains for a [sequence-to-sequence](https://en.wikipedia.org/wiki/Seq2seq) task such as the translation of a sentence from a source language to a target language. The learning mechanism is a two-phase recurrent algorithm, for the encoder and the decoder respectively, where each phase is a sequence of iterations over an RNN.

The structure of its RNN is as follows:
- Words are passed as part of the input, represented by an [embedding](https://en.wikipedia.org/wiki/Word_embedding) of dimensionality $$d$$;
- The neurons used in the model are not stateless (as is a regular neuron with an activation function), but have an internal state. Two common examples are [Long Short-Term Memory neurons (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) and [Gated Recurrent Units (GRU)](https://en.wikipedia.org/wiki/Gated_recurrent_unit). The set of state variables of the neurons in a single layer is the layer's **hidden state**. In this example, as in many applications, the focus is on GRU, as it includes a single state variable in each neuron (versus two in its LSTM counterpart) and three weight matrices (versus four), making it faster to train;
- The network has three fully connected layers:
	- an output layer of the size of the vocabulary, whose output value, after a [softmax](https://en.wikipedia.org/wiki/Softmax_function), is the probability of each word being the predicted word;
	- a single hidden layer of size $$d_h$$, referring to the hidden space of the current iteration;
	- an input layer of length $$d_h+d$$ (for both GRU and LSTM neurons, as the cell state of the LSTM is not part of the input), referring to the concatenation of the previous iteration's hidden space (covered next) and the embedding of the current input word. On the first iteration, the previous hidden space is initialized randomly or with a zero vector;

This structure is easily represented by the following picture:

{: .picture}
<img width="33.8%" height="33.8%" src="{{ site.assets }}/learning-from-sequences/encoder_decoder.svg"/>

{: .caption}
The recurrent neural network trained at each step of the encoder-decoder architecture. Its input layer is the concatenation of the hidden layer of the previous iteration (green) and the embedding of the current input word (blue). The hidden layer (green) holds the states of the GRU or LSTM neurons, and the output layer (red) holds the model output. 

The training consists of an encoding and a decoding phase:
- During encoding, the RNN processes all the words of the input sequence iteratively, and reuses (concatenates) the hidden state of an iteration as input to the next one. The output of the RNN is discarded, as only the hidden states are needed. When all words have been processed, the hidden state of the RNN is passed as input to the first iteration of the decoder;
- On its first step, the decoder (a second RNN with the same structure) takes the last hidden space of the encoder and concatenates it with the *Beginning of Sentence (BOS)* flag (a special token added to the vocabulary, with its own embedding). The output is the first predicted word in the target language. All following steps are similar: the output and hidden state of a step are passed as input to the next one, and the model is trained based on the loss on the output (translated word). During training, the ground-truth word is usually passed as input instead of the output, a method called [teacher forcing](https://en.wikipedia.org/wiki/Teacher_forcing). The last iteration's output is the *End of Sentence (EOS)* flag, built similarly to *BOS*. Both RNNs are trained together on the loss of all iterations, as a regular Deep Neural Network unrolled over the iterations ([backpropagation through time](https://en.wikipedia.org/wiki/Backpropagation_through_time)).

This algorithm can be illustrated as follows, where the thick arrow passes the last hidden state of the encoder to the first iteration of the decoder:

{: .picture}
<img width="96.4%" height="96.4%" src="{{ site.assets }}/learning-from-sequences/encoder_decoder_2.svg"/>

{: .caption}
Training workflow of an encoder-decoder architecture that learns the translation of the English sentence "Hello World." into the French sentence "Bonjour le monde.".

Two relevant remarks about the Encoder-Decoder architecture:
- Once the network is trained, a new sentence is translated by running decoder iterations until the *EOS* flag is output;
- An improvement based on the concept of an **[Attention Mechanism](https://en.wikipedia.org/wiki/Attention_(machine_learning))** delivers better results by using the hidden space of every encoder iteration (not only the last) in the decoding steps, to increase the model's capabilities (original paper: [Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473));

For the sake of brevity, these details are omitted here, and the [PyTorch tutorial "NLP from scratch: translation with a sequence to sequence network and attention"](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) covers the implementation details of regular and attention-based encoder-decoders.

Note that accurate results for long sentences are hard to obtain, as recurrent neural networks rely on recurrent steps, and information tends to get *lost* after many iterations. Also, acceleration and scaling are limited, as recurrent iterations cannot be parallelized: each iteration depends on the output of the previous one. Therefore, within a sentence, the only option is to parallelize the model itself across its parameter/activation dimensions, which leads to no speedup on *small* models.

## Transformer 

In 2017, researchers at Google introduced the [Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning)) (original paper [Attention is all you need (2017, Google, arXiv)](https://arxiv.org/abs/1706.03762)), overcoming many of the previous issues while demonstrating better results. The transformer architecture is the following:

{: .picture}
<img width="43.9%" height="43.9%" src="{{ site.assets }}/learning-from-sequences/transformer.svg"/>

{: .caption}
The transformer architecture. Grey regions represent one encoder layer (left) and one decoder layer (right), each repeated N times. Adapted from: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, arXiv)</a>


The left- and right-hand side components refer to the encoder and decoder, respectively.
The next sections describe these components, and omit the implementation details to focus on computational complexity. Its implementation is detailed in [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

### Word and Positional Embedding

The first unit of importance in the transformer is the embedding unit combined with the positional encoding (the red boxes and the circles above them in the previous picture). The transformer model has no recurrence or convolution, so a **positional encoder** is needed to learn the context of a sequence based on the order of its words. Without it, the model can only learn from the input as a set of values, not as a sequence, and inputs with swapped tokens would yield the same output. According to the paper, the positional encoding of a word at position $$pos$$ of a sentence, at each dimension of an embedding of size $$d$$, is:

- $$PE_{(pos,2i)} = sin\left(\frac{pos}{10000^{2i/d}}\right)$$ for an even dimension $$2i$$, and
- $$PE_{(pos,2i+1)} = cos\left(\frac{pos}{10000^{2i/d}}\right)$$ otherwise.

In practice, the positional encoding is given by $$sine$$ and $$cosine$$ waves with a different frequency for each pair of dimensions, and a different offset (sine or cosine) for each dimension of a pair. As an example, for a word at position $$pos$$ (x axis), the values at dimensions $$4$$, $$5$$, $$6$$ and $$7$$ are displayed below, where the second pair of dimensions has a lower frequency than the first:


{: .picture}
<img width="69.2%" height="69.2%" src="{{ site.assets }}/learning-from-sequences/transformer-embedding.svg"/>

{: .caption}
Output of the positional encoding of the transformer for dimensions 4 to 7 of an embedding array of size $$d=20$$, for a word whose position in the sentence is given by the x axis. Adapted from: <a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">The Annotated Transformer</a>

The authors chose these functions because the encoding of a position shifted by a fixed offset is a linear function of the encoding of the position, so the model may learn to attend by relative positions. Learned positional embeddings gave nearly identical results, but the sinusoids may let the model extrapolate to sentences longer than the ones seen in training (Section 3.5 in the paper).

### Attention Mechanism

The main component of the transformer is the attention mechanism, which determines how the words in the input and output sentences interact. In brief, it is the component that *learns* the relationship between words in such a way that it learns the relevant bits of information in each context (hence the name *Attention* mechanism). The transformer architecture includes not one, but several of these mechanisms, executed in parallel, to allow the model to learn multiple relevant aspects of the input. This **Multi-head Attention Mechanism** answers, for each of its $$h$$ heads, *What part of the input should this head focus on?*

Consider a single attention head for now. Take a sentence of $$n$$ words: how is each word related to every other word in that sentence? A single attention mechanism then computes the $$n \times n$$ matrix storing these inter-word importance metrics. Here is an example:

{: .picture}
<img width="33.4%" height="33.4%" src="{{ site.assets }}/learning-from-sequences/transformer_attention.svg"/>

{: .caption}
Attention weights of one attention head for the sentence "The big red dog", of length $$N=4$$. Each row holds the softmax-normalized relevance of that word to every word of the sentence, itself included, and sums to 1.

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

The term $$ \frac{1}{\sqrt{D^{QK}}} $$ helps keep the range of values roughly unchanged even for large $$D^{QK}$$; without it, large dot products would push the softmax into regions where it has extremely small gradients (Section 3.2.1 in the paper). For each query, the final value is computed as the sum of the input values weighted by the attention scores: $$Y_q = \sum_k A_{q,k} \, V_k $$. Both steps, the attention scores and the weighted sum, are illustrated in the following picture:

{: .picture}
<img width="33.7%" height="33.7%" src="{{ site.assets }}/learning-from-sequences/transformer_attention_lbdl.svg"/>

{: .caption}
The attention operator matches every query $$Q_q$$ with all keys $$K_1, ..., K_{N^{KV}}$$ to obtain normalized attention scores $$A_{q,1},...,A_{q,N^{KV}}$$ (left), and then averages the values $$V_1,...,V_{N^{KV}}$$ with these scores to compute $$Y_q$$ (right). Adapted from: [The Little Book of Deep Learning](https://fleuret.org/public/lbdl.pdf).


Notice that in the transformer diagram, each attention module has three inputs (arrows) referring to these three variables. The attention mechanism at the start of each encoder and decoder retrieves keys, values and queries from its own input (hence the name **self-attention**), learning the context of the source and target languages, respectively. However, in one occurrence of the attention mechanism, the keys and values are provided by the encoder and the queries by the decoder (the **encoder-decoder attention**, also called **cross-attention**). This is the module that is trained for the translation at hand. 


Finally, the multi-head attention mechanism combines all attention modules, and is defined as:

$$
MHA(K, V, Q) = [head_1, ..., head_h]W^{MHA}   \,\text{ and }\,  head_i = Attention(KW^K_i, VW^V_i, QW^Q_i)
$$

I.e. it is a concatenation of all attention heads, and the parameters learnt are the weights of the key ($$W^K_i$$), value ($$W^V_i$$) and query ($$W^Q_i$$) spaces of each head $$i$$, and a final transformation of the multi-head concatenation $$W^{MHA}$$. In terms of computational complexity, the attention mechanism on the encoder is trained on the complete input sequence at once (as illustrated in the "The big red dog" example), instead of looping through all words in the input sequence. Therefore, **the attention mechanism replaces the recurrent (RNN) iterations of an encoder with a set of matrix-matrix multiplications**.


The **Masked Multi-head Attention** component on the decoder is similar to the regular MHA, but replaces the upper triangle of the attention matrix with zeros, to hide the next words from the model. To this end, the dot products of the upper triangle are set to $$-\infty$$ before the softmax.

Decoding is performed one word of the output sequence at a time, with previously seen words added to the attention array and the following words set to zero. During training, the whole target sentence is known, so the mask allows the decoder to compute all iterations at once. Applied to the French translation of the previous example, the four iterations are:

{: .picture}
<img width="30.5%" height="30.5%" src="{{ site.assets }}/learning-from-sequences/transformer_attention_masked.svg"/>

{: .caption}
Attention weights of the masked attention of the decoder for the sentence "Le gros chien rouge". The algorithm performs four iterations, one per word. Attention is computed for every iterated word. Masking sets the attention scores of unseen words to $$-\infty$$ before the softmax, which gives them an attention weight of zero. Each vector sums to 1.


### Other components

The other components of the transformer are not unique to it, and have been used previously in other machine learning models:
- The *Feed Forward* is a regressor (single hidden-layer DNN) that transforms the attention vectors into a form that is valid as input to the decoder or to the next computation phase;
- The *Add & Norm* component adds the input of the previous component to its output (a [residual connection](https://en.wikipedia.org/wiki/Residual_neural_network)) and normalizes the result ([layer normalization](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization)), which keeps the training of the stacked layers stable;
- The *Linear* transformation component on the decoder expands the space into an array of size equal to the target vocabulary (French in the example); 
- The *Softmax* operation transforms the output of the previous layer into a probability distribution. The word with the highest probability is picked as output;


### Computational Complexity

Besides the great reduction in the number of iterations on the encoder side, the authors compare the computational complexity of four different models:

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|:--|:-:|:-:|:-:|
| **Self-Attention** | $$O(n^2 \cdot d)$$ | $$O(1)$$ | $$O(1)$$ |
| Recurrent | $$O(n \cdot d^2)$$ | $$O(n)$$ | $$O(n)$$ |
| Convolutional | $$O(k \cdot n \cdot d^2)$$ | $$O(1)$$ | $$O(log_k(n))$$ |
| **Self-Attention** (restricted) | $$O(r \cdot n \cdot d)$$ | $$O(1)$$ | $$O(n/r)$$ |

{: .caption}
Computational complexity of four different learning models. Key: $$n$$: sequence length, $$d$$: representation dim., $$k$$: kernel size, $$r$$: size of neighbourhood. Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, arXiv)</a>


For the RNN used in the previous Sequence-to-Sequence mechanism, the number of operations is in the order of $$d^2$$ multiplications (multiplication of weights in a fully-connected layer) for each of the $$n$$ words in the sentence, therefore $$O(n d^2)$$. In the self-attention mechanism discussed in the Transformer, there are several operations on the attention matrix ($$n^2$$) and the key and query vectors of embedding size $$d$$, therefore yielding a complexity of $$O(n^2 d)$$.

The claim is that $$O(n^2 d)$$ is better than $$O(n d^2)$$. This sounds illogical in most Machine Learning problems, as the input size is typically much larger than the dimensionality of the embedding. However, $$n$$ here is the number of words in a sentence (in the order of $$n \approx 70$$) and $$d$$ is the size of the embedding ($$d=512$$ in the original Transformer, and $$d=1024$$ in its big variant). Thus, self-attention needs fewer operations than recurrence when the sentence is shorter than the embedding size, as in most translation tasks (Section 4 in the paper).

The two other columns of the table explain the gain in parallelism. A self-attention layer connects all words with a constant number of sequential operations, while a recurrent layer needs one sequential step per word. Also, the constant path length between any two words makes long-range dependencies easier to learn (Section 4 in the paper). For very long sequences, the restricted self-attention (last row), where each word only attends to its neighbourhood, reduces the complexity at the cost of longer paths between distant words.

To summarize, the encoder-decoder architecture of the transformer allows a faster non-recurrent training of input sequences, and a faster training of output sentences, making it much more efficient than the sequence-to-sequence approaches discussed initially.

## BERT: Bidirectional Encoder Representations from Transformers

Going back to the previous section, the main rationale of the Transformer's Encoder-Decoder is that:
- The encoder learns the context of the input language (English in the previous example);
- The decoder learns the task of mapping the input to the output language (the English-to-French translation);

So, the encoder is trained efficiently and learns a *context*. The main question is then "Can only the Encoder's context be used to learn complex tasks?". This led to the introduction of the BERT model in 2018 (original paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Google AI](https://arxiv.org/abs/1810.04805)).

BERT is a stack of Transformer encoders that learns language contexts and performs interpretation tasks. As the encoder has no mask, the context of each word is learnt from the words before and after it (hence *Bidirectional*). The following picture illustrates the model, where each block is one encoder layer of the transformer (the grey region on the left of the transformer picture):

{: .picture}
<img width="83.9%" height="83.9%" src="{{ site.assets }}/learning-from-sequences/bert.svg"/>

{: .caption}
The BERT model, as a stack of Transformer encoders. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

The training is performed in two phases: a pre-training phase learns the contexts of the language, and a [fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) phase adapts the trained model to the task being solved. The pre-training is covered first.

### Pre-training

The pre-training phase is based on the simultaneous resolution of two [self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning) prediction tasks:
1. Masked [language model](https://en.wikipedia.org/wiki/Language_model): given a sentence with words replaced by a flag (masked), the model is trained against the same sentence with those words in place, with a loss computed only on the masked words (15% of the words in the paper);
2. Next sentence prediction: given two sentences, the model is trained to guess whether the second sentence follows from the first or not;

Here are two training examples:
```
Input = [CLS] the man went to [MASK] store [SEP] He bought a gallon [MASK] milk [SEP]
Output = [IsNext] the man went to the store [SEP] He bought a gallon of milk [SEP]

Input = [CLS] the man [MASK] to the store [SEP] penguins [MASK] to jump [SEP]
Output = [NotNext] the man went to the store [SEP] penguins like to jump [SEP]
```

Note that the Yes/No flag related to the second task is passed as the first embedded word in the output. The layout of the input data is:

{: .picture}
<img width="90.7%" height="90.7%" src="{{ site.assets }}/learning-from-sequences/bert_input.svg"/>

{: .caption}
Input of the BERT model. Position embeddings are learned, one vector per position (up to 512), unlike the fixed sine and cosine encoding of the transformer described above. Segment embeddings flag each word as part of the first or second sentence. Token embeddings are the text embeddings of the input data. The input to the model is the sum of these three embeddings. Adapted from: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

As a side note, the authors trained this model on BooksCorpus (800M words) and English Wikipedia (2,500M words), using 12 (BERT-base) or 24 (BERT-large) BERT layers and batches of 256 sequences of 512 tokens each. As attention is quadratic in the sequence length, 90% of the training steps use shorter sequences of 128 tokens.

### Fine-tuning

The fine-tuning phase adds an extra layer to the pre-trained BERT model, to train the model for the task at hand. This approach, known as [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning), has been applied previously in other models, particularly in [Convolutional Neural Nets](https://en.wikipedia.org/wiki/Convolutional_neural_network). There, the first layers are pre-trained and learn edges and lines, and the remaining layers are added later and trained for the object-specific detection.

In the original paper, the authors successfully applied this model to four families of tasks:
1. Sentence Pair Classification Tasks: classification of context from pairs of sentences. Examples: do sentences agree/disagree, does the second sentence follow from the previous one, do they describe a similar context, etc. Similarly to the pre-training phase of the BERT models, the label is passed as the first embedded symbol of the output; 
2. Single Sentence Classification Tasks: similar to the previous, but applied to a single sentence. Used for learning contexts like detecting hate speech, etc.;
3. Question Answering Tasks: allows one to pass as input the concatenation of a question and a paragraph that contains the answer to the question. The output of the model is the input with the embeddings representing the start and end words of the answer replaced by a marker. As an example:
	- input:  "[CLS] How old is John [SEP] His name is John. John is a Swiss carpenter and he is 23 years old";
	- output: "[CLS] How old is John [SEP] His name is John. John is a Swiss carpenter and ###he is 23 years old###";
4. Single Sentence Tagging Tasks: retrieves the classes of individual words, e.g. Name, Location, Job, etc. (as in [named-entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)), by replacing each relevant word with its class id. As an example:
	- input: "[CLS] His name is John. John is a Swiss carpenter and he is 23 years old";
	- output: "[CLS] His name is [NAME]. [NAME] is a [NATIONALITY] [OCCUPATION] and he is [AGE] years old".

The inputs and outputs of the fine-tuning for these tasks are illustrated in the following picture: 

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/learning-from-sequences/bert_tasks.svg"/>

{: .caption}
Input and output of the fine-tuning phase of a BERT network, applied to four types of tasks: (a) sentence pair classification (MNLI, QQP, QNLI, STS-B, MRPC, RTE, SWAG), (b) single sentence classification (SST-2, CoLA), (c) question answering (SQuAD v1.1) and (d) single sentence tagging (CoNLL-2003 NER). Adapted from: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

As a final note, the information encoded by BERT is useful but, on its own, insufficient to perform a translation task, as BERT has no decoder. However, "BERT pretraining allows for a better initialization point for [a Neural Machine Translation] model" (source: [Clinchant et al., On the use of BERT for Neural Machine Translation, arXiv](https://arxiv.org/abs/1909.12744)).

To find more about the topic, the [Dive into Deep Learning](https://d2l.ai/) book by Zhang, Lipton, Li and Smola implements the encoder-decoder (chapter 10), the Transformer (chapter 11) and BERT (chapter 15).
