---
layout: post
title:  "Learning from sequences: Encoder-Decoder, Transformers and BERT"
categories: [machine learning, supercomputing]
tags: [machinelearning]
---

In this post, we will look into alternative methods for ML training on sequential data, with an emphasis on efficiency and parallelism.

## Encoder-Decoder and Sequence-to-Sequence

The Encoder-Decoder (original paper [Sequence to Sequence Learning with Neural Networks (Google, arXiv)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)) is a model that learns an encoding and a decoding task applied to two sequences, i.e. it trains for a sequence-to-sequence task such as the translation of a sentence from a source language to a target language. The learning mechanism is a two-phase recurrent algorithm, for the encoder and the decoder respectively, where each phase is a sequence of iterations over a recurrent neural network.

The structure of its Recurrent Neural Network (RNN) is as follows:
- Words are passed as part of the input, represented by an embedding of dimensionality $$d$$;
- The neurons used in the model are not stateless (as is a regular neuron with an activation function), but have an internal state. Two common examples are [Long Short-Term Memory neurons (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) and [Gated Recurrent Units (GRU)](https://en.wikipedia.org/wiki/Gated_recurrent_unit). The set of state variables of the neurons in a single layer is the layer's *hidden state*. In this example, as in most applications, we'll focus on GRU, as it includes a single state variable in each neuron (versus two in its LSTM counterpart), making it faster to train;
- The network has three fully connected layers:
	- an output layer of size $$d$$, whose output value is the embedding of the predicted/ground-truth word;
	- a single hidden layer of size $$h$$, referring to the hidden space of the current iteration;
	- an input layer of length $$h+d$$ (when using GRU neurons) or $$2h+d$$ (LSTM), referring to the concatenation of the previous iteration's hidden space (we'll cover this next) and the embedding of the current input word. On the first iteration, the previous hidden space is initialized randomly or with a zero vector;

This structure is easily represented by the following picture:

{: style="text-align:center;"}
<img width="45%" height="45%" src="{{ site.assets }}/learning-from-sequences/encoder_decoder.png"/>

{: style="text-align:center; font-size: small;"}
The recurrent neural network trained at each step of the encoder-decoder architecture. The blue area represents input neurons. The green area represents a single hidden layer formed by the states of the GRU/LSTM neurons of the RNN. The red area represents the model output. The concatenation of the hidden layer of the previous iteration and the current iteration's input is used as model input. 

The training consists of an encoding and a decoding phase:
- During encoding, the RNN processes all the words of the input sequence iteratively, and reuses (concatenates) the hidden state of an iteration as input to the next one. The output of the RNN is discarded, as we are simply training the hidden states. When all words have been processed, the hidden state of the RNN is passed as input to the first iteration of the decoder;
- On its first step, the decoder takes the last hidden space of the encoder and concatenates it with the *Beginning of Sentence (BOS)* flag (user-defined, usually a vector of only zeros or ones). The output is the first predicted word in the target language. All following steps are similar: the output and hidden state of a step are passed as input to the next one, and the model is trained based on the loss on the output (translated word). The last iteration's output is the *End of Sentence (EOS)* flag, built similarly to *BOS*. The RNN is trained at every iteration, as in a regular Deep Neural Network.

This algorithm can be illustrated as:

{: style="text-align:center;"}
<img width="90%" height="90%" src="{{ site.assets }}/learning-from-sequences/encoder_decoder_2.png"/>

{: style="text-align:center; font-size: small;"}
The training workflow of an encoder-decoder architecture, learning the translation of the English sentence "Hello World." into the French sentence "Bonjour le monde.".

Two relevant remarks about the Encoder-Decoder architecture:
- Once the network is trained, a new sentence is translated by running decoder iterations until the *EOS* flag is output;
- An improvement based on the concept of an **Attention Mechanism** delivers better results by using the hidden space of every encoder iteration (not just the last) in the decoding steps, to increase the model's capabilities (original paper: [Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473));

For the sake of brevity, we will omit these details and refer you to the [PyTorch tutorial "NLP from scratch: translation with a sequence to sequence network and attention"](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) if you are curious about the implementation details of regular and attention-based encoder-decoders.

Note that accurate results for long sentences are hard to obtain, as recurrent neural networks rely on recurrent steps, and information tends to get *lost* after many iterations. Also, acceleration and scaling are limited, as we can't parallelize recurrent iterations: each iteration depends on the output of the previous one. Therefore, the only option is to parallelize the model itself across its parameter/activation dimensions, which leads to no speedup on *small* models.

## Transformer 

In 2017, researchers at Google introduced the Transformer (original paper [Attention is all you need (2017, Google, arXiv)](https://arxiv.org/abs/1706.03762)), overcoming many of the previous issues while demonstrating better results. The transformer architecture is the following:

{: style="text-align:center;"}
<img width="35%" height="35%" src="{{ site.assets }}/learning-from-sequences/transformer.png"/>

{: style="text-align:center; font-size: small;"}
The transformer architecture. Grey regions represent the Encoder (left) and Decoder (right) architectures.
Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, arXiv)</a>


The left- and right-hand side components refer to the encoder and decoder, respectively.
We will describe these components in the next sections, and omit the implementation details to focus on computational complexity. If you're curious about its implementation, have a look at [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

### Word and Positional Embedding

The first unit of importance in the transformer is the embedding unit combined with the positional encoding (red boxes and circles in the previous picture). The transformer model has no recurrence or convolution, so we need a **positional encoder** to learn the context of a sequence based on the order of its words. Without it, the model can only learn from the input as a set of values, not as a sequence, and inputs with swapped tokens would yield the same output. According to the paper, the positional encoding of a word at position $$pos$$ of a sentence, at each dimension of an embedding of size $$d$$, is:

- $$PE_{(pos,2i)} = sin\left(\frac{pos}{10000^{2i/d}}\right)$$ for an even dimension $$2i$$, and
- $$PE_{(pos,2i+1)} = cos\left(\frac{pos}{10000^{2i/d}}\right)$$ otherwise.

In practice, the positional encoding is given by $$sine$$ and $$cosine$$ waves with a different frequency and offset for each dimension. As an example, for a word at position $$pos$$ (x axis), the values at dimensions $$4$$, $$5$$, $$6$$ and $$7$$ are:


{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/learning-from-sequences/transformer-embedding.png"/>

The output of the positional encoding in the transformer architecture, for dimensions 4 to 7 of the embedding array, for a word whose position in the sentence is given by the x axis. Source: <a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">The Annotated Transformer</a>

### Attention Mechanism

The main component of the transformer is the attention mechanism, which determines how the words in the input and output sentences interact. In brief, it's the component that *learns* the relationship between words in such a way that it learns the relevant bits of information in each context (hence the name *Attention* mechanism). The transformer architecture includes not one, but several of these mechanisms, executed in parallel, to allow the model to learn multiple relevant aspects of the input. This **Multi-head Attention Mechanism** answers, for each of its $$h$$ heads, *What part of the input should I focus on?*

Let's look at a single attention head for now. Take a sentence of $$N$$ words: how is each word related to every other word in that sentence? The output of a single attention mechanism is then the $$N \times N$$ matrix storing these inter-word importance metrics. Here's an example:

{: style="text-align:center; font-size: small;"}
<img width="40%" height="40%" src="{{ site.assets }}/learning-from-sequences/transformer_attention.svg"/>

{: style="text-align:center; font-size: small;"}
The attention mechanism output. For the sentence "The big red dog", of length $$N=4$$, each row of the attention matrix holds the normalized relevance metric of that word to every other word in the sentence. Source: unknown. 

How does this mechanism work? According to the paper, each attention head is formulated as:

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

The term $$ \frac{1}{\sqrt{D^{QK}}} $$ helps keep the range of values roughly unchanged even for large $$D^{QK}$$. For each query, the final value is computed as the sum of the input values weighted by the attention scores: $$Y_q = \sum_k A_{q,k} \, V_k $$. 

{: style="text-align:center;"}
<img width="50%" height="50%" src="{{ site.assets }}/learning-from-sequences/transformer_attention_lbdl.png"/>

{: style="text-align:center; font-size: small;"}
The attention operator can be interpreted as matching every query $$Q_q$$ with all the keys $$K_1, ..., K_{N^{KV}}$$ to get normalized attention scores $$A_{q,1},...,A_{q,N^{KV}}$$ (left), and then averaging the values $$V_1,...,V_{N^{KV}}$$ with these scores to compute the resulting $$Y_q$$ (right). Source: [The Little Book of Deep Learning]({{ site.resources_permalink }}). 


Notice that in the transformer diagram, each attention module has three inputs (arrows) referring to these three variables. The attention mechanism at the start of each encoder and decoder retrieves keys, values and queries from its own input, learning the context of the source and target languages, respectively. However, in one occurrence of the attention mechanism, the keys and values are provided by the encoder and the queries by the decoder: this is the module that is trained for the translation at hand. 


Finally, the multi-head attention mechanism combines all attention modules, and is defined as:

$$
MHA(K, V, Q) = [head_1, ..., head_h]W^{MHA}   \,\text{ and }\,  head_i = Attention(KW^K_i, VW^V_i, QW^Q_i)
$$

I.e. it's a concatenation of all attention heads, and the parameters learnt are the weights of the key ($$W^K_i$$), value ($$W^V_i$$) and query ($$W^Q_i$$) spaces of each head $$i$$, and a final transformation of the multi-head concatenation $$W^{MHA}$$. In terms of computational complexity, the attention mechanism on the encoder is trained on the complete input sequence at once (as illustrated in the "The big red dog" example), instead of looping through all words in the input sequence. Therefore, **the attention mechanism replaces the recurrent (RNN) iterations of an encoder with a set of matrix-vector multiplications**.


The **Masked Multi-head Attention** component on the decoder is similar to the regular MHA, but replaces the upper triangle of the attention matrix with zeros, to hide the next words from the model. Decoding is performed one word of the output sequence at a time, with previously seen words added to the attention array and the following words set to zero. Applied to the previous example, the four iterations are: 

{: style="text-align:center; font-size: small;"}
<img width="30%" height="30%" src="{{ site.assets }}/learning-from-sequences/transformer_attention_masked.png"/>

{: style="text-align:center; font-size: small;"}
Input of the masked attention mechanism on the decoder for the sentence "Le gros chien rouge". The algorithm performs four iterations, one per word. Attention is computed for every word iterated. The mask component of the attention mechanism refers to replacing the positions of unseen words in the attention matrix with zeros. Source: unknown.


### Other components

The other components of the transformer are not unique to it, and have been used previously in other machine learning models:
- The *Feed Forward* is a regressor (single hidden-layer DNN) that transforms the attention vectors into a form that is valid as input to the decoder or to the next computation phase;
- The *Linear* transformation component on the decoder expands the space into an array of size equal to the target vocabulary (French in the example); 
- The *Softmax* operation transforms the output of the previous layer into a probability distribution. The word with the highest probability is picked as output;


### Computational Complexity

Besides the great reduction in the number of iterations on the encoder side, the authors compare the computational complexity of four different models:

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/learning-from-sequences/transformer_table.png"/>

{: style="text-align:center; font-size: small;"}
Comparative table of the computational complexity of four different learning models. Key: $$n$$: sequence length, $$d$$: representation dim., $$k$$: kernel size, $$r$$: size of neighbourhood. Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, arXiv)</a>


For the RNN used in the previous Sequence-to-Sequence mechanism, the number of operations performed is in the order of $$d^2$$ multiplications (multiplication of weights in a fully-connected layer of a DNN) for each of the $$n$$ words in the sentence, therefore $$O(n d^2)$$. In the self-attention mechanism that we discussed in the Transformer, we have several operations on the attention matrix ($$n^2$$) and the key and query vectors of embedding size $$d$$, therefore yielding a complexity of $$O(n^2 d)$$. 

The claim is that $$O(n^2 d)$$ is better than $$O(n d^2)$$. This sounds illogical in most Machine Learning problems, as the input size is typically way larger than the dimensionality of the embedding. However, remember that $$n$$ here is the number of words in a sentence (in the order of $$n \approx 70$$) and $$d$$ is the size of the embedding ($$d \approx 2000$$).

To summarize, the encoder-decoder architecture of the transformer allows a faster non-recurrent training of input sequences, and a faster training of output sentences, making it much more efficient than the sequence-to-sequence approaches discussed initially.

## BERT: Bidirectional Encoder Representation from Transformers

Note: the original BERT paper is also detailed in the <a href="{{ site.publications_permalink }}">publications bookmark</a> section.

Going back to the previous section, the main rationale of the Transformer's Encoder-Decoder is that:
- The encoder learns the context of the input language (English in the previous example);
- The decoder learns the task of mapping the input to the output language (the English-to-French translation);

So, the encoder is trained efficiently and learns a *context*. The main question is then "Can we use only the Encoder's context and learn complex tasks?". This led to the introduction of the BERT model (original paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Google AI](https://arxiv.org/abs/1810.04805)). BERT is a stack of Transformer encoders that learns language contexts and performs interpretation tasks.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="{{ site.assets }}/learning-from-sequences/bert.png"/>

{: style="text-align:center; font-size: small;"}
The BERT model, as a stack of Transformer encoders.

The training is performed in two phases: a pre-training phase learns the contexts of the language, and a fine-tuning phase adapts the trained model to the task being solved. Let's start with the pre-training.

### Pre-training

The pre-training phase is based on the simultaneous resolution of two self-supervised prediction tasks:
1. Masked language model: given a sentence with words replaced by a flag (masked), train the model against the same sentence with those words in place;
2. Next sentence prediction: given two sentences, train the model to guess whether the second sentence follows from the first or not;

Here are two training examples:
```
Input = [CLS] the man went to [MASK] store [SEP] He bought a gallon [MASK] milk [SEP]
Output = [IsNext] the man went to the store [SEP] He bought a gallon of milk [SEP]

Input = [CLS] the man [MASK] to the store [SEP] penguins [MASK] to jump [SEP]
Output = [NotNext] the man went to the store [SEP] penguins like to jump [SEP]
```

Note that the Yes/No flag related to the second task is passed as the first embedded word in the output. The layout of the input data is:

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/learning-from-sequences/bert_input.png"/>

{: style="text-align:center; font-size: small;"}
The input of the BERT model. Position embeddings are similar to those of the transformer model, discussed above. Segment embeddings flag each word as part of the first or second sentence. Token embeddings are the text embeddings of the input data. The datapoint input to the model is the sum of these three embeddings. Source: <a href="https://arxiv.org/abs/1810.04805">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018, Google, arXiv)</a>

As a side note, the authors trained this model on BooksCorpus (800M words) and English Wikipedia (2,500M words), using 24 BERT layers and batches of 256 sequences of 512 tokens each.

### Fine-tuning

The fine-tuning phase adds an extra layer to the pre-trained BERT model, in order to train the model for the task at hand. This approach has been applied previously in other models, particularly in Convolutional Neural Nets, where the first layers are pre-trained and learn edges and lines, and the remaining layers are added later and trained for the object-specific detection.

In the original paper, the authors successfully applied this model to four families of tasks:
1. Sentence Pair Classification Tasks: classification of context from pairs of sentences. Examples: do sentences agree/disagree, does the second sentence follow from the previous one, do they describe a similar context, etc. Similarly to the pre-training phase of the BERT models, the label is passed as the first embedded symbol of the output; 
2. Single Sentence Classification Tasks: similar to the previous, but applied to a single sentence. Used for learning contexts like detecting hate speech, etc.;
3. Question Answering Tasks: allows one to pass as input the concatenation of a question and a paragraph that contains the answer to the question. The output of the model is the input with the embeddings representing the start and end words of the answer replaced by a marker. As an example:
	- input:  "[CLS] His name is John. John is a Swiss carpenter and he is 23 years old [SEP] How old is John";
	- output: "[CLS] His name is John. John is a Swiss carpenter and ###he is 23 years old### [SEP] How old is John";
4. Single Sentence Tagging Tasks: retrieves the classes of individual words, e.g. Name, Location, Job, etc., by replacing each relevant word with its class id. As an example:
	- input: "[CLS] His name is John. John is a Swiss carpenter and he is 23 years old";
	- output: "[CLS] His name is [NAME]. [NAME] is a [NATIONALITY] [OCCUPATION] and he is [AGE] years old".

The inputs and outputs of the fine-tuning for these tasks are illustrated in the following picture: 

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/learning-from-sequences/bert_tasks.png"/>

{: style="text-align:center; font-size: small;"}
Input and output of the fine-tuning phase of a BERT network, applied to four different interpretation tasks.
Source: <a href="https://arxiv.org/abs/1810.04805">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018, Google, arXiv)</a>

As a final note, the information encoded by BERT is useful but, on its own, insufficient to perform a translation task. However, "BERT pre-training allows for a better initialization point for [a] Neural Machine Translation model" (source: [Clinchant et al., On the use of BERT for Neural Machine Translation, arXiv](https://arxiv.org/abs/1909.12744)). 
