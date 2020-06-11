---
layout: post
title:  "AI Supercomputing II: Limitations, Encoder-Decoder, Transformers and BERT"
categories: [machine learning, supercomputing]
tags: [machinelearning]
---


In our [previous post]({{ site.baseurl }}{% post_url 2020-05-12-AI-Supercomputing %}), we've discuss different techniques and levels of parallelism (model, data, pipeline, CPU offloading) and showed that efficient parallelism (with almost-linear scaling) at scale is achievable in Machine Learning problems. However, recursive models --- such as the ones used in translation and text interpretation tasks --- are not easy to parallelize. In this post we explain why.


# Encoder-Decoder and Sequence-to-Sequence

The Encoder-Decoder (original paper [Sequence to Sequence Learning with Neural Networks, Google, arXiv](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)) is a learning model that learns a encoding and a decoding tasks applied to two sequences, i.e. a sequence-to-sequence task such as the translation of a sentence from a given language to a target language. The learning mechanism is a two-phase recursive algorithm, for the encoder and decoder respectively, where each phase is a sequence of iterations over a recursive neural network.

The structure of the Recursive Deep Neural Network (RNN) is as follows. Words are represented by an embedding of dimensionality $d$. The neuros of the model are not stateless (e.g. like in a regular neuron with an activation function), but have an internal state. Two common examples are [Long Short-Term Memory neurons (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) or [Gated Recurrent Units (GRU)](https://en.wikipedia.org/wiki/Gated_recurrent_unit). As a side note, GRU tends to be used in most cases as it includes a single (versus two on the LSTM) state variables, making training faster. The network has fully-connected network layers:
- an input layer of length $2d$ (when utilizing GRU neurons) or $3d$ (LSTM), referring to the concatenation of the embedding of the previous iteration's hidden space (we'll cover this next), and the embeding of the current word being input;
	- on the first iteration, the input is initialized randomly or with a zero vector;
- a single hiddle layer of size $d$, refering to the hidden space of the current iteration;
- an output layer of size $d$, whose output value is the embedding of the predicted/groundtruth word;

This structure can be represented by the following picture:

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/Encoder_Decoder.png"/><br/>
<br/><small>The recursive deep neural network trained on each step of the encoder-decoder architecture. The blue areas represent input neurons. The green area represent a single hidden layer. The red area represents the model output. Neurons are typically LSTMs or GRUs, and the concatenation of their internal states forms the hidden space used as input on the next iteration
</small>
</p>

The training follows in two phases. During the encoding, the RNN processes iteratively all the words in the input sequence, and reutilizes the hidden state of an iteration as part of the input of the next one. The output of the RNN is discarded, we're simply training the hidden states. When all workds have been processed, the hidden state of the RNN is past as input to the first iteration on the decoder, concatenated with the *Beginning of Sentence (BOS)* flag. The rest follows as: the output sequence is used in iteration $t$ as output and in iteration $t+1$ as input. The last iterations input is the flat *End of String (EOS)*, that may be analogous to *BOS*. Training of the RNN iterations happens at every iterations, as in a normal [Deep Neural Network]({{ site.baseurl }}{% post_url 2018-02-27-Deep-Neural-Networks %}). This algorithm can de illustrated as:

<p align="center">
<br/>
<img width="90%" height="90%" src="/assets/AI-Supercomputing/Encoder_Decoder_2.png"/><br/>
<br/><small>The workflow of an encoder decoder training by learning from the translation of the english sentence "Hello World." to the Frence sentence "Bonjour le monde." .
</small>
</p>

Two relevant remarks about the Encoder-Decoder architecture:
- Once the network is trained, the translation of a new sentence is executed similarly by running encoder iterations until the flag *EOS* is output;
- An improvement based on the concept of **Attention Mechanism** has been introduced  (original paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)) that combines the hidden space of every encoder iteration with the input of the decoding steps, in order to increase the model capatiblities;

For the sake of brevity, we will ommit these details and refer you to the [Pytorch turorial "NLP from scratch: translation with a sequence to sequence network and attention"](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) if you are curious about implementation details. We now go back to the subject of this post and the topic of computational complexity.

Parallelism/scaling/acceleration of such sequence-to-sequence models is an issue. There are three underlying reasons behind it:
- encoding/decoding is a recursive algorithm, and we can't parallelize recursive iterations (each iteration depends on the hidden state of the previous);
- the DNN underlying the RNN has only a single layer with a dimension of $d$, referring to an embedding that usually is of small size. Therefore, it does not *allow* for good parallelism such by dimensionality of pipelining;
- input and output sequences have different lengths, therefore it is very hard to bach sizes per input or output;
	- in practice, some batching is possible by grouping sentences by encoder length size first, and for each encoder, by decoder length size. This is however very inneficient as we require a high number of sentences for every encoder/decoder lengths pair to fully utilize our compute resources. Not impossible, but very unlikely.

# Transformer 

In 2017 the staff at Google introducted the Transformer (original paper [Attention is all you need (2017, Google, Arxiv)](https://arxiv.org/abs/1706.03762)), overcoming many of the previous issues, while providing better results.  The transformer architecture is the following:
<p align="center">
<br/>
<img width="35%" height="35%" src="/assets/AI-Supercomputing/transformer.PNG"/><br/>
<br/><small>The transformer architecture. Grey regions represent the Encoder (left) and Decoder (right) architectures.
<br/>Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, Arxiv)</a>
</small>
</p>

The left and right hand side components refer to the encoder and decoder, specifically.
We will describe these components in the next sections, and ommit the implementation details to focus on computational complexity. If you're curious about implementation, have a look at [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

### Word and Positional Embeddig

The first unit of importance in the transformer is the embedding unit with the positional encoding (red box in the previous picture). The rtransformer model has no recurrence or convolution, so we need a **positional encoder** to give context of order of words in sentence. According to the paper, the embedding of a given word in the position $pos$ in a sentence is an array with size $d$ whose value at each dimension $i$ is:

$$
PE_{(pos,2i)} = sin\left(\frac{pos}{10000^{2i/d}}\right) \,\text{ and }\, PE_{(pos,2i+1)} = cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

In practice, the embedding is given by $sine$ and $cosine$  waves with a different frequency and offset for each dimension. As an example, for a work with positioning $pos$, then the values at positions $4$ to $7$ of the embedding is:

<p align="center">
<br/>
<img width="60%" height="60%" src="/assets/AI-Supercomputing/transformer-embedding.png"/><br/>
<br/><small>The output of the embedding in the transformer architecture, for dimensions 4 to 7 of the embedding array, for a word with a sentence positioning related to the x axis. <br/>Source: <a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">The Annotated Transformer</a>
</small>
</p>

### Attention Mechanism

The main component of the transformer is the attention mechanism. In brief, it's the component that *learns* the relationship between words in such a way that it learns the relevant bits of information on each context (thus the naming *Attention* mechanism). The transformer architecture includes not one, but several of these mechanisms, executed in parallel, to allow the model to learn multiple relevant aspects of the input. This **Multi-head Attention Mechanism** solves for $n$ heads, *What part of the input should I focus on?*

As an example, take the sentence of length $N$ words, how is each word related to every other word on that sentence? The output of the attention mechanism is then the $N \times N$ matrix storing these inter-word importance metric. Here's an example:

<p align="center">
<br/>
<img width="40%" height="40%" src="/assets/AI-Supercomputing/transformer_attention.PNG"/><br/>
<br/><small>The attention mechanism output. For the sentence of length $N=4$ "The big red dog" the output at every row of the attention matrix is how the normalized relevance metric of each word to every other word. Source: unkown.
</small>
</p>

How does this mechanism work? According to the paper, each attention head is formulated as:

$$
Attention(K, V, Q) = softmax\left(QK^T / \sqrt{d_k}\right) V
$$

Where $K$, $V$ and $Q$ refer to the key, value and query. Notice that the attention mechanism has three inputs (arrows in the diagram) referring to these three variables. This is a similar naming to a regular python dictionary and a C++ hash-map or hash-table, as they are all *key-value* stores that are *queried* for a given value (thus the naming and the usage of  $K$, $V$ and $Q$). The technique used and the main component of the attention mechanism is the **scaled dot-product** $QK^T$.  

The dot product of two vectors is computed as $a . b = \mid \mid a \mid \mid \, . \, \mid \mid b \mid \mid \, cos\, \theta_{ab}$, and provides the cosine of the angle between two vectors. Therefore, if two vectors have a simillar representation in space, the angle formed between them is small and its cosine is large. Here's a graphical representation of this logic:


<p align="center">
<br/>
<img width="40%" height="40%" src="/assets/AI-Supercomputing/Transformer-Attention-Mech-dot-product.png"/>
<br/><small>An example of the dot-product method inside the attention mechanism, applied to a mapping of 4 key vectors $K_1-K_4$ and a query vector $Q$. Left: a graphical representation of the query vector and the key vectors. Right: the angle and cosine of the angles formed between query and vectors.</a>
</small>
</p>

Therefore, similar vectors yield large angle-cosine values. These values are then scaled by $\sqrt{d_k}$ by the attention mechanism, introduced for numerical stability (detailed in the paper). The output then is past to a $softmax$ function that normalizes the vector and converts into a probability distribution, and finally multiplied by $V$ to retried the values scaled to the probability distribution of a given key. In practice, this is continuous implementation of the key-value store found in regular programming languages (on a discrete set of values). I.e. instead of outputting the value in a given position in a map related to the query, it outputs the probability of each value to relate to a given query, and retrieves all values and their respective importance, based on that query.

Finally, the multi-head attention architecture determines how the words in input and output sentences interact, and is defined as:

$$
MHA(K, V, Q) = [head_0,.., head_n]W^M   \,\text{ and }\,  head_i = Attention(KW^K_i, VW^V_i, QW^Q_i)
$$

I.e. it's a concatenation of all attention heads, and the trained parameters are the weights of the keys ($W^K_i$), values ($W^V_i$) and query ($W^Q_i$) space of each head $i$, and a final transformation of the multi-head concatenation $W^M$. In terms of computational complexity, the attention mechanism on the encoder is trained on the complete input sequence at once (as illustrated in the "the big red dog" example), instead of looping through all words in the input sequence. Therefore,  **the attention mechanism replaces the recursive (RNN) iterations on an encoder, by a set of matrix-vector multiplications**.


The **Masked Multi-head Attention** on the decoder is similar to the regular MHA, but replaces upper diagonal of attention vector by zeros, to hide next word from the model. Decoding is performed with a word of the sequence of a time, with previously seen word added to the attention array, and attention of the following words set to zero. Applied to the previous example, the four iterations are: 

<p align="center">
<br/>
<img width="30%" height="30%" src="/assets/AI-Supercomputing/transformer_attention_masked.png"/><br/>
<br/><small>Input of the masked attention mechanism for the sentence "Le gros chien rouge". Algorithm performs four iterations, one per work. Attention is computer for every word iterated. The *Mask* component of the attention mechanism refers to the unseen words replaced by zero. Source: unkown.
</small>
</p>

### Other components

The other components on the transformer are not unique, and have been used previously in other machine learning models:
- The *Feed Forward* is a regressor (single hidden-layer DNN) that transforms the attention vectors into a form that is valid as input to the decoder or to the next computation phase;
- The *Linear* transformation component on the decoder expands the space into an array of size equals to the target (French in the example)) vocabulary; 
- The *Softmax* transforms the previous array tranforms into a probability distribution. The word with the highest probability is picked as output;


### Computational Complexity

Besides the great reduction in the number of iterations on the encoder size, the authors inspect the computational complexity of four comparative models:

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/transformer_table.png"/><br/>
<br/><small>Comparative table of computational complexity of four different learning models. Key: $n$: sequence length, $d$: representation dim., $k$: kernel size; $r$: size of neighbourhood.
<br/>Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, Arxiv)</a>
</small>
</p>

For the RNN using in previous Sequence-to-Sequence mechanism, the number of operations performed is in the order of $d^2$ multiplications (multiplication of weights in fully connected layer of DNN) for each of the $n$ words in the sentence, therefore $O(n^2 d)$. On the self-attenion mechanism that we discussed in the Transformer, we have several operations of the attention matrix $n^2$ and the key or query vector of embedding size $d$, therefore yielding a complexity $O(n d^2$). 

The claim is that the $O(n^2 d)$ is better than $O(n d^2)$. This sound illogical as typically the input size is way larger than the dimensionality of the embedding. However, remember that $n$ here is the number of words in a sentence (in the order of $n \approx 70$) and $d$ is the size of the embedding ($d \approx 2000$).

To summarize, the encoder-decoder architecture of the transformer allows a faster non-recursive training of input sequences, and a faster training of output sentences, making it much more efficient than the sequence-to-sequence approach discussed initially.

# BERT: Bidirectional Encoder Representation from Transformers

Attending to the previous topic, the main rationale of the Transformer's Encoder-Decoder is that:
- The encoder learns the context of the input language (English in the example);
- The decoder learns the task of the input-to-output languages (translation in ta  the English-to-French translation;

The encoder is efficiently trained and learns a *context*. So the main question is "Can we get rid of the Decoder and use only the Encoder to learn complex tasks?". This led to the introduction of the BERT models (original paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Google AI](https://arxiv.org/abs/1810.04805)). BERT is a stack of Transformer encoders that Learns language contexts and performs interpretation tasks.


<p align="center">
<br/>
<img width="60%" height="60%" src="/assets/AI-Supercomputing/BERT.PNG"/><br/>
<br/><small>The BERT model, as a stack of Transformer encoders.
</small>
</p>


The training is performed in two phases. A pre-training phase learns the contexts of the language. And a fine-training adapts the trained model to the task being solved. Let's start with the first.

### Pre-training

The pre-training phases is based on the resolution of the two self-supervised prediction tasks, simultaneously:
1. Masked language model: given a sentence with works replaced with a flag (or masked), train it against the same sented with those words in place;
2. Next sentence prediction: Given two sentences, train the model to guess if the second sentence follows from the first or not;


Example of two training examples:
```
Input = [CLS] the man went to [MASK] store [SEP] He bought a gallon [MASK] milk [SEP]
Output = [IsNext] the man went to the store [SEP] He bought a gallon of milk [SEP]

Input = [CLS] the man [MASK] to the store [SEP] penguins [MASK] to jump [SEP]
Output = [NotNext] the man went to the store [SEP] penguins like to jump [SEP]
```

Note that the Yes/No flag to the second task is past as the first embedded word. The layout of the input data is:

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/BERT_input.PNG"/><br/>
<br/><small>The input of the BERT model. Position Emdebbings tell the position of a word in sentence. Segment embeddings flag the sentence as first or seconds sentence. Token embedding are the text-embeddings of the input data. The datapoint being input is the concatenation of these three embeddings.
<br/>Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, Arxiv)</a>
</small>
</p>

As a sinde note, the authors trained this model on BooksCorpus (800M words) and English Wikipedia (2,500M words), using 24 BERT layers, batches of 256 sentences with 512 tokens each.

### Fine-tuning

The fine-tuning phase adds an extra layer to the pre-trained BERT model, in order to fine-tune the model to the task at hand. This approach has been applied previously in other models, particularly on Conv Neural Nets, where the first layers are pre-trained and represent edges and lines, and the remaining layers are used on object-specific detection tasks.

In the original paper, the authors demonstrated this model successfully being fine-tuned to four families of tasks:
1. Sentence Pair Classification Tasks: classifying a pair of sentences as agreeing/disagreeing, following from the previous or, similar content or not, etc. Similarly to the pre-training phase of the BERT models, the label is past as the first embedded symbol of the output; 
2. Single Sentence Classification Tasks: similar to the previous, but applied to a single sentence. Used for learning contexts like finding hateful speech, etc.;
3. Question Answering Tasks: allows one to pass as input the concatenation of the question and the paragraph where that answer is available. Outputs the input with a flag altering the embedding of the first and last word of the answer, in that paragraph. Example: for the input "[CLS] His name is John. John is a Swiss carpenter and he is 23 years old [SEP] How old is John", the output is "[CLS] His name is John. John is a Swiss carpenter and ###he is 23 years old### [SEP] How old is John";
4. Single Sentence Tagging tasks: retrieves the classes of individual words e.g. Name, Location, Job, etc, by tagging each word by it's class id. As an example, for the same input "[CLS] His name is John. John is a Swiss carpenter and he is 23 years old", it would output "[CLS] His name is [NAME]. [NAME] is a [NATIONALITY] [OCCUPATION] and he is [AGE] years old".

The input and output models of the fine-tuning of these tasks are illustrated in the following picture: 

<p align="center">
<br/>
<img width="60%" height="60%" src="/assets/AI-Supercomputing/BERT_tasks.png"/><br/>
<br/><small>Input and output of the fine-tuning phase of a BERT network, applied to four different interpretation tasks.
<br/>Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, Arxiv)</a>
</small>
</p>

As a final note, information encoded by BERT is useful but, on its own, insufficient to perform a translation task (due to no left-to-right prediction). However, "BERT pre-training allows for a better initialization point for [an] Neural Machine Translation model" (source: [Clichant et al.,On the use of BERT for Neural Machine Translation, arXiv](https://arxiv.org/abs/1909.12744)). 


# Microsoft ZeRO and DeepSpeed

This section is being written and will be available soon. For now, here are some references:
- [ZeRO \& DeepSpeed announcement page](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/);
- [Turing-NLG blog post](ttps://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/);
- [www.deepspeed.ai/](https://www.deepspeed.ai/);
- [DeepSpeed github docs](https://github.com/microsoft/DeepSpeed\#further-reading);
- [ZeRO paper](https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/);

[//]: <> **State of the Art**: \pause Bert-large (0.3B), GPT-2 (1.5B), Megatron-LM (8.3B), T5 (11B). ZeRO can run 100B parameters but they didn't, takes longer than a year for training! So they ran 17B.

[//]: <> **ZeRO (Zero Redundancy Optimizer) combines the efforts explained in the AI Supercomputing posts ll**: \textit{"[...] achieves the computation/communication efficiency of DP while achieving memory efficiency of MP, [...] retaining the computational granularity and communication volume of DP using a dynamic communication schedule during training"

[//]: <>[Video](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

[//]: <> ZeRO removes the memory redundancies across data-parallel processes by partitioning the model states—parameters, gradients, and optimizer (Adam) state—across data parallel processes instead of replicating them. 

[//]: <> partitions optimizer states, gradients and parameters; We show that ZeRO can be combined with any model parallelism; We call this ZeRO-powered data parallelism, which allows per-device memory usage to scale linearly with the degree of data parallelism and incurs similar communication volume as data parallelism. 


[//]: <> ### Superlinear speed-up

[//]: <> <p align="center">
[//]: <> <br/>
[//]: <> <img width="45%" height="45%" src="/assets/AI-Supercomputing/ZeRO_superlinear_speedup_60B_parameter.PNG"/><br/>
[//]: <> <br/><small>The transformer  attention mechanism masked.</a>
[//]: <> </small>
[//]: <> </p>

[//]: <> super-linear speedup in the regime of 64-400 GPUs. This is a property of ZeRO-DP which reduces the memory footprint of the model states as we increase the DP degree, allowing us to fit larger batch sizes per GPU"
