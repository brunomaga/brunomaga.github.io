---
layout: post
title:  "AI Supercomputing (part 2): Limitations, Encoder-Decoder, Transformers and BERT"
categories: [machine learning, supercomputing]
tags: [machinelearning]
---


In our [previous post]({{ site.baseurl }}{% post_url 2020-05-12-AI-Supercomputing %}), we discussed different techniques and levels of parallelism (model, data, pipeline, CPU offloading) and showed that efficient parallelism (with almost-linear scaling) at scale is possible in Machine Learning problems. However, recursive models --- such as the ones used in translation and text interpretation tasks --- are not easy to parallelize. In this post we explain why.


## Encoder-Decoder and Sequence-to-Sequence

The Encoder-Decoder (original paper [Sequence to Sequence Learning with Neural Networks (Google, arXiv)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)) is a learning model that learns an encoding and a decoding task applied to two sequences, i.e. it trains for a sequence-to-sequence task such as the translation of a sentence from a given language to a target language. The learning mechanism is a two-phase recursive algorithm, for the encoder and decoder respectively, where each phase is a sequence of iterations over a recursive neural network.

The structure of its Recursive Deep Neural Network (RNN) is as follows:
- Words are past as part of the input, represented by an embedding of dimensionality $d$;
- The neurons used in the model are not stateless (e.g. like in a regular neuron with an activation function), but have an internal state. Two common examples are [Long Short-Term Memory neurons (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) and [Gated Recurrent Units (GRU)](https://en.wikipedia.org/wiki/Gated_recurrent_unit). The set of state variables of the neurons on a single are the layer's *hidden state*. In this example, as in most application, we'll focus on GRU as it includes a single state variable on each neuron (versus two on the LSTM counterpart), making it faster to train;
- The network has three fully connected layers:
	- an output layer of size $h$, whose output value is the embedding of the predicted/groundtruth word;
	- a single hiddle layer of size $d$, refering to the hidden space of the current iteration;
	- an input layer of length $h+d$ (when utilizing GRU neurons) or $2h+d$ (LSTM), referring to the concatenation of the embedding of the previous iteration's hidden space (we'll cover this next), and the embeding of the current word being input. On the first iteration, the input is initialized randomly or with a zero vector;

This structure is easily represented by the following picture:

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/Encoder_Decoder.png"/><br/>
<br/><small>The recursive deep neural network trained on each step of the encoder-decoder architecture. The blue area represent input neurons. The green area represents a single hidden layer formed by the states of the GRU/LSTM neurons of the RNN. The red area represents the model output. The concatenation of the hidden layer of the previous iteration and the current iteration's input are used as model input.
</small>
</p>

The training follows with an encoding and decoding phase:
- During the encoding, the RNN processes all the words on the input sequence iteratively, and reutilizes (concatenates) the hidden state of an iteration as input of the next one. The output of the RNN is discarded, as we are simply training the hidden states. When all words have been processed, the hidden state of the RNN is past as input to the first iteration of the decoder;
- The decoder takes on its first step the last encoder's hidden space and concatenates it with the *Beginning of Sentence (BOS)* flag (user-defined, usually a vector with only zeros or ones). The output is the first predicted word in the target language. All following steps are simillar: the output and hidden state of a single step is past as input to the next one, and the model can then be trained based on the loss on the output (translated word). The last iteration's output is the flag *End of String (EOS)*, built similarly to *BOS*. Training of the RNN iterations happens at every iteration, as in a normal [Deep Neural Network]({{ site.baseurl }}{% post_url 2018-02-27-Deep-Neural-Networks %}).

This algorithm can be illustrated as:

<p align="center">
<br/>
<img width="90%" height="90%" src="/assets/AI-Supercomputing/Encoder_Decoder_2.png"/><br/>
<br/><small>The workflow of an encoder-decoder architecture training by learning the translation of the english sentence "Hello World." to the Frence sentence "Bonjour le monde." .
</small>
</p>

Two relevant remarks about the Encoder-Decoder architecture:
- Once the network is trained, the translation of a new sentence is executed by running encoder iterations until the flag *EOS* is output;
- An improvement based on the concept of **Attention Mechanism** delivers improved results by utilising the hidden space of every encoder iteration (not just the last) on the decoding steps, in order to increase the model capatiblities (original paper: [Bahdanau et al. Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473));

For the sake of brevity, we will ommit these details and refer you to the [Pytorch turorial "NLP from scratch: translation with a sequence to sequence network and attention"](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) if you are curious about the implementation details of regular and attention-based encoder-decoders. Let's go back to the main subject of this post and the topic of computational complexity.

Parallelism, scaling and acceleration of such sequence-to-sequence models is an issue. There are four main reasons that explain this:
- encoding/decoding is a recursive algorithm, and we can't parallelize recursive iterations, as each iteration depends on the hidden state of the previous one;
- the DNN underlying the RNN architecture has only a single layer, therefore model parallelism like pipelining (covered in our [previous post]({{ site.baseurl }}{% post_url 2020-05-12-AI-Supercomputing %})) won't provide any gains;
- the hidden layer is composed of $h$ neurons, and $h$ is usually a value small enought to allow for acceleration at the layer level (e.g. the model parallelism showned in the our [previous post]({{ site.baseurl }}{% post_url 2020-05-12-AI-Supercomputing %});
- input and output sequences have different lengths, and each batch needs to be a set of input and output sentences of similar lenghts, which makes the batches small and inefficient to parallelize. In practice, some batching is possible by grouping sentences first by length of the encoder inputs, and for each encoder, group by length of decoder inputs. This is however very inneficient as we require an extremmly high number of sentences so that all groups of encoder/decoder pairs are large enough to fully utilize the compute resources at every training batch. Not impossible, but very unlikely.

## Transformer 

In 2017 the staff at Google introducted the Transformer (original paper [Attention is all you need (2017, Google, Arxiv)](https://arxiv.org/abs/1706.03762)), overcoming many of the previous issues, while demonstrating better results. The transformer architecture is the following:
<p align="center">
<br/>
<img width="35%" height="35%" src="/assets/AI-Supercomputing/transformer.PNG"/><br/>
<br/><small>The transformer architecture. Grey regions represent the Encoder (left) and Decoder (right) architectures.
<br/>Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, Arxiv)</a>
</small>
</p>

The left and right hand side components refer to the encoder and decoder, specifically.
We will describe these components in the next sections, and ommit the implementation details to focus on computational complexity. If you're curious about its implementation details, have a look at [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

#### Word and Positional Embeddig

The first unit of importance in the transformer is the embedding unit combined with the positional encoding (red boxes in the previous picture). The transformer model has no recurrence or convolution, so we need a **positional encoder** to learn the context of a sequence based on the order of its words. According to the paper, the embedding of a given word in the position $pos$ in a sentence, is an array with size $d$ whose value at each dimension $i$ is:

$$
PE_{(pos,2i)} = sin\left(\frac{pos}{10000^{2i/d}}\right) \,\text{ and }\, PE_{(pos,2i+1)} = cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

In practice, the embedding is given by $sine$ and $cosine$  waves with a different frequency and offset for each dimension. As an example, for a word with positioning $pos$, then the values at dimensions $4$ to $7$ of the embedding is:

<p align="center">
<br/>
<img width="60%" height="60%" src="/assets/AI-Supercomputing/transformer-embedding.png"/><br/>
<br/><small>The output of the positional encoding in the transformer architecture, for dimensions 4 to 7 of the embedding array, for a word with a sentence-positioning related to the x axis. <br/>Source: <a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">The Annotated Transformer</a>
</small>
</p>

#### Attention Mechanism

The main component of the transformer is the attention mechanism, that determines how the words in input and output sentences interact. In brief, it's the component that *learns* the relationship between words in such a way that it learns the relevant bits of information on each context (thus the naming *Attention* mechanism). The transformer architecture includes not one, but several of these mechanisms, executed in parallel, to allow the model to learn multiple relevant aspects of the input. This **Multi-head Attention Mechanism** solves for $n$ heads, *What part of the input should I focus on?*

Let's look at a single head of attention mechanism for now. Take the sentence of length $N$ words, how is each word related to every other word on that sentence? The output of a single attention mechanism is then the $N \times N$ matrix storing these inter-word importance metric. Here's an example:

<p align="center">
<br/>
<img width="40%" height="40%" src="/assets/AI-Supercomputing/transformer_attention.PNG"/><br/>
<br/><small>The attention mechanism output. For the sentence of length $N=4$ "The big red dog" the output at every row of the attention matrix is the normalized relevance metric of that word to every other word in the sentence. Source: unknown.
</small>
</p>

How does this mechanism work? According to the paper, each attention head is formulated as:

$$
Attention(K, V, Q) = softmax\left(QK^T / \sqrt{d_k}\right) V
$$

Where $K$, $V$ and $Q$ refer to the key, value and query latent spaces. This is a similar naming to a regular python dictionary and a C++ hash-map or hash-table, as they are all *key-value* stores that are *queried* for a given value (thus the usage of  $K$, $V$ and $Q$ variable convention). 

Notice that in the transformer diagram, each attention module has three inputs (arrows) referring to these three variables. The attention mechanism at the start of each encoder and decoder is retrieving keys, values and queries from its own input, learning the context of the source and target languages, respectively. However, in one occurence of the attention mechanism, the keys and values are provided by the encoder, and the query by the decoder, relating to the module that is trained for the translation at hand. 

The technique used and the main component of the attention mechanism is the **scaled dot-product** $QK^T$. The dot product of two vectors is computed as $a . b = \mid \mid a \mid \mid \, . \, \mid \mid b \mid \mid \, cos\, \theta_{ab}$, and provides the cosine of the angle between two vectors. Therefore, if two vectors have a simillar representation in space, the angle formed between them is small and its cosine is large. Here's a graphical representation of this logic:


<p align="center">
<br/>
<img width="40%" height="40%" src="/assets/AI-Supercomputing/Transformer-Attention-Mech-dot-product.png"/>
<br/><small>An example of the dot-product method inside the attention mechanism, applied to a mapping of 4 key vectors $K_1$ to $K_4$ and a query vector $Q$. Left: a graphical representation of the query vector and the key vectors. Right: the angle and cosine of the angles formed between query and vectors.
</small>
</p>

The logic is the following: similar vectors yield large angle-cosine values. These values are then scaled by $\sqrt{d_k}$ by the attention mechanism, introduced for numerical stability (detailed in the paper). The output is then past to a $softmax$ function that normalizes the vector and converts into a probability distribution, and finally multiplied by $V$ to retrieve the values scaled to the probability distribution of a given key. In practice, this can be interpreted as an implementation of a key-value store found in regular programming languages (on a discrete set of values), but on a continuous space. I.e. instead of outputting the value in a given position in a map related to the query, it outputs the probability of each value to relate to a given query, and retrieves all values and their respective importance, based on that query.

Finally, the multi-head attention mechanims combines all attention mechanism modues, and is defined as:

$$
MHA(K, V, Q) = [head_0,.., head_n]W^{MHA}   \,\text{ and }\,  head_i = Attention(KW^K_i, VW^V_i, QW^Q_i)
$$

I.e. it's a concatenation of all attention heads, and the parameters learnt are the weights of the keys ($W^K_i$), values ($W^V_i$) and query ($W^Q_i$) space of each head $i$, and a final transformation of the multi-head concatenation $W^{MHA}$. In terms of computational complexity, the attention mechanism on the encoder is trained on the complete input sequence at once (as illustrated in the "the big red dog" example), instead of looping through all words in the input sequence. Therefore,  **the attention mechanism replaces the recursive (RNN) iterations on an encoder, by a set of matrix-vector multiplications**.


The **Masked Multi-head Attention** component on the decoder is similar to the regular MHA, but replaces the diagonal of the attention mechanism matrix by zeros, to hide next word from the model. Decoding is performed with a word of the output sequence of a time, with previously seen words added to the attention array, and the following words set to zero. Applied to the previous example, the four iterations are: 

<p align="center">
<br/>
<img width="30%" height="30%" src="/assets/AI-Supercomputing/transformer_attention_masked.png"/><br/>
<br/><small>Input of the masked attention mechanism on the decoder for the sentence "Le gros chien rouge". The algorithm performs four iterations, one per word. Attention is computed for every word iterated. The mask component of the attention mechanism refers to replacing unseen words by zero. Source: unknown.
</small>
</p>

#### Other components

The other components on the transformer are not unique, and have been used previously in other machine learning models:
- The *Feed Forward* is a regressor (single hidden-layer DNN) that transforms the attention vectors into a form that is valid as input to the decoder or to the next computation phase;
- The *Linear* transformation component on the decoder expands the space into an array of size equals to the target vocabulary (French in the example); 
- The *Softmax* operation transforms the previous array into a probability distribution. The word with the highest probability is picked as output;


#### Computational Complexity

Besides the great reduction in the number of iterations on the encoder size, the authors compare the computational complexity of four comparative models:

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/transformer_table.png"/><br/>
<br/><small>Comparative table of computational complexity of four different learning models. Key: $n$: sequence length, $d$: representation dim., $k$: kernel size; $r$: size of neighbourhood.
<br/>Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, Arxiv)</a>
</small>
</p>

For the RNN used in previous Sequence-to-Sequence mechanism, the number of operations performed is in the order of $d^2$ multiplications (multiplication of weights in fully connected layer of DNN) for each of the $n$ words in the sentence, therefore $O(n^2 d)$. On the self-attention mechanism that we discussed in the Transformer, we have several operations of the attention matrix $n^2$ and the key and query vectors of embedding size $d$, therefore yielding a complexity of $O(n d^2$). 

The claim is that the $O(n^2 d)$ is better than $O(n d^2)$. This sound illogical in most Machine Learning problems as typically the input size is way larger than the dimensionality of the embedding. However, remember that $n$ here is the number of words in a sentence (in the order of $n \approx 70$) and $d$ is the size of the embedding ($d \approx 2000$).

To summarize, the encoder-decoder architecture of the transformer allows a faster non-recursive training of input sequences, and a faster training of output sentences, making it much more efficient than the sequence-to-sequence approaches discussed initially.

## BERT: Bidirectional Encoder Representation from Transformers

Attending to the previous topic, the main rationale of the Transformer's Encoder-Decoder is that:
- The encoder learns the context of the input language (English in the previous example);
- The decoder learns the task of the input-to-output languages (the English-to-French translation);

So, the encoder is efficiently trained and learns a *context*. So the main question is "Can we use only the Encoder's context and learn complex tasks?". This led to the introduction of the BERT model (original paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Google AI](https://arxiv.org/abs/1810.04805)). BERT is a stack of Transformer encoders that Learns language contexts and performs interpretation tasks.


<p align="center">
<br/>
<img width="60%" height="60%" src="/assets/AI-Supercomputing/BERT.PNG"/><br/>
<br/><small>The BERT model, as a stack of Transformer encoders.
</small>
</p>

The training is performed in two phases. A pre-training phase learns the contexts of the language. And a fine-tuning phase adapts the trained model to the task being solved. Let's start with the pre-training.

#### Pre-training

The pre-training phase is based on the simultaneous resolution of two self-supervised prediction tasks:
1. Masked language model: given a sentence with works replaced with a flag (or masked), train it against the same sentence with those words in place;
2. Next sentence prediction: Given two sentences, train the model to guess if the second sentence follows from the first or not;


Example of two training examples:
```
Input = [CLS] the man went to [MASK] store [SEP] He bought a gallon [MASK] milk [SEP]
Output = [IsNext] the man went to the store [SEP] He bought a gallon of milk [SEP]

Input = [CLS] the man [MASK] to the store [SEP] penguins [MASK] to jump [SEP]
Output = [NotNext] the man went to the store [SEP] penguins like to jump [SEP]
```

Note that the Yes/No flag related to the second task is past as the first embedded word in the output. The layout of the input data is:

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/BERT_input.PNG"/><br/>
<br/><small>The input of the BERT model. Position Emdebbings are similar to the transformer model, discussed above. Segment embeddings flag each word as part of the first or second sentence. Token embedding are the text-embeddings of the input data. The datapoint being input to the model is the concatenation of these three embeddings.
<br/>Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, Arxiv)</a>
</small>
</p>

As a side note, the authors trained this model on BooksCorpus (800M words) and English Wikipedia (2,500M words), using 24 BERT layers, batches of 256 sentences with 512 tokens each.

#### Fine-tuning

The fine-tuning phase adds an extra layer to the pre-trained BERT model, in order to train the model to the task at hand. This approach has been applied previously in other models, particularly on Convolutional Neural Nets, where the first layers are pre-trained and learnt edges and lines, and the remaining layers are added later and used on training of the object-specific detection.

In the original paper, the authors demonstrated this model successfully being applied to four families of tasks:
1. Sentence Pair Classification Tasks: classification of context from pairs of sentences. Examples: do sentences agree/disagree, does the second sentence follow from the previous one, do they describe a simillar context, etc. Similarly to the pre-training phase of the BERT models, the label is past as the first embedded symbol of the output; 
2. Single Sentence Classification Tasks: similar to the previous, but applied to a single sentence. Used for learning contexts like finding hateful speech, etc.;
3. Question Answering Tasks: allows one to pass as input the concatenation of the question and a paragraph where the answer to the question is available. The output of the model is the input with the embeddings representing the start and end word of the answer replaced by a marker. As an example:
	- input:  "[CLS] His name is John. John is a Swiss carpenter and he is 23 years old [SEP] How old is John";
	- output: "[CLS] His name is John. John is a Swiss carpenter and ###he is 23 years old#### [SEP] How old is John";
4. Single Sentence Tagging tasks: retrieves the classes of individual words e.g. Name, Location, Job, etc, by replacing each relevant word with its class id. As an example:
	- input: "[CLS] His name is John. John is a Swiss carpenter and he is 23 years old";
	- output: "[CLS] His name is [NAME]. [NAME] is a [NATIONALITY] [OCCUPATION] and he is [AGE] years old".

The input and output models of the fine-tuning of these tasks are illustrated in the following picture: 

<p align="center">
<br/>
<img width="60%" height="60%" src="/assets/AI-Supercomputing/BERT_tasks.png"/><br/>
<br/><small>Input and output of the fine-tuning phase of a BERT network, applied to four different interpretation tasks.
<br/>Source: <a href="https://arxiv.org/abs/1706.03762">Attention is all you need (2017, Google, Arxiv)</a>
</small>
</p>

As a final note, information encoded by BERT is useful but, on its own, insufficient to perform a translation task. However, "BERT pre-training allows for a better initialization point for [an] Neural Machine Translation model" (source: [Clichant et al.,On the use of BERT for Neural Machine Translation, arXiv](https://arxiv.org/abs/1909.12744)). 


## Further reading

Although this post covered some advanced content, the limits of supercomputing in Machine Learning has been pushed to a far greater extent. If you are interested in work in state-of-art ML supercomputing, see the [ZeRO and DeepSpeed work at Microsoft](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) ( [Turing-NLG blog post](ttps://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/), [www.deepspeed.ai/](https://www.deepspeed.ai/), [ZeRO paper](https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/)), the [Megatron work at NVIDIA](https://github.com/NVIDIA/Megatron-LM) and the [GPT work at OpenAI](https://en.wikipedia.org/wiki/GPT-3). 


[//]: <> #### Superlinear speed-up

[//]: <> <p align="center">
[//]: <> <br/>
[//]: <> <img width="45%" height="45%" src="/assets/AI-Supercomputing/ZeRO_superlinear_speedup_60B_parameter.PNG"/><br/>
[//]: <> <br/><small>The transformer  attention mechanism masked.</a>
[//]: <> </small>
[//]: <> </p>

[//]: <> super-linear speedup in the regime of 64-400 GPUs. This is a property of ZeRO-DP which reduces the memory footprint of the model states as we increase the DP degree, allowing us to fit larger batch sizes per GPU"

