---
layout: post
title:  "AI Supercomputing II: Seq-to-Seq, Encoder-Decoder, Transformers and BERT"
categories: [machine learning, supercomputing]
tags: [machinelearning]
---


In our [previous post]({{ site.baseurl }}{% post_url 2020-05-12-AI-Supercomputing %}), we've shows that:
- ML is very parallel due to efficient Matrix-vector multiplication;
- Memory is limited but we overcome it with \textbf{CPU-offloading};
- We can \textbf{pipeline} parallelism;
- We can parallelize data and use mean batch gradients;
- We can parallelize the model (locally e.g. CNN);

*Any ML model that is not covered?*


# Lmitations of parallelism (Encoder-Decoder, Seq-to-Seq)


<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/Encoder_Decoder.pdf"/><br/>
<br/><small>Model Parallelism.</a>
</small>
</p>

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/encoder_decoder_3.png"/><br/>
<br/><small>Model Parallelism.</a>
</small>
</p>

- encoding/decoding is a recursive algorithm $\rightarrow$ iterations can't the parallelized;
- Single hidden layer with *small* embedding  $\rightarrow$ no performance gain on parallelizing layers;
- Inputs/outputs of different lengths $\rightarrow$  only matching batch sizes can be parallelized;

...also important: *Attention Mechanism*

also, long sentences lead to vanishing/exploding gradients.

# Transformer 

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/transformer.PNG"/><br/>
<br/><small>The transformer architecture.</a>
</small>
</p>


[Vaswani et al. (Google), Attention is all you need, Arxiv](https://arxiv.org/abs/1706.03762)


## Encoder

Model has no recurrence or convolution, so we need a \textbf{positional encoder} to give context of order of words in sentence. Example: \textit{My \textbf{dog} is loud} vs \textit{I look like a \textbf{dog}}. %Dimensionality of PE is the same as embeddings $d$ so that they can be summed.

$$
 PE_{(pos,2i)} = sin\left(\frac{pos}{10000^{2i/d}}\right) \text {\hspace{0.25cm} and \hspace{0.25cm}} PE_{(pos,2i+1)} = cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

**Multi-head Attention** solves for $n$ heads, *What part of the input should I focus on?*

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/transformer_attention.PNG"/><br/>
<br/><small>The transformer architecture.</a>
</small>
</p>

$$
Attention(K, V, Q) = softmax\left(QK^T / \sqrt{d_k}\right) V
$$

$$
MHA(K, V, Q) = [head_0,.., head_n]W^M \text{, \hspace{0.2cm}} head_i = Attention(KW^K_i, VW^V_i, QW^Q_i)
$$

**Feed Forward** is a regressor (single hidden-layer DNN) that transforms the attention vectors into a form that is valid as input to the decoder.

\end{tiny}

## Decoder 

**Masked Multi-head Attention** similar to regular MHA, but replaces upper diagonal of attention vector by zeros, to hide next word from model model.

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/transformer_attention_masked.png"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>

Multi-head attention determines how the words in input \& output sentences interact.

*Linear* expands the space into an array of size equals to French vocabulary. 

*Softmax* tranforms into a prob. distribution. Word with highest probability is picked.

**Computational Complexity**:


<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/transformer_table.png"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>

$n$: sequence length, $d$: representation dim., $k$: kernel size; $r$: size of neighbourhood.

RNN: $d^2$ multiplications (multiplication of weights in fully connected layer of DNN) for each of the $n$ words in the sentence

Self-Attn (Encoder):  Attention matrix $n^2$, where each element of attention matrix has embedding $d$

amount of computation that can be parallelized, as measured by the minimum number of sequential operations

Why is it that $n^2 d$ is better than $n d^2$?

Sentences length $n \approx 70$, and word embeddings $d \approx 2000$.

**Parallelism:** till limited Dec. batch size, but no more Enc. recursion!

**Rationale:**
- Encoder learns English language (context);
- Decoder learnt the English-to-French translation;

Can we get rid of the Decoder and use only the Encoder to learn complex tasks?

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/transformer_3d_original.png"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>

# BERT: Bidirectional Encoder Representation from Transformers

BERT is a stack of Transformer encoders. Learns language \textit{context}.


<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/BERT.PNG"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>


Pre-Training: 2 self-supervised prediction tasks at same time:
- tasks: Masked Language Model; and Next Sentence Prediction;
- trained on Wikipedia, 24 BERT layers, batches of 256 sentences * 512 tokens;

Input = [CLS] the man went to [MASK] store [SEP]
\hspace{0.7cm}he bought a gallon [MASK] milk [SEP]
Label = IsNext

Input = [CLS] the man [MASK] to the store [SEP]
\hspace{0.7cm}penguin [MASK] are flight \#\#less birds [SEP]
Label = NotNext

### Input Layout

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/BERT_input.PNG"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Google AI](https://arxiv.org/abs/1810.04805)

- Fine-Tuning:  Adding one layer to a pre-trained BERT to learn to solve most tasks.

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/BERT_tasks.png"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>

Information encoded by BERT is useful but, on its own, insufficient to perform a translation task [due to no left-to-right prediction]. However, "BERT pre-training allows for a better initialization point for [an] NMT model", \href{https://arxiv.org/abs/1909.12744}{Clichant et al.,On the use of BERT for Neural Machine Translation, arXiv}

# Microsoft ZeRO \& DeepSpeed

Remember BERT is a stack of Transformer Encoders, i.e. a sequence of matrix-vector multiplication?

Remember Data Parallelism (DP) and Model Parallelism (MP)?

Remember the inputs and outputs on each layer of forward and backward propagation?

**ZeRO (Zero Redundancy Optimizer) combines all**: \textit{"[...] achieves the computation/communication efficiency of DP while achieving memory efficiency of MP, [...] retaining the computational granularity and communication volume of DP using a dynamic
communication schedule during training"

[Video](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

%ZeRO removes the memory redundancies across data-parallel processes by partitioning the model states—parameters, gradients, and optimizer (Adam) state—across data parallel processes instead of replicating them. 

%partitions optimizer states, gradients and parameters

%We show that ZeRO can be combined with any model parallelism
 
%We call this ZeRO-powered data parallelism, which allows per-device memory usage to scale linearly with the degree of data parallelism and incurs similar communication volume as data parallelism. 

Sources:
- [ZeRO \& DeepSpeed announcement page](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/);
- [Turing-NLG blog post](ttps://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/);
- [www.deepspeed.ai/](https://www.deepspeed.ai/);
- [DeepSpeed github docs](https://github.com/microsoft/DeepSpeed\#further-reading);
- [ZeRO paper](https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/);

**State of the Art**: \pause Bert-large (0.3B), GPT-2 (1.5B), Megatron-LM (8.3B), T5 (11B). ZeRO can run 100B parameters but they didn't, takes longer than a year for training! So they ran 17B.

### Superlinear speed-up

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/ZeRO_superlinear_speedup_60B_parameter.PNG"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>

super-linear speedup in the regime of 64-400 GPUs

This is a property of ZeRO-DP
which reduces the memory footprint of the model states as we increase the DP degree, allowing
us to fit larger batch sizes per GPU"

