---
layout: post
title: Publications bookmark
permalink: /publications/
---

A summary of some interesting publications I came accross. Continuously updated.

<br/>
# 2023 [Llama 2: Open Foundation and Fine-Tuned Chat Model](https://arxiv.org/abs/2307.09288)

LLama 2 is a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Llama 2-Chat is a finetuned LLM optimized for dialogue use cases. The models outperform open-source chat models, and based on
human evaluations for helpfulness and safety, it outperforms open-source models and appear to be on par with closed-source models (although may not be a suitable substitute). Results on safety human evaluation for Llama 2-Chat are presented in Figure 3. The train dataset is only publicly available sources, which does not include data from Meta’s products or services, or sources that may include users' personal information. Table 2 presents the GPU compute hours, power consumption and carbon emissions of each model

The pretraining setting and model architecture are adopted from Llama 1, i.e. bytepair encoding (BPE), pre-normalization via RMSNorm, SwiGLU activations, rotary positional embeddings, AdamW optimizer, cosine learning rate scheduler). However, the primary architectural differences from Llama 1 include **increased context length** and **grouped-query attention (GQA)**.

The finetuning was performed with supervised fine-tuning (Section 3.1), initial and iterative reward modeling (Section 3.2.2) and RLHF (Section 3.2.3). As drawback of RLHF, "initial RLHF models tended to forget the initial instruction after a few turns of dialogue (Figure 9, below, left). To address these limitations, we propose **Ghost Attention (GAtt)**, a very simple method inspired by Context Distillation (Bai et al., 2022b) that hacks the fine-tuning data to help the attention focus in a multi-stage process" (Figure 9, below, right).

{: style="text-align:center; font-size: small;"}
<img width="65%" height="65%" src="/assets/publications/llama2_gatt.png"/>

<br/>
# 2023 [LLaMA: Open and Efficient Foundation Language Models, Meta](https://arxiv.org/abs/2302.13971)

LLaMa is a collection of Large Language Models (LLM) with 7B to 65B parameters trained in public datasets, with performance superior to GPT-3 and comparable with Chinchilla-70B and PaLM-540B. Training is inspired by the Chinchilla scaling laws. The datasets used for the pre-training data are presented in Table 1, with training hyperparameters in Table 2. String are tokenized using the bytepair encoding (BPE) algorithm, with circa 1.4T tokens after tokenization.

The models architecture is made of several improvements over the original Transformer:
- **Pre-normalization [GPT3]:** training stability is improved with RMSNorm normalization at the input of each transformer sub-layer, instead of output.
- **SwiGLU activation function [PaLM]:** ReLU activation is replaced with SwiGLU to improve performance, with a dimension of $$\frac{2}{3} 4d$$ instead of $$4d$$ as in PaLM.
- **Rotary Embeddings [GPTNeo]:** positional embeddings are replaced by rotary positional embeddings (RoPE) at each layer of the output. 
- **Optimization** performed with AdamW optimizer with $$β_1 = 0.9$$, $$β2 = 0.95$$ and $$eps = 10^{−5}$$.
- **Cosine learning rate schedule** with a warmup of $$2000$$ steps, a weight decay of $$0.1$$, a gradient clipping of $$1.0$$ and a final learning of $$10%$$ of the initial value.
- **Causal multi-Head attention** inspired by Rabe and Staats (2021) and uses the backward from Dao et al. (2022), replaces the regular transformer multi-head attention. "This is achieved by not storing the attention weights and not computing the key/query scores that are masked due to
the causal nature of the language modeling task."
- **Activation checkpointing** was implemented to reduce memory. Yet it required manually implementing the Pytorch backward propagation function for the Transformer (insted of PyTorch autograd). This also required model and sequence parallelism (why?).
- **Overlap of the computation of activations and the communication between GPUs** over the network, to reduce latency.   
 
<br/>
# 2023 [Sparks of Artificial General Intelligence: Experiments with an early version of GPT-4, Microsoft](https://arxiv.org/abs/2303.12712)

A summary paper reporting early results of the experiments with GPT-4 when it was still in active development by OpenAI. The authors "demonstrate that, beyond its mastery of language, GPT-4 can solve novel and difficult tasks that span mathematics, coding, vision, medicine, law, psychology and more, without needing any special prompting. Moreover, in all of these tasks, GPT-4’s performance is strikingly close to human-level performance". The bulk of the paper contains dozens of examples that compare GPT-4 and Chat-GPT and demonstrate that GPU-4 surpasses ChatGPT in performance, in code generation, audio generation (output as musical notes), drawings (SVG, TIKZ), and mathematical resolutions (LaTeX). As weaknesses, besides the regular hallucinations it was also observed:
- Incapacity of planning correctly, when planning is not a linear path.
- Wrong complex arithmetic solver, e.g. `What's what is 151412 * 12412 / 12312 + 2? [...] is approximately equal to 152,513.676` instead of `152643.79`.
- Trained on past information only, without temporal awareness, e.g. `Whats the president of the US? Donald Trump`.
- lack of rigorous algorithms e.g. `What is the 11th letter of "abacadab"?  [..]  the 11th letter is "b."`
- ilogical reasoning/assumptions due to dataset biases, e.g. in gender: `If a man is a computer scientist, a woman is... a source of beauty and inspiration`.

But these can be overcome by including external APIs on training and making them in the query e.g.:
- `Using CALC(x) as a calculator of the expression x, what's 151412 * 12412 / 12312 + 2?`, or
- `Using SEARCH(x) to search for a query x, who's the president of the US?` or
- `Using CALENDAR(subject, date, user) and MAIL(user, text), book a meeting with the title 'subject' on the day 'date' to the user 'user', then email 'user' all the information`. 

<br/>
# 2023 [GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models, OpenAI, OpenResearch, Univ. of Pennsylvania](https://arxiv.org/abs/2303.10130)

The paper investigate the potential implications of large language models (LLMs), on the US labor market. The findings are:
- around 80% of the U.S. workforce could have at least 10% of their work tasks affected by the introduction of LLMs; 
- approximately 19% of workers may see at least 50% of their tasks impacted;
- with access to LLMs, about 15% of all worker tasks could be completed significantly faster at the same level of quality; and this share increases to 47%-56% when incorporating software and tooling built on top of LLMs.

The projected effects span all wage levels, with higher-income jobs potentially facing greater exposure to LLM capabilities.


<br/>
# 2023 [Segment Anything, Meta AI Research](https://arxiv.org/abs/2304.02643)

The Segment Anything (SA) project is a task, model, and dataset for image segmentation. SA built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M images. The model is designed to be promptable, to allow zero-shot transter to new image distributions and tasks. The authors clame that its zero-shot performance on new datasets is impressive – competitive or even superior to prior fully supervised results. SA and the dataset are released at [https://segment-anything.com](https://segment-anything.com)

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/publications/segment_anything.png"/>

<br/>
# 2023 [Retentive Network: A Successor to Transformer for Large Language Models, Microsoft and Tsinghua University](https://arxiv.org/abs/2307.08621)

(note: a simpler summary video of RetNet can be found [here](https://www.youtube.com/watch?v=JaIL1VAEwZ8))

RetNet is a multi-scale retention mechanism to substitute multi-head attention in Transformers, which has three computation paradigms:
- parallel framework, for training parallelism that utilizes GPU devices fully.
- recurrent framework for low-cost $$O(1)$$ inference, which improves decoding throughput (8.4x improvement over Transformer), latency (15.6x), and GPU memory (3.4x) without sacrificing performance, on Figure 1.
- a chunkwise recurrent representation that can perform efficient long-sequence modeling with linear complexity,  where each chunk is encoded parallelly while recurrently summarizing the chunks. It allows encoding each local block for computation speed while recurrently encoding the global blocks to save GPU memory 

Retentive network (RetNet) is a stack of $$L$$ identical blocks, which follows a similar layout (i.e.,
residual connection, and pre-LayerNorm) as in Transformer. Each RetNet block contains
two modules: a multi-scale retention (MSR) module, and a feed-forward network (FFN) module.  The MSR module calls the tokens in a sequence in an auto-regressive manner. The input vector is first created as $$X_0$$ in the shape of sequence length by hidden domain size. Then we calculate contextualized vector representations $$X_n$$ for each layer of the RetNet. Retention heads can be represented in **two alternative ways**:
1. in the **parallel representation**, where $$Retention(X) = Q K\intercal \dot D)V$$ similar to the transformer but with an extra matrix $$D$$ (Eq. 5). This is befenicial for parallel training.
2. in the **recurrent representation**, it is written as a recurrent neural net (RNN) which is beneficial for inference, and $$Retention(X_n)=Q_n S_n$$ where $$S_n$$ depends on the previous term $$S_{n-1}$$.
3. a **hybrid form** combining the previous two representations is also possible to accelerate training on large sequences. Input sequence is divided into chunks. Within each chunk, the computation is performed in the parallel representation. Cross-chunk information is passed in the recurrent representation.

Finally, the model uses $$h = d_{model}/d$$ retention heads in each layer, where $$d$$ is the head dimension. The heads use different parameter matrices $$W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$$ and scalar $$γ$$ per head. The overall architecture for a given layer $$l$$ of the RetNet is then $$Y_l = MSR(LayerNorm(X_l)) + X_l$$ and $$X_{l+1} = FFN(LN(Y_l)) + Y_l$$, ie similar to a regular transformer but replacing the attention by a retention head.

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/publications/RetNet.png"/>

<br/>
# 2023 [Operator Fusion in XLA: Analysis and Evaluation, UToronto](https://arxiv.org/abs/2301.13062)

Kernel fusion is the most significant optimization operation in [XLA](https://www.tensorflow.org/xla). This paper details XLA and key compiler passes of XLA's source code. It also presents the speedup that kernel fusion can deliver, and what low-level effects it has on hardware.

<br/>
# 2023 [LongNet: Scaling Transformers to 1,000,000,000 Tokens, Microsoft and Xi’an Jiaotong University](https://arxiv.org/abs/2307.02486)

LongNet is a Transformer variant that can scale the sequence length up to 1B tokens, and without sacrificing the performance on shorter sequences. This overcomes current limitations of attention size in regular transformers, that requires a tradeoff between computational complexity and the model expressivity.
1. LongNets have a linear computation complexity and a logarithm dependency between any two tokens in a sequence;
2. They can be served as a distributed trainer for extremely long sequences;
3. Its main trick is based on the **dilated attention**, which expands the attentive field exponentially as the distance grows, and is a direct replacement for current attention in Transformers.
   - The general principle behind dilatied attention is: attention allocation decreases exponentially as the distance between tokens grows.
   - When using multiple attention heads, the attention patterns differ among heads by shifting the attention masks.
4. Dilated attention yields a computation complexity of $$O(ND)$$, compared to $$O(N d^2)$$ in RNNs, $$O(N^2 d)$$ in vanilla attention, and $$O(N \sqrt{N} d)$$ in sparse attention.
 
{: style="text-align:center; font-size: small;"}
<img width="68%" height="68%" src="/assets/publications/longnets.png"/>
&nbsp; &nbsp; &nbsp; &nbsp;
<img width="19%" height="19%" src="/assets/publications/longnets2.png"/>

<br/>
# 2022 [TorchScale: Transformers at Scale, Microsoft](https://arxiv.org/abs/2211.13184)

TorchScale is an open-source toolkit that allows researchers and developers to scale up Transformers efficiently and effectively. 
TorchScale adopts [Magneto](https://arxiv.org/abs/2210.06423) (below) as the default model backbone, thus supporting several applications applications, including language modeling, machine translation, vision pretraining, speech recognition, and multi-modal pretraining, just like Magneto.
- **Stability:** to handle models unstability in optimization as the model grows, TorchScale follows the theoretical derivation of [DeepNet](https://arxiv.org/abs/2203.00555) (below), just like Magneto.
- **Efficiency:** TorchScale implements sparsity via Mixture of Experts as in [X-MoE](https://arxiv.org/abs/2204.09179) , a variant of sparse MoE model. "Torchscale supports both Top-1 and Top-2 routing algorithms (?), which balance the performance and the computation cost.  This allows
Transformers to scale up to billions or trillions of parameters without much additional computation
cost". Gradient clipping plays an important role in the performance of sparse MoE models, and this was overcome by a new method created by the authors - SparseClip.
  - Note: Gradient clipping is standard practice for deep neural models to alleviate the gradient explosion problem. It is even more important for the sparse MoE models, which are more unstable in training.
- **Transformer variants in the toolkit** available at [https://github.com/microsoft/torchscale](https://github.com/microsoft/torchscale): deep models (DeepNet),  Foundation Transformers (Magneto), sparsity ( X-MoE), RetNet, LongNet, parameter stability (SparseClip), ...  
  
<br/>
# 2022 [Foundation Transformers (Magneto), Microsoft](https://arxiv.org/abs/2210.06423)

Magneto is a "foundation transform for true general-purpose modeling, which serves as a go-to architecture for various tasks and modalities with
guaranteed training stability". Specifically, the paper introduces Sub-LayerNorm (which adds an extra LayerNorm
to each sublayer) for good expressivity. Also introduces a novel initialization strategy theoretically derived from [DeepNet](https://arxiv.org/abs/2203.00555) (below) for stable scaling up. Experiments demonstrate superior performance to the standard Transformer across language, translation, vision, speecha and multimodal tasks. There are only lines of code changes on top of the vanilla Transformer architecture: the Sub-LN structure, and the weights of query projection and key projection are not scaled during initialization following DeepNet (below).
 
{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/publications/magneto.png"/>

<br/>
# 2022 [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)

This paper introduces a normalization function (**DeepNorm**) to modify the residual connection in Transformer, accompanyed with theoretically derived initialization, in order to stabilize extremely deep Transformers.
- Background: previous work had sown that better initialization methods improve the stability of the training of Transformer.
- DeepNorm works by introducing a new normalization function at residual connections, which has theoretical justification of bounding the model update by a constant.
- "The proposed method combines the best of two worlds, i.e., good performance of Post-LN and stable training of Pre-LN (picture above), making DeepNorm a preferred alternative.". 
- Figure 2 shows the `deepnorm` (the normalization layer function), `deepnorm_init` (the weights initialization) and constants.


<br/>
# 2022 [Contrastive Deep Supervision, Tsinghua University, Intel Corporation, and Xi’an Jiaotong](https://arxiv.org/abs/2207.05306)

From the abstract: "the traditional training method only supervises the neural network at its last layer and propagates the supervision layer-by-layer, which leads to hardship in optimizing the intermediate layers. Recently, deep supervision has been proposed to add auxiliary classifiers to the intermediate layers of deep neural networks. By optimizing these auxiliary classifiers with the supervised task loss, the supervision can be applied to the shallow layers directly. However, deep supervision conflicts with the well-known observation that the shallow layers learn low-level features instead of task-biased high-level semantic features. To address this issue, this paper proposes a novel training framework named Contrastive Deep Supervision, which supervises the intermediate layers with augmentation-based contrastive learning".  The rationale is that contrastive learning can provide better supervision for intermediate layers than the supervised task loss. Contrastive learning "regards two augmentations from the same image as a positive pair and different images as negative pairs. During training, the neural network is trained to minimize the distance of a positive pair while maximizing the distance of a negative pair. As a result, the network can learn the invariance to various data augmentation, such as Color Jitter and Random Gray Scale". Contrastive Deep Supervision starts from those advancements, and optimizes the intermediate layers with contrastive learning instead of traditional supervised learning. As shown in the figure above, "several projection heads are attached in the intermediate layers of the neural networks and trained to perform contrastive learning. These projection heads can be discarded in the inference period to avoid additional computation and storage. Different from deep supervision which trains the intermediate layers to learn the knowledge for a specific task, the intermediate layers in our method are trained to learn the invariance to data augmentation, which makes the neural network generalize better. Besides, since contrastive learning can be performed on unlabeled data, the proposed contrastive deep supervision can also be easily extended in the semi-supervised learning paradigm". Finally, contrastive deep supervision can be further utilized to boost the performance of knowledge distillation.

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/publications/contrastive_deep_supervision.png"/>

<br/>
# 2022 [Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing, Microsoft](https://arxiv.org/abs/2204.09817)

A multi-model approach for text and images applied to health (radioligy), based on contrastive-learning in self-supervised vision-language processing (VLP). As background, one strong motivation is the lack of medical data and the need to self-supervise (and annotate) it, which are expensive and time-consuming. These lead to a interest in multi-model self-supervised learning and cross-model weak supervision, in particular paired text-image data. The paper focus on self-supervised vision-language learning, by jointly learning image and representations for several use cases such as zero-/few-shot image classification, report generation, error detection, and disease localisation.
 
It introduces a new chest X-ray (CXR) domain-specific language model (CXR-BERT), a self-supervised VLP task for the biomedical use case (BioViL), and a Local Alignment Chest X-ray dataset, MS-CXR. The CXR-BERT is pre-trained with a randomly initialised BERT model via Masked Language Modelling (MLM) (largely following the RoBERTa pretraining configurations), and later fine-tuned with domain-specific data.  BioViL uses a convolutional neural network image encoder Eimg, the CXR-BERT text encoder, and projection models to learn representations in a joint space. The CNN model allows them to obtain a grid of local image embeddings, which is fine-grained enough to be useful for segmentation (e.g. 16×16). Each encoder is followed by a modality-specific two-layer perceptron projection model, which projects the encoded modality to a joint space of 128 dimensions. To align the representations and learn a joint embedding, it uses two loss terms based on a symmetric contrastive loss for global alignment of the image and text. After joint training, it uses text prompts to cast the zero-shot classification problem into an image–text similarity task. Results demonstrate CXR-BERT having superior performance and improved vocabulary.
 
{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/publications/biovil.png"/>


<br/>
# 2022 [Emergent Abilities of Large Language Models, Google Research & Stanford](https://openreview.net/forum?id=yzkSU5zdwD)

The paper discusses instead the phenomenon of **emergent abilities** of large language models. An ability is emergent if it is not present in smaller models but is present in larger models, and not extrapolated from scaling laws. *Phase transition* is the scale at which such abilities are exposed. Scale in this context may represent different compute budgets, data quality or other factors - the paper foccuses not on ideal training but on the discussion of such phenomena. As a disclaimer, "model scale is not the singular factor for unlocking an emergent ability" and "the science of training large language models progresses, certain abilities may be unlocked for smaller models with new architectures, higher-quality data, or improved training procedures".

The first analysis of emergent abilities focuses the prompting paradigm, where outcome is emergent when a model has random performance until a certain scale, after which performance increases to well-above random. This was analysed on 8 different models:

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/publications/Emergent_Abilities_1.png"/>

A similar analysis with augmented prompting exposes the emergent property as related to when the model output starts having a positive effect (e.g. being able to do arithmetic only after a certain scale). A multi-step reasoning by providing a chain-of-thoughts as a sequence of intermediatte steps was also analysed, and claimed to be exposed only after $$10^{23}$$ FLOPS or approx. 100B parameters. Such scale is also required for intruction following tasks (ie new tasks without prior few-shots exemplars, and only by reading a set of instructions). Program execution tasks require $$9 x 10^{19}$$ FLOPS or 40M parameters (for a 8-digit addition) or larger. For model calibration (the ability of a model responding as True of False (or the correctness probability) to which questions they'll be able to predict correctly) requires $$3*10^{23}$$ FLOPS or 52B parameters. It is summarized as:

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/publications/Emergent_Abilities_2.png"/>


<br/>
# 2022 [Rethinking Attention with Performers, Google, Cambridge, DeepMind and Alan Turing Institute](https://arxiv.org/abs/2009.14794)

From the abstract: Performers are "Transformer architectures which **can estimate regular
(softmax) full-rank-attention Transformers with provable accuracy**, but using only
linear (as opposed to quadratic) space and time complexity, without relying on
any priors such as sparsity or low-rankness. To approximate softmax attention kernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+)".

A clearer explanation can be found on this [google research post](https://blog.research.google/2020/10/rethinking-attention-with-performers.html):

**Bidirectional attention**, where there's no notion of past and future: by decouplin matrices $$Q′$$ and $$K′$$ used in lower rank decomposition of $$A$$ and conducting matrix multiplications in the order indicated by dashed-boxes, we obtain a linear attention mechanism, never explicitly constructing $$A$$ or its approximation:

{: style="text-align:center; font-size: small;"}
<img width="75%" height="75%" src="/assets/publications/performers.jpg"/> 

{: style="text-align:center; font-size: small;"}
**Left:** Standard attention module computation, where the final desired result is computed by performing a matrix multiplication with the attention matrix $$A$$ and value tensor $$V$$. **Right:** By decoupling matrices $$Q′$$ and $$K′$$ used in lower rank decomposition of $$A$$ and conducting matrix multiplications in the order indicated by dashed-boxes, we obtain a linear attention mechanism, never explicitly constructing $$A$$ or its approximation.

**Unidirectional (causal) attention**, where tokens do not attend to other tokens appearing later in the sequence: the previous approach is modified to use prefix-sum computations, which only store running totals of matrix computations rather than storing an explicit lower-triangular regular attention matrix.

{: style="text-align:center; font-size: small;"}
<img width="75%" height="75%" src="/assets/publications/performers2.gif"/> 

{: style="text-align:center; font-size: small;"}
**Left:** Standard unidirectional attention requires masking the attention matrix to obtain its lower-triangular part. **Right:** Unbiased approximation on the LHS can be obtained via a prefix-sum mechanism, where the prefix-sum of the outer-products of random feature maps for keys and value vectors is built on the fly and left-multiplied by query random feature vector to obtain the new row in the resulting matrix.


<br/>
# 2022 [Training Compute-Optimal Large Language Models, arXiv](https://arxiv.org/abs/2203.15556)

Heavily related to HPC's performance modelling applied to large language models. The authors revisit the question "Given a fixed FLOPs budget, how should one trade-off model size and the number of training tokens?" to which they present three approaches: (1) fix model sizes and vary number of training tokens; (2) vary model sizes for 9 different FLOP counts; (3) fit a parametric loss function to the values retrived from the 2 approaches. Estimates were collected from a total of 400 runs. 

 The main conclusion is that current large language models are under-performing as they only scaled the model size and not the data size. For compute-optimal training, the model size and number of training tokens should be scalled equally. This hypothesis is demonstrated with a "compute-optimal" model Chinchilla, with the same "compute budget" as Gopher (70B parameters) and 4× more more data. Chinchilla outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on several evaluation tasks. 

To be compute optimal (in terms of accuracy vs energy cost), Kaplan et al. (2020) claims that models should not be trained to their lowest possible loss, and for a 10× increase in computational budget, the model should increase by 5.5× and the training tokens by 1.8x. In this paper, the authors defend that model size and training tokens should be scaled in equal proportions. 

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/publications/Training_Compute_Optimal_Large_Language_Models.png"/> 

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/publications/Training_Compute_Optimal_Large_Language_Models_2.png"/> 

<br/>
# 2021 [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning, Microsoft](https://arxiv.org/abs/2104.07857)

Towards allowing very large models on (memory-limited) GPUs, ZeRO-Infinity is a system technology that leverages GPU, CPU, and NVMe memory to allow for unprecedented model scale on limited resources, without code refactoring. It achieves excellent training throughput and scalability, unencumbered by the limited CPU or NVMe bandwidth.  An open source implementation of ZeRO-Infinity is available through DeepSpeed.

ZeRO-Infinity is built
on top of [ZeRO-3](https://arxiv.org/abs/1910.02054) which partitions all model states to remove
memory redundancy, and its main goal is to offload all of the partitioned model states to CPU or
NVMe memory, or keep them on the GPU based on the memory
requirements. Offloading techniques for different data types are detailed in section 5.

<br/>
# 2021 [GSPMD: General and Scalable Parallelization for ML Computation Graphs, Google](https://arxiv.org/pdf/2105.04663.pdf)

also covered on a [google blog post](https://blog.research.google/2021/12/general-and-scalable-parallelization.html).

GSPMD (General and Scalable Parallelization for ML Computation Graphs) is an open-source, automatic, compiler-based parallelization system based on the [XLA compiler](https://www.tensorflow.org/xla). Because different model architectures may be better suited to different parallelization strategies, GSPMD is designed to support a large variety of parallelism algorithms appropriate for different use cases (e.g. data parallelism for small models, pipelining parallelism for larger models, or a combination of both).

In GSPMD, each tensor will be assigned a sharding property, either explicitly by the user as initial annotations, or by the sharding completion pass. The sharding property specifies how the data is distributed across devices. GSPMD defines three types of sharding: replicated (all devices have the same full data), tiled (a tiled sharding of the tensor, without data suplication), and partially tilled (an extension to [GShard](https://arxiv.org/abs/2006.16668), where tensor is tilled among subgroups of processors, that then have a different tilling within each subgroup). 

The sharding properties are user-defined with `mesh_split(tensor, device_mesh, dims_mapping)` that allows a tensor to be across the device mesh and a mapping from each data tensor dimension (i) to an optional device mesh dimension. This simple API is general enough to express
all types of sharding, across the dimension(s) of batch, features, channels and/or others. The automatic partitioner in GSPMD is implemented as transformation/compiler passes in the XLA compiler (Section 3.5), using information about the operator (e.g. dot product is a generalized matrix multiply) or using iterative methods where  shardings assigned by the pass are refined incrementally over the iterations. 

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/publications/GSPMD.png"/>

{: style="text-align:center; font-size: small;"}
**Left:** A simplified feedforward layer of a Transformer model. Blue rectangles represent tensors with dashed red & blue lines overlaid representing the desired partitioning across a 2x2 mesh of devices. **Right:** A single partition, after GSPMD has been applied. **Source**: <a href="https://blog.research.google/2021/12/general-and-scalable-parallelization.html">google research post</a>.

<br/>
# 2021 [Skilful precipitation nowcasting using deep generative models of radar, Google Deepmind](https://www.nature.com/articles/s41586-021-03854-z)

Current weather predictions are done by using numerical weather predictions, by solving physical equations that descrive radar-based wind estimates. Alternative methods use machine learning to capture non-linear behaviour that is not described by the mathematical formalism of the weather-regulating equations. Two main problems arise: poor performance on rarer medium-to-heavy rain events, and weather forecast at high resolution for short-term intervals (2 hours, a.k.a. nowcasting).
This paper solves demonstrates improvements in the skill of probabilistic precipitation nowcasting, by using an approach known as generative modelling, based on a deep generative model (DGM) for the probabilistic nowcasting of precipitation.

The DGM presented is a statistical model that learns the probability distributions of data and uses it to generate samples. It also has the ability to simulate many samples from the conditional distribution of future radar given historical radar, thus generating forecasts (similar to ensemble methods). The model predicts $$N$$ future radar fields given $$M$$ past, using radar-based estimates of surface precipitation $$X_T$$ at a time $$T$$. Learning is performed similarly to existing work on generative adversarial networks (GANs) for video generation. The GAN model is composed as a generator trained using two discriminators (spatial and temporal) and an additional regularization term. "The generator comprises the conditioning stack which processes past four radar fields that is used as context. [...] This stack produces a context representation that is used as an input to the sampler" (that samples from a standard Gaussian).The spatial and temporal discriminator are identical, except that the temporal discrimination uses a 3D kernel to account for the temporal dimension. During evaluation, the generator architecture is the same, but the full radar observations and latent variables distribution of width and heigh $$1/32$$ times smaller than then radar observations are used as inputs to the conditioning stack and latent conditioning stack, respectively. "In particular, the latent conditioning stack allows for spatiotemporally consistent predictions for much larger regions than those on which the generator is trained".
 
These predictions focus on medium to heavy rain scenarios, as they are the most impactful for society.
The model accurately capture large-scale events, while also predicting rainfall uncertainty and generating many alternative rain scenarios (known as ensemble predictions), with consistent predictions of large regions and with lead times from 5–90 min ahead. 
Results are validated by 50 expert meteorologists that would opt in 89% of situations by this model predictions, compared to competitive methods (PySTEPS, Unet and MetNet) used for nowcasting predictions. As future work, the authors suggest specializing this model for improved long-term predictions.

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/publications/GANs_for_weather_nowcasting_nature.png"/>

<br/>
# 2021 [Reduced, Reused and Recycled: The Life of a Dataset in Machine Learning Research, Google and Univ. California, NeurIPS 2021](https://arxiv.org/abs/2112.01716)

Winner of the "Datasets and Benchmarks Best Paper Award" at NeurIPS 2021. Abstract: "We study how dataset usage patterns differ across machine learning subcommunities and across time from 2015-2020. We find increasing concentration on fewer and fewer datasets within task communities, significant adoption of datasets from other tasks, and concentration across the field on datasets that have been introduced by researchers situated within a small number of elite institutions." 

{: style="text-align:center; font-size: small;"}
<img width="75%" height="75%" src="/assets/publications/reduced_recycled_datasets.png"/> 

<br/>
# 2021 [MLP-Mixer: An all-MLP Architecture for Vision, Google, NeurIPS 2021](https://arxiv.org/abs/2105.01601)

The paper argues that neither (CNNs) convolution CNNs or (Transformers) attention are necessary for computer vision setups. To that extent, it presents MLP-mixers, a Multi-Layer Perceptron only architecture. "MLP-Mixer contains two types of layers: one with MLPs applied independently to image patches (i.e. "mixing" the per-location features), and one with MLPs applied across patches (i.e. "mixing" spatial information)." Results are competitive with existing methods.
 
{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/publications/mlp_mixer.png"/> 

<br/>
# 2021 [Pay Attention to MLPs, Google, NeurIPS 2021](https://arxiv.org/abs/2105.08050)

The paper introduces gMLP (gated MLPs) and show that they can perform as well as Transformers in language and vision applications. It claims that "self-attention is not critical for Vision Transformers, as gMLP can achieve the same accuracy". In some BERT tasks it performed better than Transformers, and on finetuning tasks, it performed worse (but this can be overcome by making the gMLP model substantially larger).

The gMLPs have no self-attention, and instead rely on channel projections and spatial projections with static parameterization. It consists of a stack of $$L$$ blocks with identical size and structure. Each block is defined as:

$$
Z = σ(XU), \,\,\,\,\,\,\,\, \tilde{Z} = s(Z), \,\,\,\,\,\,\,\, Y = \tilde{Z} V
$$

where $$σ$$ is an activation function, $$U$$ and $$V$$ are linear projections along the channel dimension, and $$s(·)$$ is a layer which captures spatial interactions. When $$s$$ is an identity mapping, the above transformation degenerates to a regular FFN, ie no cross-token communication. Here, $$s(·)$$ is a spatial depthwise convolution (Section 2.1), which, unlike Transformers, does not require position embeddings because that is captured in $$s(·)$$.

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/publications/pay_attention_to_mlps.png"/> 

<br/>
# 2021 [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Google, ICLR 2021](https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1)

An extension of the transformer architecture to images. Works by passing as input to the transformer a sequence of linear embeddings of image patches. Paper demonstrates better results on classification tasks, compared to CNNs, ResNets and native attention mechanism (that do not scale well as pixels attend to other pixels leading to a quadratic complexity). Transformers lack the inductive bias of CNNs (e.g. translation equivariance and locality), and therefore do not generalize well when training on insufficient amounts of data. Class is added similarly to BERT as the *class* token. VTs use 1D positional encodings, since performance of 2D encoders did not deliver significant performance gains. Only MLP layers are local and translationally equivariant, yielding an inductive bias much smaller than CNNs. The *hybrid architecture* mode uses feature maps of a CNN instead of raw image patches as input. Similar to the original NLP transformer, it scales well and delivers a reduced training time compared to CNN-based architectures. Performance increases with dataset size. 

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/publications/visual_transformer.png"/> 

<br/>
# 2021 [Finetuned Language Models Are Zero-Shot Learners, Google, ICLR 2022](https://arxiv.org/abs/2109.01652)

The paper presents a simple method for improving the zero-shot learning abilities of language models. It shows that instruction tuning -- finetuning language models on a collection of tasks described via instructions -- substantially improves zero-shot performance on unseen tasks.
The intuition is that performing instruction tuning—finetuning of the model with datasets expressed via natural language instructions, substantially improves the zero-shot performance of the model.
For each dataset, the authors manually compose ten unique templates that use natural language instructions to describe the task for that dataset.

{: style="text-align:center; font-size: small;"}
<img width="67%" height="67%" src="/assets/publications/finetune_language_models.png"/> 

<br/>
# 2020 [Scaling Laws for Neural Language Models, John Hopkins, OpenAI](https://arxiv.org/abs/2001.08361)

Abstract: We study empirical scaling laws for language model performance on the cross-entropy loss.
The loss scales as a power-law with model size, dataset size, and the amount of compute
used for training, with some trends spanning more than seven orders of magnitude. Other
architectural details such as network width or depth have minimal effects within a wide
range. Simple equations govern the dependence of overfitting on model/dataset size and the
dependence of training speed on model size. These relationships allow us to determine the
optimal allocation of a fixed compute budget. Larger models are significantly more sampleefficient, such that optimally compute-efficient training involves training very large models
on a relatively modest amount of data and stopping significantly before convergence.

{: style="text-align:center; font-size: small;"}
<img width="75%" height="75%" src="/assets/publications/scaling_laws.png"/>

**Keypoints:**
- Model performance depends most strongly on scale, which consists of three factors: the number of model parameters N, the size of the dataset D, and the amount of compute C used for training. Performance has a power-law relationship with each of the three scale factors (Fig.1).
- Within reasonable limits, performance depends very weakly on other architectural hyperparameters such as depth vs. width.
- Performance improves predictably as long as we scale up N and D in tandem,
but enters a regime of diminishing returns if either N or D is held fixed while the other increases.
- When we evaluate models on text with a different distribution
than they were trained on, the results are strongly correlated to those on the training validation set with
a roughly constant offset in the loss, i.e. incurs a constant
penalty but improves in line with the performance of the training set.
- When working within a fixed compute budget C but without any other restrictions on the model size N or available data D, we attain optimal performance by training very large models
and stopping significantly short of convergence.
- The ideal batch size for training these models is roughly a power of the loss only

{: style="text-align:center; font-size: small;"}
<img width="75%" height="75%" src="/assets/publications/scaling_laws_2.png"/>

<br/>
# 2020 [Language Models are Few-Shot Learners (GPT-3), OpenAI](https://arxiv.org/abs/2005.14165)

Up until now, substantial gains on many NLP tasks were achieved by pre-training on a large corpus of text followed by fine-tuning on a specific task. This method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. This paper shows that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art finetuning approaches. This paper presents and tests GPT-3 (an autoregressive LLM with 175 billion parameters, 10x larger than previous models) in the few-shot setup. 

**GPT-3 architecture** uses the same model and architecture as [GPT-2](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf), including the modified initialization, pre-normalization, and reversible tokenization described therein, with the exception that we use alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to the [Sparse Transformer](https://arxiv.org/abs/1904.10509). 
- Table 2.1 includes the 8 GPT-3 models built and their sizes/hyperparameters.
- Fig. 2.2 shows the total compute used during training. Based on the analysis in [Scaling Laws For Neural Language Models](https://arxiv.org/abs/2001.08361) we train much larger models on many fewer tokens than is typical. 
- Fig 3.1 shows the pattern of smooth scaling of performance with compute. Performance (cross-entropy loss) follows a power-law trend with the amount of compute used for training. 


**Background (Fig 2.1):**
- Fine-Tuning (FT) has been the most common approach in recent years, and involves updating the weights of
a pre-trained model by training on a supervised dataset specific to the desired task.
- Few-Shot (FS) is the term we will use in this work to refer to the setting where the model is given a few
demonstrations of the task at inference time as conditioning [RWC+19], but no weight updates are allowed.
- One-Shot (1S) is the same as few-shot except that only one demonstration is allowed, in addition to a natural
language description of the task
- Zero-Shot (0S) is the same as one-shot except that no demonstrations are allowed,  and the model is only given
a natural language instruction describing the task.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/publications/gpt3_fig21.png"/> 

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/publications/gpt3_fig11.png"/> 

**Tasks tested and performance**:
- On NLP tasks it achieves promising results in the zero-shot and one-shot settings, and in the the few-shot setting is sometimes competitive with or even occasionally surpasses state-of-the-art.
- It also displays one-shot and few-shot proficiency at tasks designed to test rapid adaption or on-the-fly reasoning,
which include unscrambling words, performing arithmetic, and using novel words in a sentence after seeing them defined only once.
- Fig 3.3 to 3.12 show that  GPT3’s performance grows with model size, suggesting that language models continue to absorb knowledge as their capacity increases. Results plotted for the the TriviaQA, translation, [Winograd Schema Challenge](https://arxiv.org/abs/1907.10641), PIQA, comprehension, SuperGLUE, ANLI Round 3, arithmetic, word scrambling, and SAT tasks; on zero-, one- and few-shot training, respectively.
- Fig 3.12 shows that people’s ability to identify whether news articles are model-generated (measured by the ratio of correct
assignments to non-neutral assignments) decreases as model size increases.
or QuAC. 
- Fig 4.2 plots the benchmark contamination analysis. Data contamination has a minimal effect on GPT-3’s performance on most datasets, but the authors identify a few datasets where it could be inflating results:
- Chaper 5 details the limitations. GPT-3 struggles with natural language inference tasks like the ANLI dataset, and some reading comprehension datasets like RACE


<br/>
# 2020 [Graph Transformers Networks, Korea University](https://arxiv.org/abs/1911.06455)

One limitation of most GNNs is that they assume the graph structure to be fixed and homogeneous, ie similar types of nodes and edges. From the abstract: "Graph Transformer Networks (GTNs) are capable of
generating new graph structures, which involve identifying useful connections
between unconnected nodes on the original graph, while learning effective node
representation on the new graphs in an end-to-end fashion. Graph Transformer layer,
a core layer of GTNs, learns a soft selection of edge types and composite relations
for generating useful multi-hop connections". 
- GTNs perform Meta-Path Generation: a meta-path defines a composite relation $$R = t_1 ◦ t_2 \, ... \, ◦ \, t_l$$ between node $$v_1$$ and $$v_{l+1}$$, where $$R_1 ◦ R_2$$ denotes the composition of relation $$R_1$$ and $$R_2$$.
- GTNs use graph convolutional network (GCN) to learn useful representations for node classification in an end-to-end fashion.

{: style="text-align:center; font-size: small;"}
<img width="75%" height="75%" src="/assets/publications/graph_transformer_networks.png"/> 

<br/>
# 2019 [No Language Left Behind: Scaling Human-Centered Machine Translation, Meta, Berkeley and Johns Hopkins](https://arxiv.org/abs/2207.04672)

No Language Left Behing (NLLB) is an open-source projects that provides an ML model capable of delivering igh-quality translations between 200 languages—including low-resource languages like Asturian, Luganda, Urdu and more. The model (NLLB-200) is a conditional
model based on Sparsely Gated Mixture of Experts that is "trained on data obtained
with novel and effective data mining techniques tailored for low-resource languages". It also presents "architectural and training improvements to counteract overfitting while training on thousands of tasks".
- Background (wikipedia): Mixture of experts (MoE) is a machine learning technique where multiple expert networks (learners) are used to divide a problem space into homogeneous regions. It differs from ensemble techniques in that typically only one or a few expert models will be run, rather than combining results from all models. An example from computer vision is combining one neural network model for human detection with another for pose estimation.


<br/>
# 2019 [Generating Long Sequences with Sparse Transformers, OpenAI](https://arxiv.org/abs/1904.10509)

The paper introduces several sparse factorizations of the attention matrix that reduce the quadratic complexity on memory and runtime Transformers to $$O(n \sqrt{n})$$. It also allows for larger sequences. These work by separating the full attention computation into several faster attention operations which, when combined, can **approximate the dense attention** operation.  The authors claim that sparsity in attention is a natural pattern and show (by visual inspection) various examples where  most layers had sparse attention patterns across most data points, suggesting that adding sparsity to the attention would not signficantly affecting performance. In other layers, however, they noticed global patterns and data-dependent sparsity, whose performance could be affected by sparsity in the attention matrix.

As a reminder, full self-attention for autoregressive models defines $$S_i = {j : j ≤ i}$$, allowing every $$i$$-th element to attend to all previous
positions and its own position. In this paper, **Factorized self-attention** has $$p$$ separate attention heads, where the $$m$$-th head defines a subset of the indices $$A_i^{(m)} ⊂ {j : j ≤ i}$$ and let  $$S_i = A_i^{(m)}$$. The problem here is to find efficient choices for the subset $$A$$. Section 4.3 details 2D factorization methods via strided attention, or fixed patterns (figure below).  

To reduce memory, the authors propose using gradient checkpointing, i.e. recomputing forward-pass attention weights during background pass.
To efficiently perform vectorized computation on sparse memory representations, the authors implemented several transpose and slicing operations on the GPU (Section 5.5, note: I'd say they used CUDA shared memory for efficient random memory access). 

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/publications/sparse_transformers.png"/> 

<br/>
# 2019 [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, Google and Toyota](https://arxiv.org/abs/1909.11942)

ALBERT ("A Lite BERT") lowers memory consumption and increase the training speed of BERT. It allows for better scaling, establishes new record performance in several benchmarks, and with fewer parameters than BERT. An ALBERT configuration similar to BERT-large has 18x fewer parameters and can be trained about 1.7x faster. The tecnniques introduced are:
1. **Factorized embedding parameterization**, for parameter reduction.
"Instead of projecting the one-hot vectors directly into the hidden space of size $$H$$, we first project them into a lower dimensional embedding space of size $$E$$, and then project it to the hidden space. By using this decomposition, we reduce the embedding parameters from $$O(V × H)$$ to $$O(V × E + E × H)$$". This separation of the size of the hidden layers from the size of vocabulary embedding, makes it easier to grow the hidden size without significantly increasing the parameter size of the vocabulary embeddings.
2. **Cross-layer parameter sharing**, for parameter reduction. The authors mention that the parameter reduction also acts as regularisation/generalisation (reduces overfitting as the model learns a representation that generalizes well for all tasks). It does not improve the performance of the model though: "This approach slightly diminishes the accuracy, but the more compact size is well worth the tradeoff". This technique prevents the parameter from growing with the depth of the network. As a practical example, take a BERT model with 12 layers ie 12 Transformer encoder blocks: instead of learning unique parameters for each layer, ALBERT learns parameters of the first layer reuse the block in the remaining 11 layers. 
3. **Self-supervised loss for sentence-order prediction (SOP)**, for performance improvement. Instead of BERT's additional loss called next-sentence prediction (NSP, a binary classification loss for predicting whether two segments appear consecutively in the original text), the authors propose SOP, focused on inter-sentence coherence which is designed to address the ineffectiveness of the NSP in BERT. The SOP loss uses as positive examples the same technique as BERT (two consecutive segments from the same document), and as negative examples the same two consecutive segments but with their order swapped. This forces the model to learn finer-grained distinctions about discourse-level coherence properties.

<br/>
# 2019 [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, Microsoft](https://arxiv.org/abs/1910.02054)

ZeRO (as in Zero Redundancy Optimizer) is a parallelism method that "eliminates memory redundancies in data- and model-parallel training while retaining low communication volume and high computational granularity, allowing us to scale the model size proportional to the number of devices with sustained high efficiency". The results show the (at the time) largest language model ever created (17B parameters), beating Bert-large (0.3B), GPT-2 (1.5B), Megatron-LM (8.3B), and T5 (11B). It also demonstrates super-linear speedup on 400 GPUs (due to an increase of batch size per accelerator). 

As motivation, the authors first emphasize that state-of-art model parallelism splits the model vertically (on each layer), leading to high communication and scaling limitations. Conversely, data parallelism has good compute/communication efficiency but poor memory efficiency. They also analyse "memory consumption of the existing systems on model training and classify it into two parts: 1) For large models, the majority of the memory is occupied by model states which include the optimizer states (such as momentum and variances in Adam), gradients, and parameters. 2) The remaining memory is consumed by activation, temporary buffers and unusable fragmented memory ([...] residual states)." ZeRO-DP claims to have the computation/efficiency of Data Parallelism (DP) while achieving memory efficiency of Model Parallelism (MP).  This is achieved by three cumulative optimizations: Optimizer State Partitioning ($$P_{os}$$, 4x memory reduction and same communication as DP), Gradient Partitioning ($$P_{os+g}$$, 8x memory reduction, same comm.) and Parameter Partitioning ($$P_{os+g+p}$$, memory reduction linear with number of accelerations $$N_d$$, 50\% increase in communication). ZeRO-DP is at least as memory-efficient and scalable as MP, or more when MP can't divide the model evenly. 

This is achieved by "removing the memory state redundancies across data-parallel processes by partitioning the model states instead of replicating them, and [..] using a dynamic communication schedule during training". In practice, non-overlapping subsets of layers are delegated to different accelerators. Different optimization levels refer to what content is split or kept across GPUs, as in the figure below. Content that is not replicated but is instead divided in synchronized with dynamic communication across connecting layers. A parameter defining the level of optimization defines the trade-off between variables replicated across accelerators (just like Data Parallelism) and variables split across accelerators (as in Model Parallelism).  

At runtime, each processor is allocated a subset of data (DP) and a subset of the model (MP). When that data goes through its layers it will broadcast its layers parameters to other accelerators on the forward pass. Each GPU will run its own data using the received parameters. During the backward pass, gradients will be reduced. See bottom figure and [video here](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/).   

Finally, ZeRO can be complemented with techniques that reduce activation memory (compression, checkpointing, live analysis). CPU offloading is not recommended or used as "50% of training time can be spent on GPU-CPU-GPU transfers" and this would penalize performance heavily. As a final insight, when compared to MP, "Zero-DP has better scaling efficiency than MP because MP reduces the granularity of the computation while also increasing the communication overhead" and "Zero-R removes the memory redundancies in MP by partitioning the activations checkpoints across GPUs, and uses allgather to reconstruct them on demand". 

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/publications/zero.png"/> 

**ZeRO forward pass:** the initial portion of model ($$M_0$$) assigned to $$GPU_0$$. It broadcasts its model parameters $$M_0$$ to all GPUs (red arrows). Each GPU will do a forward pass of *their own data* on the received parameters. As we move forward in the model, other GPUs similarly communicate their parameters. The partial activations for each layer are stored by all GPUs. The loss is then computed for each GPU's data.  

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/publications/zero2.png"/> 

{: style="text-align:center; font-size: small;"}
image credit: adapted from images in [Microsoft Research Blog video](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) 

**ZeRO backward propagation:** on the first iteration of the Backwards pass, GPUs 0,1 and 2 hold the gradients of the last GPU's model layers $$M_3$$ for data points 0, 1 and 2. Combined with the partial activation stored, the partial gradient updates can be computed locally. An all-reduce of all updates will compute the averaged gradient update for model portion $$M_3$$ in $$GPU_3$$ (green arrows). All remaining layers follow analogously.  

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/publications/zero3.png"/> 

{: style="text-align:center; font-size: small;"}
image credit: adapted from images in [Microsoft Research Blog video](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) 


<br/>
# 2018 [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

This paper introduces efficient distributed intra-layer model parallelism for Multi-Layer Perceptrons and Transformer attention mechamism on GPT2, BERT and bidirectional Transformers. The method present is an orthogonal effort and can be combined with existing data, pipeline or model parallelism techniques. The approach approximates closely the pattern of linear scaling (Fig. 1). "We show that the existing BERT architecture results in model degradation as the size increases. We overcome this challenge by rearranging the layer normalization and residual connection in the transformer layers and show that with this change, results for the downstream tasks on development sets improve monotonically as the model size
increases." It also overcomes the limitations of data parallelims (where the model must fit entirely in one worker) and the idleness across time in pipeline parallelism. 

Section 3 includes details on the parallelism technique. On the MLP block, take each block being described as $$Y = GeLU(XA)$$:
- the typical approach is to split the weight matrix A along its rows and input X along its columns as (for 2 processors $$1$$ and $$2$$): $$X=[X_1, X_2]$$ and $$A=[A_1, A_2]^T$$. This partitioning will result in $$Y = GeLU(X_1A_1 + X_2A_2)$$. Since $$GeLU$$ is a nonlinear function, $$GeLU(X_1A_1+ X_2A_2) \neq GeLU(X_1A_1) + GeLU(X_2A_2)$$ and this approach will require a synchronization point (to sum both partial sums of products) before the $$GeLU$$ function.
- Another option is to split $$A$$ along its columns, i.e. it's a feature- (not row-) wise partitioning. This allows the $$GeLU$$ nonlinearity to be independently applied to the output of each partitioned GEMM: $$[Y1, Y2] = [GeLU(XA1), GeLU(XA2)]$$. This removes the synchronization point.

This approach splits both GEMMs in the MLP block across GPUs and requires only a single all-reduce operation in the forward pass (g operator) and a single all-reduce in the backward pass (f operator).

The same logic applies to the attention heads, where we split the key, value and query matrices similartly to the matrix $$A$$ above. A similar logic follows for the embeddings: "in transformer language models, the output embedding layer shares weights with the input embedding, requiring modifications to both. We parallelize the input embedding weight matrix $$E_{H×v}$$ along the vocabulary dimension $$E = [E1, E2]$$ (column-wise)". To reduce the communication in the output of the model (logits) the authors replace communication of logits by scalar losses. Finally, there are 4 total communication operations in the forward and backward pass of a single model parallel transformer layer (Fig. 4).

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/publications/MegatronLM.png"/> 

<br/>
# 2018 [Averaging Weights Leads to Wider Optima and Better Generalization (Stochastic Weight Averaging), Cornel & Samsumg AI](https://arxiv.org/abs/1803.05407)

The authors present SWA, a "simple averaging of multiple points along the trajectory of SGD, with a cyclical or constant learning rate, that leads to better generalization than conventional training" and provides "much flatter solutions than SGD". The rationale is: (1) SGB with constant or cyclical LR  traverse regions of weight space that correspond to high-performing networks, never reaching their central points. (2) Fast Gradient Ensembles (FGE) for $$k$$ models required $$l$$ times more computation. SWA is an approximation of FGE with the efficiency of a single model, with a better solution that SGD. The algorithm is the following: Starting from $$\hat{w}$$ we continue training, using a cyclical or constant learning rate schedule: 

- When using a cyclical learning rate we capture the models $$w_i$$ that correspond to the minimum values of the learning rate, i.e. the values at then end of each cycle (at the lowest learning rate value); 

- For *high constant* learning rates we capture models at each epoch. 

Next, we average the weights of all the captured networks wi to get our final model $$w_{SWA}$$. For cyclical learning rate schedule, the SWA algorithm is related to FGE, except that instead of averaging the predictions of the models, we average their weights, and we use a different type of learning rate cycle.  

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/publications/SWA.png"/> 


<br/>
# 2018 [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism, Google](https://arxiv.org/abs/1811.06965)

GPipe is a method for pipeline parallelism that allows the scaling of neural networks that are expressed as a sequence of layers. The method partitions the original model into subsets of consecutive layers across difference accelerators. This allows for batch pipelining but sequentially feeding to each accelerator one subset of the mini-batch at a time (a micro-batch), and pipelining the whole mini-batch as a sequence of micro-batches. The method applies synchronous (mini-batch) gradient descent with batch accumulation for all micro-batches. During the backward pass, gradients for each micro-batch are computed based on the same model parameters used for the forward pass. At the end of each mini-batch, gradients from all M micro-batches are accumulated and applied to update the model parameters across all accelerators. The authors claim that GPipe's pipelining (model parallelism) can also be complemented with data parallelism for further training scale. Regular pipelining struggles with RAM issues: when running several micro-batches per mini-batches, it is required to accumulate several input activations (on the forward phase) for the backward phase. Activations (not parameters) are the main memory-consuming factor in CNNs. Therefore, instead of keeping all those activations in memory, "in order to reduce activation memory requirements, [...] during forward computation, each accelerator only stores output activations at the partition boundaries, During the backward pass, the accelerator recomputes the composite forward function". Relating to efficiency and idleness, the "bubble" overhead in the picture can be considered negligible when M ≥ 4 × K, for M micro-batches and K accelerators. "This is also partly because re-computation during the backward pass can be scheduled earlier, without waiting for the gradients from earlier layers". Benchmark results demonstrate increases peroformance and an almost linear speedup on: image classification (AmoebaNet model) of 480x480 input images, and multilingual translation (128-layer Transformer) tasks. A comparison of runtime against Data Parallelism was not provided. As an important remark, this work was compared with PipeDream that does not follow the Bulk Synchronous Parallel. Moreover, due to the design of overlapping forward and backward passes in PipeDream, it requires maintaining multiple versioned copies of the model parameters. This prevents the PipeDream model to scale as well as GPipe. 

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/publications/gpipe.png"/> 


<br/>
# 2018 [PipeDream: Fast and Efficient Pipeline Parallel DNN Training, Microsoft Research, Carnegie Mellon, Stanford](https://arxiv.org/abs/1806.03377)

PipeDream is a parallel pipelining method that delivers perfect overlap of communication and computation, and uses all GPUs by overlapping forward and backward passes on data. Compared to other model parallelism techniques, it fully utilises all resources. It "allows perfect overlap of communication and computation. PipeDream keeps all available GPUs productive by systematically partitioning DNN layers among them to balance work and minimize communication, versions model parameters for backward pass correctness, and schedules the forward and backward passes of different inputs in round-robin fashion to optimize time to target accuracy". On completing the forward pass for a minibatch, each stage asynchronously sends the output activations to the next stage, while simultaneously starting to process another minibatch. Backpropagation proceeds similarly. Thus, the main issue with PipeDream is weight inconsistency ("weight staleness") caused by performing backward passes of previous mini-batches while doing forward passes of the current mini-batch: "We find that allowing the backward pass for a given minibatch to use more up-to-date parameters than those used in the corresponding forward pass can be a significant problem. PipeDream maintains parameter value versions for each in-flight minibatch to combat this problem". This leads to an increase of memory requirements. However, it only communicate data between neighboring GPUs, yielding less communication than distributed data parallel, that must communicate all parameters. Finally, PipeDream provides also data parallelism by being able to merge and divide layers across different GPUs. This is supported by: (1) an automatic partitioning scheme to delegate work to compute resources and (2) a work scheduler ("one-forward-one-backward") that alternates between running a forward and a backward tasks on the queue of tasks available on each GPU to provide a good global flow of the minibatches. A small memory efficiency is achieved by pre-allocating and reusing the GPU memory required for the activations, parameters and intermediate states required in the pipeline, avoiding dynamic allocations. "Experiments with five different DNNs on two different clusters show that PipeDream is up to 5x faster in time-to-accuracy compared to data-parallel training." Tasks performed: image classification with VGG16 and Inception-v3 models, and video classification with the S2VT model. 

{: style="text-align:center; font-size: small;"}
<img width="45%" height="45%" src="/assets/publications/pipedream2.png"/>  <img width="45%" height="45%" src="/assets/publications/pipedream3.png"/> 


<br/>
# 2018 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Google](https://arxiv.org/abs/1810.04805)

(also covered in a [different post]({{ site.baseurl }}{% post_url 2020-05-28-AI-Supercomputing-2 %}) )

Existing standard language models are unidirectional and that's a major limitation in performance, e.g. attending to previous tokens in the self-attention layers in the Transformer. This is an issue for many problems like question answering, it is crucial to incorporate context from both directions. BERT removes this unidirectionality by using a masked language model instead, that allows it to train a deep bidirectional Transformer. BERT model architecture is a multi-layer bidirectional sequence of Transformer encoder blocks. BERT models are trained in 2 steps: pre-training and fine-tuning. During pre-training, the model is trained on *unlabeled data* on different datasets. During fine-tuned, the pre-trained model is trained for a given specific task. Apart from output layers, the same architectures are used in both pre-training and fine-tuning. During fine-tuning, all parameters are fine-tuned. The input sentence may be a single sentence or a pair of sentences (e.g. question/answer) packed together. Words are embedded with WorkPiece embeddings. [CLS] is the first token of every sentence. [SEP] is a special separator token. To each token (word embedding) it is also added a learned embedding to indicate if it belongs to sentence A or B. Each input is then the sum of its position embedding, segment embedding and token embedding (Fig. 2). The pre-training happens in two unsupervised tasks: (1) Masked LM, by masking of 15% of input tokens at random and trying to predict them, and (2) and Next Sentence Prediction, by passing sentence pairs and predicting whether the second sentence is a logic follow up from the first, or not. The fine-tuning happens differently for every task: we pass the specific inputs and outputs to the BERT and do a regular training. The input is the sequences A and B and separators. The output is the answer to the task by: replacing [CLS] by the sentence or sentence-pair label when the task is to classify a sentence or pair or sentences; replacing the stard and end tokens to indicate the span of output answer tokens that answers the question passed in the input (when input is a question/answer pair, Fig 1); or the class of each word for Named Entity Recognition tasks.

{: style="text-align:center; font-size: small;"}
<img width="75%" height="75%" src="/assets/publications/bert.png"/> 

{: style="text-align:center; font-size: small;"}
<img width="75%" height="75%" src="/assets/publications/bert2.png"/> 


<br/>
# 2017 [Mixed Precision Training, Baidu and NVIDIA](https://arxiv.org/abs/1710.03740)

A method for training using half-precision floating point numbers, without losing model accuracy or having to modify hyperparameters. Weights, activations, and gradients are stored in IEEE halfprecision format. This nearly halves memory requirements and speeds up arithmetic. Due to the reduced range of 16- vs 32-bit representation, three techniques are proposed to prevent the loss of critical information (or numerical overflows):
1. Maintaining a single-precision copy of weights that accumulates the gradients after each optimizer step (this copy is rounded to half-precision for the forward- and back-propagation).
  - Ie, use single-precision for master weights and updates loss-scaling, and accumulating FP16 products into FP32. And half-precision otherwise. This requires memory *duplication* in two representations for the FP32 master copy of weights: weights, activations and gradients are stored as FP16, and in order to match the accuracy of the FP32 networks, an FP32 master copy of weights is maintained and updated with the weight gradient during the optimizer step.
  - The intuiting behind requiring FP32 for master weights is to include small values that cant be represented by FP16, and because the ratio of the weight value to the weight update is very large.
2. performing loss-scaling to preserve gradient values with small magnitudes.
  - Rationale: FP16 exponent bias centers the range of normalized value exponents to $$[−14, 15]$$ while gradient values in practice tend to be dominated by small magnitudes (negative exponents). 
  - Scaling up the gradients will shift them to occupy more of the representable range and preserve values that are otherwise lost to zeros
  - To implement scaling, scale the loss value
computed in the forward pass by shifting the gradient values into FP16-representable range, prior to back-opagation. By chain rule back-propagation
ensures that all the gradient values are scaled by the same amount. Finally, weight gradients must be unscaled before weight update to maintain the update magnitudes as in FP32 training.
  - Any scale factor can be used (without any loss), as long as its product with the maximum absolute gradient value is below 65504.
3. Using half-precision arithmetic that accumulates into single-precision outputs, which are converted to half precision before storing to memory.
  - Different arithmetics (vector dot-products, reductions, and point-wise operations) require different treatment.
  - "To maintain model accuracy, we found that some networks require that FP16 vector dot-product accumulates the partial products into an FP32 value, which is converted to FP16 before writing to memory. Without this accumulation in FP32, some FP16 models *did not match the accuracy* of the baseline models."
  - Large reductions (sums across elements of a vector) e.g. batch-normalization, should be carried out in FP32.
  - Point-wise operations, such as non-linearities and element-wise matrix products, are memory bandwidth limited. Since arithmetic precision does not impact the speed of these operations, either FP16 or FP32 math can be used.


<br/>
# 2018 [Group Normalization, Facebook AI Research](https://arxiv.org/abs/1803.08494)

This paper presents Group normalization. GN surpasses Batch Normalization particularly on small batch sizes, due to error increasing rapidly when the batch size becomes smaller, caused by inaccurate batch statistics estimation. This limits BN's udage for training larger models and trasferring features. The rationale is that BN exhibits drawbacks that are also caused by its distinct behavior of normalizing along the batch dimension. In particular, it is required for BN to work with a "sufficiently large batch size". 

Layer Normalization and Instance Normalization also avoid normalizing along the batch dimension. These methods are effective for training sequential models (RNN/LSTM) or generative models (GANs), but both have limited success in visual recognition, for which GN presented better results. 

Formulation: if $$i = (iN, iC, iH, iW)$$ is a 4D vector indexing the image features in (N batch size, Channels , Height, Widht), the mean is $$/mu_i = 1/m \sum_{k \in S_i x_k}$$ and the standard deviation is $$\sigma_i = \sqrt{1/m \sum_{k \in S_i} (x_k - \mu_i)^2 }$$: 
- Batch norm: $$S_i = \{ k \mid k_C = i_C \}$$ ie the output is a vector of the same length as channel count; 
- Layer norm: $$S_i = \{ k \mid k_N = i_N \}$$ ie the output is a vector of the same length as batch size; 
- Instance norm: $$S_i = \{ k \mid k_N = i_N, k_C = i_C \}$$ ie the output is a matrix of size $$N \times C$$; 
- Group norm: $$S_i = \{ k \mid k_N = i_N, \frac{k_C}{C/G} = \frac{i_C}{C/G} \}$$ ie the output is a matrix of size $$N \times C/G$$, for $$G$$ groups;

{: style="text-align:center; font-size: small;"}
<img width="75%" height="75%" src="/assets/publications/group_normalization.png"/>  <img width="23%" height="23%" src="/assets/publications/group_normalization_2.png"/> 

<br/>
# 2016 [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

GCNs are a variant of the Convolutional Neural Networks that operate on graphs. Similarly to CNNs, GCNs learn the features by inspecting neighboring nodes. The main difference is that CNNs are meant to operate on regular Euclidean structures (e.g. images), while GNNs are a generalized applicable to an arbitrary structure or order.

The background is the following. Similarly to the CNN downsampling/upsampling layers, there are several layers in the CGN model. We represent the hidden representation at a given layer as $$H^{(l)}$$. We also add the representation of the graph structure in matrix form $$A$$, e.g. an adjacency matrix or some function of it. Finally, we write the output of each layer as 

$$
H^{(l+1)}=f(H^{(l)},A) = σ(AH^{(l)}W^{(l)})
$$

with $$H^{(0)}=X$$ (i.e. the input data), and where W$$^{(l)}$$ is a weight matrix for the $$l$$-th neural network layer and $$σ(⋅)$$ is an activation function (e.g. ReLU). To address two limitations (include the node itself on all multiplications with $$A$$ and normalizing (by scaling) $$A$$, the final formula seen in the paper is then extended to:

$$
f(H^{(l)},A)=σ(\hat{D}^{−\frac{1}{2}} \hat{A} \hat{D} ^{−\frac{1}{2}} H^{(l)} W^{(l)}),
$$

with $$\hat{A}=A+I$$, where $$I$$ is the identity matrix and $$\hat{D}$$ is the (diagonal) [degree matrix](https://en.wikipedia.org/wiki/Degree_matrix) of $$\hat{A}$$. Note that $$D^{-\frac{1}{2}}$$ is the matrix with the reciprocal of the square root of each term in the diagonal. The layer-wise propagation rule is 

$$
h^{(l+1)}_{v_i}=σ(\sum_j \frac{1}{c_{ij}} h^{(l)}_{v_j} W^{(l)})
$$  

where $$j$$ indexes the neighboring nodes of $$v_i$$ and $$c_{ij}$$ is a normalization constant for the edge $$(v_i,v_j)$$.


Sections 2.2 and 2.3 provide theoretical background and section 3 demonstrates an example on the task of node classification, using softmax of the output as in regular CNNs. I used a separate [blog post from the author](https://tkipf.github.io/graph-convolutional-networks/) or [this post from Francesco Casalegno](https://towardsdatascience.com/graph-convolutional-networks-deep-99d7fee5706f) for a better explanation.

{: style="text-align:center; font-size: small;"}
<img width="65%" height="65%" src="/assets/publications/GraphConvNets.png"/> 

<br/>
# 2016 [Attention is all you need (Transformer), Google, NeurIPS 2017](https://arxiv.org/abs/1706.03762)

(also covered in a [different post]({{ site.baseurl }}{% post_url 2020-05-28-AI-Supercomputing-2 %}) )

State-of-art transduction models are based on recurrent encoder-decoder architectures (possibly with Attention Mechanisms). The Transformer uses only attention mechanisms, and no recurrence or convolutions. Results show it to be of better performance, more parallelizable (due to non-recurrence in model), and faster to train. Contrarily to recurrent models, the whole source sentence (in the encoder) and target sentence (in the decoder) are fed at once. Therefore, backpropagation happens on a single step as well. Because the concept of word sequence provided by the recurrence was removed, Transformers use positional encoding of the input embeddings based on the combination of sine and cosine waves of different frequencies. The encoder and decoder are composed of a stack of 6 layers each. Each encoder layer includes a multi-heard attention module and a feed forward network. The decoder includes also a third module, a *masked* multi-head attention, that ensures that sentence does not learn from subsequent words in sentence. An attention head is a mapping of a query to a set of key-value pairs. Key-Value pairs are output by the encoder, and Queries are output by the decoder. The formulation of this *dot-product attention* is: $$Attention (Q, K, V) = softmax( QK^T / \sqrt{d_k}) V$$. Here, the dot-product of all queries and the key ($$QK^T$$) gives a value referring to how well aligned the query vectors are for a given key. This is then converted into a distribution ($$softmax$$) and then used extract the most meaningfull value $$V$$ (by multiplying). This is effectively an indexing mechanism (similar to a dictionary $$value = query\_dict[key]$$) but in a continuous space. The scaling factor $$\sqrt{d_k}$$ is used to avoid having really small gradients for large values of $$d_k$$ (dimensionality of keys). The multi-head attention heads allows the model to jointly attend to information from different (8) representation. It is formulated as $$MultiHead(Q,K, V) = Concat(head_1, ..., head_h)W^O$$ where $$head_i = Attention(QW^Q_i ,KW^K_i , VW^V_i)$$, ie it's the linearly-transformed (projected) concatenation of the attention heads with projected Q, K, and V. In terms of performance, self-attention layers have complexity $$O(n^2 d)$$ per layer, compared to $$O(n d^2)$$ in recurrent models (for sequence length $$n$$ and representation dimension $$d$$) - which is typically faster as $$n < d$$ in most use cases. It also requires no recurrence and no attention connectivity between previous words in a sentence. 

{: style="text-align:center; font-size: small;"}
<img width="45%" height="45%" src="/assets/publications/transformer.svg"/> 

<br/>
# 2015 [Distilling the Knowledge in a Neural Network, Google](https://arxiv.org/abs/1503.02531), and 
# 2021 [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/), and 
# 2020 [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525) 

The **distillation** method is based on first training a *cumbersome* model (e.g. an ensemble of models, dropout, etc), and once trained, transfer the knowledge/skill of to a smaller model. In the use case of classification, "an obvious way to transfer the generalization ability of the cumbersome model to a small model is to use the class probabilities produced by the cumbersome model as 'soft targets' for training the small model" (while using the same training set). In the use case where the cumbersome model is an ensemble, we can use the geometric mean as target of the samll one. The rationale is that "when the soft targets have high entropy, they provide much more information per training case than hard targets and much less variance in the gradient between training cases, so the small model can often be trained on much less data than the original cumbersome model and using a much higher learning rate." I.g. distillation relies on the fact that the soft max assignments of the large network is a much better label for the input than the Hard MI, thus the smaller network has now the same input trained against a "cleaner" output, and thus requires less complexity to perform equally or better than the larger model . 

Applied to the MNIST data case, the previous approach (Caruana et al.) was to use the output of the logits (input to final layer) rather than the probabilities produced by the softmax (as they're too small), and minimize the squared difference between the logits of the cumbersome and the small model. The authors distillation method is to "raise the temperature of the final softmax until the cumbersome model produces a suitably soft set of targets", and "then use the same high temperature when training the small model to match these soft targets".  

Few variants of distillation: offline when only the post-training result of the big network is provided to the small network (with the same data, as above), or online when both networks train at the same time. Also possible, for the use case of deep networks, is to use ‘soft labels’ as the output of the bigger network after every X layers, and train the smaller network to learn to replicate the big network's outputs at every level, and not just the final loss. 

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/publications/model_distillation.png"/> 

<br/>
# 2015 [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) 

quote: *Training of DNNs is complicated by the fact that the inputs to each layer are affected by the parameters of all preceding layers – so that small changes to the network parameters amplify as the network becomes deeper. The change in the distributions of layers’ inputs presents a problem because the layers need to continuously adapt to the new distribution.* However, [...] *it has been long known (LeCun et al., 1998b; Wiesler & Ney, 2011) that the network training converges faster if its inputs are whitened – i.e., linearly transformed to have zero means and unit variances, and decorrelated.* 

Batch Normalization is a technique for training very deep neural networks that standardizes the inputs of the network layers, at every mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks. Batch Normalization "reduces the internal covariate shift". "Covariates" is just another name for the input "features". Covariate shift means the distribution of the features is different in different parts of the training/test data, breaking the i.i.d assumption used across most of ML. This problem occurs frequently in medical data (where you train on samples from one age group, but want to classify something from another age group), or finance (due to changing market conditions). Internal covariate shift refers to covariate shift occurring within a neural network, i.e. going from (say) layer 2 to layer 3. This happens because, as the network learns and the weights are updated, the distribution of outputs of a specific layer in the network changes. This forces the higher layers to adapt to that drift, which slows down learning. BN helps by making the data flowing between intermediate layers of the network look like whitened data, this means you can use a higher learning rate. Since BN has a regularizing effect it also means you can often remove dropout. 

In the results, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. 


<br/>
# 2015 [Siamese neural networks for one-shot image recognition, CS Toronto, ICML 2015](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

The paper describes **siamese neural networks** for efficient **one-shot learning** (not to be confused with zero-shot learning), the task of classification under the restriction that we may only observe a single example of each possible class before making a prediction about a test instance. The model learns to identify input pairs according to the probability that they belong to the same class or different classes. This model can then be used to evaluate new images, exactly one per novel class, in a pairwise manner against the test image. The pairing with the highest score according to the verification network is then awarded the highest probability for the one-shot task. 

In terms of structure, a siamese neural network consists of twin networks which **accept distinct inputs** but are joined by an energy function at the top. The parameters between the twin networks are tied (ie the **weights are shared**), guaranteeing that two extremely similar images could not possibly be mapped by their networks to different locations in feature space because each network computes the same function. I.e. the network is symmetric. For the task of image classification, the authors used a convolutional neural network as the base model of the siamese twins. The units in the final convolutional layer are flattened into a single vector. This convolutional layer is followed by a fully-connected layer, and then one more layer computing the induced distance metric between each siamese twin, which is given to a single sigmoidal output unit. The loss functions is a **binary cross-entropy** with a regularizer. 

The optimization follows a standard backpropagation where the gradient is additive across the twin networks due to the tied weights. Final results show that the model outperforms all available baselines by a significant margin and come close to the best numbers achieved by the previous authors. 

{: style="text-align:center; font-size: small;"}
<img width="47%" height="47%" src="/assets/publications/siamese_networks.png"/> $$\, \, \,$$ <img width="47%" height="47%" src="/assets/publications/siamese_networks_2.png"/> 

<br/>
# 2015 [Neural Machine Translation by Jointly Learning to Align and Translate (and Attention Mechanism), D. Bahdanau, K. Cho, Y. Bengio](https://arxiv.org/abs/1409.0473) (also covered in a [different post]({{ site.baseurl }}{% post_url 2020-05-28-AI-Supercomputing-2 %}))

In most encoder-decoder models, encoders encode a sentence into a vector of fixed-length, from which a decoder generates the translation. Thus, neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector. Here authores claim that fixed-length arrays are a bottleneck in performance on encoder-decoder architectures, particularly for long lentences. Therefore, the authors [quote] "propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly [...] The new architecture consists of a **bidirectional RNN as an encoder (BiRNN) and an uni-directional RNN decoder** that emulates searching through a source sentence during decoding.". A BiRNN consists of a forwards a a backward RNNs, containing the **summaries of the preceeding words and the following words**. The *annotation* of each word is the concatenation of the forward and backward states. The decoder receives the output of the previous decoded word, a hidden state for time $$i$$ (e.g. LSTM hidden state) and the context vector from a sequence of annotations - computed as a *weighted* sum of annotations. In practice, the encoder encodes the input sentence into a sequence of vectors and the decoder chooses a subset of these vectors adaptively while decoding the translation. 

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/publications/attention_mech.png"/> 

<br/>
# 2015 [Spatial Transformer Networks, Google DeepMind, NeurIPS 2015](https://arxiv.org/abs/1506.02025) 

{: style="text-align:center; font-size: small;"}
<img width="85%" height="85%" src="/assets/publications/STN.png"/> 

<br/>
# 2014 [Deeply-supervised Nets, USCD and Microsoft](https://arxiv.org/abs/1409.5185)

The rationale of Deeply-supervised nets is the following: in general, a discriminative classifier trained on highly discriminative features will display better performance than a discriminative classifier trained on less discriminative features. If the features in question are the hidden layer feature maps of a deep network, this observation means that the performance of a discriminative classifier trained using these hidden layer feature maps can serve as a proxy for the quality/discriminativeness of those hidden layer feature maps, and further to the quality of the upper layer feature maps. The basic network architecture will be similar to the standard one used in the CNN framework. Our additional deep feedback is brought in by associating a companion local output with each hidden layer. Backpropagation of error now proceeds as usual, with the crucial difference that we now backpropagate not only from the final layer but also simultaneously from our local companion output. Results suggests that it acts as a kind of feature regularization (which leads to significant reduction to the testing error but not necessarily to the train error) and it results in faster convergence, especially in presence of small training data.   

{: style="text-align:center; font-size: small;"}
<img width="65%" height="65%" src="/assets/publications/deeply_supervised_nets.png"/> 

<br/>
# 2014 [Generative Adversarial Networks (GANs), Univ Montreal, NeurIPS 2014](https://arxiv.org/abs/1406.2661)

(also detailed on a [different blog post]({{ site.baseurl }}{% post_url 2020-02-01-Generative-Adversarial-Networks %})) 

A new generative model composed of two models trained simultaneously: a generative model G that captures the data distributed, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework is the minimax 2-player game. The adversarial framework comes from the generative model facing the discrinative model that learns wether a sample is from the model distribution or the data distribution. *"The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency. Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles."* The generative model generates samples by passing a random noise through a multilayer perceptron. The discriminative model is also a multilayer perceptron. Because both models are connected deep neural networks, training is performed regularly via backpropagation. 

{: style="text-align:center; font-size: small;"}
<img width="65%" height="65%" src="/assets/Generative-Adversarial-Networks/GAN.png"/> 

{: style="text-align:center; font-size: small;"}
image credit: Benjamin Striner, lecture notes CMU 11-785) 

<br/>
# 2014 [Sequence to Sequence Learning with Neural Networks, Google, NeurIPS 2014](https://arxiv.org/abs/1409.3215)

(also detailed on a [different blog post]({{ site.baseurl }}{% post_url 2019-10-12-Variational-Autoencoders %})) 

A sequence-to-sequence or Encoder-(to-)Decoder architecture built on Deep Neural Networks of LSTM neurons, demonstrating efficient results on an English-to-French translation task. The main idea is that both Encoder and Decoder are RNNs that use LSTM neurons and its hidden states as a fixed-dimensional vector representation of the sequence so far. That representation is then passed it (i.e. concatenated) to the next token of the sentence. Token [EOS] delimited end of input and output sentences. 

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/publications/seq2seq.png"/> 

<br/>
# 2014 [Dropout: a simple way to prevent neural networks from overfitting, Univ. Toronto, Journal of ML Research 2014](https://jmlr.org/papers/v15/srivastava14a.html)

A method that drops neurons (in different layers) with a given **probability $$p$$** during train time. For each training minibatch, a new network is sampled. Dropout can be improved by adding max-norm regularization, decaying learning rate and high momentum. **At test time, all neurons are used, with outgoing weights multiplied by $$p$$**. Dropout helps **reducing overfitting**, as the network learns to never rely on any given activations, so it learns "redundant" ways of solving the task with multiple neurons. It also leads to sparse activations, similar to a regularization (L2). Dropping 20% of input units and 50% of hidden units was often found to be optimal in the original publication. It's computationally less expensive than regular model averaging of multiple trained DNNs. However, it takes 2-3 times longer to train than single fully-connected DNNs because requires way more epochs, as parameter updates are very noisy. Because a fully connected layer occupies most of the parameters, it is prone to overfitting. Therefore, dropout **increases model generalization**. 

{: style="text-align:center; font-size: small;"}
<img width="50%" height="50%" src="/assets/publications/dropout.png"/> 

<br/>
# 2013 [Auto-Encoding Variational Bayes (Variational Autoencoders), Universiteit van Amsterdam, 2013 ](https://arxiv.org/abs/1312.6114)

and [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) from the same authors. Also detailed on a [different blog post]({{ site.baseurl }}{% post_url 2019-10-12-Variational-Autoencoders %}). 

The VAE aims at approximating the distribution of the weights that generates an input, similarly to other variational inference. Te intractable true posterior $$p_{\theta}(z \mid x)$$ is approximated by $$q_\phi(z \mid x)$$ (the Encoder), whose parameters $$\phi$$ are not computed by a closed-form expectation but by the Encoder DNN instead. $$p_\theta(x \mid z)$$ is the Decoder, that given a $$z$$ will produce/generate the output which is a distribution over the possible values of x. Given a datapoint $$x$$ the encoder produces produces a distribution over the possible values of the code $$z$$ from which the datapoint $$x$$ could have been generated. The VAE proposed includes a DNN decoder, a DNN decoder, with parameters $$\theta$$ and $$\phi$$, where $$p_\theta(x \mid z)$$ is a Gaussian/Bernoulli with distribution parameters computed from $$z$$. Therefore the VAE can be viewed as two coupled, *independent* parameterized models: the encoder/recognition models, and the decoder/generative model (trained together), where the encoder delivers to the decoder an approximation to its posterior over latente random variables. One advantage of the VAE framework, relative to ordinary Variational Inference, is that the encoder is now a (stochastic) function of the input variables, in contrast to VI where each data-case has a separate variational distribution, which *is inefficient for large datasets*. Finally, the authors noticed that the sampling induces sampling noise in the gradients required for learning (or that because $$z$$ is randomly generated and cannot be backpropagated), and to can counteract that variance they use the “reparameterization trick”. It goes as follows: the sample vector $$z$$ that is typically sampled from the mean vector $$\mu$$ and variance $$\sigma$$ in the Gaussian scenario in now described as $$ z = \mu + \sigma \cdot \epsilon$$ where $$\epsilon$$ is always the standard gaussian ie $$\epsilon \sim N(0,1)$$. 

The loss function is a sum of two terms:

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/publications/vae_loss.png"/> 

The first term is the reconstruction loss (or expected negative log-likelihood of the i-th datapoint), comparing the model output with the model input and can be the losses we used in the autoencoders(such as L2 loss). The second term is the Kullback-Leibler divergence between the encoder’s distribution $$q_\theta(z\mid x)q$$ and $$p(z)$$, measuring how much information is lost (in units of nats) when using $$q$$ to represent $$p$$. It is one measure of how close $$q$$ is to $$p$$. 

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/publications/vae.png"/> 

{: style="text-align:center; font-size: small;"}
VAE vs AE structures. image credit: [Data Science Blog: Variational Autoencoders, by Sunil Yadav](https://data-science-blog.com/blog/2022/04/19/variational-autoencoders/)

<br/>
# 2011 [Popular Ensemble Methods: An Empirical Study, 2011 ](https://arxiv.org/abs/1106.0257)

A summary of results and conclusions on ensemble methods (bagging, boosting) on DNNs and decision trees. Bagging ensemble generally produces a classifier that is more accurate than a standard classifier. About Boosting: for a few data sets Boosting produced dramatic reductions in error (even compared to Bagging), but for other data sets it actually increases in error over a single classifier (particularly with neural networks). Alternatively, an **ensemble of similar DNNs initialized with different random seeds is surprisingly effective**, often producing results as good as Bagging. An ideal ensemble consists of highly correct classifiers that disagree as much as possible.

**Bagging trains the several different models with different datapoints** randomly sampled (**with replacement**, ie same samples can be redrawn) from the same dataset.  Bagging is effective on “unstable” learning algorithms (such as DNNs) where small changes in the training set result in large changes in predictions.  

**Boosting produces a series of classifiers**. The training set used for each member of the series is **chosen based on the performance of the earlier classifier(s) in the series**. Examples that are incorrectly predicted by previous classifiers in the series are chosen more often than those correctly predicted. Thus Boosting attempts to produce new classifiers that are better able to predict examples for which the current ensemble’s performance is poor. Ada-Boosting can use the approach of (1) selecting a set of examples based on the probabilities of the examples, or (2) simply using all of the examples and weight the error of each example by the probability for that example (i.e., examples with higher probabilities have more effect on the error) -- easier as these probabilities are incorporated in the dataset. 

{: style="text-align:center; font-size: small;"}
<img width="45%" height="45%" src="/assets/publications/ensemble_methods.png"/> 

<br/>
# 2011 [Cyclical Learning Rates for Training Neural Networks, US Naval Research Lab, 2017](https://arxiv.org/abs/1506.01186)

The author claims that cyclic learning rates improve time to convergence and increases accuracy of most models. It suggests triangular scheduler as a efficient method with similar results to other non-triangular cyclic schedulers. The paper also provides a method to find a good initial learning rate by doing several training short sessions (8 iterations) with different learning rates and picking the best initial learning rate from the analysis. Finally, provides "rule of thumb" parameters for min and max learning rates in the triangular scheduler proposed. 

<br/>
# 2006 [Connectionist Temporal Classification: Labelling Unsegmented: Sequence Data with Recurrent Neural Networks, IDSIA Switzerland, ICML 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

The paper presents a network and a loss function for the prediction on **sequences of labels from unsegmented input data**. The overcomes limitations of recursive neural networks that requires well-segmented data. Moreover, it has not been possible to apply RNNs directly to sequence labelling. The problem is that the standard neural network objective functions are defined separately for each point in the training sequence; in other words, RNNs can only be trained to make a series of independent label classifications. The basic idea behind CTC is to interpret the **network outputs as a probability distribution over all possible label sequences, conditioned on a given input sequence**. Given this distribution, an objective function can be derived that directly maximises the probabilities of the correct labellings. Since the objective function is differentiable, the network can then be trained with standard backpropagation through time. A CTC network has a softmax output layer, **with one more unit than there are labels in L**. The activations of the first $$L$$ units are interpreted as the probabilities of observing the corresponding labels at particular times. The activation of the extra unit is the probability of observing a ‘blank’, or no label. These outputs define the probabilities of all possible ways of aligning all possible label sequences with the input sequence. **The total probability of any one label sequence can then be found by summing** the probabilities of its different alignments. 

The main formulation is of the objective function is:  let $$y = N_w(x)$$ be the sequence of network outputs, and by $$y^t_k$$ the activation of output unit $$k$$ at time $$t$$, ie the probability of observing label $$k$$ at time $$t$$. Let $$L'^T$$ be the sequence of $$T$$ over the $$L' = L ∪ \{blank\}$$. $$\, \, \,$$ Then: $$p(π \mid x) = \prod_{t=1}^T y^{t}_{π_t}, ∀π ∈ L'^T $$.  From this equation we observe that that **the model assumes frames to be independent**. We define the conditional probability of a given labelling $$l ∈ L^{≤T}$$ as the sum of the probabilities of all the paths corresponding to it: $$p(l \mid x) = \sum_{\pi} p(\pi \mid x)$$. The classified is simply $$ h(x) = argmax_l \, p(l \mid x)$$. 

To efficiently calculate individual labellings, the authors describe the CTC forward-backward algorithm. Training follows the maximum likelihood principle. Experiments compare CTC with *framewise* method of Hidden Markov Model on the decoding of speech. For fairness, CTC and HMM used the same RNN architecture: bidirectional Long Short-Term Memory. her architecture could have been used instead. BLSTM was chosen because experiments with standard BRNNs and unidirectional networks gave worse results on the same task. Results show improved accuracy of CTC over HMM.    

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/publications/CTC.png"/> 

<br/>
# 2006 [Dimensionality Reduction by Learning an Invariant Mapping (contrastive loss), New York Uni, CVPR 2006](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)]

 The paper presents Dimensionality Reduction by Learning an Invariant Mapping (DrLIM) - for learning a globally coherent non-linear function that maps the data evenly to the output manifold. The problem is to find a function that maps high dimensional input patterns to lower dimensional outputs, given neighborhood relationships between samples in input space. It also presents the **Contrastive Loss Function**. The underlying rationale is that a meaningful mapping from high to low dimensional space maps similar input vectors to nearby points on the output manifold and dissimilar vectors to distant points. Therefore, the contrastive loss function runs over pairs of samples. The training is done by (1) collecting images of similar classes using prior knowledge, (2) pair a sample with all other training samples, and (3) traing them against the **binary classification** (1 or 0) to label them as belonging to the same or different classes, respectively. The neural network used is a **"siamese" architecture**, consisting of two copies of the function which share the same set of parameters, and a cost module. A loss module whose input is the output of this architecture is placed on top of it. The input to the entire system is a pair of images and a label Y. The images are passed through the functions, yielding two outputs $$G(X_1)$$ and $$G(X_2)$$. The cost module then generates the distance $$D_W(G_W(X_1), G_W(X_2))$$. The loss function combines $$D_W$$ with the label to produce the scalar loss $$L_S$$ or $$L_D$$. The **partial loss functions** $$L_S$$ and $$L_D$$ refer to the loss functions to optimize for similar and dissimilar objects. Experiments demonstrate the effectiveness of the method by learning a shift invariant mapping of MNIST samples and a learning temporal neighborhoods and lighting invariance of single objects (airplane). 

{: style="text-align:center; font-size: small;"}
<img width="45%" height="45%" src="/assets/publications/contrastive_loss.png"/>  <img width="47%" height="47%" src="/assets/publications/contrastive_loss_2.png"/> 
