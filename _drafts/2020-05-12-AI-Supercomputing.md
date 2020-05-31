---
layout: post
title:  "AI Supercomputing: from Linear Regression to BERT"
categories: [machine learning, supercomputing]
tags: [machinelearning]
---


[coordinate-ascent]: {{ site.baseurl }}{% post_url 2018-02-17-Supervised-Learning %}

tion can be performed using the [coordinate ascent][coordinate-ascent] method. 


Machine Learning is driven by mathematical models that try to *learn* from data. The complexity of the data is continuously increasing, due to higher-resolution photos, larger textual databases, higher number of observable features on input datapoints, etc. The computing power available *tends* to follow, or somehow adapt, as observed by [Moore's law](https://en.wikipedia.org/wiki/Moore%27s_law). However, in the occurence of very large datasets and very computationally-expensive learning models, the learning process is limited due to insufficient memory or an infeasibly-high training time. This is where **AI Supercomputing --- or technically speaking Parallel/Distributed computing of Machine Learning algorithms** ---  comes into place. 

AI Supercomputing focuses on how to distribute data (models, inputs) and computation across several compute units (vector units, CPU cores, GPU cores, and machines). The goal is to distribute the data in such way that the learning algorithm can be execution simultaneously (*parallelized*) with different datasets across all processors, speeding up the solution and reducing memory usage per machine. On the other hand, distributing data that are computationally dependent --- or equally, that need to be in the same memory region (machine) to be computed together at some point --- introduces another layer of analysis of efficiency due to the overhead on the communication required to move variables across machines or compute units. So the ideal algorithm can be characterized as the one that allows for:

1. homogeneous distribution of data across memory units, i.e. balanced memory usage;
2. homogeneous amount of computation assigned to each compute unit, i.e. balanced computation; and
3. a minimal amount of communication across memory/compute units, or ideally a zero communication overhead if overlaping of communication and computation is possible.

When these three properties are achieved, then we guarantee the **linear scaling** of the algorithm. This means that, by increasing (e.g. doubling) the compute resources, we decrease (halve) the computation. In practice, perfectly-linear scaling is very hard, but quasi-linear scaling is commonly achieved on cleverly designed algorithms. Let's start with the basics.


# Linear Regression and Mutual Exclusion

That a basic linear example of input variables $x$, labels $y$, learnt weights $w$ and a loss functions set at the Mean Absolute Error.
We want to minimize:

$$
MAE(w)  = \frac{1}{N} \sum_{n=1}^N | y_n - f(x_n) | \text{, where } f(x_n) = \sum_{j=1}^{M} w_j x_j
$$


To speed-up the solution, we can *parallelize* both sums in $MAE$ and $f(x_n)$ with several compute units (let's say $T$ compute threads) and decompose the computation as:

$$
\begin{align*}
MAE(w) & = MAE (w_{thread_1}) + MAE (w_{thread_2}) + ... + MAE (w_{thread_T})\\
& = \sum_{n=1}^{\lfloor N/T \rfloor} |y_n - f(x_n)| +  \sum_{n=\lfloor N/T \rfloor +1}^{2\lfloor N/T \rfloor} |y_n - f(x_n)| + ... + \sum_{n=(T-1)\lfloor N/T \rfloor +1}^{N} |y_n - f(x_n)|
\end{align*}
$$

\pause
\vspace{0.15cm}
This operation is \textit{memory-safe} for $f(x_n)$, but unsafe for $MAE(w)$. \pause Options:

\small
\begin{enumerate}
\item base case, no parallelism. \textbf{Slow!} \texttt{(AI\_SC\_1.cpp)}
\pause
\item each thread updates the MAE sum continuously. \textbf{Wrong!} \texttt{(AI\_SC\_2.cpp)}
\pause
\item same as before, with a \textit{mutual-exclusion} control. \textbf{Very slow!} \texttt{(AI\_SC\_3.cpp)}
\pause
\item same as before, with sums computed independently. \textbf{Good!} \texttt{(AI\_SC\_4.cpp)}
\end{enumerate}
}

\end{frame}

\begin{frame}{From CPU to GPU to TPU/IPU}
\Wider[2.5em]{
\small
\vspace{0.1cm}
\textbf{\alert{Take-home message:}} In Machine Learning:
\begin{enumerate}
\item (matrix-vector) computations can de decomposed and fully-parallelized;
\item Reductions are uncommon, so there's almost no synchronization overhead;
\end{enumerate}
\pause

\vspace{0.08cm}
\textbf{\alert{Exception:}} MCMC methods, due to parallel random number generation.
\begin{itemize}
\item choose between parallelism \small{(different seeds)} and reproducibility {\small(prev. option 3)}.
\end{itemize}


\pause \centering
\vspace{0.1cm}
\textbf{\alert{Rule:}} total parallel compute power (\textbf{FLOPS}, not Ghz) is the relevant hardware spec.

\pause
\centering
\vspace{0.2cm}
\begin{columns}
\column{0.45\textwidth}
\includegraphics[width=1.05\textwidth]{microsoft-sync-computing/figures/a53-power-curve.png}
\column{0.55\textwidth}

\pause	
\begin{tiny}
\begin{tabular}{l r r r}
\hline
 & \textbf{base CPU} & \textbf{FLOPS (32 bit)} & \textbf{Max RAM} \\
\hline
Intel Xeon 8180 & 28x 2.5 Ghz & 1.36 TFLOPS & 768 GB\\
Tesla K80 & 4992x 0.56 Ghz & 8.73 TFLOPS & 2x12 GB \\
GraphCore & * 1216 x 1.6Ghz & 31.1 TFLOPS &  ** 304 MiB \\
\hline
\end{tabular}

\vspace{0.2cm}
* TPUs use Accumulating Matrix Product (AMP) units, allowing 16 single-precision floating point operations per clock cycle.

\vspace{0.1cm}
** Small memory compensated by low latency.

\vspace{0.1cm}
\textbf{CPU:} 64- and 32-bit ops;
\textbf{GPU:} 64, 32, 16 (NVIDIA Pascal);
\textbf{TPU:} 64, 32, 16, ...

\vspace{0.3cm}
{\color{gray}\textbf{Source:} \href{https://www.graphcore.ai/products/ipu}{Dissecting the
Graphcore IPU Architecture via Microbenchmarking, Citadel Technical Report, 7 December 2019}}




\end{tiny}

\vspace{0.25cm} \centering \pause
\textbf{\alert{Memory is limited and expensive.}}
\end{columns}
}
\end{frame}

\begin{frame}{Deep Neural Nets on GPUs. GPU-offloading (vDNN)}
\Wider[2.5em]{
\centering
\small
\vspace{0.4cm}
GPUs are faster, but... \alert{how to overcome the memory limitations?}
\vspace{-0.2cm}

\pause
$$
x^{(l)} = f^{(l)} (x^{(l-1)}) = \phi ((W^{(l)})^T x^{(l-1)}) \text{ \hspace{0.2cm} and \hspace{0.2cm} }
L = \frac{1}{N} \sum_{n=1}^N | y_n - f^{(L+1)} \circ ... \circ f^{(2)} \circ f^{(1)} (x_n^{(0)}) |
$$

\pause
\vspace{0.1cm}
\includegraphics[width=0.85\textwidth]{microsoft-sync-computing/figures/vDNN.png}

\pause
\vspace{0.2cm}
vDNN\footnote{ Source, References: \textbf{vDNN}:
\href{https://arxiv.org/pdf/1602.08124.pdf}{Rhu et al., vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design, Proc. 49th Annual IEEE/ACM Symposium on Microarchitecture (MICRO)};
 \textbf{vDNN+}: \href{https://www.cse.iitb.ac.in/~shriramsb/submissions/GPU_mem_ML.pdf}{Shriram et al, Dynamic Memory Management for GPU-based training of Deep Neural Networks, Proc. IPDPS 2019} 	
}: 
Keep model in CPU memory, and active layer in GPU memory.

For larger networks, use hard drive (SSD).
}
\end{frame}

\begin{frame}{Deep Neural Nets on GPUs. GPU-offloading (vDNN)}
\Wider[3.5em]{
\centering \scriptsize
\textbf{Forward pass}

\includegraphics[width=0.75\textwidth]{microsoft-sync-computing/figures/vDNN2.png}

\pause
\vspace{0.1cm}
\textbf{Backward propagation}

\includegraphics[width=0.75\textwidth]{microsoft-sync-computing/figures/vDNN3.png}

\vspace{-0.6cm}
\scriptsize
$$
\delta_j^{(l)} =  \frac{\partial L_n}{\partial z_j^{(l)}} = \sum \frac{\partial L_n}{\partial z_k^{(l+1)}} \frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}} = \sum_k \delta_k^{(l+1)} W_{j,k}^{(l+1)} \phi '(z_j^{(l)})
\text{ \hspace{0.3cm}\, where \hspace{0.3cm} } 
z_j^{(l)} =  (W^{(l)})^T x^{(l-1)}
$$

}
\end{frame}

\begin{frame}{Pipeline Parallelism (G-Pipe, PipeDream)}
\Wider[2.5em]{
\vspace{0.5cm}
\centering
\includegraphics[width=0.38\textwidth]{microsoft-sync-computing/figures/Pipedream_DNN_pipeline.PNG}
\hspace{0.5cm}
\pause
\includegraphics[width=0.53\textwidth]{microsoft-sync-computing/figures/Pipedream_DNN_pipeline_parallel.PNG}

\small \pause
\vspace{0.2cm}
Backward prop. starts after all forward pass. finishes. Can we do better?

\pause
\vspace{0.6cm}

\includegraphics[width=0.53\textwidth]{microsoft-sync-computing/figures/Pipedream_DNN_pipeline_parallel_Microsoft.PNG}

{\tiny \color{gray}
\vspace{0.5cm}
\href{https://arxiv.org/abs/1811.06965}{Google, GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism, ArXiv};

\href{https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/}{Microsoft, PipeDream: Generalized Pipeline Parallelism for DNN Training} }
}
\end{frame}


\begin{frame}{Data Parallelism}
\centering
\vspace{1.2cm}
\includegraphics[width=1.0\textwidth]{microsoft-sync-computing/figures/DNN_data_parallelism.pdf}

\pause
\vspace{0.8cm}
\textbf{\alert{Issue:}} Not memory efficient. Model is duplicated!

\vspace{0.6cm}
{\tiny \color{gray}
Source: \href{https://arxiv.org/abs/1811.03600}{Google Labs, Measuring the Effects of Data Parallelism on Neural Network Training
, arXiv}}
\end{frame}


\begin{frame}{Model Parallelism (DNN)}
\centering
\includegraphics[width=0.9\textwidth]{microsoft-sync-computing/figures/DNN_model_parallelism.pdf}

\pause
\vspace{1cm}
\textbf{\alert{Issue:}} Not communication efficient!
\end{frame}


\begin{frame}{Partial Model Parallelism (CNN)}
\Wider[3em]{
%Input: N points x C channels x W width x H height
%weights: F filters x C channels x (K x K) filter size
%output: N  x Filter x Widht' x Height'
\small
\vspace{0.1cm}

\begin{center}
\vspace{0.1cm}
\includegraphics[width=1.0\textwidth]{microsoft-sync-computing/figures/argonne_parallel_2.PNG}
\end{center}

\scriptsize
\textbf{LEFT:} Parallelism of image across \alert{four processors};  {\color{red}Red box:} center of 3x3 convolution filter; {\color{red}red arrow}: data movement; {\color{violet}violet region:} elements to be communicated at every step so perform filter of elements at the border. \textbf{RIGHT:} communication across \alert{two processors}. {\color{red}Red arrow:} forward phase dependencies; {\color{blue}Blue arrow:} back-propagation dependencies;

%\textbf{Equations:}

\small
\vspace{0.5cm}
{\small Equation 1:} \hspace{0.1cm} $ y_{k,f,i,j} = \sum_{c=0}^{C-1} \sum_{a=-O}^{O} \sum_{b=-O}^{O} x_{k,c,i+a,j+b} w_{f,c,a+O,b+O} $

{\small Equation 2:} \hspace{0.1cm}  $ \frac{dL}{dw_{f,c,a,b}} = \sum_{k=0}^{N-1} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \frac{dL}{dy_{k, f, i, j}} x_{k, c, i+a-O, j+b-O} $

{\small Equation 3:} \hspace{0.1cm}  $ \frac{dL}{dx_{k,c,i,j}} = \sum_{j=0}^{F-1} \sum_{a=-O}^{O} \sum_{b=-O}^{O} \frac{dL}{dy_{k, f, i-a, j-b}} w_{f, c, a+O, b+O} $

\vspace{0.2cm}
{\tiny \color{gray}
Source: \href{https://arxiv.org/pdf/1903.06681.pdf}{Dryden et al., Improving Strong-Scaling of CNN Training by Exploiting Finer-Grained Parallelism, Proc. IPDPS 2019} }

}
\end{frame}


\begin{frame}[standout]
\centering
\vspace{1cm}
So far, we know that:

\vspace{1.2cm}
\begin{small}
\begin{itemize}
\item ML is very parallel due to efficient Matrix-vector multiplication;
\item Memory is limited but we overcome it with \textbf{CPU-offloading};
\item We can \textbf{pipeline} parallelism;
\item We can parallelize data and use mean batch gradients;
\item We can parallelize the model (locally e.g. CNN);
\end{itemize}
\end{small}

\pause
\vspace{1.2cm}
\textbf{\alert{Any ML model that is not covered?}}

\begin{tabular}{l l}
\parbox{0.45\textwidth}{
}
\end{tabular}

\vspace{1.5cm}

\small brunomaga.github.io
\end{frame}


\begin{frame}{Limitations of parallelism (Encoder-Decoder, Seq-to-Seq)}
\Wider[2.5em]{
\small 

\begin{center}
\includegraphics[width=0.48\textwidth]{microsoft-sync-computing/figures/Encoder_Decoder.pdf}
\pause \hspace{1.2cm} \vspace{0.3cm}
\includegraphics[width=0.65\textwidth]{microsoft-sync-computing/figures/encoder_decoder_3.png}
\end{center}

\scriptsize 
\vspace{-0.6cm}
\begin{itemize}
\item \pause \textbf{encoding/decoding is a recursive algorithm} $\rightarrow$ iterations can't the parallelized;
\item \pause \textbf{Single hidden layer with \textit{small} embedding}  $\rightarrow$ no performance gain on parallelizing layers;
\item \pause \textbf{Inputs/outputs of different lengths} $\rightarrow$  only matching batch sizes can be parallelized;
\end{itemize}

\pause
\centering \alert{...also important: \textbf{Attention Mechanism}}

%\pause Also, long sentences lead to vanishing/exploding gradients.

}
\end{frame}

\begin{frame}{Transformer}
\Wider[4em]{
\vspace{-0.05cm}
\begin{columns}
\column{0.33\textwidth}
\includegraphics[width=1.23\textwidth]{microsoft-sync-computing/figures/transformer.PNG}

\vspace{-0.1cm}
{\color{gray}\tiny (\href{https://arxiv.org/abs/1706.03762}{Vaswani et al., Attention is all you need, Arxiv})}
\column{0.55\textwidth}

\begin{small}
\pause\textbf{\alert{Encoder}}
\end{small}

\begin{tiny}

\pause Model has no recurrence or convolution, so we need a \textbf{positional encoder} to give context of order of words in sentence. Example: \textit{My \textbf{dog} is loud} vs \textit{I look like a \textbf{dog}}. %Dimensionality of PE is the same as embeddings $d$ so that they can be summed.

$ PE_{(pos,2i)} = sin\left(\frac{pos}{10000^{2i/d}}\right) \text {\hspace{0.25cm} and \hspace{0.25cm}} PE_{(pos,2i+1)} = cos\left(\frac{pos}{10000^{2i/d}}\right) $\\

\pause \vspace{0.3cm}\textbf{Multi-head Attention} solves for $n$ heads, \textit{What part of the input should I focus on?}\\\vspace{-0.3cm}
\begin{center}
\includegraphics[width=0.6\textwidth]{microsoft-sync-computing/figures/transformer_attention.PNG}
\end{center}

\vspace{-0.5cm}
$$
Attention(K, V, Q) = softmax\left(QK^T / \sqrt{d_k}\right) V
$$

\vspace{-0.7cm}
$$
MHA(K, V, Q) = [head_0,.., head_n]W^M \text{, \hspace{0.2cm}} head_i = Attention(KW^K_i, VW^V_i, QW^Q_i)
$$

\vspace{-0.1cm}
\pause \textbf{Feed Forward} is a regressor (single hidden-layer DNN) that transforms the attention vectors into a form that is valid as input to the decoder.

\end{tiny}

\vspace{0.15cm}
\begin{small}
\pause\textbf{\alert{Decoder}}
\end{small}

\begin{tiny}

\pause \textbf{Masked Multi-head Attention} similar to regular MHA, but replaces upper diagonal of attention vector by zeros, to hide next word from model model. \\\vspace{-0.3cm}
\begin{center}
\includegraphics[width=0.4\textwidth]{microsoft-sync-computing/figures/transformer_attention_masked.png}
\end{center}

\vspace{-0.3cm}
\pause \textbf{Multi-head attention} determines how the words in input \& output sentences interact.

\vspace{0.1cm}
\pause \textbf{Linear} expands the space into an array of size equals to French vocabulary. 

\vspace{0.1cm}
\textbf{Softmax} tranforms into a prob. distribution. Word with highest probability is picked.\\
\end{tiny}

\end{columns}
}
\end{frame}


\begin{frame}{Transformer (2)}
\Wider[3.5em]{
\vspace{0.2cm}

\begin{columns}
\column{0.33\textwidth}
\includegraphics[width=1.23\textwidth]{microsoft-sync-computing/figures/transformer.PNG}

\vspace{-0.1cm}
{\color{gray}\tiny (\href{https://arxiv.org/abs/1706.03762}{Vaswani et al., Attention is all you need, Arxiv})}

\column{0.55\textwidth}

\small 
\textbf{\alert{Computational Complexity}}

\includegraphics[width=1.\textwidth]{microsoft-sync-computing/figures/transformer_table.png}

\begin{tiny}
$n$: sequence length, $d$: representation dim., $k$: kernel size; $r$: size of neighbourhood.

\end{tiny}

%RNN: $d^2$ multiplications (multiplication of weights in fully connected layer of DNN) for each of the $n$ words in the sentence

%Self-Attn (Encoder):  Attention matrix $n^2$, where each element of attention matrix has embedding $d$

%amount of computation that can be parallelized, as measured by the minimum number of sequential operations

\vspace{0.3cm}
\begin{scriptsize}
\pause \textbf{{Why is it that $n^2 d$ is better than $n d^2$?}}

\pause Sentences length $n \approx 70$, and word embeddings $d \approx 2000$.
\end{scriptsize}
\pause

\vspace{0.5cm}
\pause
\textbf{\alert{Parallelism}}

\begin{scriptsize}
Still limited Dec. batch size, but no more Enc. recursion!
\end{scriptsize}

\vspace{0.5cm}
\pause
\textbf{\alert{Rationale}}

\begin{scriptsize}
\begin{itemize}
\item Encoder learns English language (context);
\item Decoder learnt the English-to-French translation;
\end{itemize}
\end{scriptsize}

\vspace{0.3cm}
\pause
\textbf{Can we get rid of the Decoder} and use only the Encoder to learn complex tasks?
\end{columns}

}
\end{frame}


%\begin{frame}%{Transformer 3D}
%\centering \small
%\includegraphics[width=0.75\textwidth]{microsoft-sync-computing/figures/transformer_3d.jpg}
%\end{frame}

\begin{frame}{BERT \small (Bidirectional Encoder Representation from Transformers)}
\Wider[3em]{

\small

\begin{center}
BERT is a stack of Transformer encoders. Learns language \textit{context}.

\vspace{0.1cm}
\includegraphics[width=0.9\textwidth]{microsoft-sync-computing/figures/BERT.PNG}
\end{center}

\pause

\textbf{Pre-Training: 2 self-supervised prediction tasks at same time}:
\begin{itemize}
\item \pause tasks: Masked Language Model; and Next Sentence Prediction;
\item \pause trained on Wikipedia, 24 BERT layers, batches of 256 sentences * 512 tokens;
\end{itemize}

\pause
\vspace{0.2cm}
\begin{columns}
\column{0.45\textwidth}
\texttt{\tiny
Input = [CLS] the man went to [MASK] store [SEP]\\
\hspace{0.7cm}he bought a gallon [MASK] milk [SEP]\\
Label = IsNext\\}

\column{0.45\textwidth}
\texttt{\tiny
Input = [CLS] the man [MASK] to the store [SEP]\\
\hspace{0.7cm}penguin [MASK] are flight \#\#less birds [SEP]\\
Label = NotNext\\}
\end{columns}

\pause
\vspace{0.2cm}
\begin{columns}
\column{0.1\textwidth}
{\small \textbf{Input Layout}}

\column{0.71\textwidth}

\includegraphics[width=1.0\textwidth]{microsoft-sync-computing/figures/BERT_input.PNG}
\end{columns}

\vspace{0.2cm}
{\tiny \color{gray}
\href{https://arxiv.org/abs/1810.04805}{BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Google AI Language}
}

}
\end{frame}


\begin{frame}{BERT \small (Bidirectional Encoder Representation from Transformers) (2)}
\Wider[3em]{

\small

\textbf{Fine-Tuning:}  Adding one layer to a pre-trained BERT to learn to solve most tasks.

\begin{center}
\vspace{-0.2cm}
\includegraphics[width=0.6\textwidth]{microsoft-sync-computing/figures/BERT_tasks.png}
\end{center}

\vspace{-0.1cm}
\pause
\tiny
Information encoded by BERT is useful but, on its own, insufficient to perform a translation task [due to no left-to-right prediction]. However, "BERT pre-training allows for a better initialization point for [an] NMT model", \href{https://arxiv.org/abs/1909.12744}{Clichant et al.,On the use of BERT for Neural Machine Translation, arXiv}

}
\end{frame}



%\begin{frame}%{Transformer 3D}
%\centering \small
%\includegraphics[width=0.75\textwidth]{microsoft-sync-computing/figures/transformer_3d.jpg}
%\end{frame}

\begin{frame}{Microsoft ZeRO \& DeepSpeed}
\Wider[2.5em]{

\scriptsize	

\pause
Remember BERT is a stack of Transformer Encoders, i.e. a sequence of matrix-vector multiplication?

\pause
\vspace{0.3cm}
Remember Data Parallelism (DP) and Model Parallelism (MP)?

\vspace{0.2cm}
\begin{columns}
\column{0.55\textwidth}
\includegraphics[width=1.0\textwidth]{microsoft-sync-computing/figures/DNN_data_parallelism.pdf}
\column{0.38\textwidth}
\includegraphics[width=1.0\textwidth]{microsoft-sync-computing/figures/DNN_model_parallelism.pdf}
\vspace{0.5cm}
\end{columns}

\pause
\vspace{0.3cm}
Remember the inputs and outputs on each layer of forward and backward propagation?

\includegraphics[width=0.45\textwidth]{microsoft-sync-computing/figures/vDNN2.png}
\hspace{1cm}
\includegraphics[width=0.45\textwidth]{microsoft-sync-computing/figures/vDNN3.png}

\pause
\vspace{0.2cm}
\textbf{ZeRO (Zero Redundancy Optimizer) combines all}: \textit{"[...] achieves the computation/communication efficiency of DP while achieving memory efficiency of MP, [...] retaining the computational granularity and communication volume of DP using a dynamic
communication schedule during training"}

\pause
\centering
\vspace{0.2cm}
\scriptsize
Video: 
\href{https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/}{ZeRO \& DeepSpeed: New system optimizations enable training models with over 100B parameters}

%ZeRO removes the memory redundancies across data-parallel processes by partitioning the model states—parameters, gradients, and optimizer (Adam) state—across data parallel processes instead of replicating them. 

%partitions optimizer states, gradients and parameters

%We show that ZeRO can be combined with any model parallelism
 
%We call this ZeRO-powered data parallelism, which allows per-device memory usage to scale linearly with the degree of data parallelism and incurs similar communication volume as data parallelism. 

\vspace{0.2cm}
{\color{gray}\tiny
Sources: \href{https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/}{ZeRO \& DeepSpeed announcement page}, \href{https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/}{Turing-NLG blog post}; \href{https://www.deepspeed.ai/}{www.deepspeed.ai/}; \href{https://github.com/microsoft/DeepSpeed\#further-reading}{DeepSpeed github docs}; \href{https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/}{ZeRO paper};
} 
}
\end{frame}

\begin{frame}{Microsoft ZeRO \& DeepSpeed}
\Wider[2.5em]{
\vspace{0.2cm}
\textbf{State of the Art}: \pause Bert-large (0.3B)\pause , GPT-2 (1.5B)\pause , Megatron-LM (8.3B)\pause , T5 (11B). \pause ZeRO can run 100B parameters \pause but they didn't, takes longer than a year for training! \pause So they ran 17B.

\centering \pause
\vspace{0.3cm}
\includegraphics[width=0.9\textwidth]{microsoft-sync-computing/figures/ZeRO_superlinear_speedup_60B_parameter.PNG}

\vspace{0.2cm} \pause
\textbf{super-linear speedup in the regime of 64-400 GPUs}

\small \vspace{0.1cm} \pause
\textit{"This is a property of ZeRO-DP
which reduces the memory footprint of the model states as we increase the DP degree, allowing
us to fit larger batch sizes per GPU"}
}
\end{frame}

\begin{frame}[standout]
\centering
\Wider[2.5em]{
\vspace{1cm}
\centering Thank you
\vspace{1.5cm}

\small
Linear Regression $\cdot$ CPU offloading $\cdot$ Pipeline parallelism $\cdot$
\\data parallelism $\cdot$ model parallelism $\cdot$ Encoder-Decoder $\cdot$
\\Transformer $\cdot$ BERT $\cdot$ ZeRO $\cdot$ super-linear speedup

\vspace{1.5cm}

%\small brunomaga.github.io
}
\end{frame}
