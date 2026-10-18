# TODO

Open items that the text rewrite and the picture work found but did not fix. Each item names the post, the problem
and a proposed fix. The user picks the items to fix before the next pass.

## Code issues

Bugs in the code and data files of the posts (under `assets/`), and in the styles of the site (under `_sass/`).

### Distributed Sorting Algorithms (2014-06-21-distributed-sort.md)

- `assets/distributed-sort/distributedmemorysorter.cxx`, `oddEvenSort_MPI`: the loop runs exactly `mpiSize` phases,
  which sorts the data only when every rank holds the same number of elements. Its own test, `test_OddEvenSort`,
  gives rank r `1000000 + 3r` elements. A simulation with unequal counts fails to sort in about a third of the
  cases after p phases. Fix: balance the counts first, or run phases until an even and an odd phase in a row change
  nothing (an all-reduce of a "changed" flag per phase).
- `assets/distributed-sort/distributedmemorysorter.cxx`, sample sort: lines 687 to 691 take the samples at
  floor((m-1) i/p) for i = 1 to p (the comment says p-1, and only p-1 are gathered), and line 805 picks the splitters
  `recvSamples[i*p-1]`, the largest sample of each group. Both shift the splitters upward. A simulation with 2,000
  elements per rank gives rank 0 1.44 n/p elements and the last rank 0.67 n/p at p = 64 (at most 1.54 n/p), and 1.17
  and 0.86 n/p at p = 16. Fix: take p regular samples per rank at i m/p, for i = 0 to p-1, and the splitters at
  k p + p/2 - 1, as in PSRS (Shi and Schaeffer, 1992), which gives about 1.0 n/p on the first and the last ranks, and
  at most 1.13 n/p at p = 64. The example of the post's picture does not change (splitters 4 and 10, counts 9, 4
  and 8).

### Distributed Load Balancing and Spatial Slicing (2015-12-01-slicing.md)

- `assets/slicing/slicing.tar.gz`, `DistributedLoadBalancing.cxx` line 20 and `DistributedSorting.cxx` line 28: the
  checksum test is inverted. It warns "checksum ... failed" when the sums before and after are equal
  (`std::abs(pre_sum - post_sum) < 0.000001`), and stays silent when they differ. The warnings are off by default
  (`outputWarnings = false`). Fix: warn when the difference is larger than a relative tolerance, e.g.
  `std::abs(pre_sum - post_sum) > 1e-9 * std::abs(pre_sum)`, since the sum of many doubles changes with the order
  of the additions.
- Optional, `assets/slicing/slicing.tar.gz`, `DistributedSorting.cxx` line 414: the splitters are chosen as in the
  sample sort of the sorting post, and shift upward in the same way. The balance step then evens out the counts, but
  more data moves. Fix: as for the sorting post.

### The Leaky Integrate-and-Fire Neuron and the Brunel Network (2016-02-08-lif-brunel.md)

- `assets/lif-brunel/lif-brunel.tar.gz`, `include/Neuron.hpp` line 6 includes itself, which shows as the self-loop of
  `diagram_class.svg`. Fix: delete the line and the duplicate include in `src/Neuron.cpp`, then regenerate the graph.
- Same archive, `src/Network.cpp`, `make_connections`: each pair of neurons is connected with the probability
  epsilon, so the in-degrees are binomial around 800 and 200, while Brunel's model and the caption of the network
  picture give every neuron exactly C_E excitatory and C_I inhibitory inputs. Fix: draw exactly C_E and C_I inputs per
  neuron.
- Same archive, `Physics.hpp`: `NUMBER_OF_NEURONS` defaults to 12,500, while the tables and plots of the post use
  10,000 neurons. Fix: a default of 10,000, or `-n 10000` in the instructions.
- Optional, same archive, `step_analytic` computes `exp(-dt/tau)` at every step. Fix: compute it once.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- `assets/unsupervised-learning/kohonen_mnist.zip`, `kohonen.py`, `name2digits`: the function loads `hash.mat`,
  which is not in the zip, so the script stops with a `FileNotFoundError` before the training starts. Fix: add
  `hash.mat` to the zip, or set the digits directly, e.g. `targetdigits = [0, 1, 2, 3]`, the digits of the post's
  figure.

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- `assets/bayesian-linear-regression/bayesian_linear_regression.ipynb`, predictive distribution cell: the data
  have a noise standard deviation of 0.25 (`create_toy_data(sinusoidal, 25, 0.25)`), but the model uses
  `BayesianRegression(alpha=1e-3, beta=2.)`, i.e. a noise standard deviation of 0.71. The predictive band is thus
  three times too wide. Fix: `beta=16.` (1 / 0.25^2).
- Same notebook, posterior cell: `plt.contour(w0, w1, multivariate_normal.pdf(...))` uses the default levels, and
  draws the contour at density 0 where the pdf underflows. Fix: pass
  `levels=pdf.max() * np.exp(-0.5 * np.array([9, 4, 1]))`, or draw the contours of the log-density.
- Same notebook, `GaussianFeature` docstring: "exp(-0.5 * (x - m) / v)" misses the square that the code applies.
  Fix: "exp(-0.5 * (x - m)^2 / v)". `predict(..., sample_size)` says it samples "from the predictive
  distribution", but it returns the noise-free lines `X @ w` for weights drawn from the posterior. Fix: "samples of
  the mean function X @ w, with w drawn from the posterior".

### Building a GPT model in PyTorch from scratch (2023-02-28-gptlite.md)

- Post, section "Multi-Head Masked Attention": the uniform attention matrix is created on `device`, but `x` and
  `tril` on the CPU, so on a GPU machine `wei @ x` and `wei.masked_fill(tril==0, ...)` fail with a device mismatch.
  The alternative notation applies the mask and the softmax to the `wei` of the previous snippet, which gives the
  uniform matrix only because the unmasked entries of each row are equal. Fix: drop `device=device` from the first
  snippet, and start the second one with `wei = torch.zeros((S,S))`.
- Post, section "Training Loop": `eval_loss` runs with dropout on, since the model never switches to eval mode, and
  the loop defines the function again at every step. Fix: define `eval_loss` once before the loop, and call
  `m.eval()` at its start and `m.train()` before it returns, as `assets/gptlite/main.py` does.
- Post, section "Final Model": the comment "one layer normalization layer after transformer blocks and one before
  linear layer that outputs the vocabulary" describes the single `self.ln` as two layers. Fix: "one layer
  normalization after the transformer blocks, before the linear layer that outputs the vocabulary".
- Post, section "Tokenization": the code block that lists the 65 characters is marked `python`, so the highlighter
  marks "!" as an error. Fix: remove the language tag `python` from the opening fence.
- `assets/gptlite/gptlite.py`, `GPTlite.__init__`: the comment `# Add this` after `self.ln = nn.LayerNorm(d_model)`
  is a leftover editing note. Fix: remove it.
- `assets/gptlite/utils.py`, `sin_cos_positional_embeddings`: no code calls it, and it differs from the formula of
  the transformer: the base is 1000 instead of 10000, and the odd dimension 2i+1 uses the exponent (2i+1)/E instead
  of the 2i/E of its pair. Fix: remove the function, or set `div_term = 10000 ** (2 * (torch.arange(E) // 2) / E)`.
- `assets/gptlite/utils.py`, `get_gpt3_small_model_parameters`: the docstring dates the GPT-3 paper to 2021 instead
  of 2020, and `batch_size = 500_000` counts tokens in the paper (0.5M tokens, about 244 sequences of 2048 tokens),
  while `get_batch` reads it as a number of sequences. The only caller overrides it. Fix: "Brown et al, 2020", and
  `batch_size = 500_000 // 2048`, or a comment that the value counts tokens.

### GPT in C++: LibTorch vs. PyTorch, TorchScript and torch.compile (2023-06-27-gptlite-cpp.md)

- Post and `assets/gptlite-cpp/gptlite.cpp`, `Head::forward`: the scores are scaled by `std::pow(E,-0.5)`, with
  E = `n_embd` = 768, instead of `head_size`^-0.5 = 1/8, as in the Python code and the paper. Fix:
  `wei = wei * std::pow(head_size, -0.5);`.
- `assets/gptlite-cpp/main.cpp` and `main.py`: `main.cpp` times with `clock()`, the CPU time of all the threads of
  the process, and `main.py` with `time.time()`, the wall-clock time. Neither waits for the GPU
  (`torch::cuda::synchronize()`, `torch.cuda.synchronize()`) before it reads the clock. The C++ and Python
  throughputs thus come from different clocks. E.g. LibTorch is 11% faster than PyTorch 2.1 on the inference of the
  wide DNN, whose forward pass takes 0.35 s and launches 6 kernels, a gap that the Python overhead cannot explain.
  Fix: time both with the wall-clock time (`std::chrono::steady_clock` in C++), synchronize the GPU before each
  reading of the clock, and rerun the benchmark.
- `assets/gptlite-cpp/main.cpp` and `main.py`, `benchmark_train`: a softmax is applied to the logits before
  `cross_entropy`, which applies a log-softmax itself, so the model trains on a doubly normalized output. The
  runtime barely changes, but the training is wrong. Fix: remove the softmax line in both files.
- `assets/gptlite-cpp/main.py`: it imports `block_size` and `n_embd` from `assets/gptlite/gptlite.py`, which no
  longer defines them, so it stops with an `ImportError`. The current Python `GPTlite` also takes other constructor
  arguments, returns logits of shape (B, S, C), and has biases in its projections, so it is not the model of the C++
  code. `torch_compile_kwargs` sets `mode='default-overhead'`, which is not a mode of `torch.compile`; with
  `torch_compile_kwargs` set, the script also writes none of the `.pt` files that `./main [jit_folder]` loads. Fix:
  keep a copy of the GPTlite code of the GPT-lite post in `assets/gptlite-cpp`, so that Python and C++ run the same
  model, and set `torch_compile_kwargs = None` to export the TorchScript models, or a valid mode such as
  `dict(mode='reduce-overhead', fullgraph=False)`.
- Post and `assets/gptlite-cpp`: the post shows `gptlite.h` and `benchmark.h` as header-only files, `GPTlite` built
  without a device, and benchmark functions without a model name. The repository splits the code into `.h` and
  `.cpp` files, passes the device to `GPTlite`, runs 10 times more inference iterations than training iterations
  (`epochs_multiplier = 10`), and uses the Adam betas (0.5, 0.5) in C++ and the defaults (0.9, 0.999) in Python.
  Fix: show the repository code in the post, and use the same optimizer settings in both languages.
- `assets/gptlite-cpp/README.md` names the post by an old title, "Building a GPT model in C++, and benchmarking
  LibTorch, PyTorch, TorchScript and torch.compile", and `gptlite.h` dates the GPT-3 paper to 2021 instead of 2020.
  Fix: the current title, and 2020.

### Data parallelism, sharding and CPU offloading (2023-08-18-data-parallelism.md)

- `assets/distributed-training/gptlite_ds.py` imports `n_layer`, `n_embd`, `n_head`, `block_size` and `Block` from
  `assets/gptlite/gptlite.py`, which no longer defines the first four, and whose `Block` now takes
  `(d_model, n_heads, d_head, dropout_p)`. `train.py` thus stops with an `ImportError`, for this post and for the
  pipeline parallelism post. Fix: define the GPT-3 Small dimensions in `gptlite_ds.py`, and build the blocks with
  `Block(n_embd, n_head, n_embd // n_head, dropout)`.
- Post, PyTorch and DeepSpeed training loops: `torch.nn.functional.cross_entropy(outputs, labels)` and
  `torch.nn.CrossEntropyLoss()` receive GPTlite logits of shape (B, S, C) and labels of shape (B, S). Cross-entropy
  reads dimension 1 (S) as the classes and stops with a shape error. `train.py` uses `CrossEntropyLoss_FlatView`
  instead. Fix: flatten in the post, `loss = criterion(outputs.flatten(0, 1), labels.flatten())`, or show
  `CrossEntropyLoss_FlatView`.

### Pipeline parallelism (1F1B, Zero Bubble, DualPipe) (2023-08-30-pipeline-parallelism.md)

- Toy PyTorch pipeline in the post (`pipeline_step_no_overlap`): stage 0 and the middle stages overwrite `act` with
  the stage output. A middle stage then sends `act.grad` of its output, which is `None`, so a pipeline with more than
  2 stages stops at the first backward pass. Stage 0 also calls `requires_grad_(True)` on its input, which fails for
  integer token ids, and the receive buffers take the shape of the input `x_mb`, not of the activations. Fix: keep
  the input and the output of the stage apart, as `pytorch_pipeline_train_step()` in
  `assets/distributed-training/gptlite_ds.py` does (`act_in`, `act_out`, then `dist.send(act_in.grad, ...)`).
- `GPTlite.to_layers()` in the post starts with a lambda that sums the embeddings. `PipelineModule` does not
  register the parameters of a layer that is not an `nn.Module`, so the optimizer does not update the embedding
  tables, and `partition_method="parameters"` counts them as 0, as the partition output in the results shows.
  `gptlite_ds.py` returns the module `emb_sum` instead. Fix: show the `gptlite_ds.py` version,
  `[self.emb_sum, *self.blocks, self.ln, self.lm_head]`, adapt the paragraph after it, and rerun the partition
  example of the results with the `LayerSpec` version that the runs use.
- The activation checkpointing snippet reads `args.activation_checkpoint_interval` inside `get_model()`, where
  `args` is not defined. Fix: `'activation_checkpoint_interval': activation_checkpoint_interval`, as in
  `gptlite_ds.py`.
- Post and `train.py`: `step_count = len(train_dataset) // engine.gradient_accumulation_steps()` is the number of
  `train_batch()` calls per epoch only with one pipeline and micro-batches of 1 sample. With 4 stages on 8 GPUs (2
  pipelines), each epoch of the loop runs 2 passes over the data. Fix:
  `step_count = len(train_dataset) // engine.train_batch_size()`.
- `train.py` also uses `CrossEntropyLoss_FlatView` for the benchmark model, whose outputs have shape
  (batch, classes), so `B, S, C = logits.shape` stops with a `ValueError`. Fix:
  `F.cross_entropy(logits.flatten(0, -2), labels.flatten())`, which accepts both shapes.
- The `ImportError` of `gptlite_ds.py` (see the data parallelism entry) also stops the code of this post, and the
  post's `LayerSpec(Block, n_embd, n_head)` follows the old `Block` signature. Fix: as in the data parallelism
  entry.

### Tensor parallelism (2023-09-02-tensor-parallelism.md)

- `Megatron_FeedForward.__init__` builds `nn.Dropout(dropout)`, but `dropout` is not one of its arguments, while
  `Megatron_MHA` takes it as an argument. The class only works with a global `dropout`. Fix: add a `dropout`
  argument to `Megatron_FeedForward.__init__`.
- A comment of `Megatron_MHA` says "then we all-reduce (Megatron_g) to combine ranks", in the first person. Fix:
  "then Megatron_g all-reduces (sums) the outputs across ranks".

### Mixture-of-Experts and Expert Parallelism (2024-01-19-mixture-of-experts.md)

- The single-node `MoE` builds its experts with `FeedForward(input_size, output_size, dropout=dropout)`, but the
  `FeedForward` of the GPTlite post takes only `n_embd`, as `MoE_dist` uses it. The post defines no `FeedForward`
  with an output size, and `MoE_deep` needs one for its last layer. Fix: define
  `FeedForward(input_size, output_size, dropout)` in the post, next to the `Router`.
- `MoE_dist` sets `capacity = int(S / self.num_experts * self.capacity_factor)` with `k=2`. Each row sends `k*S`
  tokens to the experts, `k*S/N` per expert on average, so the default capacity factor of 1.25 drops at least 37.5%
  of the assignments, even with a perfect balance. GShard sets the capacity of top-2 gating to `2T/N`, and the
  `top2gating` of DeepSpeed doubles the capacity factor. Fix:
  `capacity = int(self.k * S / self.num_experts * self.capacity_factor)`.
- `MoE_dist` has no load balancing loss, so the router can send most tokens to a few experts. The text states this.
  Proposed fix: compute the Switch loss in `forward()` from `probs` and `topk_experts`, and return it with the
  output.

### Variable sequence lengths, curriculum learning, adaptive batch size and LR (2024-05-10-training-variable-length-curriculum.md)

- `sample_sort` assigns each row to a rank by its first column only. Rows with the same key, such as samples of
  the same length, all go to the same rank, and a dataset with fewer distinct keys than ranks leaves some ranks
  empty. The result is still sorted, but the memory and the time of the sort are unbalanced. Fix: sample and
  bucket on a unique key, such as `key * num_rows + row_id` for integer keys.
- The `assert` in `collate_fn_wrapper` lost its line break: the post puts 21 spaces between
  `assert sample_padding_fn is not None,` and the message. The DeepSpeed source ends the first line with `\` and
  puts the message on the next line. The code runs, but the line reads like a typo. Fix: restore the two lines.

### Sequence and context parallelism with Ulysses and Ring attention (2024-08-30-sequence-context-parallelism.md)

- `isend_k_and_v` passes the group ranks `(rank + 1) % P` and `(rank - 1) % P` as the `peer` of `dist.P2POp`, but
  `torch.distributed` reads `peer` as a global rank (`isend`: "Destination rank on global process group
  (regardless of group argument)"). The code works only for a group of the global ranks 0 to P-1, not for the
  second sequence-parallel group of a data- and sequence-parallel run. Fix:
  `dst = dist.get_global_rank(group, (rank + 1) % P)`, and the same for `src`.
- The ring code calls `fa._flash_attn_forward` with 8 return values and a `window_size` tuple, and
  `fa._flash_attn_backward` with the same tuple, as in flash-attn 2.6. In current flash-attn (2.8),
  `_flash_attn_forward` returns 4 values and takes `window_size_left` and `window_size_right`. The text names the
  version. Fix: pin `flash-attn==2.6.*`, or update the calls.
- Both attention implementations run without a causal mask, while the GPTlite blocks they replace are causal. The
  text states it. Fix: `causal=True` in the `flash_attn_func` of Ulysses; in the ring, the causal mask on the
  diagonal block, and no computation for the blocks above the diagonal.
- The comments of the ring `MultiHeadAttention.forward` use `E` for both the embedding size and the head size:
  `(B, S/P, E) -> (H, B, S/P, E)`. Fix: `(H, B, S/P, d_head)` and `(B, S/P, H, d_head)`.
- The ring `MultiHeadAttention.forward` is indented by 5 spaces. Fix: 4 spaces.
- The docstring of `dist_view_swap` and its line `H, B, S, E = full_shape` call the last dimension E, but
  `MultiHeadAttention` calls the function on tensors whose last dimension is the head size. Fix: `(H, B, S, d)` in the
  docstring, and `H, B, S, d = full_shape`.

### Diffusion models, from DDPMs to DiTs (2025-02-25-diffusion.md)

- The decoder comment in `DiT.__init__` (in `diffusion/dit.py` and in the post) quotes the DiT paper with other
  words. The paper reads "We use a standard linear decoder to do this; we apply the final layer norm (adaptive if
  using adaLN) and linearly decode each token into a p×p×2C tensor". Fix: quote these words, or drop the quotation
  marks.
- The EMA comment "with a warmup as in diffusers" (in `diffusion/diffusion.py` and in the post): the decay
  (1+n)/(10+n) is the default of the diffusers `EMAModel`, which gives the name "warmup" (`use_ema_warmup`) to
  another formula. Fix: "0.9999 as in the paper, ramped up as in the default EMAModel of diffusers".

### Faster attention (GQA, FlashAttention, sparse, linear and latent attention) (2025-12-30-inference-attention.md)

- `indexer_loss` in the DSA code block masks the scores with -1e9, which does not fit in float16 (its largest value
  is 65,504), while the comment of `scaled_dot_product_attention` in the same post gives this as the reason to use
  `torch.finfo(dtype).min`. The queries of the first block of CSA have rows that are fully masked. Fix: replace both
  -1e9 of `indexer_loss` with `torch.finfo(<tensor>.dtype).min`.

### Pruning and quantization (_drafts/2026-04-22-inference-pruning-quantization.md)

- `int8_matmul` (in `inference/quantization.py` and in the post) multiplies INT32 tensors with `@`, which PyTorch
  runs on the CPU only: on a GPU it fails with `"addmm_cuda" not implemented for 'Int'`. `W8A8Linear` and
  `LLMInt8Linear` call it, and `quantization.py` runs the model on the GPU when there is one. Fix: on a GPU, use
  `torch._int_mm` on the INT8 tensors (it needs more than 16 rows and sizes that are multiples of 8), or multiply
  in FP64, which is exact for these sums.
- Optional, the FP8 KV cache experiment of `quantization.py` applies `fake_quantize_fp8` to the whole cache after
  every decode step, with one scale per tensor that follows the largest value so far. When the scale grows, the
  older entries are rounded again, which adds error that a real FP8 cache, which stores each entry once, does not
  have (1.4 times the error in a test on random data). Fix: quantize only the new entries, with a fixed scale per
  layer, for example taken from the prompt.
- Optional, the comment `# ... (ReLU(z / s) = ReLU(z) / s for s > 0)` of `quantize_awq` (`inference/quantization.py`
  line 178, and in the post) writes the input as z, which is the zero point in the text. Fix:
  `ReLU(y / s) = ReLU(y) / s`, as the text writes it.

### Publications bookmark (publications.md)

- `_sass/site/_labels.scss`: the rules `.label-table td`, `td:first-child` (`width: 1%`), `td + td` and the phone
  rule (`.label-table tbody, tr, td { display: block }`) also match the table inside the notes of DeepSeek-V2, the
  only table in the notes. Its first column shrinks, so $$2\,n_h\,d_h$$ wraps over two lines, its cells lose their
  padding and borders, and on phones the table breaks into blocks (also on master). Fix: child selectors for the
  label table only, `.label-table > tbody > tr > td` (and the same for `:first-child`, `+ td` and `:empty`), and in
  the `$on-palm` rule `.label-table > tbody`, `.label-table > tbody > tr` and `.label-table > tbody > tr > td`.

## Diagram issues

Pictures that are wrong or disagree with the post text (under `assets/`). Items marked "Optional" improve a picture
that is not wrong.

### All pictures

- The kernel fusion post on expert parallelism (2026-10-09-kernel-fusion-expert-parallelism.md): its 11 pictures,
  `inference/imgs/ep_*.svg`, are copies of the figures of the paper repository and keep the Times New Roman font of
  the paper figures. They show at widths set by hand (40% to 100%), so their text sizes differ from the other
  pictures. Fix: redraw them at page scale and in the fonts of CLAUDE.md, in a tab `kernel-fusion-expert-parallelism`
  of `inference/inference.drawio`, from the draw.io or TikZ sources of the paper repository (CLAUDE.md, Pictures,
  sections 2 to 4), as the tensor parallelism post's pictures are, and check their captions.
- Style details that the restyle left, from an automated check of the 230 draw.io figures: 178 boxes in 15 figures
  keep the rounded corners of the original with a corner radius of 2 to 4.5 px instead of 5 px (for example the
  GPT-lite pictures, `diffusion/playground_v3.svg` and `publications/visual_transformer.svg`), and the 12 chat
  bubbles of `publications/llama2_gatt.svg` 15 px; 147 arrows in 19 figures are curved but are neither pointers nor
  loops (for example in `deep-neural-networks/densenet.svg`, `lif-brunel/diagram_class.svg` and
  `publications/graphconvnets.svg`); 176 filled cells of `distributed-training/imgs/mixture_of_depths.svg`, and a few
  in other figures, have no outline; `learning-from-sequences/transformer_attention.svg` (#97afd2, a heat map shade)
  and `diffusion/sora_vae.svg` (#333333) use colors outside the palette; 8 figures have lines thinner than 0.7 px;
  `publications/attention_mech.svg` has 8 arrow heads smaller than size 6; `vae-gan/gan.svg` has a gradient glow
  around the prior. Fix: set them to the values of CLAUDE.md, then run the check again.

### Distributed Matrix Transpose Algorithms (2015-04-21-matrix-transpose.md)

- `matrix-transpose/all_to_all.svg`: the caption writes "block $$j$$ of process $$i$$ goes to process $$j$$, as its
  block $$i$$", while the post calls the processes ranks, numbers them $$r$$ and $$s$$, and uses $$i$$ and $$j$$ for
  vertices. Fix: "`MPI_Alltoall` on 6 ranks: block $$s$$ of rank $$r$$ goes to rank $$s$$, as its block $$r$$. The
  blocks of the first rank, A0 to A5 (red frame), become the first block of every rank." The label "processes" of the
  picture can stay, since the mixture-of-experts post shows the same picture and says processes.
- `matrix-transpose/matrix-transposition-crs.svg`: the labels "Metadata counts (MPI_Alltoall)" and "Metadata sparse
  matrix (MPI_Alltoallv)", and the group label "Metadata" over "Row id", "Column id" and "Element", copy the XCSR
  wording. The CSR transpose sends the element counts, then the elements with their row and column indices, and the
  post uses "metadata" only for XCSR. Fix: "Element counts (MPI_Alltoall)" and "Elements with their indices
  (MPI_Alltoallv)", without the group label. Optional: "Row index" and "Column index", as in the text.
- `matrix-transpose/crs_format.svg`, `xcrs_format.svg`: the array of column indices is labeled `disps`, MPI's name
  for displacements, which reads as offsets. Optional fix: `col_idx`, as in the matmul post.
- Optional, `matrix-transpose/xcrs_transpose_3.svg`: the labels "Objects to be sent" and "Objects to be received"
  name the values of the view swap, which the post calls values. Fix: "Values to be sent" and "Values to be
  received".

### Distributed Load Balancing and Spatial Slicing (2015-12-01-slicing.md)

- `slicing/balance.svg`, its caption, and `split.svg` call the items elements ("19 sorted elements", "7, 6, 6
  elements", "18 elements"), while the text and the caption of `sbs.svg` call them objects. Fix: "objects" in the
  labels and in the caption.
- `slicing/n_body_slicing.svg` has no caption, and does not name the source of its N-body render. Fix: a caption,
  with a credit if the render comes from a source.
- `slicing/mini-*.svg`: the four layouts have 10 (single axis), 35 (NUG, 5 x 7), 16 (ORB) and 35 (STR, 5 x 7)
  slices. Optional fix: 16 slices in all four (4 x 4 for NUG and STR), so that they compare on equal terms.

### The Leaky Integrate-and-Fire Neuron and the Brunel Network (2016-02-08-lif-brunel.md)

- `lif-brunel/raster_plot_g4.5_f0.9.svg`, `raster_plot_g5_f2.svg`, `raster_plot_g3_f2.svg`: the plots do not name
  their state. The post used to call them "slow", "regular" and "fast" dynamics, but their parameters are those of
  Brunel's states SI slow (g = 4.5, eta = 0.9), AI (g = 5, eta = 2) and SR (g = 3, eta = 2); the text now uses these
  names. Fix: a caption under each plot, e.g. "SI slow (g = 4.5, eta = 0.9)", since CLAUDE.md puts titles in captions.
- Same three plots: the label "time interval (ms)" overlaps the bars at the right end of the lower panel, and
  "spikes count" reads better as "spike count (per ms)". Fix: move the label below the axis.

### Numerical Simulation of Detailed Neuron Models (2016-08-15-neuron-electrical-activity.md)

- `neuron-electrical-activity/brain_scales.svg`: the LadyofHats drawing is not restyled and not at page scale. It is
  716 px wide and shows at 50%, so its sans-serif panel letters (a to d) and ion labels show at 61% of their size.
  Fix: keep the drawing, scale it to about 440 px, redraw the letters and labels in Source Sans 3 (14 px and 12 px),
  and set P = W/880.
- Caption of `neuron-electrical-activity/brain_scales.svg`: the first credit, "Mariana Villarreal (LadyofHats),
  Wikimedia Commons", has no title and no link, and the author's name is Mariana Ruiz Villarreal. Panel (b) is her
  public-domain drawing "Complete neuron cell diagram en.svg". Fix: "Adapted from: [Complete neuron cell diagram,
  Mariana Ruiz Villarreal (LadyofHats), Wikimedia
  Commons](https://commons.wikimedia.org/wiki/File:Complete_neuron_cell_diagram_en.svg); and Blausen.com staff, ...",
  after a check of the sources of panels (c) and (d).
- Caption of `neuron-electrical-activity/neuron_model_scales.svg`: the credit "J. M. Eppler, Introduction to NEST,
  First community workshop on NESTML, 2015" has no link. Fix: a link to the slides, if they are online.
- Caption of `neuron-electrical-activity/net_current_conservation.svg`: "the axial currents $$i_a$$ balance the
  membrane current $$i_m$$", but the text defines $$i_m$$ as a current density (mA/cm²), and the axial currents
  balance its integral over the membrane. Fix: "the axial currents $$i_a$$ balance the membrane current density
  $$i_m$$, integrated over the membrane area".
- `neuron-electrical-activity/compartment_rc_circuit.svg` and its caption name the capacitor $$C$$, each resistor
  $$R$$ and the total current $$I$$. In the text, the capacitance of compartment $$j$$ is $$c_j$$, $$R$$ is the gas
  constant, and $$I$$ is the inhibitory population and the identity matrix. One resistor stands for all the channels
  of one type, not for one ion channel, and the caption has no final period. Fix: the label c_j, and the caption "The
  electrical circuit of one compartment. The membrane is the capacitor $$c_j$$. Each channel type $$s$$ is a resistor
  $$R_s$$ in series with a battery $$E_s$$, its reversal potential. Arrows mark gated channels, whose resistance
  changes over time. The leak $$L$$ has a constant resistance. $$I = i_{m,j} A_j$$ is the total current through the
  membrane."

### Variable Timestep Simulation of the Electrical Activity of Neurons (2017-02-17-neuron-variable-timestep.md)

- `neuron-variable-timestep/cvode_pulse_currents_table.svg`, `cvode_dependency_on_variation_solution.svg` and
  `cvode_dependency_on_events_arrival.svg`: the currents are in mA and µA (2.7 mA, "threshold current (0.206mA)",
  "0.1µA" to "1mA"), as in the preprint. NEURON's `IClamp` amplitude is in nA, and the threshold current of a cortical
  pyramidal cell model is of the order of 0.1 to 1 nA, so 0.206 mA is likely 0.206 nA. The text gives no current
  values. Fix, after a check of the simulation scripts: mA to nA and µA to pA in the labels.
- `benchmark_two_rows_2.svg`: the legend writes (1a) as "BSP CN Exponential" and (2a) as "BSP Deriv. Implicit†",
  while the text writes `cnexp` and `derivimplicit`. The dagger has no key: it is the footnote of the preprint's
  caption (Newton iterations solve each mechanism implicitly, but the coupling between mechanisms stays explicit),
  which the text explains for `derivimplicit`. Fix: "(1a) BSP cnexp" and "(2a) BSP derivimplicit", without the
  dagger.
- Optional, `benchmark_two_rows_2.svg`: the orange "region of similar runtime growth" does not follow from the plotted
  runtimes. Fix: check it against the benchmark data, or remove it.
- `neuron_203076.svg`, `neuron_138083.svg`, `neuron_208223.svg` and `neuron_87784.svg`: the last 10 ms bar of each
  event histogram leaves out the events after 7.5 s, and the burst bars near 0 s are cut at the top of the plot (198,
  203, 44 and 47 events) without a mark. Fix: redraw the histograms from the event data of the simulations, with the
  count on each cut bar.
- `cvode_pulse_currents_table.svg`: the CVODE step counts cover 1.1 s of simulation, while the 48,001 steps of
  backward Euler cover 1.2 s. The values are those of the table of the preprint. Fix, after a check of the simulation
  scripts: step counts for the same time span.
- `neuron_208223.svg` and `neuron_87784.svg` are shown for the median 1% of neurons, which receive 541 to 558 events,
  but their 10 ms bars add up to 584 and 586 events. The same sums give 6,146 and 5,574 for the top neurons, which
  agree with the text. Fig. 8 of the preprint has the same mismatch, and the text now gives both counts. Fix: show
  two neurons of the median 1%, from the event data, and drop the sentence with the counts.
- `benchmark_two_rows_2.svg`: the text gives the speed-ups of the preprint, 7.7 to 1.8 at the moderate rate, but the
  chart reads about 1.5 on 65,536 neurons. Fix: check the speed-ups with the benchmark data.
- The 13 pictures without a caption, most of them figures of the preprint, have no credit. The kernel fusion posts
  have the same open point. Fix: a caption under each picture, ending with "Source: [Fully-Asynchronous
  Fully-Implicit Variable-Order Variable-Timestep Simulation of Neural Networks](link)", if the user wants credits
  for the own preprint.

### Linear Regression, Matrix Factorization and Loss Functions (2017-08-17-linear-regression-matrix-factorization.md)

- Optional, `linear-regression-matrix-factorization/matrix-factorization.svg`: the rating 6 (movie 1, user 7) is
  outside a scale of 1 to 5 stars. Fix: a rating from 1 to 5.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- `unsupervised-learning/gaussianscatterpca.svg`: the arrow heads end past the scaled eigenvectors, at 3.75 and 1.75
  instead of 3 and 1, the lengths in the Wikipedia figure it follows. Fix: arrow heads that end at the tips of the
  vectors.
- Optional, `unsupervised-learning/k-means_convergence.gif`: an animation of 15 frames. Fix: an SVG with a few
  iterations side by side, computed by running k-means on the same data.

### Deep Neural Networks, Backpropagation, Dropout, CNNs and Embeddings (2018-03-27-deep-neural-networks.md)

- `deep-neural-networks/fully_sparse_connectivity.svg` and its caption keep the symbols of Figure 9.2 of the Deep
  Learning book: inputs x1 to x5 and outputs s1 to s5. The text writes the input of a layer as $$x^{(0)}$$ and its
  output as $$x^{(1)}$$, as `dnn.svg` does. Fix: inputs x₁⁽⁰⁾ to x₅⁽⁰⁾ and outputs x₁⁽¹⁾ to x₅⁽¹⁾ in both rows, and
  the caption "... where the input $$x_3^{(0)}$$ affects only the outputs $$x_2^{(1)}$$, $$x_3^{(1)}$$ and
  $$x_4^{(1)}$$. Bottom: a fully connected layer, where $$x_3^{(0)}$$ affects every output."

### Mathematics for ML (2018-04-20-algebra-for-ml.md)

- `algebra-for-ml/newton_method.gif`: the legend is in German ("Funktion", "Tangente"), and the animation numbers the
  first guess x1, while the text starts at $$x^{(0)}$$. The caption translates both words and explains the numbering,
  but writes the text's first guess as $$x_0$$. Fix: "(the text's $$x^{(0)}$$)" in the caption, then one SVG with the
  function, the first three tangents and the guesses x⁽⁰⁾ to x⁽³⁾, labeled in English.
- `algebra-for-ml/svd_example.svg`: the panels are crops of the "Lenna" test image, which IEEE no longer allows in its
  publications since April 1, 2024, and the panels with 50 and 85 singular values look like ranks 25 to 35. Optional
  fix: the same experiment (all 256, then 20, 50 and 85 singular values) on another test image, with the computed
  results.
- Optional, `algebra-for-ml/mml_eigenvalues_and_eigenvectors.svg`: the plots have unequal axis scales. Fix: the same
  scale on both axes.
- `algebra-for-ml/autodiff.svg`: the square-root node (cell `autodiff-x12`) writes √ in Source Sans 3 and the
  argument as a raised dot, so it renders as a small √ under a dot. Fix: the label "√( · )" in
  NewComputerModernSansMath only, like "exp( · )" and "cos( · )", then export the picture again.

### Statistics for ML (2018-06-15-statistics-for-ml.md)

- `statistics-for-ml/bivariate_gaussian.svg`: the picture labels the marginals p(X) and p(Y), while the text and the
  caption write p(x) and p(y), and the caption calls a conditional distribution "a slice" of the joint density. Fix:
  p(x) and p(y) in the picture, and "a normalized slice" in the caption.
- `statistics-for-ml/confidence_interval.svg`: the drawing has no source, and the label "0.95" is crossed by the
  center line. Fix: name the source of the drawing, or redraw it, and move the label. Optional: the curve is centered
  at X̄, as in many textbooks, but the sampling distribution of X̄ is centered at μ; a curve centered at μ, with the
  interval drawn around one sample mean X̄, is exact.
- Caption of `statistics-for-ml/kde.svg`: "each divided by $$n$$", while the text and the formula write the number of
  samples as $$N$$. Fix: "each divided by $$N$$".

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- `bayesian-linear-regression/linear_bayesian_predictive_distribution_*.svg`: the model assumes a noise standard
  deviation of 0.71 (noise precision beta = 2, as the caption states), but the data have a noise standard deviation of
  0.25. After 25 samples, the band is thus 0.76 to 0.88 wide on each side of the mean, three times the scatter of the
  samples. Fix: redraw with beta = 16 (band 0.27 to 0.32), see the code issues above, and change the caption.
- Caption of the same pictures: it gives the prior precision $$\alpha = 10^{-3}$$ and the noise precision
  $$\beta = 2$$ of Bishop's PRML and of the notebook, which the post never defines: it writes the prior with its
  covariance and the noise with its variance $$\sigma^2$$. Fix: "The model has the prior $$\mathcal{N}(0, 10^3 I)$$
  and the noise variance $$\sigma^2 = 0.5$$", or $$\sigma^2 = 1/16$$ after the redraw with beta = 16.

### Exponential Family of Distributions (2019-03-20-exponential-family.md)

- `exponential-family/common_distributions.svg`: the text says that the binomial approaches the normal for large n
  (de Moivre-Laplace theorem), as the source post by Sean Owen does, but the figure has no arrow from Binomial to
  Normal; it has the arrows from Hypergeometric and from Poisson. Fix: add the arrow from Binomial to Normal.
- Optional: the caption starts with "The 15 most common probability distributions", but the source post only calls
  them common. Fix: "Fifteen common probability distributions and their relationships", with the same source.

### Variational Inference (ELBO, CAVI) and Gaussian Mixture Models (2019-07-01-variational-inference-gmm.md)

- `variational-inference-gmm/gmm_iterations.svg`: the caption calls the gray ellipses "2σ contours of the variational
  approximating factors", as the source (Blei et al., Figure 3) does, and says that the figure uses the 2D version of
  the model. But the ellipses are elongated and, at iteration 50, as large as the clusters. In the model of the text,
  the factor of each mean is an isotropic Gaussian whose variance s_k² = 1/(1/σ² + Σ_i φ_ik) shrinks as the cluster
  gains points, so the figure comes from a model with unknown covariances. Fix: "The gray ellipses are the fitted
  components at initialization and after 20, 28, 35 and 50 iterations" in the caption, or redraw the figure with the
  model of the text in 2D: unit-variance components, drawn as circles of radius 2 around each m_k.

### Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) (2019-10-12-vae-gan.md)

- `vae-gan/gan.svg`: the discriminator has two output arrows, "1 (real)" and "0 (generated)", but D outputs one
  number, the probability that its input is real, as the caption says. Fix: one output arrow labeled "D(x):
  probability that x is real (1) or generated (0)".
- `vae-gan/vae.svg`: the labels "μz", "Σz" and "z ∼ 𝒩(μz, Σz)" keep the symbols of the source blog. The text writes
  the outputs of the encoder as the mean $$\mu$$ and the variance $$\sigma^2$$, and never uses $$\Sigma$$. Fix: "μ",
  "σ²" and "z ∼ 𝒩(μ, σ²)".
- Optional, `vae-gan/vae.svg`: the encoder box shared by the AE and the VAE paths is labeled q_φ(z|x), a distribution,
  but the AE encoder is a deterministic function. Fix: label the encoder "Encoder φ".

### Learning from sequences with Encoder-Decoder, Transformers and BERT (2020-02-28-learning-from-sequences.md)

- Optional, `learning-from-sequences/encoder_decoder.svg`: the output layer has as many neurons as the input
  embedding and the hidden state, but the decoder outputs a softmax over the target vocabulary, which is much
  larger. Fix: draw the output layer wider than the other layers, with "softmax over the vocabulary" next to it.
- Caption of `learning-from-sequences/transformer_attention.svg`: it writes the sentence length as N, while the text
  writes n. Fix: n.
- Optional, `learning-from-sequences/encoder_decoder.svg`: the label "hidden space t−1" names the hidden state of the
  previous step, as the text calls it. Fix: "hidden state t−1".
- Optional, caption of `learning-from-sequences/transformer_attention_lbdl.svg`: it calls $$A_{q,k}$$ "normalized
  attention scores", while the text calls them attention weights, and calls scores the values before the softmax.
  Fix: "to obtain the attention weights $$A_{q,1},...,A_{q,N^{KV}}$$ (left), and then averages the values
  $$V_1,...,V_{N^{KV}}$$ with these weights".
- Optional, caption of `learning-from-sequences/transformer_attention_masked.svg`: "The algorithm performs four
  iterations, one per word" holds only at inference: in training, the mask lets the decoder compute the four vectors
  at once, as the text says. Fix: "At inference, the decoder performs four iterations, one per word; in training, the
  mask computes the four vectors at once."

### GPT in C++: LibTorch vs. PyTorch, TorchScript and torch.compile (2023-06-27-gptlite-cpp.md)

- `gptlite-cpp/benchmark_wide.svg` and `benchmark_deep.svg`: the throughput axes say "samples/sec", but the values
  are iterations per second on batches of 2048 samples, as the code prints them ("epochs/second"). Fix: label the
  axes "iterations/sec (batches of 2048 samples)", or multiply the values by 2048.
- Optional, caption of `gptlite_compact.svg` and `benchmark_model.svg`: it starts with "This post details", and calls
  the left model "the GPT-3 Small model", while it is GPTlite with the dimensions of GPT-3 Small. Fix: "Left: GPTlite
  with N decoder blocks, with the dimensions of GPT-3 Small. Right: the benchmark DNN with L layers of width W. The
  left diagram is adapted from Attention is all you need."
- Optional, caption of `gptlite-cpp/benchmark_gptlite.svg`: "GPTlite Small" is a model name that the posts use
  nowhere else. Fix: "GPTlite (with the dimensions of GPT-3 Small)".

### Data parallelism, sharding and CPU offloading (2023-08-18-data-parallelism.md)

- Optional, `distributed-training/workflow_sharding.svg` shows ZeRO-3 as the ZeRO paper describes it: each process
  owns whole layers, broadcasts their parameters and reduces their gradients. DeepSpeed ZeRO-3 and PyTorch FSDP split
  every layer into one shard per process, with an all-gather of the parameters and a reduce-scatter of the
  gradients. The text explains both. Fix: redraw each layer as four shards, with all-gathers and reduce-scatters.
- `distributed-training/imgs/vdnn2.svg` and `vdnn3.svg` label the last layer "Layer(N)", and the caption says "at
  layer N", while the text numbers the layers from 1 to $$L+1$$, and `deepspeed_stages.svg` uses N for the processes.
  Fix: "Layer(L+1)", and the caption "at its last layer, $$L+1$$".
- Optional, `distributed-training/deepspeed_stages.svg`: the number of processes is N_d (and gpu_N-1), while
  `data_parallelism.svg` and the text use P; the text states that N_d = P. The caption calls the stages
  "ZeRO FSDP". Fix: write P in the figure, and "the three stages of ZeRO" in the caption.
- Optional, the caption of `distributed-training/imgs/benchmark_gptlite.svg` does not say how the memory is averaged.
  Fix: "GPTlite on one node with 8 GPUs, in fp32, with micro-batches of 1 sample: average and peak allocated memory
  per GPU, and throughput over all GPUs".
- Optional, caption of `distributed-training/imgs/data_parallelism.svg`: it names "the DDP data layout" before the
  text introduces DDP, while the layout is the same for DDP, FSDP and ZeRO. Fix: "The data layout of data
  parallelism, split across $$P = 4$$ processes color-coded as blue, yellow, red and green."

### Pipeline parallelism (1F1B, Zero Bubble, DualPipe) (2023-08-30-pipeline-parallelism.md)

- `distributed-training/imgs/pipeline_8_stages.svg`, `pipeline_4_stages.svg` and `pipeline_2_stages.svg` label their
  rows "Worker 1" to "Worker 8", and the caption says "8 workers", while the text says GPUs and the other schedules
  label their rows "GPU 0" to "GPU 3" (CLAUDE.md keeps "Worker" for the PipeDream pictures). The caption also calls
  the center and right panels "2 data-parallel groups of 4 pipeline stages" and "4 data-parallel groups of 2 pipeline
  stages", but in Megatron-LM and DeepSpeed a data-parallel group holds the GPUs of one stage: the panels show 2
  pipelines of 4 stages and 4 pipelines of 2 stages. Fix: rows "GPU 0" to "GPU 7", and the caption "... on 8 GPUs.
  ... Center: 2 pipelines of 4 stages. Right: 4 pipelines of 2 stages."
- Optional, the caption of `distributed-training/pipedream_dnn_pipeline_parallel_microsoft.svg` says that the
  forward passes in flight "belong to different micro-batches". PipeDream pipelines mini-batches and updates the
  weights after each one, as the text says. Fix: "different mini-batches".

### Mixture-of-Experts and Expert Parallelism (2024-01-19-mixture-of-experts.md)

- Optional, `moe_2017_dean.svg` writes the gate as G(x) and the number of experts as n, and `moe_gshard.svg` writes E
  experts and N layers, as their papers do. The text writes the gate as g(x) and the number of experts as N, and
  names the notation of the first figure. Fix: relabel the figures to g(x) and N experts, and the GShard layers to D
  (L is the loss in the post), with "The encoder has $$D$$ Transformer layers" in the caption.
- Optional, `distributed-training/imgs/moe_switch_transformers.svg`: the label "Switching FFN Layer", while the
  method and the caption say Switch FFN layer (the figure of the paper has the same mismatch). Fix: "Switch FFN
  Layer".
- Optional, caption of `distributed-training/imgs/moe_switch_transformers_2.svg`: it writes the expert capacity as
  (total_tokens / num_experts) × capacity_factor, while the Terminology box of the picture writes tokens_per_batch.
  Fix: tokens_per_batch in the caption.

### Variable sequence lengths, curriculum learning, adaptive batch size and LR (2024-05-10-training-variable-length-curriculum.md)

- `distributed-training/imgs/variable_batch_lr_pipeline.svg`: the left panel is titled "Micro-batches for regular use
  case", while the text names it DDP; the caption of `variable_attn_matrix.svg` says "the non-pipeline micro-batch 1
  above". Fix: "Micro-batches for DDP", and "micro-batch 1 of the DDP panel above" in the caption.
- Optional, caption of `distributed-training/imgs/variable_batch_lr.svg`: it does not say which of the two scaling
  rules of the text gives the LRs of the picture. Fix: "and thus a larger LR (linear scaling)".
- Optional, `distributed-training/imgs/variable_batch_lr_pipeline.svg`: the right panel packs consecutive samples
  (0 to 2, 3 to 5, ...) into micro-batches, while `batch_by_seqlens` of DeepSpeed, with
  `required_microbatches_of_same_size`, interleaves them (`batch[b::n]`). Interleaved, the 16 samples fit 4
  micro-batches of 4 samples (30, 28, 26 and 30 tokens), of shape 4 x 11 x E, with 35% padding. Fix: none if the
  picture only shows the constraint; otherwise say in the caption that consecutive samples are packed.
- Optional, caption of `distributed-training/imgs/torch_compile_dataset.svg`: it says both processes and ranks, while
  the text says ranks. Fix: "on 3 ranks" and "to the ranks (color-coded)".

### Sequence and context parallelism with Ulysses and Ring attention (2024-08-30-sequence-context-parallelism.md)

- `distributed-training/imgs/ulysses_sequence_parallelism.svg`: the label "Local attention shape: B x H/P x S x E"
  and the axis E of the middle cube give the head size as E, the embedding size. An all-to-all keeps the number of
  elements per process, B·(S/P)·E before it and B·(H/P)·S·d after it, with Hd = E, so the label is H times too large.
  Fix: "Local attention shape: B x H/P x S x d", and the axis d on the middle cube.
- Optional, `distributed-training/imgs/context_parallelism.svg` writes the head size as E and leaves out the scale
  1/sqrt(d) of the scores. The text names the label. Fix: write the head size as d.

### Diffusion models, from DDPMs to DiTs (2025-02-25-diffusion.md)

- `diffusion/masked_autoencoders_cropped.svg` and `masked_autoencoders.svg` are now the same picture, shown twice in
  the post. Fix: keep one file and link both places to it, or show the picture once.
- Optional, `diffusion/dit_moe.svg`: the dashed residual connections start after the first AdaLN, while the code of
  the post adds the input of the block, and three experts are gated, while the code routes each token to 2 experts.
  The picture follows the figure of the paper, which draws both this way. Fix: check both against the code, and say in
  the caption where the code differs.
- Optional, `diffusion/masked_autoencoders.svg`, `masked_autoencoders_cropped.svg` and `align_your_latents.svg` (with
  its caption) write the number of frames as T, the symbol of the diffusion steps in the post, and
  `anymate_anyone.svg` writes the frames, height and width as t, h, w and 2w, while t is the diffusion step. The text
  uses F, H and W, as the Latte figures do. These are figures from the papers. Fix: F for the frame axis of the two
  MAE figures, B·F in `align_your_latents.svg` and its caption, and F, H, W and 2W in `anymate_anyone.svg`; or keep
  the paper labels.
- Optional, caption of `diffusion/diffusion.svg`: "A diffusion model is a $$T$$-step Markov chain", while the text
  counts two Markov chains. Fix: "A diffusion model has two Markov chains of $$T$$ steps: a forward process $$q$$
  that adds noise, and a trainable reverse process $$p_\theta$$ that removes it."

### Prefill vs decode, roofline, KV cache, batching and disaggregation (2025-06-07-inference-prefill-decode-roofline-cache.md)

- `inference/imgs/kv_cache.svg`: the panel with the KV cache writes the length of the cache plus the new token as S
  ("KT: h x S", "qKT: 1 x S", "V: S x h"), while the code of the post writes the new tokens as S and the cached ones
  as P, as `static_cache.svg` does ("cache: P x h", "new tensor: (P+1) x h"). Fix: "KT: h x (P+1)", "qKT: 1 x (P+1)"
  and "V: (P+1) x h" in that panel, and "$$h$$ is the head size and $$P$$ the number of cached tokens" in the
  caption.
- Optional, `inference/imgs/roofline.svg`: the label "overlap computation and communication" reads as transfers
  between devices, while the roofline counts the memory transfers of one device; the caption says "Below the ridge
  point ... beyond it", while the text says left and right. Fix: "overlap computation and memory transfers", and
  "Left of the ridge point ... right of it" in the caption.
- Optional, `inference/imgs/semantic_caching.svg`: the label d names the size of the query embedding, while d is the
  width of the LLM in the post. Fix: "dim", as in the code, or no label.
- `inference/imgs/prefill_decode.svg` writes "(n - 1)" with a hyphen. Fix: a true minus sign, "(n − 1)".

### Faster attention (GQA, FlashAttention, sparse, linear and latent attention) (2025-12-30-inference-attention.md)

- Optional, `inference/imgs/mla.svg`: the absorb = True panel shows a whole sequence, while weight absorption serves
  decode, with one new token (H x 1 x (P+1) scores over P+1 latent vectors). The panels are labeled "usually prefill"
  and "usually decode". Fix: redraw the absorb = True panel for one decode token.

### Kernel Fusion for Megatron-LM Tensor Parallelism and MoE Expert Sharding (2026-08-18-kernel-fusion-tensor-parallelism.md)

- `inference/imgs/tp_megatron_lm.svg` and `tp_expert_sharding.svg` label the reduction "sum-reduce", and the caption
  of `tp_expert_sharding.svg` says "a sum-reduce adds their partial outputs", while the text says all-reduce. The sum
  ends on every GPU ("Y_A = Y_B", "replicated"), which is an all-reduce, as in Megatron-LM. Fix: "all-reduce" in both
  pictures and in the caption.
- Caption of `inference/imgs/tp_expert_sharding.svg`: the credit "Inspired by Figure 2 of [...]." does not follow the
  credit form of CLAUDE.md. Fix: "Inspired by: [Accelerating MoE Model Inference with Expert
  Sharding](https://arxiv.org/abs/2503.08467)".
- The captions of `tp_megatron_lm.svg`, `tp_timeline.svg`, `tp_matmuls.svg` and `tp_block_specialization.svg`,
  redrawn from the figures of the post's paper, and of `tp_results.svg`, drawn from its data, have no credit. The
  paper is the author's own and not public yet. Fix: "Adapted from: [Towards Efficient LLM Inference via Full-Stack
  Fusion of Computation, Communication, and In-Network Computing](link)" once the paper has a public link; the user
  decides.
- Optional, `inference/imgs/tp_expert_sharding.svg` and its caption write the weights of an expert as W_i and W_o,
  and the tokens and outputs as x, y, y_A and y_B, while the rest of the post writes the same matrices as W_1, W_2, X
  and Y. The text names the mapping of the weights. Fix: relabel the figure and the caption to W_1, W_2, X, Y, Y_A
  and Y_B, and drop "(the W_1 and W_2 above)" from the text.
- Optional, `inference/imgs/tp_megatron_lm.svg` names the GPUs A and B and writes their slices with subscripts (H_A,
  (W_1)_A), while the text numbers the GPUs g = 1 to G, writes their slices with a superscript, $$H^{(g)}$$, and uses
  subscripts for tiles. Fix: "GPU 1" and "GPU 2", H⁽¹⁾, W₁⁽¹⁾, W₂⁽¹⁾ and so on, "(GPU 1 in green, GPU 2 in blue)" in
  the caption, and the same GPU names in `tp_expert_sharding.svg`.
- Optional, `inference/imgs/tp_timeline.svg`: the fused row applies the bias and the activation once, after all the
  reduce steps, while the text and `tp_block_specialization.svg` apply them to each band, right after the all-reduce
  of that band. The figure of the paper has the same simplification. Fix: a small bias and activation step after
  each reduce step, or keep the picture as a schematic.

### Kernel Fusion for MoE Expert Parallelism (2026-10-09-kernel-fusion-expert-parallelism.md)

- `inference/imgs/ep_all_to_all_reduce.svg` names the buffer of the partial result of rank x as Σ_{x/B}. This index
  gives W buffers for B = 1, and a single buffer for B = W, the opposite of the text. The picture (W = 4, B = 2) puts
  σ_0 and σ_1 in Σ_0, and σ_2 and σ_3 in Σ_1. Fix: Σ_{⌊xB/W⌋}.
- `inference/imgs/ep_stages.svg` draws the eight stages on a time axis from 0 to 300 µs. In the paper, this figure is
  the decode step, and its timings are placeholders, not measurements. The caption does not say so. Fix: remove the
  time axis, or replace the timings with measured ones.
- `inference/imgs/ep_results.svg`: the right panel plots decode times that are placeholders, not measurements, as the
  caption says. Fix: remove the right panel and its sentence in the caption until the decode step is measured.
- Caption of `inference/imgs/ep_data_movement.svg`: "$$\star$$ marks memory of the shared heap", while the text
  defines the symmetric heap of SHMEM. Fix: "$$\star$$ marks memory on the symmetric heap", and the same words in the
  legend at the redraw.
- Caption of `inference/imgs/ep_all_to_all_reduce.svg` and `ep_all_to_all_unique.svg`: $$E_x$$ names the expert ids
  and weights of rank $$P_x$$, while everywhere else the subscript of E is an expert index ($$E_0$$ to $$E_5$$). Fix:
  $$R_x$$, in the caption and in the two figures at the redraw.
- The captions write one quantity, the sum of the expert outputs of one rank for one token, as $$A_0$$
  (`ep_routing_table.svg`), $$\sigma_C$$ (`ep_all_to_all_inc.svg` and `ep_timeline.svg`) and $$\sigma_x$$
  (`ep_all_to_all_reduce.svg`), and the subscript of σ is a token in the second and a rank in the third. Fix:
  $$\sigma_A^{(0)}$$, $$\sigma_C^{(x)}$$ and $$\sigma^{(x)}$$, in the captions and in the figures at the redraw.
- The captions have no credit to the post's paper, whose figures the pictures copy. The paper is the author's own and
  not public yet. Fix: "Source: [MoeInc: an Expert Parallelism MegaKernel with Overlap of Matrix, Vector and
  In-Network Compute](link)" once the paper has a public link, and "Adapted from:" after the redraw; the user
  decides.
- Optional, `inference/imgs/ep_timeline.svg` labels the dispatch call of P_2 as put_signal_01, but the call goes to
  the INC switch, while the caption says that a subscript names the destination of the call. Fix: put_signal*
  without a subscript, or a caption that says the subscript names the final destinations P_0 and P_1.
- `inference/imgs/ep_communication_models.svg` labels the transfers "W hops of T tokens per link" and "INC switch
  (fewer hops, less data)". There, a hop means one packet on a link, but the text and the paper say that both fused
  implementations cross the same number of hops. The text says "one packet per link instead of W". Fix at the
  redraw: "W packets of T tokens per link", "1 packet of ..." and "INC switch (fewer packets, less data)".

### Knowledge distillation, speculative decoding and multi-token decoding (_drafts/2026-03-17-inference-distillation-speculative-decoding.md)

- `inference/imgs/layerskip.svg` labels the drafts "drafts of the first E = 4 layers". E is the embedding dimension
  in the code and in the eagle, medusa and mtp_parallel figures. The figure also names the predictions of the target
  p_1 to p_3, while p is the distribution of the draft model in spec_sampling. Fix: "drafts of the first 4 layers",
  and y_1 to y_3 (or no symbol) for the predictions.
- Optional, `inference/imgs/mtp_parallel.svg` writes the tokens as x_t, as Gloeckle et al. do, while eagle, layerskip
  and mtp_sequential write them as t_i. The text follows each figure. Fix: keep the paper notations, or write t_i in
  mtp_parallel and in the formulas of the parallel heads.

### Pruning and quantization (_drafts/2026-04-22-inference-pruning-quantization.md)

- Caption of `inference/imgs/hadamard_rotation.svg`: "the INT4 levels are closer together (5.7 instead of 12.6
  apart)" gives the distances between the drawn level lines in pixels. In the units of x, the INT4 step is 40/7 =
  5.71 before the rotation and 18.03/7 = 2.58 after it. Fix: "(2.6 instead of 5.7 apart)".
- `inference/imgs/pruning_structures.svg` writes the number of heads as H ("H·d_head × E", "E × H·d_head"), while H
  is the Hessian of GPTQ in the post, and the text writes the number of heads as $$n_\text{heads}$$, as the code
  does. Fix: "n_heads·d_head × E" and "E × n_heads·d_head".
- `inference/imgs/gptq.svg`: the labels "err = W[:, i] − q" and "H = 2 X XT, from the calibration inputs X" take the
  columns of X as the inputs, and q as the quantized column. The text, the code and `llm_int8.svg` take the rows of X
  as the inputs, the text writes the quantized column as $$\hat{W}_{:,i}$$, and `int4_packing.svg` uses q for the
  integers. Fix: "err = W[:, i] − Ŵ[:, i]" and "H = 2 XT X".
- Optional, `inference/imgs/quantization_granularity.svg`: the per-group panel colors its 12 groups with 6 colors
  (rows 1 and 3, and rows 2 and 4, repeat their colors). In the per-row panel each color is one scale, so the
  per-group panel reads as 6 scales. Fix: a black frame around each group, or 12 distinct fills.

### Publications bookmark (publications.md)

- Optional, `publications/llama_3_multi_modal.svg`: the connectors "Every 4th layer" and "Every ~3rd layer" have arrow
  heads at both ends, as in Fig. 28 of the paper, while the forward pass sends data one way. Fix: one arrow head,
  toward the cross-attention layers.
- `publications/comet_2.svg`: GEMM #0 and #1 mix the rows of two experts after a global sort by source rank. The
  picture follows Fig. 5 of the paper, whose text does not say whether the sort is global or per expert. Fix: check
  the code of Comet, and sort per expert if the code does.
- Caption of `publications/performers2.gif`: no credit. Fix: "Source: [Rethinking Attention with
  Performers](https://research.google/blog/rethinking-attention-with-performers/)". Optional: a static SVG of the
  prefix sums instead of the animation.
- Caption of `publications/latentmoe.svg`: the tokens are projected into the latent dimension $$\ell$$ "for expert
  routing and computation", while the router scores the full $$d$$-dimensional token (the notes, panel (b) of the
  picture and the paper). Fix: "for the all-to-all dispatch, the routed experts and the combine (the router still
  scores the full $$d$$-dimensional token)".
- Caption of `publications/tpla.svg`: it says that the heads are distributed across devices during decoding, which
  describes GLA. In TPLA, every device computes all the heads on its slice of the latent KV cache, and an all-reduce
  sums the outputs (the notes, and panel (c) of the picture). Fix: "every device computes all the attention heads on
  its local slice of the latent KV cache, and an all-reduce sums the partial outputs."
- Caption of `publications/mamba.svg`: "times batch size $$B$$", while $$B$$ is the input matrix of the SSM in the
  same caption and in the notes. Fix: "($$DN$$, times the batch size and the sequence length $$L$$)".
- `publications/dylora.svg`: the label "W ∈ ℝ^(m×d)" names the frozen weight W, while the notes and the paper write
  $$W_0$$. Fix: "W₀ ∈ ℝ^(m×d)".
- Optional, `publications/performers.svg`: "O(Lmd)", "L × m" and "m × L" write the number of random features as m,
  as the blog figure does, while the notes and the paper write r. The notes now say "($$r$$ is $$m$$ in the first
  figure)". Fix: "O(Lrd)", "L × r" and "r × L", then remove that remark from the notes.
- Optional, `publications/lora.svg`: "W ∈ ℝ^(d×d)", with d under x, shows the square case of the paper's figure,
  while the notes write $$W \in \mathbb{R}^{d \times k}$$, with the input dimension k. The notes now say "The figure
  below shows the square case, $$k = d$$." Fix: "W ∈ ℝ^(d×k)" and k under x, then remove that sentence from the
  notes; or keep both as they are.
- Optional, `publications/zeroplusplus.svg`: the labels follow Fig. 2(a) of ZeRO++, but call "scale" a range of
  values (9.22 = max |x|, with q = round(127 x / 9.22); and for a block, max − min = 9.07, with
  q = round(255 (x − zp) / 9.07) − 128). The text of the paper calls qwZ symmetric, with one scaling coefficient per
  block, while its figure uses a zero point. Fix: rename the label to "range", or add "(the scale is the range of
  the block)" to the caption.
- Optional, caption of `publications/flashattention.svg`: "(dotted box)", but the box is dashed. Fix: "(dashed box)".

## Content issues

Content problems that need the author's decision, such as overlap between posts.

### Variable Timestep Simulation of the Electrical Activity of Neurons (2017-02-17-neuron-variable-timestep.md)

- The preprint estimates the speed-up of FAP CVODE on its mix of firing rates as the mean of the speed-ups at the
  five rates, weighted by the shares of neurons: 224.5 to 11.9 with exact event delivery, and 228.5 to 24.6 with
  grouping per $$\Delta t$$. The runtime of a mixed network is the weighted sum of the runtimes, so its speed-up is
  the weighted harmonic mean of the speed-ups, which the text bounds by 28.5 on 1,024 neurons and 6.7 on 65,536
  neurons. The text gives both, and says that a brain-like mix, with more quiet neurons, gives a higher speed-up.
  Proposed fix: keep the estimate of the text; the alternative is to give only the numbers of the preprint.
- The default absolute tolerance of CVODE in NEURON: the NEURON Book and the CVode documentation give 1e-2, while the
  source of NEURON (`netcvode.cpp`) sets 1e-3, the value that the preprint calls the default and uses in its tests.
  The text gives both. Proposed fix: keep the text as it is, or keep only the value of the code.

### Linear Regression, Matrix Factorization and Loss Functions (2017-08-17-linear-regression-matrix-factorization.md)

- Optional: "dropout" links to a Medium article, in this post and in the deep neural networks post. Proposed fix:
  link the paper of Srivastava et al. (2014), https://jmlr.org/papers/v15/srivastava14a.html, or, in this post, the
  dropout section of the deep neural networks post.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- The definition of PCA that the post credits to "Gerstner's lecture notes" is the first sentence of the Wikipedia
  article on PCA. Proposed fix: check the lecture notes, or credit Wikipedia.
- Optional: independent sections reuse symbols. Λ is the matrix of eigenvalues of ICA and the neighbourhood
  function Λ(j, j*) of the SOM, U is the rotation of ICA and the matrix of eigenvectors of spectral clustering, and d
  is the degree of the polynomial kernel and the distance d(·,·). Proposed fix: h(j, j*) for the SOM (Kohonen's
  notation), H for the eigenvectors of spectral clustering, and ‖·‖ or dist(·,·) for the distances.

### Mathematics for ML (2018-04-20-algebra-for-ml.md)

- Overlap with the deep neural networks post (2018-03-27-deep-neural-networks.md): both posts explain
  backpropagation as the chain rule and introduce automatic differentiation. Proposed fix: this post keeps the math
  (the chain rule through K layers and the worked example with its computation graph); the deep neural networks
  post keeps the algorithm, its costs and the PyTorch details.

### Statistics for ML (2018-06-15-statistics-for-ml.md)

- Overlap with the exponential family post (2019-03-20-exponential-family.md), which also covers the canonical form,
  the derivatives of the log-partition function and maximum likelihood, and with the variational inference post
  (2019-07-01-variational-inference-gmm.md), which also derives the ELBO. The Bernoulli example and the list of
  members of the family are now only in the exponential family post. This post keeps a short version of the rest and
  links to the dedicated post. Proposed fix: keep the short versions as a reference sheet and the full derivations in
  the dedicated posts; the alternative is to keep only the links here.
- Optional, the post could be about 200 words shorter without these sentences, which other posts do not need: the
  costs of the Cholesky decomposition, Gaussian processes and Kalman filters, the FFT paragraph of the KDE, the
  parameter count of the Bayesian network, the mini-batch sentence of the central limit theorem, and the conjugate
  pairs. Proposed fix: the user decides.

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- The rewrite removed three parts that repeat other posts: the refresher on the Gaussian pdf (in the statistics
  post), the section "Adding regularization", whose ridge solution is the MAP estimate of the next section, and the
  appendix of linear algebra rules (the derivations now state the two gradient rules they use; the mathematics for
  ML post and The Matrix Cookbook cover the rest). Proposed fix: keep them removed; restore the appendix if it
  should stay as a refresher.

### Variational Inference (ELBO, CAVI) and Gaussian Mixture Models (2019-07-01-variational-inference-gmm.md)

- The rewrite replaced the derivation of the CAVI update with the chain rule of probability, from the CMU 10-708
  lecture notes, by the shorter derivation of section 10.1.1 of Bishop's PRML: as a function of one factor, the ELBO
  is a negative KL divergence. It also removed the recap of the chain rule, and keeps the definition of the
  conditional probability in one line, since the introduction and the ELBO derivation use it. The CMU notes stay
  linked. Proposed fix: keep the shorter derivation.

### Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) (2019-10-12-vae-gan.md)

- The rewrite removed the section "Detour: why maximize the Expected Value?", which explained with games of chance
  that the law of large numbers makes the average of many trials approach the expected value. The losses in the
  post are expectations estimated with mini-batch averages, which the VI and statistics posts cover. Proposed fix:
  keep it removed.

### Data parallelism, sharding and CPU offloading (2023-08-18-data-parallelism.md)

- The post holds a hidden HTML comment, from the mixed precision paragraph to the end of the activation
  checkpointing section, still in the first and second person. The visible text now introduces ZeRO++ and
  activation checkpointing in a few sentences. The hidden text sets buckets of `4e5` elements, while the configs
  use `5e4`, and calls `to_layers()` `lo_layers()`. Proposed fix: delete the comment, or rewrite and publish it.
- The link to vDNN+, `ieeexplore.ieee.org/document/8820980`, could not be checked from the sandbox (IEEE Xplore,
  dblp and Crossref do not load). Proposed fix: check by hand that it is "Dynamic Memory Management for GPU-Based
  Training of Deep Neural Networks" (Shriram S B, Garg and Kulkarni, IPDPS 2019), or link its DOI.

### Pipeline parallelism (1F1B, Zero Bubble, DualPipe) (2023-08-30-pipeline-parallelism.md)

- Optional, results, memory per GPU: the first GPU used more memory than the others, also with the deep benchmark
  model. The text now says that 1F1B explains at most a few MB of this gap for that model, and that the cause of the
  rest is not known. Proposed fix: measure the peak memory of every GPU (`torch.cuda.max_memory_allocated()` on each
  rank, and `nvidia-smi` for the CUDA contexts on GPU 0), and give the numbers and the cause in the post.
- The DualPipeV row of the summary table does not list the doubled point-to-point communication, which the text
  names, under "Main cost". Proposed fix: add "2× point-to-point communication" to that cell.
- The post keeps the workaround `outputs = outputs.requires_grad_(True)` for the error of the swapaxes lambda. If
  `outputs` has no `grad_fn`, this call lets the backward pass run, but no gradient reaches the layers before it.
  Proposed fix: check that the parameters of the last stage get gradients with the workaround (`p.grad` after
  `train_batch()`), and remove the sentence if they do not.

### Tensor parallelism (2023-09-02-tensor-parallelism.md)

- The last section was named "Final remarks and code", but the repository holds no code for this post: no file
  defines `Megatron_f`, `Megatron_g`, `Megatron_FeedForward` or `Megatron_MHA`. Proposed fix: add these classes
  to `assets/distributed-training` with a small test on 2 GPUs, or keep the code in the post only.

### Diffusion models, from DDPMs to DiTs (2025-02-25-diffusion.md)

- The link for the EMA of the weights (openreview.net, `2M9CUnYnBA`) opens a browser check, so it could not be
  verified. Proposed fix: open it by hand, and replace it if it does not point to the source of the EMA.

### Kernel Fusion for MoE Expert Parallelism (2026-10-09-kernel-fusion-expert-parallelism.md)

- The post does not mention DySHARP (Zhang et al., "Accelerating MoE with Dynamic In-Switch Computing on
  Multi-GPUs", ISCA 2026), the closest related work, which the paper compares with at length. Proposed fix: one
  linked sentence after the DeepSeek-V3 example, saying that DySHARP adds multicast and reduction to the switch but
  still uses 2 of 4 SM groups to issue the dispatch and the combine; check its public link and title first.

### Pruning and quantization (_drafts/2026-04-22-inference-pruning-quantization.md)

- The draft describes the experiments of `inference/pruning.py` and `inference/quantization.py` on GPTlite (pruning
  with distillation, Wanda, ShortGPT, weight-only and weight-activation quantization, KV cache quantization), but
  shows none of their results. Fix: run the two scripts and add a table of their outputs (parameters, size,
  validation loss, time per token, matching tokens), or present the code without the comparisons.
- Optional: once both drafts are published, "knowledge distillation" in the introduction could link to the
  distillation post instead of Wikipedia.

### Publications bookmark (publications.md)

- DeepSeek-V3.2-Exp: the title and the author line write the hyphens as non-breaking hyphens (U+2011),
  "DeepSeek‑V3.2‑Exp" and "DeepSeek‑AI", while the rest of the page uses ASCII hyphens, so a search for
  "DeepSeek-V3.2-Exp" misses the title. Proposed fix: ASCII hyphens.
- NSA: the paper gives 260B pre-training tokens in its introduction and 270B in its section 4.1, and the notes say
  260B. Proposed fix: keep 260B, or write "260B tokens (270B in Sec. 4.1)".
- DeepSeek-V3.2: the notes say that it is the first DeepSeek model to integrate thinking into tool use, and that
  DeepSeek-V3.2-Speciale has no tool use. Both come from the release note of DeepSeek, not from the report.
  Proposed fix: link the note (https://api-docs.deepseek.com/news/news251201), or keep the notes as they are.
- Optional, links that redirect: blog.research.google/2020/10/... (Performers) and blog.research.google/2021/12/...
  (GSPMD) go to research.google/blog/..., www.tensorflow.org/xla (operator fusion in XLA, and GSPMD) goes to
  openxla.org/xla, and papers.nips.cc/paper/5346-... (the sequence to sequence paper, in the notes on neural machine
  translation) goes to proceedings.neurips.cc/.... The GPT-3 notes link GPT-2 to a third-party copy of its paper
  (insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf), which is the right paper. Proposed fix: use the
  target URLs, and the PDF of OpenAI for GPT-2 after a check.

## Title issues

A new title keeps the URL, which comes from the file name (`permalink: /:title` in `_config.yml`). A title, with the
category that the home page puts before it, fits on one line of the home page on a desktop screen (under 750 px).

### All posts

- The capitalization is mixed: the posts up to 2019 use title case, most posts since 2023 use sentence case, and
  a few titles mix both. Fix: one style for all titles.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- To do later. The title names only PCA, but the post also covers ICA, k-means, Gaussian mixtures, hierarchical
  clustering, DBSCAN, spectral clustering, self-organizing maps, t-SNE, UMAP and self-supervised learning.
  Proposed: "Dimensionality Reduction, Clustering and Self-Supervised Learning" (652 px).
