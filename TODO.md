# TODO

Open items that the text rewrite and the picture work found but did not fix. Each item names the post, the problem
and a proposed fix. The user picks the items to fix before the next pass.

## Code issues

Bugs in the code and data files of the posts (under `assets/`).

### Distributed Sorting Algorithms (2014-06-21-distributed-sort.md)

- `assets/distributed-sort/distributedmemorysorter.cxx`, `oddEvenSort_MPI`: the loop runs exactly `mpiSize` phases,
  which sorts the data only when every rank holds the same number of elements. Its own test, `test_OddEvenSort`,
  gives rank r `1000000 + 3r` elements. A simulation with unequal counts fails to sort in about a third of the
  cases after p phases. Fix: balance the counts first, or run phases until an even and an odd phase in a row change
  nothing (an all-reduce of a "changed" flag per phase).

### Distributed Load Balancing and Spatial Slicing (2015-12-01-slicing.md)

- `assets/slicing/slicing.tar.gz`, `DistributedLoadBalancing.cxx` line 20 and `DistributedSorting.cxx` line 28: the
  checksum test is inverted. It warns "checksum ... failed" when the sums before and after are equal
  (`std::abs(pre_sum - post_sum) < 0.000001`), and stays silent when they differ. The warnings are off by default
  (`outputWarnings = false`). Fix: warn when the difference is larger than a relative tolerance, e.g.
  `std::abs(pre_sum - post_sum) > 1e-9 * std::abs(pre_sum)`, since the sum of many doubles changes with the order
  of the additions.

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

### Diffusion models, from DDPMs to DiTs (2025-02-25-diffusion.md)

- The decoder comment in `DiT.__init__` (in `diffusion/dit.py` and in the post) quotes the DiT paper with other
  words. The paper reads "We use a standard linear decoder to do this; we apply the final layer norm (adaptive if
  using adaLN) and linearly decode each token into a p×p×2C tensor". Fix: quote these words, or drop the quotation
  marks.
- The EMA comment "with a warmup as in diffusers" (in `diffusion/diffusion.py` and in the post): the decay
  (1+n)/(10+n) is the default of the diffusers `EMAModel`, which gives the name "warmup" (`use_ema_warmup`) to
  another formula. Fix: "0.9999 as in the paper, ramped up as in the default EMAModel of diffusers".

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

## Diagram issues

Pictures that are wrong or disagree with the post text (under `assets/`). Items marked "Optional" improve a picture
that is not wrong.

### All pictures

- The text rewrite (PR #92) changes the text of the posts, and the picture restyle (branch
  `claude/project-thread-56yymq`) changes their pictures, img tags and captions, so both change the same Markdown
  files. The restyled pictures already use the symbols of the text rewrite: ranks and p in the sorting pictures, x_i,
  j* and w_j* in the Kohonen pictures, and "Exact" in the legend of the LIF solutions. Fix, in the order the user
  chose: merge the picture branch first, then merge the main branch into the text rewrite, keeping its text and the
  pictures, img tags and captions of the main branch, and check each caption against the new text.
- The two kernel fusion posts (2026-08-18-kernel-fusion-tensor-parallelism.md and
  2026-10-09-kernel-fusion-expert-parallelism.md): their 17 pictures, `inference/imgs/tp_*.svg` and `ep_*.svg`, are
  not restyled and keep the Times New Roman font of the paper figures. 16 are copies of the figures of the paper
  repositories, and `tp_expert_sharding.svg` comes from a tab of `inference/inference.drawio` without a figure group.
  All show at widths set by hand (40% to 100%), so their text sizes differ from the other pictures. Fix: redraw them
  at page scale and in the fonts of CLAUDE.md, in tabs of `inference/inference.drawio` named after the posts, from the
  draw.io or TikZ sources of the paper repositories (CLAUDE.md, Pictures, sections 2 to 4), and check their captions.
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

- `matrix-transpose/crs_format.svg`, `xcrs_format.svg`: the array of column indices is labeled `disps`, MPI's name
  for displacements, which reads as offsets. Optional fix: `col_idx`, as in the matmul post.

### Distributed Load Balancing and Spatial Slicing (2015-12-01-slicing.md)

- `slicing/mini-*.svg`: the four layouts have 10 (single axis), 35 (NUG, 5 x 7), 16 (ORB) and 35 (STR, 5 x 7)
  slices. Optional fix: 16 slices in all four (4 x 4 for NUG and STR), so that they compare on equal terms.

### The Leaky Integrate-and-Fire Neuron and the Brunel Network (2016-02-08-lif-brunel.md)

- `lif-brunel/raster_plot_g4.5_f0.9.svg`, `raster_plot_g5_f2.svg`, `raster_plot_g3_f2.svg`: the plots do not name
  their state. The post used to call them "slow", "regular" and "fast" dynamics, but their parameters are those of
  Brunel's states SI slow (g = 4.5, eta = 0.9), AI (g = 5, eta = 2) and SR (g = 3, eta = 2); the text now uses these
  names. Fix: a caption under each plot, e.g. "SI slow (g = 4.5, eta = 0.9)", since CLAUDE.md puts titles in captions.
- Same three plots: the label "time interval (ms)" overlaps the bars at the right end of the lower panel, and
  "spikes count" reads better as "spike count (per ms)". Fix: move the label below the axis.
- `lif-brunel/action_potential.gif`: a "MakeAGIF.com" watermark in the bottom right corner, and the arrows that show
  the direction of the current are missing in the first and last frames. Fix, asked by the user for after the merge: a
  static SVG in which the arrows are always visible.

### Numerical Simulation of Detailed Neuron Models (2016-08-15-neuron-electrical-activity.md)

- `neuron-electrical-activity/brain_scales.svg`: the LadyofHats drawing is not restyled and not at page scale. It is
  716 px wide and shows at 50%, so its sans-serif panel letters (a to d) and ion labels show at 61% of their size.
  Fix: keep the drawing, scale it to about 440 px, redraw the letters and labels in Source Sans 3 (14 px and 12 px),
  and set P = W/880.

### Variable Timestep Simulation of the Electrical Activity of Neurons (2017-02-17-neuron-variable-timestep.md)

- `neuron-variable-timestep/cvode_pulse_currents_table.svg`, `cvode_dependency_on_variation_solution.svg` and
  `cvode_dependency_on_events_arrival.svg`: the currents are in mA and µA (2.7 mA, "threshold current (0.206mA)",
  "0.1µA" to "1mA"), as in the preprint. NEURON's `IClamp` amplitude is in nA, and the threshold current of a cortical
  pyramidal cell model is of the order of 0.1 to 1 nA, so 0.206 mA is likely 0.206 nA. The text gives no current
  values. Fix, after a check of the simulation scripts: mA to nA and µA to pA in the labels.
- `benchmark_two_rows_2.svg`: the legend entry "(2a) BSP Deriv. Implicit†" has no key for the dagger. The preprint's
  caption defines it: Newton iterations solve each mechanism implicitly, but the coupling between mechanisms stays
  explicit. Fix: a key line in the legend, "† implicit per mechanism, explicit across mechanisms".
- Optional, `benchmark_two_rows_2.svg`: the orange "region of similar runtime growth" does not follow from the plotted
  runtimes. Fix: check it against the benchmark data, or remove it.
- `neuron_203076.svg`, `neuron_138083.svg`, `neuron_208223.svg` and `neuron_87784.svg`: the last 10 ms bar of each
  event histogram leaves out the events after 7.5 s, and the burst bars near 0 s are cut at the top of the plot (198,
  203, 44 and 47 events) without a mark. Fix: redraw the histograms from the event data of the simulations, with the
  count on each cut bar.
- `cvode_pulse_currents_table.svg`: the CVODE step counts cover 1.1 s of simulation, while the 48,001 steps of
  backward Euler cover 1.2 s. The values are those of the table of the preprint. Fix, after a check of the simulation
  scripts: step counts for the same time span.

### Linear Regression, Matrix Factorization and Loss Functions (2017-08-17-linear-regression-matrix-factorization.md)

- Optional, `linear-regression-matrix-factorization/matrix-factorization.svg`: the rating 6 (movie 1, user 7) is
  outside a scale of 1 to 5 stars. Fix: a rating from 1 to 5.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- `unsupervised-learning/gaussianscatterpca.svg`: the arrow heads end past the scaled eigenvectors, at 3.75 and 1.75
  instead of 3 and 1, the lengths in the Wikipedia figure it follows. Fix: arrow heads that end at the tips of the
  vectors.
- Optional, `unsupervised-learning/k-means_convergence.gif`: an animation of 15 frames. Fix: an SVG with a few
  iterations side by side, computed by running k-means on the same data.

### Mathematics for ML (2018-04-20-algebra-for-ml.md)

- `algebra-for-ml/newton_method.gif`: the legend is in German ("Funktion", "Tangente"), and the animation numbers the
  first guess x1, while the text starts at x0. The caption translates both words and explains the numbering. Fix: one
  SVG with the function, the first three tangents and the guesses x0 to x3, labeled in English.
- `algebra-for-ml/svd_example.svg`: the panels are crops of the "Lenna" test image, which IEEE no longer allows in its
  publications since April 1, 2024, and the panels with 50 and 85 singular values look like ranks 25 to 35. Optional
  fix: the same experiment (all 256, then 20, 50 and 85 singular values) on another test image, with the computed
  results.
- Optional, `algebra-for-ml/mml_eigenvalues_and_eigenvectors.svg`: the plots have unequal axis scales. Fix: the same
  scale on both axes.

### Statistics for ML (2018-06-15-statistics-for-ml.md)

- `statistics-for-ml/bivariate_gaussian.svg`: the picture labels the marginals p(X) and p(Y), while the text and the
  caption write p(x) and p(y), and the caption calls a conditional distribution "a slice" of the joint density. Fix:
  p(x) and p(y) in the picture, and "a normalized slice" in the caption.
- `statistics-for-ml/confidence_interval.svg`: the drawing has no source, and the label "0.95" is crossed by the
  center line. Fix: name the source of the drawing, or redraw it, and move the label. Optional: the curve is centered
  at X̄, as in many textbooks, but the sampling distribution of X̄ is centered at μ; a curve centered at μ, with the
  interval drawn around one sample mean X̄, is exact.

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- `bayesian-linear-regression/linear_bayesian_predictive_distribution_*.svg`: the model assumes a noise standard
  deviation of 0.71 (noise precision beta = 2, as the caption states), but the data have a noise standard deviation of
  0.25. After 25 samples, the band is thus 0.76 to 0.88 wide on each side of the mean, three times the scatter of the
  samples. Fix: redraw with beta = 16 (band 0.27 to 0.32), see the code issues above, and change the caption.

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
- Optional, `vae-gan/vae.svg`: the encoder box shared by the AE and the VAE paths is labeled q_φ(z|x), a distribution,
  but the AE encoder is a deterministic function. Fix: label the encoder "Encoder φ".

### Learning from sequences with Encoder-Decoder, Transformers and BERT (2020-02-28-learning-from-sequences.md)

- Optional, `learning-from-sequences/encoder_decoder.svg`: the output layer has as many neurons as the input
  embedding and the hidden state, but the decoder outputs a softmax over the target vocabulary, which is much
  larger. Fix: draw the output layer wider than the other layers, with "softmax over the vocabulary" next to it.
- Caption of `learning-from-sequences/transformer_attention.svg`: it writes the sentence length as N, while the text
  rewrite writes n. Fix: n, at the merge with the text rewrite.

### GPT in C++: LibTorch vs. PyTorch, TorchScript and torch.compile (2023-06-27-gptlite-cpp.md)

- `gptlite-cpp/gptlite_compact.svg`, and the three GPTlite figures that this post reuses: the post-norm "Add & Norm"
  boxes and the sinusoidal positional encoding disagree with the code, which is pre-norm, has a final layer
  normalization and learns its position embeddings. Fix: as for the GPT-lite post.
- `gptlite-cpp/benchmark_model.svg`: the output is labeled "Output Probabilities", but the model ends with a ReLU,
  not a softmax, and the ReLUs after the layers are not drawn. Fix: label the output "Output", and each layer
  "Linear + ReLU (size W)".
- `gptlite-cpp/benchmark_wide.svg` and `benchmark_deep.svg`: the throughput axes say "samples/sec", but the values
  are iterations per second on batches of 2048 samples, as the code prints them ("epochs/second"). Fix: label the
  axes "iterations/sec (batches of 2048 samples)", or multiply the values by 2048.
- Optional, caption of `gptlite_compact.svg` and `benchmark_model.svg`: it starts with "This post details", and calls
  the left model "the GPT-3 Small model", while it is GPTlite with the dimensions of GPT-3 Small. Fix: "Left: GPTlite
  with N decoder blocks, with the dimensions of GPT-3 Small. Right: the benchmark DNN with L layers of width W. The
  left diagram is adapted from Attention is all you need."

### Data parallelism, sharding and CPU offloading (2023-08-18-data-parallelism.md)

- `distributed-training/workflow_sharding.svg` and its caption show ZeRO-3 as the ZeRO paper describes it: each
  process owns whole layers, broadcasts their parameters, and "gathers" their gradients, which is a reduction (a
  sum) to the owner. DeepSpeed ZeRO-3 and PyTorch FSDP split every layer into one shard per process, with an
  all-gather of the parameters and a reduce-scatter of the gradients. The text explains both. Optional fix: label
  the arrows "reduce gradients" instead of "gather gradients" and "send grads", or redraw each layer as four shards,
  with all-gathers and reduce-scatters.
- Optional, `distributed-training/deepspeed_stages.svg`: the number of processes is N_d (and gpu_N-1), while
  `data_parallelism.svg` and the text use P; the text states that N_d = P. The caption calls the stages
  "ZeRO FSDP". Fix: write P in the figure, and "the three stages of ZeRO" in the caption.
- Optional, `workflow_data_parallelism.svg` and `workflow_sharding.svg` write "optimiser", while
  `deepspeed_stages.svg` and the text write "optimizer". Fix: "optimizer".
- Optional, `distributed-training/benchmark_gptlite.svg` has no caption, and does not say how the memory is
  averaged. Fix: a caption such as "GPTlite on one node with 8 GPUs, in fp32, with micro-batches of 1 sample:
  average and peak allocated memory per GPU, and throughput over all GPUs".

### Pipeline parallelism (1F1B, Zero Bubble, DualPipe) (2023-08-30-pipeline-parallelism.md)

- Optional, the caption of `distributed-training/pipedream_dnn_pipeline_parallel_microsoft.svg` says that the
  forward passes in flight "belong to different micro-batches". PipeDream pipelines mini-batches and updates the
  weights after each one, as the text says. Fix: "different mini-batches".

### Tensor parallelism (2023-09-02-tensor-parallelism.md)

- `distributed-training/argonne_parallel_2.svg`: the right panel is labelled "(a) Sample parallelism" and shows data
  parallelism, where an all-reduce sums the weight gradients. The caption calls it "communication between
  processors p0 and p1", as if it belonged to the spatial split on the left, and the panel letters run (b), (a)
  from left to right. The text now explains the right panel. Fix: caption "RIGHT: sample (data) parallelism on
  processors p0 and p1, where an all-reduce sums the weight gradients", and swap the letters, or crop the right
  panel.
- Optional, the captions of `dnn_model_parallelism.svg` and `argonne_parallel_2.svg` write the processors as p0,
  p1 and p0-p3, while the figures write P0 to P3 with subscripts. Fix: P0 to P3 in the captions.

### Mixture-of-Experts and Expert Parallelism (2024-01-19-mixture-of-experts.md)

- `distributed-training/imgs/moe_megablocks.svg`: the router outputs the expert indices 1 2 0 2 1 2 for "the quick
  brown fox jumped over", which send "quick", "fox" and "over" to expert 2 and "brown" to expert 0. The permutation,
  the computation and the output put "brown" at Expert-2, and "quick" and "fox" at Expert-0. Fix: indices
  1 0 2 0 1 0. Check whether Figure 1 of the MegaBlocks paper has the same error.
- `distributed-training/imgs/moe_gshard.svg`: the three titles write "Transfomer". Fix: "Transformer".
- Optional, `moe_2017_dean.svg` writes the gate as G(x) and the number of experts as n, and `moe_gshard.svg` writes E
  experts and N layers, as their papers do. The text writes the gate as g(x) and the number of experts as N, and
  names the notation of the first figure. Fix: relabel the figures to g(x) and N experts, and the GShard layers to L.

### Variable sequence lengths, curriculum learning, adaptive batch size and LR (2024-05-10-training-variable-length-curriculum.md)

- Optional, `distributed-training/imgs/dnn_multiproc.svg`, `dnn_multiproc_compiled.svg` and
  `dnn_multiproc_optimized.svg` name the two processes "Node 1" and "Node 2", while the captions and the text say
  2 GPUs. The text names the labels. Fix: "GPU 0" and "GPU 1".
- Optional, `distributed-training/imgs/variable_attn_matrix.svg` names the slices of the mask "layer 0" to
  "layer 3", which reads as network layers, and calls the causal masks "diagonal mask". The text names the
  slices. Fix: "sample 0" to "sample 3", and "causal mask".

### Sequence and context parallelism with Ulysses and Ring attention (2024-08-30-sequence-context-parallelism.md)

- `distributed-training/imgs/ulysses_sequence_parallelism.svg`: the left label reads "same input to both procs
  (split by time dimension)", but 4 processes hold different chunks of the input. The bottom label reads "Local
  attention weights KQ^T : B*H/P, S, S, E", but the attention weights are QK^T, of shape B*H/P x S x S. Fix:
  "input split along the sequence across 4 processes" and "Local attention weights QK^T: B*H/P x S x S".
- Optional, `distributed-training/imgs/context_parallelism.svg` writes the head size as E and leaves out the scale
  1/sqrt(d) of the scores. The text names the label. Fix: write the head size as d.

### Diffusion models, from DDPMs to DiTs (2025-02-25-diffusion.md)

- `diffusion/masked_autoencoders_cropped.svg` and `masked_autoencoders.svg` are now the same picture, shown twice in
  the post. Fix: keep one file and link both places to it, or show the picture once.
- Optional, `diffusion/dit_moe.svg`: the dashed residual connections start after the first AdaLN, while the code of
  the post adds the input of the block, and three experts are gated, while the code routes each token to 2 experts.
  The picture follows the figure of the paper, which draws both this way. Fix: check both against the code, and say in
  the caption where the code differs.
- Optional, `diffusion/masked_autoencoders.svg`, `masked_autoencoders_cropped.svg` and `align_your_latents.svg`
  (with its caption) write the number of frames as T, the symbol of the diffusion steps in the post. The text uses
  F, as the Latte figures do. These are figures from the papers. Fix: relabel the frame axis of the two MAE figures
  as F, or keep the paper labels.

### Faster attention (GQA, FlashAttention, sparse, linear and latent attention) (2025-12-30-inference-attention.md)

- Optional, `inference/imgs/mla.svg`: the absorb = True panel shows a whole sequence, while weight absorption serves
  decode, with one new token (H x 1 x (P+1) scores over P+1 latent vectors). The panels are labeled "usually prefill"
  and "usually decode". Fix: redraw the absorb = True panel for one decode token.

### Kernel Fusion for Megatron-LM Tensor Parallelism and MoE Expert Sharding (2026-08-18-kernel-fusion-tensor-parallelism.md)

- Optional, `inference/imgs/tp_results.svg` names the fused kernel "Ours" in its legend, and its caption says
  "fused (ours)", in the first person. Fix: "Fused" in the legend and "fused" in the caption.
- Optional, `inference/imgs/tp_expert_sharding.svg` and its caption write the weights of an expert as W_i and W_o,
  while the rest of the post writes the same matrices as W_1 and W_2. The text names the mapping. Fix: relabel the
  figure and the caption to W_1 and W_2, and drop "(the W_1 and W_2 above)" from the text.

### Kernel Fusion for MoE Expert Parallelism (2026-10-09-kernel-fusion-expert-parallelism.md)

- `inference/imgs/ep_all_to_all_reduce.svg` names the buffer of the partial result of rank x as Σ_{x/B}. This index
  gives W buffers for B = 1, and a single buffer for B = W, the opposite of the text. The picture (W = 4, B = 2) puts
  σ_0 and σ_1 in Σ_0, and σ_2 and σ_3 in Σ_1. Fix: Σ_{⌊xB/W⌋}.
- `inference/imgs/ep_stages.svg` draws the eight stages on a time axis from 0 to 300 µs. In the paper, this figure is
  the decode step, and its timings are placeholders, not measurements. The caption does not say so. Fix: remove the
  time axis, or replace the timings with measured ones.
- `inference/imgs/ep_results.svg`: the right panel plots decode times that are placeholders, not measurements, as the
  caption says. Fix: remove the right panel and its sentence in the caption until the decode step is measured.
- Optional, `inference/imgs/ep_timeline.svg` labels the dispatch call of P_2 as put_signal_01, but the call goes to
  the INC switch, while the caption says that a subscript names the destination of the call. Fix: put_signal*
  without a subscript, or a caption that says the subscript names the final destinations P_0 and P_1.

### Knowledge distillation, speculative decoding and multi-token decoding (_drafts/2026-03-17-inference-distillation-speculative-decoding.md)

- `inference/imgs/layerskip.svg` labels the drafts "drafts of the first E = 4 layers". E is the embedding dimension
  in the code and in the eagle, medusa and mtp_parallel figures. The figure also names the predictions of the target
  p_1 to p_3, while p is the distribution of the draft model in spec_sampling. Fix: "drafts of the first 4 layers",
  and y_1 to y_3 (or no symbol) for the predictions.
- Optional, `inference/imgs/mtp_parallel.svg` writes the tokens as x_t, as Gloeckle et al. do, while eagle, layerskip
  and mtp_sequential write them as t_i. The text follows each figure. Fix: keep the paper notations, or write t_i in
  mtp_parallel and in the formulas of the parallel heads.

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

## Content issues

Content problems that need the author's decision, such as overlap between posts.

### Mathematics for ML (2018-04-20-algebra-for-ml.md)

- Overlap with the deep neural networks post (2018-03-27-deep-neural-networks.md): both posts explain
  back-propagation as the chain rule and introduce automatic differentiation. Proposed fix: this post keeps the math
  (the chain rule through K layers and the worked example with its computation graph); the deep neural networks
  post keeps the algorithm, its costs and the PyTorch details.

### Statistics for ML (2018-06-15-statistics-for-ml.md)

- Overlap with the exponential family post (2019-03-20-exponential-family.md), which also covers the canonical form,
  the Bernoulli example, the derivatives of the log-partition function and maximum likelihood, and with the
  variational inference post (2019-07-01-variational-inference-gmm.md), which also derives the ELBO. This post keeps
  a short version of each and links to the dedicated post. Proposed fix: keep the short versions as a reference
  sheet and the full derivations in the dedicated posts; the alternative is to keep only the links here.

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- The rewrite removed three parts that repeat other posts: the refresher on the Gaussian pdf (in the statistics
  post), the section "Adding regularization", whose ridge solution is the MAP estimate of the next section, and the
  appendix of linear algebra rules (the derivations now state the two gradient rules they use; the mathematics for
  ML post and The Matrix Cookbook cover the rest). Proposed fix: keep them removed; restore the appendix if it
  should stay as a refresher.

### Variational Inference (ELBO, CAVI) and Gaussian Mixture Models (2019-07-01-variational-inference-gmm.md)

- The rewrite replaced the derivation of the CAVI update with the chain rule of probability, from the CMU 10-708
  lecture notes, by the shorter derivation of section 10.1.1 of Bishop's PRML: as a function of one factor, the ELBO
  is a negative KL divergence. It also removed the recaps of conditional probability and of the chain rule. The CMU
  notes stay linked. Proposed fix: keep the shorter derivation.

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

### Pipeline parallelism (1F1B, Zero Bubble, DualPipe) (2023-08-30-pipeline-parallelism.md)

- Results, memory per GPU: the first GPU used more memory than the others, also with the deep benchmark model, and
  the post says that 1F1B causes such an imbalance. With micro-batches of 1 sample, the activations of the deep
  benchmark model take about 0.5 MB per micro-batch and stage, so 1F1B explains a few MB at most for that model,
  and the cause of a larger gap is not known. Proposed fix: measure the peak memory of every GPU
  (`torch.cuda.max_memory_allocated()` on each rank, and `nvidia-smi` for the CUDA contexts on GPU 0), then give
  the numbers in the post, or drop the benchmark model from that sentence.
- The post keeps the workaround `outputs = outputs.requires_grad_(True)` for the error of the swapaxes lambda. If
  `outputs` has no `grad_fn`, this call lets the backward pass run, but no gradient reaches the layers before it.
  Proposed fix: check that the parameters of the last stage get gradients with the workaround (`p.grad` after
  `train_batch()`), and remove the sentence if they do not.

### Tensor parallelism (2023-09-02-tensor-parallelism.md)

- The last section was named "Final remarks and code", but the repository holds no code for this post: no file
  defines `Megatron_f`, `Megatron_g`, `Megatron_FeedForward` or `Megatron_MHA`. Proposed fix: add these classes
  to `assets/distributed-training` with a small test on 2 GPUs, or keep the code in the post only.

### Pruning and quantization (_drafts/2026-04-22-inference-pruning-quantization.md)

- The draft describes the experiments of `inference/pruning.py` and `inference/quantization.py` on GPTlite (pruning
  with distillation, Wanda, ShortGPT, weight-only and weight-activation quantization, KV cache quantization), but
  shows none of their results. Fix: run the two scripts and add a table of their outputs (parameters, size,
  validation loss, time per token, matching tokens), or present the code without the comparisons.

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
