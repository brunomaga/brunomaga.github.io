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

### Distributed Matrix Transpose Algorithms (2015-04-21-matrix-transpose.md)

- `assets/matrix-transpose/matrix-transposer.zip`, `MatrixTransposer.hpp`, use-case 1 (`Transpose(int row_count, ...,
  MPI_Comm comm)`, lines 54 to 70): `MPI_Comm_size(MPI_COMM_WORLD, &mpi_size)` and `MPI_Allgather(...,
  MPI_COMM_WORLD)` ignore the `comm` argument, so the row offsets are wrong (or the call hangs) when the function is
  called on a sub-communicator. Fix: use `comm` in both calls, as use-case 2 does.
- `assets/matrix-transpose/matrix-transposer.zip`, `main.cpp`: `main` returns without calling `MPI_Finalize`, which
  the MPI standard requires before a process exits (Open MPI reports an error for it). Fix: call `MPI_Finalize()`
  before `return 0;`.
- Optional, `assets/matrix-transpose/matrix-transposer.zip`, `main.cpp`: the benchmark times the transpose with
  `clock()`, i.e. the CPU time of rank 0, and the loop over `scale` never frees `cell_counts` (a leak per iteration).
  Fix: time with `MPI_Wtime()`, and add `delete [] cell_counts;` next to the other `delete []` calls at the end of the
  loop.
- Optional, `assets/matrix-transpose/matrix-transposer.zip`, `MatrixTransposer.hpp`: `QSortCompare` returns
  `a->row - b->row`, a difference of unsigned ids, as an `int`, which gives the wrong order when two ids differ by
  2^31 or more; and the MPI counts are `int` numbers of bytes, so one call cannot send more than 2 GB to a rank. Fix:
  `return (a->row > b->row) - (a->row < b->row);`, and the same for `col`; for the counts, send the cells with a
  derived datatype (`MPI_Type_contiguous`) or with the large-count calls of MPI 4.0 (`MPI_Alltoallv_c`).

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
- `assets/slicing/slicing.tar.gz` is an uncompressed POSIX tar archive with a `.tar.gz` name: `tar -xzf
  slicing.tar.gz` stops with "gzip: stdin: not in gzip format" (only `tar -xf` works). Fix: compress it (`gzip -c
  slicing.tar > slicing.tar.gz`), or rename it to `slicing.tar` and change the link of the post.
- `assets/slicing/slicing.tar.gz`, `SortBalanceSplit.cxx` lines 71 and 76, and `Slicing.cxx` lines 161 and 162
  (`getUniverseBoundaries`): the functions take an `MPI_Comm` argument, but start the recursion and reduce the
  boundaries on `MPI_COMM_WORLD`, and line 173 gathers the final slices on `MPI_COMM_WORLD`. A call on a
  sub-communicator hangs or mixes ranks. Fix: pass `mpiComm` to `sortTileRecursive_aux` and to the `MPI_Allreduce`
  calls, and pass the top-level communicator down for the final `MPI_Allgather`.
- `assets/slicing/slicing.tar.gz`, `Slicing.cxx` line 138: the maximum coordinates start at
  `std::numeric_limits<double>::min()`, the smallest positive double, so the universe maximum of a dimension is wrong
  (about 0) when all its coordinates are negative. Fix: start at `std::numeric_limits<double>::lowest()` (or
  `-std::numeric_limits<double>::max()`).
- Optional, `assets/slicing/slicing.tar.gz`: the `samplingRate` argument of `SortBalanceSplit::sortTileRecursive` and
  `SerialSlicing::sortTileRecursive` is checked but not used, as the sampling code is commented out, so the serial
  slicing of the post runs on all the data. When it is restored, `Slicing::getSamples` (`Slicing.cxx` line 13)
  allocates `(long long)(n*samplingRate*3)` doubles but writes `3*ceil(n*samplingRate)` (e.g. 7 for 9 when `n = 10`
  and `samplingRate = 0.25`). Fix: allocate `3 * (long long) ceil(n * samplingRate)` doubles and restore the sampling,
  or remove the argument.
- Optional, `assets/slicing/slicing.tar.gz`, `Slicing.cxx` line 54: the Windows branch calls `qsort_s(*base, ...)`,
  which dereferences a `void*` and does not compile. Fix: `qsort_s(base, ...)`.
- `assets/slicing/slicing.tar.gz`, `SortBalanceSplit.cxx` (`getSubSlicesDetails`): every slice gets `(int)(p / S_k)`
  ranks, with only a warning when $$S_k$$ does not divide $$p$$. For $$p = 8$$ and $$S_k = 3$$, the case that
  `slicing/split.svg` shows, the slices are ranks 0-1, 2-3 and 4-5: ranks 6 and 7 belong to no slice and never call
  `MPI_Comm_split`, so the other ranks wait in that collective. Fix: give slice $$s$$ the ranks
  `ceil(s * p / S_k)` to `ceil((s + 1) * p / S_k) - 1` (3, 3 and 2 ranks for $$p = 8$$ and $$S_k = 3$$, as the
  picture; no slice is empty, as $$p \ge S_k$$), with `numberOfCPUs` the size of that range.

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
- `assets/lif-brunel/lif-brunel.tar.gz`, `include/Simulation.hpp` and `src/Simulation.cpp`: both `Simulation`
  constructors take `unsigned int relative_strength`, while `UserArguments` and `Network` use a `double`. A
  command-line value `-G 4.5` is truncated to `g = 4`, so the SI slow plot (`raster_plot_g4.5_f0.9`) may have been run
  with `g = 4`. Fix: declare `relative_strength` as `double` in both constructors, then rerun the SI slow plot.
- Same archive, `src/Network.cpp`, `Network::update`, variable-step branch: after a spike, `Neuron::update` has
  already moved the neuron's time past the refractory period (`t_ += refractory_period_`), so the raster file records
  every spike 2 ms late, and `spike_times_[(int)last_neuron->get_t()]` ignores `spike_interval_` and can write past
  the end of the array (`time_of_simulation / spike_interval + 1` entries) for spikes near the end of the simulation.
  Fix: record the spike time (a getter for `last_spike_time_`), and compute and clamp the index as the fixed-step
  branch does.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- `assets/unsupervised-learning/kohonen_mnist.zip`, `kohonen.py`, `name2digits`: the function loads `hash.mat`,
  which is not in the zip, so the script stops with a `FileNotFoundError` before the training starts. Fix: add
  `hash.mat` to the zip, or set the digits directly, e.g. `targetdigits = [0, 1, 2, 3]`, the digits of the post's
  figure.
- `assets/unsupervised-learning/kohonen_mnist.zip`, `kohonen.py`, `name2digits` (besides the missing `hash.mat`): the
  hash `s` is a float (`s += primenumbers[i]*ord(name[i])*2.0**(i+1)`), so `t = np.mod(s, x.shape[0])` is a float and
  `x[t,:]` raises an `IndexError` (NumPy 1.12 and later refuse float indices), even once `hash.mat` is added to the
  zip. Fix: `t = int(np.mod(s, x.shape[0]))`, or set `targetdigits = [0, 1, 2, 3]` directly, as the `hash.mat` item
  proposes.

### Deep Neural Networks, Backpropagation, Dropout, CNNs and Embeddings (2018-03-27-deep-neural-networks.md)

- Deep neural networks, dropout snippet: `MyDropout` reads `self.p` but defines no `__init__` that sets it, so the
  first call fails with an `AttributeError`; also, `torch.distributions.binomial.Binomial(probs=1-self.p)` samples its
  mask on the CPU, so `X * mask` fails for an input `X` on a GPU. Proposed fix: add `def __init__(self, p=0.5):
  super().__init__(); self.p = p`, and draw the mask on the device of `X`, e.g. `mask = (torch.rand_like(X) >=
  self.p).to(X.dtype)` and `return X * mask / (1 - self.p)`.

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

- Post, section "The Causal Mask": the uniform attention matrix is created on `device`, but `x` and
  `tril` on the CPU, so on a GPU machine `wei @ x` and `wei.masked_fill(tril==0, ...)` fail with a device mismatch.
  The alternative notation applies the mask and the softmax to the `wei` of the previous snippet, which gives the
  uniform matrix only because the unmasked entries of each row are equal. Fix: drop `device=device` from the first
  snippet, and start the second one with `wei = torch.zeros((S,S))`.
- Post, section "Train loop": `eval_loss` runs with dropout on, since the model never switches to eval mode, and
  the loop defines the function again at every step. Fix: define `eval_loss` once before the loop, and call
  `m.eval()` at its start and `m.train()` before it returns, as `assets/gptlite/main.py` does.
- Post, section "Final Model": the comment "one layer normalization layer after transformer blocks and one before
  linear layer that outputs the vocabulary" describes the single `self.ln` as two layers. Fix: "one layer
  normalization after the transformer blocks, before the linear layer that outputs the vocabulary".
- Post, section "Tokenization": the code block that lists the 65 characters is marked `python`, so the highlighter
  marks "!" as an error. Fix: remove the language tag `python` from the opening fence.
- Post, section "Tokenization": the same listing leaves out the space, the character after the line break, which the
  text counts among the 65 characters. Fix: start the second line of the listing with a space (" !$&',-.3:;?ABC...").
- `assets/gptlite/gptlite.py`, `GPTlite.__init__`: the comment `# Add this` after `self.ln = nn.LayerNorm(d_model)`
  is a leftover editing note. Fix: remove it.
- `assets/gptlite/utils.py`, `sin_cos_positional_embeddings`: no code calls it, and it differs from the formula of
  the transformer: the base is 1000 instead of 10000, and the odd dimension 2i+1 uses the exponent (2i+1)/E instead
  of the 2i/E of its pair. Fix: remove the function, or set `div_term = 10000 ** (2 * (torch.arange(E) // 2) / E)`.
- `assets/gptlite/utils.py`, `get_gpt3_small_model_parameters`: the docstring dates the GPT-3 paper to 2021 instead
  of 2020, and `batch_size = 500_000` counts tokens in the paper (0.5M tokens, about 244 sequences of 2048 tokens),
  while `get_batch` reads it as a number of sequences. The only caller overrides it. Fix: "Brown et al, 2020", and
  `batch_size = 500_000 // 2048`, or a comment that the value counts tokens.
- Building a GPT model (gptlite), sections "Final Model" and "Size matters": `generate` samples with the model in
  training mode and with autograd on: dropout stays active while the trained model generates text (the post never
  calls `m.eval()`), and every generation step builds a graph for a backward pass that never comes. Proposed fix:
  decorate `generate` with `@torch.no_grad()`, and call `m.eval()` before `print(decode(m.generate(idx,
  max_new_tokens=500).tolist()[0]))`, as `assets/gptlite/main.py` does (`model.eval()` and `torch.inference_mode()`
  before its generation loop).
- Building a GPT model (gptlite), section "The Causal Mask": the comment of `out = wei @ x` gives the
  shapes "(B,S,S) @ (B,S,E)", while `wei` has the shape (S,S) and is broadcast over the batch. Proposed fix: `# dot
  product shape: (S,S) @ (B,S,E) = (B,S,E), with wei broadcast over the batch, i.e. ([4, 8, 2])`.

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
- Post and `assets/gptlite-cpp/gptlite.cpp`, `MultiHeadAttention::forward`: the loop runs over the global constant
  `n_head` (`Tensor outputs[n_head]; for (int i=0; i<n_head; i++)`), not over the `num_heads` heads that the
  constructor created, so a module built with another number of heads reads past `heads` or ignores some of them. Fix:
  `std::vector<Tensor> outputs; for (const auto& m : *heads) outputs.push_back(m->as<Head>()->forward(x));`.
- Post, TorchScript inference snippet: `JitModule model = torch::jit::load("model_jit.pt").to(device);` does not
  compile, because `torch::jit::Module::to` returns `void` (`torch/csrc/jit/api/module.h`, v2.1.0). `main.cpp` does it
  right. Fix: `JitModule model = torch::jit::load("model_jit.pt"); model.to(device);`, or
  `torch::jit::load("model_jit.pt", device)`.

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
- The PyTorch snippets never call `torch.distributed.init_process_group()`, which `DistributedDataParallel` and
  `FullyShardedDataParallel` need, and the training loop calls `optimizer.step()`, but no `optimizer` is ever created.
  Fix: add `torch.distributed.init_process_group(backend="nccl")` and `torch.cuda.set_device(local_rank)` after
  reading the environment variables, and `optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)` after wrapping
  the model.
- The torchvision detour does not match the GPTlite functions: its `get_dataset()` returns one dataset, while the
  GPTlite version returns `train_dataset, valid_dataset, vocab_size`, and `main_deepspeed` unpacks three values and
  passes the third to `get_model()`. Fix: `return dataset, None, 10` in the torchvision `get_dataset()`.

### Pipeline parallelism (1F1B, Zero Bubble, DualPipe) (2023-08-30-pipeline-parallelism.md)

- Toy PyTorch pipeline in the post (`pipeline_step_no_overlap`): stage 0 and the middle stages overwrite `act` with
  the stage output. A middle stage then sends `act.grad` of its output, which is `None`, so a pipeline with more than
  2 stages stops at the first backward pass. Stage 0 also calls `requires_grad_(True)` on its input, which fails for
  integer token ids, and the receive buffers take the shape of the input `x_mb`, not of the activations. Fix: keep
  the input and the output of the stage apart, as `pytorch_pipeline_train_step()` in
  `assets/distributed-training/gptlite_ds.py` does (`act_in`, `act_out`, then `dist.send(act_in.grad, ...)`).
- The activation checkpointing snippet reads `args.activation_checkpoint_interval` inside `get_model()`, where
  `args` is not defined. Fix: `'activation_checkpoint_interval': activation_checkpoint_interval`, as in
  `gptlite_ds.py`.
- The `ImportError` of `gptlite_ds.py` (see the data parallelism entry) also stops the code of this post, and the
  post's `LayerSpec(Block, n_embd, n_head)` follows the old `Block` signature. Fix: as in the data parallelism
  entry.
- `GPTlite.to_layers()` in the post starts with a lambda that sums the embeddings. `PipelineModule` does not register
  the parameters of a layer that is not an `nn.Module`, so the optimizer does not update the embedding tables, and
  `partition_method="parameters"` counts them as 0. The partition output in the results shows this: it comes from this
  `to_layers()` version (16 layers, with the swapaxes lambda), not from the `LayerSpec` version that the runs use.
  `gptlite_ds.py` returns the module `emb_sum` instead. Fix: show the `gptlite_ds.py` version, `[self.emb_sum,
  *self.blocks, self.ln, self.lm_head]`, adapt the paragraph after it, and rerun the partition example of the results
  with the `LayerSpec` version.
- Post and `train.py` (line 100): `step_count = len(train_dataset) // engine.gradient_accumulation_steps()` is the
  number of `train_batch()` calls per epoch only with one pipeline and micro-batches of 1 sample. With 4 stages on 8
  GPUs (2 pipelines), each epoch of the loop runs 2 passes over the data. Fix: `step_count = len(train_dataset) //
  engine.train_batch_size()`.
- `train.py` (line 66) also uses `CrossEntropyLoss_FlatView` for the benchmark model, whose outputs have shape (batch,
  classes), so `B, S, C = logits.shape` stops with a `ValueError`. Fix: `F.cross_entropy(logits.flatten(0, -2),
  labels.flatten())`, which accepts both shapes.
- `assets/distributed-training/gptlite_ds.py` (line 17) imports `n_layer, n_embd, n_head, block_size` from
  `assets/gptlite/gptlite.py`, which defines none of them at module level, so the import stops with an `ImportError`.
  Its `Block` takes `(d_model, n_heads, d_head, dropout_p)`, while `gptlite_ds.py` (lines 60 and 116) and the post's
  `LayerSpec(Block, n_embd, n_head)` pass two arguments. Fix: define the hyper-parameters in `gptlite_ds.py` (or
  import the configuration of the GPT-lite post) and call `Block(n_embd, n_head, n_embd // n_head, dropout_p)`, also
  in the post's `LayerSpec`.
- The first `get_model()` snippet of the post (DeepSpeed section) calls
  `deepspeed.runtime.utils.set_random_seed(random_seed)`, but `random_seed` is an argument of `main_deepspeed()`, not
  of `get_model()`, and `train.py` already sets the seed there (line 63). The snippet also uses `device_str` and
  `gptlite.GPTlite`, which it does not define, and its argument order `(criterion, vocab_size, ...)` differs from the
  `gptlite_ds.py` version `(vocab_size, criterion=None, ...)` that `train.py` calls. Fix: show the `get_model()` of
  `gptlite_ds.py`, without the seed call, with `device_str = _get_device_str()` and `GPTlite(...)`, and label the
  snippets with `gptlite_ds.py`, the file that holds this code.

### Tensor parallelism like Megatron-LM (2023-09-02-tensor-parallelism.md)

- `Megatron_FeedForward.__init__` builds `nn.Dropout(dropout)`, but `dropout` is not one of its arguments, while
  `Megatron_MHA` takes it as an argument. The class only works with a global `dropout`. Fix: add a `dropout`
  argument to `Megatron_FeedForward.__init__`.
- A comment of `Megatron_MHA` says "then we all-reduce (Megatron_g) to combine ranks", in the first person. Fix:
  "then Megatron_g all-reduces (sums) the outputs across ranks".
- `Megatron_FeedForward` and `Megatron_MHA`: the shards of `fc1`, `fc2`, `qkv` and `proj` are initialized, and the
  dropout masks drawn, from the default random state of each GPU. With the same seed on all GPUs of a tensor-parallel
  group, the shards start from identical values (the split layers hold copies of the same neurons) and the attention
  dropout draws the same mask for the heads of every GPU; with different seeds, the dropout after $$g$$
  (`self.dropout`, `self.out_dropout`) draws different masks on activations that all GPUs hold in full, and the copies
  diverge. Megatron-LM (Appendix B.2) seeds the dropout outside the model parallel regions identically on all workers,
  and keeps a second random number generator, seeded differently on each worker, for the dropout inside them. Fix:
  seed all GPUs identically, and initialize the shards and run the attention dropout under a second generator seeded
  with the tensor-parallel rank, as Megatron-LM does with `model_parallel_cuda_manual_seed` and
  `get_cuda_rng_tracker().fork()`.

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
- Optional: the comment `# We return the collated batch over all dataset samples referenced in the microbatches.` in
  `collate_fn_wrapper` is in the first person, which the text of the posts avoids. Fix: `# Return the collated batch
  of all dataset samples referenced in the micro-batches.`
- Optional: the comments of `VariableBatchSizeLR.step` use `-->` arrows. They are plain text in a code block, but
  they end an HTML comment early, e.g. one that hides the code. Fix: `->` in the three comments.

### Sequence and context parallelism with Ulysses and Ring attention (2024-08-30-sequence-context-parallelism.md)

- `isend_k_and_v` passes the group ranks `(rank + 1) % P` and `(rank - 1) % P` as the `peer` of `dist.P2POp`, but
  `torch.distributed` reads `peer` as a global rank (`isend`: "Destination rank on global process group
  (regardless of group argument)"). The code works only for a group of the global ranks 0 to P-1, not for the
  second sequence-parallel group of a data- and sequence-parallel run. Fix:
  `dst = dist.get_global_rank(group, (rank + 1) % P)`, and the same for `src`.
- The ring code calls `fa._flash_attn_forward` with 8 return values and a `window_size` tuple, and
  `fa._flash_attn_backward` with the same tuple, as in flash-attn 2.6. In current flash-attn (2.8),
  `_flash_attn_forward` returns 4 values and takes `window_size_left` and `window_size_right`. Fix: pin
  `flash-attn==2.6.*`, or update the calls.
- Both attention implementations run without a causal mask, while the GPTlite blocks they replace are causal. Fix:
  `causal=True` in the `flash_attn_func` of Ulysses; in the ring, the causal mask on the diagonal block, and no
  computation for the blocks above the diagonal.
- The comments of the ring `MultiHeadAttention.forward` use `E` for both the embedding size and the head size:
  `(B, S/P, E) -> (H, B, S/P, E)`. Fix: `(H, B, S/P, d_head)` and `(B, S/P, H, d_head)`.
- The ring `MultiHeadAttention.forward` is indented by 5 spaces. Fix: 4 spaces.
- The docstring of `dist_view_swap` and its line `H, B, S, E = full_shape` call the last dimension E, but
  `MultiHeadAttention` calls the function on tensors whose last dimension is the head size. Fix: `(H, B, S, d)` in the
  docstring, and `H, B, S, d = full_shape`.
- Optional: the forward pass of `RingAttention` sends k and v at its last step too, only to get its own block back
  for `save_for_backward`, and the backward pass sends k and v at its last step as well. Fix: save the local k and v
  before the loop and skip the last send, as ring-flash-attention does (`step + 1 != world_size`), and the same at
  the last step of the backward pass.

### Diffusion models, from DDPMs to DiTs (2025-02-25-diffusion.md)

- The decoder comment in `DiT.__init__` (in `diffusion/dit.py` and in the post) quotes the DiT paper with other
  words. The paper reads "We use a standard linear decoder to do this; we apply the final layer norm (adaptive if
  using adaLN) and linearly decode each token into a p×p×2C tensor". Fix: quote these words, or drop the quotation
  marks.
- The EMA comment "with a warmup as in diffusers" (in `diffusion/diffusion.py` and in the post): the decay
  (1+n)/(10+n) is the default of the diffusers `EMAModel`, which gives the name "warmup" (`use_ema_warmup`) to
  another formula. Fix: "0.9999 as in the paper, ramped up as in the default EMAModel of diffusers".

### Prefill vs decode, roofline, KV cache, batching and disaggregation (2025-06-07-inference-prefill-decode-roofline-cache.md)

- `generate_kvcache` (KV cache section) does not crop a prompt longer than `seqlen`. Its prefill attends over all the
  prompt tokens and gives every token past `seqlen-1` the position `seqlen-1`. It also returns the keys and values of
  all the prompt tokens, because `MHA` keeps only the last `max_seqlen` when a `kv_cache` is given. However,
  `generate` crops the prompt to its last `seqlen` tokens, and the text says that the cache keeps the last `seqlen`
  keys and values. So for such a prompt, the two functions differ from the first generated token. Fix: prefill only
  the last `seqlen` tokens when no cache is given, as `generate` does: `logits, kv_cache = model(prompt if kv_cache is
  not None else prompt[:, -seqlen:], kv_cache, max_seqlen=seqlen)`.
- Optional: the roofline plotting code labels the point of the vector sum "Your Code" (`label="Your Code"` and
  `f"Your Code ({code_x:.3f}, {code_y:.1f})"`), in the second person, while the text calls it the vector sum. Fix:
  `label="Vector sum"` and `f"Vector sum ({code_x:.3f}, {code_y:.1f})"`.

### Faster attention (GQA, FlashAttention, sparse, linear and latent attention) (2025-12-30-inference-attention.md)

- `indexer_loss` in the DSA code block masks the scores with -1e9, which does not fit in float16 (its largest value
  is 65,504), while the comment of `scaled_dot_product_attention` in the same post gives this as the reason to use
  `torch.finfo(dtype).min`. The queries of the first block of CSA have rows that are fully masked. Fix: replace both
  -1e9 of `indexer_loss` with `torch.finfo(<tensor>.dtype).min`.
- Optional: the DSA code block trains the indexer only during the dense warm-up, as the text before it says: in the
  sparse branch (`dense = False`) it sets `self.indexer_loss = None`, and since `x_index = x.detach()` and the top-k
  selection has no gradient, the indexer stops learning. DeepSeek-V3.2 (arXiv 2512.02556, Sec. 2.1.1, "Sparse
  Training Stage", Eq. 4) keeps training the indexer in that stage, with the KL-divergence loss restricted to the
  selected tokens. Fix: in the sparse branch, compute the logits `scores = q @ k.transpose(-1, -2) / h ** 0.5` and set
  `self.indexer_loss = indexer_loss(index_scores, scores, mask)` with the mask of the selected tokens (the function
  already takes a `[B, Sq, Sk]` mask), update the comments of `self.dense` and `self.indexer_loss`, and remove "for
  simplicity, the code trains the indexer only during the warm-up" from the text.
- The code comments call the number of cached past tokens P (`[B, G, P+S, h]`, and the docstring of `causal_mask`),
  while the text writes $$P$$ for the softmax matrix of FlashAttention and SageAttention, as the FlashAttention paper
  does. A comment of the MLA code also calls the attention weights `w_t`, while the text writes $$w^I_{t,j}$$ for the
  weights of the indexer heads. Fix: rename the past tokens to T, and the attention weights to `a_t`, in the comments
  and docstrings of the code.

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
- The GPTlite checkpoint of the inference scripts cannot be made with the code of the repository. `load_model`
  (`inference/inference_utils.py`), which `pruning.py`, `quantization.py`, `kvcache.py`, `gqa.py`, `prompt_lookup.py`
  and `speculative_decoding.py` call, builds a GPTlite with `get_gptlite_model_parameters()` (d_model 512, 6 heads,
  seqlen 128) and loads `assets/gptlite/gptlite.pth`, which is not in the repository; without it, the scripts print
  "Couldn't find ...: GPTlite has random weights" and report the results of a random model. The only training script,
  `assets/gptlite/main.py`, trains a GPTlite with `get_gptlite_distilled_model_parameters()` (d_model 256, 4 heads,
  seqlen 64) and saves it as `gptlite.pth` in the working directory: run from `assets/gptlite`, its file has the path
  that `load_model` reads but the shapes of the smaller model, so `load_state_dict` fails with a size mismatch.
  Proposed fix: a short script in `assets/inference` that trains GPTlite with `get_gptlite_model_parameters()` (for
  example with `train_steps` of `inference_utils.py`) and saves it to `GPTLITE_CKPT_PATH`, or the same change in
  `main.py` (the parameters of `get_gptlite_model_parameters()` and the path `GPTLITE_CKPT_PATH`), and one sentence in
  the inference posts that says to run it first.

## Diagram issues

Pictures that are wrong or disagree with the post text (under `assets/`). Items marked "Optional" improve a picture
that is not wrong.

### All pictures

- Style details that the restyle left, from an automated check of the 230 draw.io figures before the picture round
  of 2026-10-22, which fixed some of them: 178 boxes in 15 figures keep the rounded corners of the original with a
  corner radius of 2 to 4.5 px instead of 5 px (for example the GPT-lite pictures and
  `publications/visual_transformer.svg`); 147 arrows in 19 figures are curved but are neither pointers nor loops (for
  example in `deep-neural-networks/densenet.svg`, `lif-brunel/diagram_class.svg` and
  `publications/graphconvnets.svg`); 176 filled cells of `distributed-training/imgs/mixture_of_depths.svg`, and a few
  in other figures, have no outline; `learning-from-sequences/transformer_attention.svg` uses a color outside the
  palette (#97afd2, a heat map shade); 3 figures have lines thinner than 0.4 px. Fix: set them to the values of
  CLAUDE.md, then run the check again.
- The arrow check of the restyle (CLAUDE.md, Pictures, section 6) reported 487 findings in 71 draw.io figures before
  the picture round of 2026-10-22, which fixed some of them: 199 arrows with less than 6 px of line before the head
  (48 of them with 4.6 to 5.9 px, as the 6 px heads take more of short arrows, for example in
  `gptlite-cpp/torch_compile.svg` and the three `distributed-training/imgs/dnn_multiproc*.svg`), 197 lines through
  the text box of a label, 55 labels that run into a box, 28 heads on text and 8 overlapping boxes. Most are in dense
  pictures from papers: `matrix-transpose/xcrs_format.svg` (48), `deep-neural-networks/densenet.svg` (42) and
  `publications/performers.svg` (21); some are text boxes larger than their text. Fix: run the check again, check
  each picture at its page size, move the labels and arrows that touch, and lengthen the short arrows (move their
  bend, or the boxes they join).
- The log charts write the exponents of their tick labels at 8.4 px, under the 10 px of the smallest text size
  (CLAUDE.md, Pictures, section 4): `publications/llama3_scaling_laws.svg`, `scaling_laws.svg`, `scaling_laws_2.svg`,
  `training_compute_optimal_large_language_models.svg` and `training_compute_optimal_large_language_models_2.svg`,
  and `neuron-variable-timestep/cvode_dependency_on_variation_solution.svg` and
  `cvode_dependency_on_events_arrival.svg`. Fix: exponents of 10 px.
- Pictures to redraw, from the user's review of the four line weights (the next step of the restyle):
  - `distributed-training/imgs/workflow_data_parallelism.svg` and `workflow_sharding.svg` changed too much. Fix: the
    layout of the originals in this style, with square inputs, outputs and activations, and round layers and
    parameters.
  - `neuron-electrical-activity/neuron_model_scales.svg`: the compartmental membrane and the reaction-diffusion parts
    look poor. Fix: redraw them from the source file.
  - `distributed-training/imgs/vdnn2.svg` and `vdnn3.svg` are too tall. Fix: a lower layout, with the outlines and
    arrows of CLAUDE.md.
  - `inference/imgs/medusa.svg`, `eagle.svg` and the other pictures of the speculative decoding draft show a 4 x 1
    matrix after the target model, where the originals show a row of 3D cubes. Fix: the row of cubes again.
  - `distributed-training/imgs/dnn_model_parallelism.svg` is hard to read at its width, and its 64 links are lines
    without heads, while CLAUDE.md, Pictures, section 6, draws the links of a dense network as thin arrows with heads.
    Neither the picture nor the text says which layer is the input. Fix: draw it larger, and add the heads in the
    direction of the forward pass, once the user picks the input layer.
- Optional: two labels keep less than the 3 px of CLAUDE.md, Pictures, section 3, between their text and the outline
  of their box, as on master: 𝑠<sub>𝑡−1</sub> of `publications/attention_mech.svg` (2.4 px on each side) and
  𝑥<sub>𝑗+1</sub> of the two rows of tokens of `diffusion/latte2.svg` (2.9 px). Fix: boxes 2 px wider, or keep them.

### Distributed Matrix Multiplication and Database Joins (2014-04-02-distributed-matmul-dbs.md)

- `distributed-matmul-dbs/spmv.svg`: the panel titles "data layout on 3 ranks" and "entries of x on each rank" sit
  about 7 px right of the centres of their panels (at 160.96 and 487.06 px instead of 154.24 and 479.51 px). Fix: x
  = 84.69 for `spmv-1` and x = 403.26 for `spmv-52` in the draw.io file.
- `distributed-matmul-dbs/csr_format.svg`: the array names "values", "col_idx" and "row_ptr" are 16 px regular, while
  `matvec_row1d.svg` and `spmv.svg` name their matrices at 14 px bold. Fix: 14 px bold.
- Caption of `distributed-matmul-dbs/joins.svg`: it says "Dark grey cells are `NULL`", but the picture draws the
  `NULL` cells white, as CLAUDE.md draws unused elements. Fix: "White cells are `NULL`.", or drop the sentence.

### Distributed Matrix Transpose Algorithms (2015-04-21-matrix-transpose.md)

- `matrix-transpose/all_to_all.svg`: the caption writes "block $$j$$ of process $$i$$ goes to process $$j$$, as its
  block $$i$$", while the post calls the processes ranks, numbers them $$r$$ and $$s$$, and uses $$i$$ and $$j$$ for
  vertices. Fix: "`MPI_Alltoall` on 6 ranks: block $$s$$ of rank $$r$$ goes to rank $$s$$, as its block $$r$$. The
  blocks of the first rank, A0 to A5 (red frame), become the first block of every rank." The label "processes" of the
  picture can stay, as it is the word of the MPI tutorial that the picture follows, and the mixture-of-experts post
  shows the same picture and says processes.
- Optional, `matrix-transpose/matrix-transposition-crs.svg`: the labels "Metadata counts (MPI_Alltoall)" and
  "Metadata sparse matrix (MPI_Alltoallv)", and the group label "Metadata" over "Row id", "Column id" and "Element",
  are the paper's own words, which a picture from a source keeps (CLAUDE.md, Pictures, section 1). They copy the XCSR
  wording, while the CSR transpose sends the element counts, then the elements with their row and column indices,
  and the post uses "metadata" only for XCSR. Fix, if the user wants the words of the post: "Element counts
  (MPI_Alltoall)" and "Elements with their indices (MPI_Alltoallv)", without the group label, and "Row index" and
  "Column index", as in the text.
- Optional, `matrix-transpose/crs_format.svg` and `xcrs_format.svg`: the array of column indices is labeled `disps`,
  the paper's own label and MPI's name for displacements, which reads as offsets. Fix: `col_idx`, as in the matmul
  post.
- Optional, `matrix-transpose/xcrs_transpose_3.svg`: the labels "Objects to be sent" and "Objects to be received"
  name the values of the view swap, which the post calls values. Fix: "Values to be sent" and "Values to be
  received".
- `matrix-transpose/xcrs_format.svg`, `xcrs_transpose_1.svg` and `xcrs_transpose_3.svg`: as in the paper's figure,
  some letter boxes straddle the gaps between the blocks of the ranks or cross the outline of a block (F, J/K and R/S
  in xcrs_transpose_1; A, G, O, B/C, H/I, T and R/S in xcrs_transpose_3), and the L/M/N connector of
  xcrs_transpose_1 runs along the right outline of the block of rank 1. Fix: put each box next to its dot inside its
  own block, or in the free space beside the matrix, and route the L/M/N connector down through the empty cell of
  row 5, column 3.

### Distributed Load Balancing and Spatial Slicing (2015-12-01-slicing.md)

- `slicing/n_body_slicing.svg` has no caption, and does not name the source of its N-body render. Fix: a caption,
  such as "Slicing of an N-body simulation (left) and of a small network of neurons (right).", and "Source: ..." once
  the origin of the render is known.
- `slicing/mini-*.svg`: the four layouts have 10 (single axis), 35 (NUG, 5 x 7), 16 (ORB) and 35 (STR, 5 x 7)
  slices. Optional fix: 16 slices in all four (4 x 4 for NUG and STR), so that they compare on equal terms.
- Optional, `slicing/sbs.svg`: in step 1.1, the top side of the red emphasis frame around the first row (x) lies
  where the arms of the rank brackets end, so the arms touch the frame. Fix: move the rank brackets, the rank
  numbers and the "rank" label of step 1.1 up by 3 px.

### The Leaky Integrate-and-Fire Neuron and the Brunel Network (2016-02-08-lif-brunel.md)

- `lif-brunel/raster_plot_g4.5_f0.9.svg`, `raster_plot_g5_f2.svg`, `raster_plot_g3_f2.svg`: the plots do not name
  their state. The post used to call them "slow", "regular" and "fast" dynamics, but their parameters are those of
  Brunel's states SI slow (g = 4.5, eta = 0.9), AI (g = 5, eta = 2) and SR (g = 3, eta = 2); the text now uses these
  names. Fix: a caption under each plot, e.g. "SI slow (g = 4.5, eta = 0.9)", since CLAUDE.md puts titles in captions.
- Same three plots: the axis label "spikes count" reads better as "spike count (per ms)". Fix: relabel it.
- `lif-brunel` table captions: "source: Marc-Oliver Gewaltig, ResearchGate" and "source: Brunel 2000" are lower case
  and have no link, and the parameter table is Table 2 of NEST by Example (Gewaltig et al., 2012), not a ResearchGate
  page. Fix: "Source: [NEST by Example, Table
  2](https://www.nest-simulator.org/wp-content/uploads/2015/02/NEST_by_Example.pdf)" and "Source: [Dynamics of
  Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons, Table
  1](https://link.springer.com/article/10.1023/A:1008925309027)".
- `lif-brunel/raster_plot_g3_f2.svg`, `raster_plot_g4.5_f0.9.svg` and `raster_plot_g5_f2.svg`: the time axis of the
  spike-count histogram is not aligned with the raster's: its 0 and 1,000 ms sit about 7 to 9 px left of the
  raster's. Fix: the same x for 0 and 1,000 ms in the histogram as in the raster, in the three plots.

### Numerical Simulation of Detailed Neuron Models (2016-08-15-neuron-electrical-activity.md)

- `neuron-electrical-activity/brain_scales.svg`: the labels are restyled, but the SVG has no width (its viewBox is
  716 x 1096 px) and the post shows it at 50%, so P does not follow W/880. Fix: width="440" and the matching
  height="673.5" on the svg element, with the viewBox unchanged, so that P = 50 = W/880.
- Caption of `neuron-electrical-activity/brain_scales.svg`: the first credit, "Mariana Villarreal (LadyofHats),
  Wikimedia Commons", has no title and no link, and the author's name is Mariana Ruiz Villarreal. Panel (b) is her
  public-domain drawing "Complete neuron cell diagram en.svg". Fix: "Adapted from: [Complete neuron cell diagram,
  Mariana Ruiz Villarreal (LadyofHats), Wikimedia
  Commons](https://commons.wikimedia.org/wiki/File:Complete_neuron_cell_diagram_en.svg); and Blausen.com staff, ...",
  after a check of the sources of panels (c) and (d).
- Caption of `neuron-electrical-activity/neuron_model_scales.svg`: the credit "J. M. Eppler, Introduction to NEST,
  First community workshop on NESTML, 2015" has no link. Fix: a link to the slides, if they are online.
- `neuron-electrical-activity/neuron_model_scales.svg`: the lower zoom line from E crosses the vertical segment of the
  arrow from I to E. Fix: route the arrow from I to E on the left, next to the arrow from E to I, about 8 px apart.
- `neuron-electrical-activity/neuron_model_scales.svg`: the label "Compartmental membrane" is centred about 8 px right
  of its drawing. Fix: move it 8 px left.
- `neuron-electrical-activity/neuron_model_scales.svg`: Ca²⁺ is in the math font, while CaM and the other names of
  the reaction-diffusion part are in Source Sans 3 (the post writes $$Ca^{2+}$$). Fix: Source Sans 3, if Ca²⁺ counts
  as a name.
- Caption of `neuron-electrical-activity/net_current_conservation.svg`: "the axial currents $$i_a$$ balance the
  membrane current $$i_m$$", but the text defines $$i_m$$ as a current density (mA/cm²), and the axial currents
  balance its integral over the membrane. Fix: "the axial currents $$i_a$$ balance the membrane current density
  $$i_m$$, integrated over the membrane area".
- `neuron-electrical-activity/compartment_rc_circuit.svg`, its caption and the equation above it name the capacitor
  $$C$$, each resistor $$R$$ and the total current $$I$$. In the text, the capacitance of compartment $$j$$ is
  $$c_j$$, $$R$$ is the gas constant, and $$I$$ is the inhibitory population and the identity matrix. One resistor
  stands for all the channels of one type, not for one ion channel, and the caption has no final period. Fix, in one
  change of the picture, the caption and the equation: the label c_j in the picture, the equation $$I = c_j
  \frac{dv_j}{dt} + \sum_i g_i x_i (v_j - E_i)$$, and the caption "The electrical circuit of one compartment. The
  membrane is the capacitor $$c_j$$. Each channel type $$i$$ is a resistor $$R_i$$ in series with a battery $$E_i$$,
  its reversal potential. Arrows mark gated channels, whose resistance changes over time. The leak $$L$$ has a
  constant resistance. $$I = i_{m,j} A_j$$ is the total current through the membrane."
- `neuron-electrical-activity/compartment_rc_circuit.svg`: the open item on this picture proposes the equation $$I =
  c_j \frac{dv_j}{dt} + \sum_i g_i x_i (v_j - E_i)$$, but the text now names the gating term $$w_i$$ (content item
  "Symbols that the author uses", applied in iteration 2). Fix: write $$w_i$$ in that item's equation, $$I = c_j
  \frac{dv_j}{dt} + \sum_i g_i w_i (v_j - E_i)$$, when the picture, the caption and the equation are changed.

### Variable Timestep Simulation of the Electrical Activity of Neurons (2017-02-17-neuron-variable-timestep.md)

- `neuron-variable-timestep/cvode_pulse_currents_table.svg`, `cvode_dependency_on_variation_solution.svg` and
  `cvode_dependency_on_events_arrival.svg`: the currents are in mA and µA (2.7 mA, "threshold current (0.206mA)",
  "0.1µA" to "1mA"), as in the preprint. NEURON's `IClamp` amplitude is in nA, and the threshold current of a cortical
  pyramidal cell model is of the order of 0.1 to 1 nA, so 0.206 mA is likely 0.206 nA. The text gives no current
  values. Fix, after a check of the simulation scripts: mA to nA and µA to pA in the labels.
- Optional, `benchmark_two_rows_2.svg`: the orange "region of similar runtime growth" does not follow from the plotted
  runtimes, and it is hatched. Fix: check it against the benchmark data, or remove it; if it stays, a flat #ffe6cc
  fill with a #d79b00 outline.
- `neuron_203076.svg`, `neuron_138083.svg`, `neuron_208223.svg` and `neuron_87784.svg`: the last 10 ms bar of each
  event histogram leaves out the events after 7.5 s, and the burst bars near 0 s are cut at the top of the plot (198,
  203, 44 and 47 events) without a mark. Fix: redraw the histograms from the event data of the simulations, with the
  count on each cut bar.
- `cvode_pulse_currents_table.svg`: the CVODE step counts cover 1.1 s of simulation, while the 48,001 steps of
  backward Euler cover 1.2 s. The values are those of the table of the preprint. Fix, after a check of the simulation
  scripts: step counts for the same time span.
- The 13 pictures without a caption, most of them figures of the preprint, have no credit. The kernel fusion posts
  have the same open point. Fix: a caption under each picture, ending with "Source: [Fully-Asynchronous
  Fully-Implicit Variable-Order Variable-Timestep Simulation of Neural Networks](link)", if the user wants credits
  for the own preprint.
- `neuron-variable-timestep/neuron_208223.svg` and `neuron_87784.svg` are shown as two neurons of the median 1%, which
  received 541 to 558 events (the text, and Section 3.4 of the preprint). Their 10 ms bars add up to 584 and 586
  events. The same sums give 6,146 and 5,574 for the two top neurons, which agree with the text. Fig. 8 of the
  preprint has the same mismatch. Proposed fix: show two neurons of the median 1%, from the event data of the
  simulation. If that data cannot be found, add a caption that gives their counts (584 and 586 events).
- `benchmark_two_rows_2.svg`: the legend entry "(2a) BSP Deriv. Implicit†" has no key for the dagger. The preprint's
  caption defines it: Newton iterations solve each mechanism implicitly, but the coupling between mechanisms stays
  explicit. Fix: a key line in the legend, "† implicit per mechanism, explicit across mechanisms".
- `neuron-variable-timestep/l5_neuron_pulse1_3ma_100ms_cvode_vs_backward_euler.svg` (the 100 ms chart): the grey
  spike of atol 10⁻¹ crosses the right edge of the tick label 25. Fix: the tick labels and t (ms) under the traces,
  as in the 1000 ms chart.
- `neuron-variable-timestep/l5_neuron_pulse_1000ms_results.svg`: a grey bar is left in the inset (the magnified
  −50 mV line). Fix: remove it, as in the 100 ms chart.
- `neuron-variable-timestep/neuron_132412.svg`: the tick labels 0.1, 10 and 1,000 are missing. Fix: restore them.
- `neuron-variable-timestep/l5_neuron_pulse_1000ms_results.svg`: the zoom box is 0.8 px, while the zoom box of the
  100 ms chart is 1.4 px. Fix: 1.4 px, as its connector.
- `neuron-variable-timestep/l5_neuron_soma_voltage_per_step_6ms.svg`, right panel: a red step dot sits 0.7 px from
  the tick label 4, under the 2.5 px of section 6 of CLAUDE.md. Fix: the tick labels and t (ms) under the traces in
  both panels, as in the 1000 ms chart.

### Linear Regression, Matrix Factorization and Loss Functions (2017-08-17-linear-regression-matrix-factorization.md)

- Optional, `linear-regression-matrix-factorization/matrix-factorization.svg`: the rating 6 (movie 1, user 7) is
  outside a scale of 1 to 5 stars. Fix: a rating from 1 to 5.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- `unsupervised-learning/gaussianscatterpca.svg`: the arrow heads end past the scaled eigenvectors, at 3.75 and 1.75
  instead of 3 and 1, the lengths in the Wikipedia figure it follows. Fix: arrow heads that end at the tips of the
  vectors.
- `unsupervised-learning/kohonen2.svg`: the outline of the input-space sheet is 1.4 px, a weight that no rule of
  CLAUDE.md gives this shape. Fix: leave it, or 1.8 px, as the frame of a 2D domain.
- Optional: `unsupervised-learning/kohonen_1.svg`, `kohonen_2.svg` and `kohonen_3.svg` have no caption; the list
  before them describes the three maps. Fix: one caption under the three pictures, e.g. "The weights of the 6x6
  neurons of the SOM, each drawn as a 28x28 image, for the digits 0 to 3: a tuned SOM (left), a high learning rate
  (center) and a large neighbourhood width (right)." (no credit: the pictures come from the post's own code).
- The captions of `unsupervised-learning/gaussianscatterpca.svg` and `kohonen.svg`, and the label "Input fibres" of
  `kohonen.svg` (draw.io cell kohonen-13), use British spellings, while the text uses American ones. Fix: "centered",
  "fibers", "neighbors" and "Input fibers", and "neighborhood" in the caption proposed above for kohonen_1 to
  kohonen_3.

### Deep Neural Networks, Backpropagation, Dropout, CNNs and Embeddings (2018-03-27-deep-neural-networks.md)

- Optional, `deep-neural-networks/fully_sparse_connectivity.svg` and its caption keep the symbols of Figure 9.2 of
  the Deep Learning book: inputs x1 to x5 and outputs s1 to s5. The text writes the input of a layer as $$x^{(0)}$$
  and its output as $$x^{(1)}$$, as `dnn.svg` does, and says that the $$x_i$$ and $$s_i$$ of the picture stand for
  them. Fix: inputs x₁⁽⁰⁾ to x₅⁽⁰⁾ and outputs x₁⁽¹⁾ to x₅⁽¹⁾ in both rows, and
  the caption "... where the input $$x_3^{(0)}$$ affects only the outputs $$x_2^{(1)}$$, $$x_3^{(1)}$$ and
  $$x_4^{(1)}$$. Bottom: a fully connected layer, where $$x_3^{(0)}$$ affects every output."
- Optional, `deep-neural-networks/fully_sparse_connectivity.svg`, dense panel: the arrows from the far inputs reach
  s1 and s5 only 6 to 11 degrees apart, so 2 to 3 heads overlap into one dark mark (less at s2 and s4). CLAUDE.md
  lets heads that converge on one node of a dense network touch. Fix: end the five arrows into each output at points
  about 7 px apart along the lower half of its circle.
- `deep-neural-networks/densenet.svg`: the curved arrows into the fourth BN-ReLU-Conv box and into the Transition
  Layer arrive horizontally on their slanted (24 degree) top edges, and the lower barb of each head touches the edge.
  Fix: end each curve perpendicular to the edge (its last control point on the normal of the edge), with the whole
  head outside the shape.
- `deep-neural-networks/cnn.svg`: the fully connected layers read "Nx[512]" and "Nx[2]" under one title "Fully
  Connected", while the original picture (cnn.png at commit c1a083b4) has "1024x[512]" and "512x[2]", the shapes of
  the weights (4·4·64 = 1,024 inputs), under two titles. Fix: keep the activation shapes, which match the other
  labels, or restore the labels and the two titles of the original.

### Mathematics for ML (2018-04-20-algebra-for-ml.md)

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
- `statistics-for-ml/confidence_interval.svg`: the drawing has no source. Fix: name the source of the drawing, or
  redraw it. Optional: the curve is centered at X̄, as in many textbooks, but the sampling distribution of X̄ is
  centered at μ; a curve centered at μ, with the interval drawn around one sample mean X̄, is exact.

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- `bayesian-linear-regression/linear_bayesian_predictive_distribution_*.svg`: the model assumes a noise standard
  deviation of 0.71 (noise precision beta = 2, as the caption states), but the data have a noise standard deviation of
  0.25. After 25 samples, the band is thus 0.76 to 0.88 wide on each side of the mean, three times the scatter of the
  samples. Fix: redraw with beta = 16 (band 0.27 to 0.32), see the code issues above, and change the caption.
- Caption of `bayesian-linear-regression/linear_bayesian_predictive_distribution_*.svg`: it gives the prior precision
  $$\alpha = 10^{-3}$$ and the noise precision $$\beta = 2$$, which the text now defines as $$\alpha = 1/\sigma_w^2$$
  and $$\beta = 1/\sigma^2$$. Fix (in a picture task): "a prior precision $$\alpha = 1/\sigma_w^2 = 10^{-3}$$ and a
  noise precision $$\beta = 1/\sigma^2 = 2$$" (or 16 after the redraw with beta = 16).

### Exponential Family of Distributions (2019-03-20-exponential-family.md)

- `exponential-family/common_distributions.svg`: the source post by Sean Owen says that "as the number of trials
  increases, that binomial distribution becomes more like the normal distribution", but the picture has no arrow from
  Binomial to Normal (it has the arrows from Hypergeometric and from Poisson to Normal). Proposed fix: check the
  figure of the source post and add the arrow from Binomial to Normal if the source has it.

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
  writes n. Fix: n.
- Captions of `learning-from-sequences/transformer_attention.svg` and
  `learning-from-sequences/transformer_attention_masked.svg`: they call the softmax outputs "attention weights", and
  the masked caption calls the values before the softmax "attention scores", while the text and the caption of
  `learning-from-sequences/transformer_attention_lbdl.svg` (after The Little Book of Deep Learning) call the softmax
  outputs $$A_{q,k}$$ the attention scores. Fix: "Attention scores of one attention head ..." in the first caption,
  and in the second "Attention scores of the masked attention of the decoder ... Masking sets the dot products of
  unseen words to $$-\infty$$ before the softmax, which gives them an attention score of zero."
- Caption of the table of the computational complexity of the layer types (after Table 1 of the paper): "four
  different learning models" and "size of neighbourhood". Fix: "four layer types" and "size of the neighborhood in
  restricted self-attention", as in Table 1 of the paper.
- Optional, `learning-from-sequences/encoder_decoder.svg`: the label "hidden space t−1". Fix: "hidden state t−1".

### Building a GPT model in PyTorch from scratch (2023-02-28-gptlite.md)

- Optional, the GPT-lite pictures (`gptlite/*.svg`) and
  `learning-from-sequences/transformer.svg`: the one-line boxes (Layer Norm, Linear, Softmax, Add & Norm) are 17.5 px
  tall, so their 14 px labels sit about 2.3 px from the top and bottom outlines, where the arrows attach, and look
  tight. Fix: make the one-line boxes 20 px tall in the whole series, moving every row.

### GPT in C++: LibTorch vs. PyTorch, TorchScript and torch.compile (2023-06-27-gptlite-cpp.md)

- Optional, caption of `gptlite/gptlite.svg` and `benchmark_model.svg`: it starts with "This post details", and calls
  the left model "the GPT-3 Small model", while it is GPTlite with the dimensions of GPT-3 Small. Fix: "Left: GPTlite
  with N decoder blocks, with the dimensions of GPT-3 Small. Right: the benchmark DNN with L layers of width W. The
  left diagram is adapted from Attention is all you need."
- Optional, caption of `gptlite-cpp/benchmark_gptlite.svg`: "GPTlite Small" is a model name that the posts use
  nowhere else. Fix: "GPTlite (with the dimensions of GPT-3 Small)".
- `gptlite-cpp/benchmark_wide.svg` and `benchmark_deep.svg`, and `benchmark.xlsx` they are drawn from: the throughput
  axes say "samples/sec", but `main.cpp` and `main.py` print epochs per second, and each epoch of the two DNN
  benchmarks is one batch of 2048 samples, so the values are iterations per second (the text now says iterations/sec;
  for GPTlite, with a batch of one sequence, both units are the same). Fix: label the DNN throughput axes and the
  spreadsheet columns "iterations/sec", or multiply the DNN values by 2048 to keep samples/sec, after checking how the
  spreadsheet values were entered.
- Optional, `gptlite-cpp/benchmark_wide.svg` and `benchmark_deep.svg`, which the post links but does not show, are
  the only charts that the chart restyle of 2026-10-22 left out: their bar outlines and grid lines keep the 0.8 px of
  their earlier export. Fix: restyle them as `gptlite-cpp/benchmark_gptlite.svg` (bar outlines 1.0 px, CLAUDE.md,
  Pictures, section 6).

### Data parallelism, sharding and CPU offloading (2023-08-18-data-parallelism.md)

- Optional, `distributed-training/imgs/workflow_sharding.svg` shows ZeRO-3 as the ZeRO paper describes it: each
  process owns whole layers, broadcasts their parameters and reduces their gradients. DeepSpeed ZeRO-3 and PyTorch
  FSDP split every layer into one shard per process, with an all-gather of the parameters and a reduce-scatter of the
  gradients. The text explains both. Fix: redraw each layer as four shards, with all-gathers and reduce-scatters.
- Optional, `distributed-training/imgs/deepspeed_stages.svg`: the number of processes is N_d (and gpu_N−1), the
  symbol of the ZeRO paper, while `data_parallelism.svg` and the text use P; the caption defines N_d as the
  data-parallel degree. Fix: keep the paper's N_d and add "(N_d = P)" to the caption, or write P in the figure.
- Optional, `distributed-training/imgs/deepspeed_stages.svg`: the formulas write their products without a sign (KΨ,
  (2 + K) Ψ), while the figure of the ZeRO paper writes K*Ψ. Fix: restore the "*", if the notation of the source is
  wanted.
- Optional, `distributed-training/imgs/vdnn2.svg` and `vdnn3.svg` color the weights W purple and the gradients dX and
  dY green, while `deepspeed_stages.svg` colors the parameters blue, the gradients orange and the optimizer states
  green. Fix, for one color per meaning in the post: in vdnn2 and vdnn3, W blue, dX and dY orange, X and Y purple,
  and the color words of their captions to match. The layers are light blue #dae8fc, at the user's request: with W
  blue, check that the W arrows still stand out from them.
- Optional, the caption of `distributed-training/imgs/benchmark_gptlite.svg` does not say how the memory is averaged.
  Fix: "GPTlite on one node with 8 GPUs, in fp32, with micro-batches of 1 sample: average and peak allocated memory
  per GPU, and throughput over all GPUs".
- Optional, caption of `distributed-training/imgs/data_parallelism.svg`: it names "the DDP data layout" before the
  text introduces DDP, while the layout is the same for DDP, FSDP and ZeRO. Fix: "The data layout of data
  parallelism, split across $$P = 4$$ processes color-coded as blue, yellow, red and green."
- `distributed-training/imgs/vdnn2.svg` and `vdnn3.svg` label the last layer "Layer(N)", and the vdnn2 caption says
  "at layer N", while the text numbers the layers from 1 to $$L+1$$ and writes the number of samples as $$N$$ (under
  the MAE formula). Fix: "Layer(L+1)", and in the caption "at its last layer, $$L+1$$".
- Optional, `distributed-training/imgs/workflow_data_parallelism.svg` keeps 20 px between its process frames and
  12 px between the arrows and the frames, while `workflow_sharding.svg` now has 12 and 8 px. Fix: the same gaps in
  both.

### Pipeline parallelism (1F1B, Zero Bubble, DualPipe) (2023-08-30-pipeline-parallelism.md)

- The caption of `distributed-training/imgs/pipeline_8_stages.svg`, `pipeline_4_stages.svg` and
  `pipeline_2_stages.svg` calls the center and right panels "2 data-parallel groups of 4 pipeline stages" and "4
  data-parallel groups of 2 pipeline stages", but in Megatron-LM and DeepSpeed a data-parallel group holds the GPUs
  of one stage: the panels show 2 pipelines of 4 stages and 4 pipelines of 2 stages. Fix: "Center: 2 pipelines of 4
  stages. Right: 4 pipelines of 2 stages."
- `distributed-training/imgs/pipeline_8_stages.svg`, `pipeline_4_stages.svg` and `pipeline_2_stages.svg` draw each
  backward pass as long as a forward pass (1 time unit), while the other schedules, and section 8 of CLAUDE.md, give
  it 2 units. Fix: backward cells twice as long as the forward cells, each picture keeping its width (the 8-stage
  one goes from about 12.9 to 8.6 px per unit).

### Tensor parallelism like Megatron-LM (2023-09-02-tensor-parallelism.md)

- `distributed-training/imgs/tp_megatron_lm.svg`, moved here from the kernel fusion post, labels the reduction
  "sum-reduce", as its caption and the text of the post do ("[*sum-reduce*]" in the quote of option 1, and "the
  sum-reduce is added after the matmul"). The result is replicated on both GPUs, which is the all-reduce of $$g$$ in
  Megatron-LM, as the caption of `megatron-lm.svg` and the rest of the text say. Fix, in one change of the picture,
  its caption and the text: "all-reduce" in all three, or "sum-reduce" kept in all three; the user decides.
- `distributed-training/imgs/argonne_parallel_2.svg`: the panel letters are swapped relative to the paper, which
  names the spatial parallelism (b) and the sample parallelism (a). Fix: keep them, or swap them back, together with
  the caption and the paragraph after the picture ("the sample (data) parallelism shown in (b)").

### Mixture-of-Experts and Expert Parallelism (2024-01-19-mixture-of-experts.md)

- Optional, `distributed-training/imgs/moe_switch_transformers.svg`: the label "Switching FFN Layer", while the
  method and the caption say Switch FFN layer (the figure of the paper has the same mismatch). Fix: "Switch FFN
  Layer".
- Optional, caption of `distributed-training/imgs/moe_switch_transformers_2.svg`: it writes the expert capacity as
  (total_tokens / num_experts) × capacity_factor, while the Terminology box of the picture writes tokens_per_batch.
  Fix: tokens_per_batch in the caption.
- Caption of the Table 9 of the 2017 paper (the Markdown table under "Results" in the Sparsely-Gated MoE section): it
  is in the first person ("we sort the inputs in a training batch ... and show the words surrounding ..."). Fix, in
  the third person as pass 2 wrote it (73b4f6a): "For each expert $$i$$, the inputs of a training batch are sorted in
  decreasing order of $$G(x)_i$$, and the table shows the words around the corresponding positions in the input
  sentences."
- `distributed-training/imgs/moe_2014_ilya.svg`: the left panel writes g(x), f_i(x) and z without the superscript 1
  of the paper (g¹(x), f¹_i(x), z¹), to match the text ("$$f_i(x)$$ in the picture", $$g(x)$$, $$z$$). Fix: keep it,
  or restore the superscripts and change the text.
- `distributed-training/imgs/mixture_of_depths.svg`: the faded column uses opacity, which darkens the arrowheads where
  they overlap. Fix: grey #666666 lines and light fills instead of the opacity.
- Optional, `distributed-training/imgs/moe_gshard.svg`: the two device frames of panel (c) leave 25 px on each side
  of their boxes, as in the paper, while the frames of panels (a) and (b) leave 27 px on the left and 8 px on the
  right. Fix: the same margins in the four frames, with boxes about 97 px wide so that the picture still fits
  820 px.

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
- `distributed-training/imgs/variable_batch_lr_pipeline.svg` has no caption, unlike the ten other pictures of the
  post. Fix: add one, e.g. "Micro-batches of at most 30 tokens, on 2 processes with 2 gradient accumulation steps, for
  DDP (left) and pipeline parallelism (right). Pipeline parallelism needs the same shape in all micro-batches of a
  batch (here 3x11xE), so it packs fewer samples per micro-batch." (The left micro-batches hold 30, 28, 30 and 26
  tokens, the right ones 24, 25, 22 and 17.)
- Optional, `distributed-training/imgs/variable_batch_lr.svg` keeps 10 and 12 px between its labels and its
  tables, while `variable_batch_lr_pipeline.svg` now has 6 and 8 px. Fix: the same gaps in both.

### Sequence and context parallelism with Ulysses and Ring attention (2024-08-30-sequence-context-parallelism.md)

- `distributed-training/imgs/ulysses_sequence_parallelism.svg`: the label "local attention shape: B x H/P x S x E"
  and the axis E of the middle cube give the head size as E, the embedding size. An all-to-all keeps the number of
  elements per process, B·(S/P)·E before it and B·(H/P)·S·d after it, with Hd = E, so the label is H times too large.
  Fix: "local attention shape: B x H/P x S x d", and the axis d on the middle cube.
- Optional, `distributed-training/imgs/context_parallelism.svg` writes the head size as E. Fix: write the head size as
  d (see the content item on E).
- `distributed-training/imgs/ulysses_sequence_parallelism.svg`, `ring_attention.svg` and `ring_attention_qkv.svg`
  color the sequence chunks blue, yellow, red and green (and the ring in this order), while
  `sequence_and_data_parallelism.svg` and its caption put the blue and green processes in one sequence-parallel group
  and the red and yellow processes in the other, i.e. processes 1 (blue), 2 (green), 3 (red) and 4 (yellow). So the
  second process is green in one picture and yellow in the others. Fix: swap yellow and green in the three pictures
  (chunks and ring blue, green, red, yellow); the red process of `ring_attention_qkv.svg` then sees red, green, blue
  and yellow blocks, and the text ("the third (red) process") stays as it is.
- `distributed-training/imgs/ring_attention_qkv.svg`: the original picture has a "+" between the local Q and the
  steps, which the restyle dropped. Fix: restore the "+" signs, if the user wants them.

### Diffusion models, from DDPMs to DiTs (2025-02-25-diffusion.md)

- Optional, `diffusion/dit_moe.svg`: the dashed residual connections start after the first AdaLN, while the code of
  the post adds the input of the block, and three experts are gated, while the code routes each token to 2 experts.
  The picture follows the figure of the paper, which draws both this way. Fix: check both against the code, and say in
  the caption where the code differs.
- Optional, caption of `diffusion/diffusion.svg`: "A diffusion model is a $$T$$-step Markov chain", while the text
  counts two Markov chains. Fix: "A diffusion model has two Markov chains of $$T$$ steps: a forward process $$q$$
  that adds noise, and a trainable reverse process $$p_\theta$$ that removes it."
- Optional, `diffusion/diffusion.svg`: the dashed arrow q(x_t | x_t−1) is curved, as in Fig. 2 of DDPM, but it is
  neither a pointer nor a loop. Fix: an orthogonal dashed arrow under the nodes, with its label under it, or keep the
  curve of the paper.
- Optional, `diffusion/latte.svg`: the additions of (c) and (d) are 9.2 px wide, below the 20 px of an operator
  (CLAUDE.md, Pictures, section 7), as there is no room between the boxes. Fix: about 11 px more space around each
  addition (taller (c) and (d) frames), then draw them 20 px wide.
- Optional, `diffusion/playground_v3.svg`: the legend is one column between the panels, as in the paper's figure; a
  row is about 470 px wide and does not fit. Fix: a row under the T2i model forward panel (about 25 px taller), or
  keep the column.
- Optional, `diffusion/tora.svg`: the legend (Add, Noise patch, Motion patch) is one column, as in the paper; a row is
  about 270 px wide, and about 205 px are free beside the Trajectory Extractor box. Fix: a row under the output frames
  (about 30 px taller), or keep the column.
- Optional, `diffusion/longvilla.svg`: the GPU numbers of grid (3) are 14 px bold in cells 17 px wide and 18 px
  tall, as in the paper. Fix: keep them (they name the GPUs in their colors), or 12 px regular.
- `diffusion/latte2.svg`, panel (b): the labels x<sub>j</sub> and x<sub>j+1</sub> are 12.4 px tall, as tall as the 12.4
  and 13.8 px top faces of the cuboids that hold them, so the subscript j of x<sub>j</sub> crosses the front edge of its
  face, as in the paper's figure, and x<sub>j+1</sub> touches the back edge of its face. The cuboids match the cells of
  the frames, so their faces cannot grow without leaving the grid. Fix: keep them as in the paper, or move x<sub>1</sub>
  and x<sub>j</sub> to the front faces of the front cuboids (22 px tall) and the other two labels above their cuboids.
- Optional, `diffusion/sora_vae.svg`: the cells of the latent volume and of the latent sequence are grey #b3b3b3
  cubes with white 0.4 px lines, not a fill of the palette with #999999 lines (CLAUDE.md, Pictures, sections 5 and
  6). Fix: #f5f5f5 or the closest pastel fill, #999999 cell lines and a black 1.2 px silhouette.

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
- `inference/imgs/roofline.svg`: the ❌ sits at an arithmetic intensity of 512, right of the ridge point, where the
  roof is flat. There, moving right does not raise the attainable performance. Yet the picture draws a "move right"
  arrow from the ❌, while the text says "Computation to the right is compute-bound: only faster math, or less of it,
  helps". The author's original picture already had the ❌ at this place. Fix: move the ❌, with its arrows and their
  labels, left of the ridge point and below the slanted roof, for example to an intensity of 1 and a performance of
  1/4. There, moving right and moving up both raise the attainable performance, as the two lists of the text describe.
  Another fix is to keep the ❌ and drop the right arrow and its label.
- Optional, `inference/imgs/roofline.svg` keeps the axis titles of the author's original, "Operational Intensity
  [FLOPs / byte]" and "Performance [GFLOPs / sec 10⁹]", as the user decided on 2026-10-22, and the text names the
  operational intensity too, while the note of the picture says "increase Arithmetic Intensity" (the original said
  Operational). Fix: "Operational Intensity" again in the note, so that the picture uses one term.
- `inference/imgs/disaggregation.svg`: the "requests" box is data but has the 5 px corners of an operation, and the
  five GPU boxes, which are devices, have 5 px corners instead of the 10 px corners of a frame. Fix: square corners
  for "requests", and 10 px corners for the GPU boxes.
- Optional, `inference/imgs/prefix_caching.svg`: the labels "h₁ = hash(∅, block 1)" to "h₃ = hash(h₂, block 3)" are
  set entirely in the math font, words included. Fix: "hash" and "block 1" to "block 3" in Source Sans 3, and h₁ to
  h₃ and ∅ in the math font (section 4 of CLAUDE.md).
- `inference/imgs/chunked_prefill.svg`: the prompt boxes are 31.49 px tall and the decode boxes 29.52 px, left from
  a scaled drawing. Fix: every box of both rows 30 px tall (CLAUDE.md, Pictures, section 7).
- `inference/imgs/disaggregation.svg`: the box sizes come from a scaled drawing (requests 104.94 x 40.23 px, GPU
  boxes 148.66 x 34.98 px, blocks of the time line 92.69 x 26.23 px with 3.5 px gaps). Fix: one-line boxes 30 px
  tall (CLAUDE.md, Pictures, section 7).

### Faster attention (GQA, FlashAttention, sparse, linear and latent attention) (2025-12-30-inference-attention.md)

- Optional, `inference/imgs/mla.svg`: the absorb = True panel shows a whole sequence, while weight absorption serves
  decode, with one new token (H x 1 x (P+1) scores over P+1 latent vectors). The panels are labeled "usually prefill"
  and "usually decode". Fix: redraw the absorb = True panel for one decode token.
- The captions of `inference/imgs/csa.svg` and `inference/imgs/hca.svg` call the entry of the query's own block
  (entry 3 in CSA, entry 2 in HCA) "dark grey", but both pictures draw it white with a grey outline, as CLAUDE.md
  draws unused elements. Fix: "(white)" instead of "(dark grey)" in both captions.

### Kernel Fusion for Tensor Parallelism and MoE Expert Sharding (2026-08-18-kernel-fusion-tensor-parallelism.md)

- `inference/imgs/tp_expert_sharding.svg` labels the reduction "sum-reduce", and its caption says "a sum-reduce adds
  their partial outputs", while the text calls the sum of the partial outputs of all GPUs an all-reduce, as in
  Megatron-LM, and says that each sharded expert is an FFN with Megatron-LM tensor parallelism. Fix: "all-reduce" in
  the picture and in the caption.
- Caption of `inference/imgs/tp_expert_sharding.svg`: the credit "Inspired by Figure 2 of [...]." does not follow the
  credit form of CLAUDE.md. Fix: "Inspired by: [Accelerating MoE Model Inference with Expert
  Sharding](https://arxiv.org/abs/2503.08467)".
- The captions of `tp_timeline.svg`, `tp_matmuls.svg` and `tp_block_specialization.svg`, redrawn from the figures of
  the post's paper, and of `tp_results.svg`, drawn from its data, have no credit. The
  paper is the author's own and not public yet. Fix: "Adapted from: [Towards Efficient LLM Inference via Full-Stack
  Fusion of Computation, Communication, and In-Network Computing](link)" once the paper has a public link; the user
  decides.
- Optional, `inference/imgs/tp_expert_sharding.svg` and its caption write the weights of an expert as W_i and W_o,
  and the tokens and outputs as x, y, y_A and y_B, while the rest of the post writes the same matrices as W_1, W_2, X
  and Y. The text names the mapping of the weights. Fix: relabel the figure and the caption to W_1, W_2, X, Y, Y_A
  and Y_B, and drop "(the W_1 and W_2 above)" from the text.
- Optional, `inference/imgs/tp_expert_sharding.svg` and its caption name the GPUs A and B and the partial outputs y_A
  and y_B, while the text numbers the GPUs g = 1 to G and writes their slices with a superscript, $$H^{(g)}$$, and
  `distributed-training/imgs/tp_megatron_lm.svg`, the picture of the tensor parallelism post that the text points to,
  names them GPU 1 and GPU 2. Fix: "GPU 1" and "GPU 2" in the picture and the caption, with the partial outputs
  named after them (y⁽¹⁾ and y⁽²⁾, or Y⁽¹⁾ and Y⁽²⁾ with the item above), or drop the item.
- Optional, `inference/imgs/tp_timeline.svg`: the fused row applies the bias and the activation once, after all the
  reduce steps, while the text and `tp_block_specialization.svg` apply them to each band, right after the all-reduce
  of that band. The figure of the paper has the same simplification. Fix: a small bias and activation step after
  each reduce step, or keep the picture as a schematic.
- Optional, `inference/imgs/tp_expert_sharding.svg` comes after the results, while the text now explains expert
  sharding before them. Fix: move the picture and its caption up to the expert sharding section.

### Kernel Fusion for MoE Expert Parallelism (2026-10-09-kernel-fusion-expert-parallelism.md)

- `inference/imgs/ep_stages.svg` draws the eight stages on the time axis of the decode step, whose costs are
  placeholders, not measurements, as the caption says. Fix: remove the time axis until the decode step is measured,
  or redraw it with the measured costs.
- `inference/imgs/ep_results.svg`: the right panel plots decode times that are placeholders, not measurements, as the
  caption says. Fix: remove the right panel and its sentence in the caption until the decode step is measured.
- The captions have no credit to the post's paper. `ep_stages.svg` and `ep_pipeline.svg` are the paper's own figures,
  and the other pictures redraw its figures. The paper is the author's own and not public yet. Fix: "Source: [MoeInc:
  An Expert Parallelism MegaKernel Overlapping Matrix and Vector Units with In-Network Dispatch-and-Combine](link)"
  under `ep_stages.svg` and `ep_pipeline.svg`, and "Adapted from:" with the same link under the others, once the paper
  has a public link; the user decides.
- The paper redrew `agrs_inc.svg` and `all_to_all_inc.svg` (2026-10-26): the bracket under each INC diagram now
  names its operation a collective. `ep_all_gather_reduce_scatter_inc.svg` and `ep_all_to_all_inc.svg` do not have
  it yet. Fix: redraw them from the paper's draw.io in the next picture task.

### Knowledge distillation, speculative decoding and multi-token prediction (_drafts/2026-03-17-inference-distillation-speculative-decoding.md)

- Optional, `inference/imgs/mtp_parallel.svg` writes the tokens as x_t, as Gloeckle et al. do, while eagle, layerskip
  and mtp_sequential write them as t_i. The text follows each figure. Fix: keep the paper notations, or write t_i in
  mtp_parallel and in the formulas of the parallel heads.
- Optional, `inference/imgs/mtp_parallel.svg`: the vector z has no size label, while the author's original labeled it
  E, and `medusa.svg` and `eagle.svg` label the same size E, 12 px in the math font; the code gives z the shape
  [B, S, E]. Fix: an E label beside z, as in medusa and eagle.
- `inference/imgs/layerskip.svg` names the predictions of the full model p_1 to p_3 ("predictions that verify the
  drafts"), while $$p$$ is the distribution of the draft model and of the student in the text, in spec_sampling and in
  the `accept_sampling` code. Fix: label these cells v_1 to v_3, or leave them without a symbol; not y_1 to y_3, as
  $$y_i$$ is the softened output of the distillation section.
- `inference/imgs/layerskip.svg` and the text write the exit layer of the draft as E ("drafts of the first E = 4
  layers", "the first $$E$$ layers"), while E is the embedding size in the code of the post ([B, S, E]) and in
  `medusa.svg` and `eagle.svg`. Fix: $$\ell$$ for the exit layer, in the picture and in the text.
- Optional, `inference/imgs/mtp_sequential.svg`: the dashed frames of the output tokens are blue #6c8ebf, while those
  of the input tokens are black, and neither the caption nor the text gives the blue a meaning. Fix: black frames, or
  a caption that names the blue frames as the targets of each loss.
- `inference/imgs/medusa.svg`, `eagle.svg` and `dflash.svg` keep tokens of 36 x 30, 30 x 30 and 32 x 30 px, while
  the other pictures of the draft use 46 x 30 px. With 46 x 30 px tokens, eagle would be about 880 px wide, medusa
  about 830 px, and the token rows of dflash would be wider than its draft layers. Fix: a new layout for each
  (eagle: t6 above the output layer, or the drafter on a second row; medusa: the tree attention mask under the
  candidate tree; dflash: draft layers as wide as the row of 8 tokens, with the row labels left of the rows), or
  two token sizes in the draft.
- `inference/imgs/mtp_parallel.svg`: its one-line boxes are 44 to 46 px tall and its output tokens 60 x 30 px,
  unlike the 30 px boxes and 46 x 30 px tokens of the other pictures of the draft. Fix: boxes 30 px tall and output
  tokens 46 x 30 px.

### Pruning and quantization (_drafts/2026-04-22-inference-pruning-quantization.md)

- Optional, `inference/imgs/quantization_granularity.svg`: the per-group panel colors its 12 groups with 6 colors
  (rows 1 and 3, and rows 2 and 4, repeat their colors). In the per-row panel each color is one scale, so the
  per-group panel reads as 6 scales. Fix: a black frame around each group, or 12 distinct fills.

### GPT from scratch in C++ and CUDA vs. fused PyTorch (_drafts/2026-08-17-gpt-cpp-cuda.md)

- Optional: the matrix multiplication section has no picture. Fix: a picture of the tiles of `tiledMultiply` (the row
  of tiles of A and the column of tiles of B that one thread block reads for its tile of C), and one of the ring
  algorithm (the blocks of B passed between the ranks at each step).

### Publication Summaries (publications.md)

- Optional, `publications/llama_3_multi_modal.svg`: the connectors "Every 4th layer" and "Every ~3rd layer" have arrow
  heads at both ends, as in Fig. 28 of the paper, while the forward pass sends data one way. Fix: one arrow head,
  toward the cross-attention layers.
- `publications/llama_3_multi_modal.svg`: the legend is one column of six entries, right of the picture, as in Fig. 28
  of the paper, while section 7 of CLAUDE.md asks for a row. Fix: two rows of three entries centred under the picture
  (about 50 px taller), or keep the column of the source and note the exception in CLAUDE.md; the user decides.
- Optional, `publications/llama_3_multi_modal.svg`: the three "Image patches" boxes and the two "Image patch
  representation" boxes are data, as the square "Image representation" box, but have 5 px corners. Fix: square
  corners.
- `publications/comet_2.svg`: GEMM #0 and #1 mix the rows of two experts after a global sort by source rank. The
  picture follows Fig. 5 of the paper, whose text does not say whether the sort is global or per expert. Fix: check
  the code of Comet, and sort per expert if the code does.
- Optional, `publications/performers2.gif` is an animation from the Google Research blog, shown as it is. Fix: a static
  SVG of the prefix sums, redrawn in the style of the other pictures, with "Adapted from:" in its caption.
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
- Optional, `publications/zeroplusplus.svg`: the labels follow Fig. 2(a) of ZeRO++, but call "scale" a range of
  values (9.22 = max |x|, with q = round(127 x / 9.22); and for a block, max − min = 9.07, with
  q = round(255 (x − zp) / 9.07) − 128). The text of the paper calls qwZ symmetric, with one scaling coefficient per
  block, while its figure uses a zero point. Fix: rename the label to "range", or add "(the scale is the range of
  the block)" to the caption.
- Optional, caption of `publications/comet.svg` (Comet): "the number of SMs on Hopper(132)" lacks a space. Fix:
  "Hopper (132)".
- Optional, `publications/mutransfer.svg`: the grid lines run through the label "optimum shifts", which section 6 of
  CLAUDE.md allows for a label wider than the spacing of the grid lines. Fix: a white box behind the label.
- `publications/rqvae.svg`: the cell labels of q(z\|x) (53) and of the front face of z<sub>q</sub>(x) (e<sub>1</sub>,
  e<sub>2</sub>, e<sub>3</sub>, e<sub>53</sub>) touch or cross the 0.4 px cell lines, as they are wider than the 11 to
  12.7 px cells. Fix: redraw both grids with fewer, larger cells, each label in the same relative cell as in the paper:
  4 x 4 cells of about 15 px for q(z\|x), and 3 columns of 19 px by 4 rows of 15.4 px for the front face of
  z<sub>q</sub>(x), with the lines of its top and side faces moved to the new cell lines.
- `publications/training_compute_optimal_large_language_models_2.svg` lost the tick labels 2.5, 3.5, 4.5, 5.5, 250M, 1B
  and 5B. Fix: restore them.
- `publications/training_compute_optimal_large_language_models_2.svg`: the projection lines are green, as in the paper
  and as the caption says ("Green: the projected optimal model size ..."), while section 6 of CLAUDE.md draws the
  projections on an axis as grey dashed guide lines. Fix: keep the green lines, or grey dashed guides with a reworded
  caption; the user decides.
- `publications/scaling_laws.svg`: the three fitted power laws are legend boxes inside the plots, as in Figure 1 of
  the paper, while section 6 of CLAUDE.md puts the legend of a chart in one column outside the plot, on its right;
  three legends outside the panels do not fit 820 px. Fix: write each formula as a label next to its fit line, or
  add the legends inside the plots to section 6 as an exception.

## Content issues

Content problems that need the author's decision, such as overlap between posts.

### Distributed Sorting Algorithms (2014-06-21-distributed-sort.md)

- The text lists five steps of the sample sort, and `distributed-sort/sample_sort.svg` four, grouped as in the
  author's original picture (1. local sort; 2. samples sent to rank 0; 3. splitters picked and broadcast; 4. elements
  exchanged and sorted). The step numbers of the caption fit both. Fix: leave as is, or renumber the text list into
  the four steps of the picture.

### Distributed Load Balancing and Spatial Slicing (2015-12-01-slicing.md)

- The table of the slicing layouts lists Sort Tile Recursive before Orthogonal Recursive Bisection, while the
  pictures above it and the text number ORB as layout 3 and STR as layout 4; the worked example of the costs follows
  the table. Fix: swap the last two rows of the table, with their thumbnails, in a picture task; then name ORB before
  STR in the worked example.

### The Leaky Integrate-and-Fire Neuron and the Brunel Network (2016-02-08-lif-brunel.md)

- The spike counts of the three raster plots point to 12,500 neurons, the default of the code (see the code issues),
  while the text and the paper use 10,000. SR has 12,383 spikes per 3 ms cycle, i.e. one spike per neuron and cycle
  with 12,500 neurons; AI gives 46.3 Hz with 10,000 neurons, or about 37 Hz with 12,500; SI slow gives 13.1 or
  10.5 Hz, against the 5.5 Hz of the paper (its mean-field rate at g = 4 is 15.5 Hz, which supports the code item on
  the truncation of g). The text quotes only the numbers that do not depend on the number of neurons: the burst
  intervals, the first spikes and the 3 ms period. Proposed fix: run the three plots again with `-n 10000` and g as a
  double (the two code items), then add their firing rates to the text.
- The link to the PDF of the paper (web.stanford.edu/group/brainsinsilicon/documents/BrunelSparselyConnectedNets.pdf)
  could not be fetched from here, while the site of the lab loads. Proposed fix: open it in a browser; if it is
  gone, remove the "pdf" link and keep the link to the Springer page.

### Variable Timestep Simulation of the Electrical Activity of Neurons (2017-02-17-neuron-variable-timestep.md)

- Optional: the reference `[lytton2005independent]` points to mitpressjournals.org, an old publisher domain. Crossref
  and NEURON's list of publications give https://doi.org/10.1162/0899766053429453 for Lytton and Hines (2005), Neural
  Computation 17(4), 903–921, but the link could not be checked from here. Proposed fix: link the DOI after opening
  it in a browser.
- The pulse test has two break-even rates. The preprint (section 3.3) puts the runtime break-even of the weakest
  pulses at "circa 1600 events per second (10^3.2 Hz)", while its Fig. 7 and
  `cvode_dependency_on_events_arrival.svg` show CVODE as fast as Backward Euler at about 3,100 pulses per second, and
  990 for the strongest pulses. The post says 1,000 and 3,000, from the picture. Proposed fix: check the simulation
  data; if the preprint's 1,600 is right, write it in the text and correct the picture.
- The page is now over 6,000 words, but has no list of contents, which CLAUDE.md gives to the pages over 6,000 words.
  Proposed fix: add it after the opening paragraphs.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- Kohonen maps: `unsupervised-learning/kohonen.svg` and `kohonen2.svg` write the input as v, the winner as s and its
  weights as w_s, as their source (Ritter, Martinetz and Schulten, 1992) does, while the algorithm in the text writes
  them as $$x_u$$, $$j^*$$ and $$w_{j^*}$$. The caption bridges the two ("In the algorithm below, $$v$$ is the input
  $$x_u$$ and $$s$$ is the winner $$j^*$$"). Fix, in a text task: keep the bridge, or write the algorithm with the
  source's v, s and w_s.
- Optional, symbols with two meanings: $$E$$ is the expectation in ICA and the error of k-means, $$K$$ the kernel
  matrix and the number of components of a GMM, $$d$$ the degree of a polynomial kernel and a distance, and $$k$$ has
  several uses. Proposed fix: write the expectation as $$\mathbb{E}$$ in ICA, as the statistics posts do, and keep the
  others, which each section defines.

### Deep Neural Networks, Backpropagation, Dropout, CNNs and Embeddings (2018-03-27-deep-neural-networks.md)

- `deep-neural-networks/cnn.svg` gives no filter size, so the text assumes 3×3 filters: the first layer has 896
  parameters, 7,000 times fewer than a dense layer, and the fully connected layers hold 96.4% of the 545,218
  parameters. With 5×5 filters, these numbers become 2,432 parameters, 2,600 times fewer, and 90.7% of 579,522.
  Proposed fix: confirm the filter size of the network of the picture, and keep or change the numbers; "more than all
  the convolutional layers together" holds either way.

### Exponential Family of Distributions (2019-03-20-exponential-family.md)

- Seven formulas hold hidden `%% source: <url>` comments, which do not render but show in the source of the page.
  Three point to chapter 8 of Jordan's draft on the Berkeley server, whose course page asks readers not to link the
  chapters from public sites, and four to Columbia lecture notes (cs.columbia.edu/~jebara/4771/tutorials/lecture12.pdf).
  Proposed fix: delete the three Berkeley comments, as the copy of the chapter was deleted for the same reason; keep
  or delete the Columbia ones.
- The table of distributions has eight columns and is wider than the text column: in a 900 px window, its last
  columns are cut at the right edge. Proposed fix: wrap it in `<div style="overflow-x: auto;">`, an HTML change for
  the author to approve.

### Data parallelism, sharding and CPU offloading (2023-08-18-data-parallelism.md)

- The precision of the benchmark: the post says that all implementations use the same numerical precision, and both
  configuration files of the post set `"fp16": {"enabled": false}`, i.e. fp32, while the original text said mixed
  precision. The history of the configuration does not settle it: `fp16` was enabled in the commit of the first
  results (4ade5d7, 2023-10-08 02:31), disabled at 12:02 (75ac452), and the benchmark pictures were updated again at
  17:20 (0beddf7). The author thinks that the runs used fp16, but is not sure (2026-10-19). Proposed fix: confirm the
  precision of the runs (from their logs, or by running one configuration again); if they used fp16, write "mixed
  precision (fp16)" in the text, set `fp16` in the configuration files, and add "in fp16" to the caption of the
  benchmark (diagram item); otherwise add "in fp32" to the caption.

### Pipeline parallelism (1F1B, Zero Bubble, DualPipe) (2023-08-30-pipeline-parallelism.md)

- The post keeps the workaround `outputs = outputs.requires_grad_(True)` for the error of the swapaxes lambda. If
  `outputs` has no `grad_fn`, this call lets the backward pass run, but no gradient reaches the layers before it.
  Proposed fix: check that the parameters of the last stage get gradients with the workaround (`p.grad` after
  `train_batch()`), and remove the sentence if they do not.
- Optional, results, memory per GPU: the first GPU used more memory than the others, also with the deep benchmark
  model. The text now says that 1F1B explains only a few MB of this gap for that model, and the cause of the rest is
  not known. Fix: measure the peak memory of every GPU (`torch.cuda.max_memory_allocated()` on each rank, and
  `nvidia-smi` for the CUDA contexts on GPU 0), and give the numbers and the cause in the post.
- The post keeps the symbols of the papers: $$p$$ stages, $$m$$ micro-batches, $$v$$ chunks per GPU, and the times
  $$F$$, $$B$$ (a backward pass) and $$W$$, while the other posts of the series write $$P$$ for the number of GPUs and
  $$B$$ for the batch size. Proposed fix: keep them, and say once in the text that the post follows the notation of
  the papers; or write the times as $$T_F$$, $$T_B$$ and $$T_W$$.

### Variable sequence lengths, curriculum learning, adaptive batch size and LR (2024-05-10-training-variable-length-curriculum.md)

- Bruno, in the picture review of 2026-10-22, on torch_compile_dataset: the section "Static compilation of
  variable-shaped inputs" "makes no more sense anymore". Fix: remove that subsection of "Kernels compilation",
  with its picture (torch_compile_dataset.svg and its draw.io group), and check that the rest of "Kernels
  compilation" still reads well without it.
- `distributed-training/imgs/dnn_multiproc.svg`, `dnn_multiproc_compiled.svg` and `dnn_multiproc_optimized.svg` label
  their rows Node 1 and Node 2, the words of their source (TorchDynamo Update 9), while the text ("a 2-GPU DDP run")
  and the three captions ("on 2 GPUs") say GPUs. Fix: "a 2-node DDP run" and "on 2 nodes" in the text and the
  captions, or GPU labels in the pictures.

### Sequence and context parallelism with Ulysses and Ring attention (2024-08-30-sequence-context-parallelism.md)

- The post uses `E` for both the embedding size (introduction, `(B, S, E)`) and the head size (the attention formula,
  the Ulysses view swap `(H, B, S/P, E)` to `(H/P, B, S, E)`, the ring comments, and the pictures
  `context_parallelism.svg` and `ulysses_sequence_parallelism.svg`), and `H` for both the number of heads and the
  frame height of the video shape `(B, C, T, H, W)`. The text keeps `E` for the head size, so that it agrees with the
  pictures and the code. Fix: write the head size as `d` (the code's `d_head`) in the text, in the same change as the
  pass 2 code items (ring comments, `dist_view_swap`) and diagram items (`ulysses_sequence_parallelism.svg`,
  `context_parallelism.svg`) on this symbol; optionally write the video shape as `(B, C, T, X, Y)` for `XxY` pixels
  per frame.
- When the code item on the causal mask is fixed, the sentence "The implementations above compute attention without a
  mask" must change in the same commit. Fix: say that both implementations apply the causal mask of the GPTlite
  blocks.

### Diffusion models, from DDPMs to DiTs (2025-02-25-diffusion.md)

- Optional: `diffusion/masked_autoencoders_cropped.svg` is shown twice, with two different captions, in the section
  "Video diffusion and 3D attention" and in the further reading entry on masked autoencoders. Fix: show it once, and
  refer to it from the other place, or keep both.

### Kernel Fusion for Tensor Parallelism and MoE Expert Sharding (2026-08-18-kernel-fusion-tensor-parallelism.md)

- The post quotes the 19% that Table 2 of the paper prints for GPT-4 and GPT-3 175B without INC, but the latencies of
  the table (52.078 and 63.860 ms) give 18.45%; for Llama 3 8B without block specialization, the table prints 23.12%,
  while its latencies give 29.64%. The "at least 17%" of the post holds either way. Proposed fix: check the
  percentages of the paper, and write the corrected values in the post.

### Kernel Fusion for MoE Expert Parallelism (2026-10-09-kernel-fusion-expert-parallelism.md)

- The MoE paper repository moved from d1b9da2, the version that the post follows, to f2a3131 (2026-10-30). The paper
  now reports measured prefill and decode results and ends with a summary, presents the partial collectives under
  another name, and has new and redrawn figures. So the decode placeholders of the post (the right panel of
  `ep_results.svg` and the time axis of `ep_stages.svg`), its One-to-Many row and its names of the partial
  collectives are out of date. The third iteration of the text rewrite left this post as it is on master, as its
  rewrite was written before the post followed the public version of the paper. Fix: after 1 November 2026, at the
  user's request (2026-10-30), a task of its own syncs the text, tables, captions and pictures of the post to the
  public files of the paper at its latest commit, in the style of CLAUDE.md, and records the commit there; the new
  figures of the partial collectives stay out, unless the user changes the rules on them.
- The token names are italic in 𝐴<sub>0</sub>, Σ<sub>𝐴</sub> and the packets of the communication pictures, and upright
  in the token sets AC<sub>0</sub>, BC<sub>1</sub> and AB<sub>2</sub> of the expert boxes, in the caption of
  `ep_all_gather_reduce_scatter.svg` ($$A_0$$, $$E_0(\text{AC}_0)$$) and in the four pictures. Fix: one form for the
  token names, italic everywhere (𝐴𝐶<sub>0</sub>, the tokens 𝐴 and 𝐶) or upright everywhere, once the user picks it.

### Knowledge distillation, speculative decoding and multi-token prediction (_drafts/2026-03-17-inference-distillation-speculative-decoding.md)

- The usage comment of the Medusa code ("prefill the prompt: [1, S], with the causal_mask of the post on attention")
  relies on `causal_mask` from the post on attention, which the text no longer links (#69). Fix: in the code pass,
  write the mask in the comment as `torch.ones(S, S, dtype=torch.bool).tril()` (True where a token may attend, as in
  `tree_attention_mask`) and drop "of the post on attention", or keep the mention without a link.
- One symbol for two quantities, as in the author's original: the logits of the temperature formula of the
  distillation section are $$z$$, as in Hinton et al., and so is the output $$z_t$$ of the trunk of the parallel
  heads, as in Gloeckle et al. (also `z` in `inference/imgs/mtp_parallel.svg` and in the code of `MultiTokenHeads`).
  Proposed fix: write the logits as $$u$$ in the temperature sentence and formula, which changes no code or picture;
  or keep the notations of the two papers.

### Pruning and quantization (_drafts/2026-04-22-inference-pruning-quantization.md)

- The draft describes the experiments of `inference/pruning.py` and `inference/quantization.py` on GPTlite (pruning
  with distillation, Wanda, ShortGPT, weight-only and weight-activation quantization, KV cache quantization), but
  shows none of their results. Fix: run the two scripts and add a table of their outputs (parameters, size,
  validation loss, time per token, matching tokens), or present the code without the comparisons.
- Optional: once both drafts are published, "knowledge distillation" in the introduction could link to the
  distillation post instead of Wikipedia.

### GPT from scratch in C++ and CUDA vs. fused PyTorch (_drafts/2026-08-17-gpt-cpp-cuda.md)

- The results that need GPUs are placeholders in bold brackets: the runtimes of `tiledMultiply` and cuBLAS, of the
  ring on several GPUs, the benchmark table and the CPU runtime, the hardware and the versions, and the discussion of
  the results. The CUDA, LibTorch and NCCL programs build, but have not run on a GPU: the CPU code prints the outputs
  of PyTorch in fp32, and a CPU emulation of the CUDA kernels prints those of PyTorch in bf16. The kernels of
  `torch.compile` were listed on the CPU, and the 5 passes over the attention weights come from the source of
  TorchInductor. Fix: run the commands of the Benchmark section and of `assets/gpt-cpp-cuda/README.md` on a GPU
  node, the compiled model with `TORCH_LOGS=output_code`, then fill in the placeholders and write the discussion.
- The draft overlaps two posts. Its sections "Change of philosophy", "Kernel fusion" and "torch.compile" come from
  the post GPT in C++ (2023-06-27-gptlite-cpp.md), whose benchmark it replaces, and its section "Kernel fusion on
  several GPUs" covers the first section of the kernel fusion post for tensor parallelism. Fix: when the draft is
  published, the kernel fusion post links to it and keeps only its own points in that section, and the old post
  links to the new one in its introduction, or is removed; the user decides.

### From Mixture-of-Experts to RAG and Agentic AI Systems (_drafts/2026-10-23-moe-rag-agentic-ai.md)

- Eq. 4 prefills at step $$s$$ all the $$c_s - c_{s-1}$$ tokens appended since step $$s - 1$$, on a cache of
  $$c_{s-1}$$ tokens. These include the $$n_{s-1}$$ tokens decoded at step $$s - 1$$, and the decode has already
  computed the KV cache of all of them but the last one. Proposed fix:
  $$t_{\text{orch}}(c_s - c_{s-1} - n_{s-1} + 1,\; c_{s-1} + n_{s-1} - 1)$$ for $$s > 1$$, or a sentence saying that
  the model prefills them again, for simplicity.
- One symbol for several quantities: $$k$$ is the number of experts per token of the MoE (top-k), the number of
  chunks of RAG (top-k) and the number of agent results used; $$d$$ is the decode index of the sums and the decoded
  tokens $$d_s$$ of the context of the orchestrator (the Execution table and the aai_dynamic picture). Proposed fix:
  keep top-k as the name of the selection, and name the agent results and the decoded tokens with other letters
  (e.g. $$r$$ and $$o_s$$), in the text, the formulas, the tables and the pictures.

### Publication Summaries (publications.md)

- ZeRO-Offload++ (entry 052): the entry title links the blog on the branch `offloadpp-news` of microsoft/DeepSpeed,
  and the caption links it on `master`; both copies exist today, and the repository has moved to deepspeedai/DeepSpeed,
  where the README of the blog opens. The decision is to fix the two links, but they are in the summary line of the
  entry and in a caption, which a text task does not change. Proposed fix: link
  https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-offloadpp in both places, after a check that
  the page opens in a browser.
- LatentMoE: the accuracy configuration $$(N', K') = (512, 22)$$ is the one that the Nemotron 3 white paper reports,
  while the rule of the LatentMoE paper ($$K' = K\,d/\ell$$, with $$K = 6$$ and $$d/\ell = 4$$) gives 24, and the
  LatentMoE paper never mentions 22. Proposed fix: keep 22 and add "(the rule $$K' = K\,d/\ell$$ of the LatentMoE
  paper would give 24)", or leave the sentence as it is.
- Multi-Head LatentMoE: the sentence "HP also composes with EP to scale beyond $$N_h$$ GPUs" follows section 3.2 of
  the paper, while its conclusion says that Head Parallel "is bounded by the number of heads", and section 3.2
  requires $$P \le N_h$$ for HP alone. Proposed fix: leave the sentence, or add "(HP alone needs at most $$N_h$$
  GPUs)".

## Title issues

A new title keeps the URL, which comes from the file name (`permalink: /:title` in `_config.yml`). A title, with the
category that the home page puts before it, fits on one line of the home page on a desktop screen (under 690 px).

### All posts

- The capitalization is mixed: the posts up to 2019 use title case, most posts since 2023 use sentence case, and
  a few titles mix both. Fix: one style for all titles.
- With the 880 px page, eight post titles are longer than 690 px on the home page (they still fit its 735 px
  column in the site's font, but may wrap with a wider system font): faster attention (693 px), variable sequence
  lengths (725 px), learning from sequences (694 px), VAEs and GANs (729 px), Bayesian linear regression (718 px),
  deep neural networks (708 px), variable timestep (720 px) and the LIF neuron and Brunel network (694 px). Fix:
  shorter titles, under 690 px.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- To do later. The title names only PCA, but the post also covers ICA, k-means, Gaussian mixtures, hierarchical
  clustering, DBSCAN, spectral clustering, self-organizing maps, t-SNE, UMAP and self-supervised learning.
  Proposed: "Dimensionality Reduction, Clustering and Self-Supervised Learning" (652 px).

### Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) (2019-10-12-vae-gan.md)

- The title is 729 px wide on the home page, the widest of the posts. Fix: "Variational Autoencoders and Generative
  Adversarial Networks" (615 px with the category), as the opening spells out both acronyms.

### Sequence and context parallelism with Ulysses and Ring attention (2024-08-30-sequence-context-parallelism.md)

- The title writes "Ring attention", while the text writes the method as Ring Attention. Fix: "Sequence and context
  parallelism with Ulysses and Ring Attention" (656 px with the category).

### Pruning and quantization (_drafts/2026-04-22-inference-pruning-quantization.md)

- Optional: the other posts of the series name their methods in the title ("Faster attention (GQA, FlashAttention,
  sparse, linear and latent attention)", "Prefill vs decode, roofline, KV cache, batching and disaggregation"). Fix:
  "Pruning and quantization (Wanda, Minitron, GPTQ, AWQ, FP8, FP4)", 84 characters with "Efficient Inference: ".
