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
- Building a GPT model (gptlite), sections "Final Model" and "Size matters": `generate` samples with the model in
  training mode and with autograd on: dropout stays active while the trained model generates text (the post never
  calls `m.eval()`), and every generation step builds a graph for a backward pass that never comes. Proposed fix:
  decorate `generate` with `@torch.no_grad()`, and call `m.eval()` before `print(decode(m.generate(idx,
  max_new_tokens=500).tolist()[0]))`, as `assets/gptlite/main.py` does (`model.eval()` and `torch.inference_mode()`
  before its generation loop).
- Building a GPT model (gptlite), section "Multi-Head Masked Attention": the comment of `out = wei @ x` gives the
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

### Tensor parallelism (2023-09-02-tensor-parallelism.md)

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

### Faster attention (GQA, FlashAttention, sparse, linear and latent attention) (2025-12-30-inference-attention.md)

- `indexer_loss` in the DSA code block masks the scores with -1e9, which does not fit in float16 (its largest value
  is 65,504), while the comment of `scaled_dot_product_attention` in the same post gives this as the reason to use
  `torch.finfo(dtype).min`. The queries of the first block of CSA have rows that are fully masked. Fix: replace both
  -1e9 of `indexer_loss` with `torch.finfo(<tensor>.dtype).min`.
- The DSA code block trains the indexer only during the dense warm-up: in the sparse branch (`dense = False`) it sets
  `self.indexer_loss = None`, and since `x_index = x.detach()` and the top-k selection has no gradient, the indexer
  stops learning. DeepSeek-V3.2 (arXiv 2512.02556, Sec. 2.1.1, "Sparse Training Stage", Eq. 4) keeps training the
  indexer in that stage, with the KL-divergence loss restricted to the selected tokens, and the post's text says so
  ("the indexer learns only from its own loss"). Fix: in the sparse branch, compute the logits `scores = q @
  k.transpose(-1, -2) / h ** 0.5` and set `self.indexer_loss = indexer_loss(index_scores, scores, mask)` with the mask
  of the selected tokens (the function already takes a `[B, Sq, Sk]` mask), and update the comments of `self.dense`
  and `self.indexer_loss`; or keep the code and say before it that it trains the indexer only during the warm-up.

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
  `publications/attention_mech.svg` has 8 arrow heads smaller than size 6. Fix: set them to the values of CLAUDE.md,
  then run the check again.
- Line weights and arrow layout: only the 44 pictures of the third iteration follow sections 6 and 7 of CLAUDE.md
  (1.6 px outlines, arrows and lines, 0.8 px cells, round operations, square data, heads of size 5); the other
  draw.io figures keep the thinner lines of the originals. The arrow check also finds lines through labels and
  arrowheads that touch in some of them, for example the All-to-All lines of
  `distributed-training/imgs/moe_gshard.svg` and the arrows of `gptlite/gptlite_embedding.svg`. Fix: apply the style
  of the sample to every draw.io figure, then run the arrow check again.

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
- `lif-brunel` table captions: "source: Marc-Oliver Gewaltig, ResearchGate" and "source: Brunel 2000" are lower case
  and have no link, and the parameter table is Table 2 of NEST by Example (Gewaltig et al., 2012), not a ResearchGate
  page. Fix: "Source: [NEST by Example, Table
  2](https://www.nest-simulator.org/wp-content/uploads/2015/02/NEST_by_Example.pdf)" and "Source: [Dynamics of
  Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons, Table
  1](https://link.springer.com/article/10.1023/A:1008925309027)".

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
- `neuron-electrical-activity/compartment_rc_circuit.svg`, its caption and the equation above it name the capacitor
  $$C$$, each resistor $$R$$ and the total current $$I$$. In the text, the capacitance of compartment $$j$$ is
  $$c_j$$, $$R$$ is the gas constant, and $$I$$ is the inhibitory population and the identity matrix. One resistor
  stands for all the channels of one type, not for one ion channel, and the caption has no final period. Fix, in one
  change of the picture, the caption and the equation: the label c_j in the picture, the equation $$I = c_j
  \frac{dv_j}{dt} + \sum_i g_i x_i (v_j - E_i)$$, and the caption "The electrical circuit of one compartment. The
  membrane is the capacitor $$c_j$$. Each channel type $$i$$ is a resistor $$R_i$$ in series with a battery $$E_i$$,
  its reversal potential. Arrows mark gated channels, whose resistance changes over time. The leak $$L$$ has a
  constant resistance. $$I = i_{m,j} A_j$$ is the total current through the membrane."

### Variable Timestep Simulation of the Electrical Activity of Neurons (2017-02-17-neuron-variable-timestep.md)

- `neuron-variable-timestep/cvode_pulse_currents_table.svg`, `cvode_dependency_on_variation_solution.svg` and
  `cvode_dependency_on_events_arrival.svg`: the currents are in mA and µA (2.7 mA, "threshold current (0.206mA)",
  "0.1µA" to "1mA"), as in the preprint. NEURON's `IClamp` amplitude is in nA, and the threshold current of a cortical
  pyramidal cell model is of the order of 0.1 to 1 nA, so 0.206 mA is likely 0.206 nA. The text gives no current
  values. Fix, after a check of the simulation scripts: mA to nA and µA to pA in the labels.
- Optional, `benchmark_two_rows_2.svg`: the orange "region of similar runtime growth" does not follow from the plotted
  runtimes. Fix: check it against the benchmark data, or remove it.
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

### Linear Regression, Matrix Factorization and Loss Functions (2017-08-17-linear-regression-matrix-factorization.md)

- Optional, `linear-regression-matrix-factorization/matrix-factorization.svg`: the rating 6 (movie 1, user 7) is
  outside a scale of 1 to 5 stars. Fix: a rating from 1 to 5.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- `unsupervised-learning/gaussianscatterpca.svg`: the arrow heads end past the scaled eigenvectors, at 3.75 and 1.75
  instead of 3 and 1, the lengths in the Wikipedia figure it follows. Fix: arrow heads that end at the tips of the
  vectors.

### Deep Neural Networks, Backpropagation, Dropout, CNNs and Embeddings (2018-03-27-deep-neural-networks.md)

- `deep-neural-networks/fully_sparse_connectivity.svg` and its caption keep the symbols of Figure 9.2 of the Deep
  Learning book: inputs x1 to x5 and outputs s1 to s5. The text writes the input of a layer as $$x^{(0)}$$ and its
  output as $$x^{(1)}$$, as `dnn.svg` does. Fix: inputs x₁⁽⁰⁾ to x₅⁽⁰⁾ and outputs x₁⁽¹⁾ to x₅⁽¹⁾ in both rows, and
  the caption "... where the input $$x_3^{(0)}$$ affects only the outputs $$x_2^{(1)}$$, $$x_3^{(1)}$$ and
  $$x_4^{(1)}$$. Bottom: a fully connected layer, where $$x_3^{(0)}$$ affects every output."

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
- `statistics-for-ml/confidence_interval.svg`: the drawing has no source, and the label "0.95" is crossed by the
  center line. Fix: name the source of the drawing, or redraw it, and move the label. Optional: the curve is centered
  at X̄, as in many textbooks, but the sampling distribution of X̄ is centered at μ; a curve centered at μ, with the
  interval drawn around one sample mean X̄, is exact.

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- `bayesian-linear-regression/linear_bayesian_predictive_distribution_*.svg`: the model assumes a noise standard
  deviation of 0.71 (noise precision beta = 2, as the caption states), but the data have a noise standard deviation of
  0.25. After 25 samples, the band is thus 0.76 to 0.88 wide on each side of the mean, three times the scatter of the
  samples. Fix: redraw with beta = 16 (band 0.27 to 0.32), see the code issues above, and change the caption.
- Caption of `bayesian-linear-regression/linear_bayesian_predictive_distribution_*.svg`: it gives the prior precision
  $$\alpha = 10^{-3}$$ and the noise precision $$\beta = 2$$. The text now ties $$\alpha$$ to the prior (the $$L_2$$
  constant $$\alpha$$ equals $$1/b^2$$, the prior precision, as the notebook's `alpha`), but it never defines the
  noise precision $$\beta = 1/\sigma^2$$. Fix (in a picture task): "a prior precision $$\alpha = 1/b^2 = 10^{-3}$$ and
  a noise precision $$\beta = 1/\sigma^2 = 2$$" (or 16 after the redraw with beta = 16).

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

### GPT in C++: LibTorch vs. PyTorch, TorchScript and torch.compile (2023-06-27-gptlite-cpp.md)

- Optional, caption of `gptlite_compact.svg` and `benchmark_model.svg`: it starts with "This post details", and calls
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

### Data parallelism, sharding and CPU offloading (2023-08-18-data-parallelism.md)

- Optional, `distributed-training/workflow_sharding.svg` shows ZeRO-3 as the ZeRO paper describes it: each process
  owns whole layers, broadcasts their parameters and reduces their gradients. DeepSpeed ZeRO-3 and PyTorch FSDP split
  every layer into one shard per process, with an all-gather of the parameters and a reduce-scatter of the
  gradients. The text explains both. Fix: redraw each layer as four shards, with all-gathers and reduce-scatters.
- Optional, `distributed-training/deepspeed_stages.svg`: the number of processes is N_d (and gpu_N-1), while
  `data_parallelism.svg` and the text use P; the text states that N_d = P. The caption calls the stages
  "ZeRO FSDP". Fix: write P in the figure, and "the three stages of ZeRO" in the caption.
- Optional, the caption of `distributed-training/imgs/benchmark_gptlite.svg` does not say how the memory is averaged.
  Fix: "GPTlite on one node with 8 GPUs, in fp32, with micro-batches of 1 sample: average and peak allocated memory
  per GPU, and throughput over all GPUs".
- Optional, caption of `distributed-training/imgs/data_parallelism.svg`: it names "the DDP data layout" before the
  text introduces DDP, while the layout is the same for DDP, FSDP and ZeRO. Fix: "The data layout of data
  parallelism, split across $$P = 4$$ processes color-coded as blue, yellow, red and green."
- `distributed-training/imgs/vdnn2.svg` and `vdnn3.svg` label the last layer "Layer(N)", and the vdnn2 caption says
  "at layer N", while the text numbers the layers from 1 to $$L+1$$ and writes the number of samples as $$N$$ (under
  the MAE formula). Fix: "Layer(L+1)", and in the caption "at its last layer, $$L+1$$".

### Pipeline parallelism (1F1B, Zero Bubble, DualPipe) (2023-08-30-pipeline-parallelism.md)

- `distributed-training/imgs/pipeline_8_stages.svg`, `pipeline_4_stages.svg` and `pipeline_2_stages.svg` label their
  rows "Worker 1" to "Worker 8", and the caption says "8 workers", while the text says GPUs and the other schedules
  label their rows "GPU 0" to "GPU 3" (CLAUDE.md keeps "Worker" for the PipeDream pictures). The caption also calls
  the center and right panels "2 data-parallel groups of 4 pipeline stages" and "4 data-parallel groups of 2 pipeline
  stages", but in Megatron-LM and DeepSpeed a data-parallel group holds the GPUs of one stage: the panels show 2
  pipelines of 4 stages and 4 pipelines of 2 stages. Fix: rows "GPU 0" to "GPU 7", and the caption "... on 8 GPUs.
  ... Center: 2 pipelines of 4 stages. Right: 4 pipelines of 2 stages."

### Tensor parallelism (2023-09-02-tensor-parallelism.md)

- Picture markup of the option 1 and option 2 pictures: the `{: .picture}`, `<img>` and `{: .caption}` lines are
  indented by 2 spaces, less than the 3 of the list item "1. ", so they end the ordered list, and kramdown renders the
  second option as a new list numbered "1." (also on master). Fix: indent these lines by 3 spaces, so that each
  picture and caption stays inside its list item and the second option is numbered "2." (checked with kramdown on a
  copy).

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
- Optional, `distributed-training/imgs/context_parallelism.svg` writes the head size as E. Fix: write the head size as
  d (see the content item on E).

### Diffusion models, from DDPMs to DiTs (2025-02-25-diffusion.md)

- `diffusion/masked_autoencoders_cropped.svg` and `masked_autoencoders.svg` are now the same picture, shown twice in
  the post. Fix: keep one file and link both places to it, or show the picture once.
- Optional, `diffusion/dit_moe.svg`: the dashed residual connections start after the first AdaLN, while the code of
  the post adds the input of the block, and three experts are gated, while the code routes each token to 2 experts.
  The picture follows the figure of the paper, which draws both this way. Fix: check both against the code, and say in
  the caption where the code differs.
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
- `inference/imgs/roofline.svg`: the ❌ sits at an arithmetic intensity of 512, right of the ridge point, where the
  roof is flat. There, moving right does not raise the attainable performance. Yet the picture draws a "move right"
  arrow from the ❌, while the text says "Computation to the right is compute-bound: only faster math, or less of it,
  helps". The author's original picture already had the ❌ at this place. Fix: move the ❌, with its arrows and their
  labels, left of the ridge point and below the slanted roof, for example to an intensity of 1 and a performance of
  1/4. There, moving right and moving up both raise the attainable performance, as the two lists of the text describe.
  Another fix is to keep the ❌ and drop the right arrow and its label.

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
- `inference/imgs/tp_timeline.svg` labels the baseline collective "NCCL AllReduce". The kernel fusion posts name no
  hardware or library of the papers (Writing rules, "The author's papers"), and the text says "the communication
  library". Fix: "AllReduce".

### Kernel Fusion for MoE Expert Parallelism (2026-10-09-kernel-fusion-expert-parallelism.md)

- The eleven `inference/imgs/ep_*.svg` pictures are copies of the figures of the paper (its
  repository's `imgs/`, commit 7d15d86), not yet redrawn. The TikZ exports (`ep_pipeline.svg`, `ep_stages.svg`, `ep_results.svg`,
  `ep_data_movement.svg`) have 4 to 6 pt text at their natural size, and the post shows them at 75% to 100% of the
  page, with text of 8 to 13 px. `ep_communication_models.svg` is 1019 px wide, wider than a post, and is shown at
  100%, i.e. at 86% of its size. Fix: redraw all eleven in a "kernel-fusion-expert-parallelism" tab of
  `assets/inference/inference.drawio`, from the paper's `imgs/moe_dispatch_and_combine.drawio` and TikZ sources,
  with the sizes and fonts of CLAUDE.md (Pictures, sections 3 and 4); the four models of
  `ep_communication_models.svg` fit 880 px in one column.
- Four copies name the ranks after the hardware of the paper: "Token NPU" and "Expert NPU" in `ep_all_to_all.svg`,
  `ep_all_to_all_inc.svg`, `ep_one_to_many_inc.svg` and `ep_communication_models.svg`, and "collective push, NPU
  reduce" in `ep_communication_models.svg`. The kernel fusion posts name no hardware of the papers (Writing rules,
  "The author's papers"), and the captions say "rank". Fix: "Token rank", "Expert rank" and "rank reduce".
- `inference/imgs/ep_stages.svg` draws the eight stages on the time axis of the decode step, whose costs are
  placeholders, not measurements, as the caption says. Fix: remove the time axis until the decode step is measured,
  or redraw it with the measured costs.
- `inference/imgs/ep_results.svg`: the right panel plots decode times that are placeholders, not measurements, as the
  caption says. Fix: remove the right panel and its sentence in the caption until the decode step is measured.
- Signals are dashed arrows in `ep_all_to_all_inc.svg`, `ep_one_to_many_inc.svg` and `ep_communication_models.svg`,
  and dotted arrows in `ep_timeline.svg` and `ep_timeline_noncollective.svg`. The paper calls them dotted in all its
  captions; the captions of the post follow each picture. Fix at the redraw: dotted signals (dashPattern=1 2) in every
  picture, and "dotted" in every caption.
- The captions have no credit to the post's paper, whose figures the pictures copy. The paper is the author's own and
  not public yet. Fix: "Source: [MoeInc: an Expert Parallelism MegaKernel with Overlap of Matrix, Vector and
  In-Network Compute](link)" once the paper has a public link, and "Adapted from:" after the redraw; the user
  decides.

### Knowledge distillation, speculative decoding and multi-token decoding (_drafts/2026-03-17-inference-distillation-speculative-decoding.md)

- Optional, `inference/imgs/mtp_parallel.svg` writes the tokens as x_t, as Gloeckle et al. do, while eagle, layerskip
  and mtp_sequential write them as t_i. The text follows each figure. Fix: keep the paper notations, or write t_i in
  mtp_parallel and in the formulas of the parallel heads.
- `inference/imgs/layerskip.svg` names the predictions of the full model p_1 to p_3 ("predictions that verify the
  drafts"), while $$p$$ is the distribution of the draft model and of the student in the text, in spec_sampling and in
  the `accept_sampling` code. Fix: label these cells v_1 to v_3, or leave them without a symbol; not y_1 to y_3, as
  $$y_i$$ is the softened output of the distillation section.

### Pruning and quantization (_drafts/2026-04-22-inference-pruning-quantization.md)

- Caption of `inference/imgs/hadamard_rotation.svg`: "the INT4 levels are closer together (5.7 instead of 12.6
  apart)" gives the distances between the drawn level lines in pixels. In the units of x, the INT4 step is 40/7 =
  5.71 before the rotation and 18.03/7 = 2.58 after it. Fix: "(2.6 instead of 5.7 apart)".
- Optional, `inference/imgs/quantization_granularity.svg`: the per-group panel colors its 12 groups with 6 colors
  (rows 1 and 3, and rows 2 and 4, repeat their colors). In the per-row panel each color is one scale, so the
  per-group panel reads as 6 scales. Fix: a black frame around each group, or 12 distinct fills.
- `inference/imgs/pruning_structures.svg` writes the number of heads as H ("H·d_head × E", "E × H·d_head"), while H is
  the Hessian of GPTQ in the same post; the code names it `n_heads`. Fix: "n_heads·d_head × E" and "E ×
  n_heads·d_head".
- `inference/imgs/gptq.svg`: the labels "err = W[:, i] − q" and "H = 2 X XT, from the calibration inputs X" take the
  columns of X as the inputs, and q as the rounded column. The text, the code and `llm_int8.svg` take the rows of X as
  the inputs, the text writes the rounded column as Ŵ[:, i], and `int4_packing.svg` uses q for the integers. Fix: "err
  = W[:, i] − Ŵ[:, i]" and "H = 2 XT X".

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
- Optional, `publications/zeroplusplus.svg`: the labels follow Fig. 2(a) of ZeRO++, but call "scale" a range of
  values (9.22 = max |x|, with q = round(127 x / 9.22); and for a block, max − min = 9.07, with
  q = round(255 (x − zp) / 9.07) − 128). The text of the paper calls qwZ symmetric, with one scaling coefficient per
  block, while its figure uses a zero point. Fix: rename the label to "range", or add "(the scale is the range of
  the block)" to the caption.
- Optional, caption of `publications/flashattention.svg`: "(dotted box)", but the box is dashed. Fix: "(dashed box)".
- Optional, caption of `publications/comet.svg` (Comet): "the number of SMs on Hopper(132)" lacks a space. Fix:
  "Hopper (132)".

## Content issues

Content problems that need the author's decision, such as overlap between posts.

### Distributed Matrix Multiplication and Database Joins (2014-04-02-distributed-matmul-dbs.md)

- Optional, Cannon's algorithm needs a square grid of $$p = q^2$$ ranks. The first version of the post (fac2c1d) also
  named SUMMA, which ca5e655 dropped: "SUMMA: Scalable Universal Matrix Multiplication Algorithm" (van de Geijn and
  Watts, LAWN 96, https://www.netlib.org/lapack/lawnspdf/lawn96.pdf). SUMMA replaces the shifts with broadcasts along
  the grid rows and columns, and also works on rectangular grids (the paper runs it on a 15 x 31 mesh; the Wikipedia
  article on Cannon's algorithm says that SUMMA "overcomes the need for a square 2D grid"). Proposed fix: add one
  sentence at the end of the Cannon section, such as "Cannon's algorithm needs a square grid of ranks; [SUMMA][summa]
  replaces the shifts with broadcasts along the grid rows and columns, and also works on rectangular grids.", with
  `[summa]: https://www.netlib.org/lapack/lawnspdf/lawn96.pdf`. The other option is to leave the post as it is.

### Distributed Matrix Transpose Algorithms (2015-04-21-matrix-transpose.md)

- The formulas number the ranks from 1 ($$M_1 // M_2 // ... // M_R$$, and the row interval $$\big[ \sum^{r-1}_{s=1}
  \vert V_s \vert, \,\, \sum^{r}_{s=1} \vert V_s \vert \big)$$), as the preprint does, while every picture numbers
  them from 0 ("rank 0" to "rank 2"). Fix: number the ranks from 0 in the text, i.e. $$M_0 // M_1 // ... // M_{R-1}$$
  (and the same for $$\|$$ and for the transposed blocks) in the definitions of the local transpose and the view swap
  and in the verification, and $$\big[ \sum^{r-1}_{s=0} \vert V_s \vert, \,\, \sum^{r}_{s=0} \vert V_s \vert \big)$$
  for the row interval, as pass 2 did (1563211); or keep the notation of the preprint and say once, where the ranks
  are introduced, that they are numbered from 1 in the formulas. This rewrite keeps the author's notation, as it is
  not wrong in itself and the fix changes eight formulas.
- Optional: the paragraph on the dense transpose ("Given a dense matrix representation ...") uses the rank $$r$$,
  $$G^{\star}_{r}$$, $$V_r$$ and $$E^{\star}_{r}$$ one paragraph before the text defines the ranks and $$G_{r} =
  (V_{r}, E_{r})$$. Fix: move the two sentences "A distributed memory data layout assumes ..." and "Let $$G_{r} =
  (V_{r}, E_{r})$$ be the subset ..." to the start of that paragraph, or add there "(the subgraph held by rank $$r$$,
  defined below)".

### Distributed Load Balancing and Spatial Slicing (2015-12-01-slicing.md)

- Optional: the ORB row of the table says "requires one sorting of half of the elements of a slice per iteration",
  while each level sorts all the objects, in slices that hold half of the objects of the slice they come from (the
  cost $$n \log_2 \frac{n}{2^{i-1}}$$ of the same row). Fix: "requires, at every iteration, one sorting of each slice,
  which holds half of the elements of its parent slice".

### The Leaky Integrate-and-Fire Neuron and the Brunel Network (2016-02-08-lif-brunel.md)

- The neuron parameter table writes the membrane time constant as $$\tau_m$$ and the resting potential as $$V_E$$ (the
  symbols of NEST by Example), while the text uses $$\tau$$ and the RC picture $$u_{\text{rest}}$$; "1pF" has no
  space. Fix: $$\tau$$, $$u_{\text{rest}}$$ and "1 pF" in the table, in a table task.

### Numerical Simulation of Detailed Neuron Models (2016-08-15-neuron-electrical-activity.md)

- Symbols that the author uses for more than one quantity, kept in this pass because the formulas are the author's:
  $$A$$ (membrane area, and the matrix of the linear system $$dy/dt = Ay$$); $$I$$ (inhibitory population, total
  membrane current, identity matrix, besides $$I_k$$, $$I_{inj}$$ and $$I(V,t)$$); $$n$$ (HH gating variable, number
  of half waves, time step index $$y^n$$, order of accuracy, size of the matrix in $$O(n)$$); $$m$$ (gating variable,
  points per cable); $$h$$ (gating variable, step size in $$E(h) = Ch^n$$); $$E$$ (excitatory population, reversal
  potentials, error $$E(h)$$); $$C$$ (capacitance, constant of $$E(h)$$); $$L$$ (leak, cable length); $$R$$
  (resistance, gas constant); $$x$$ (position, gating term $$x_i$$); $$d$$ (diameter, main diagonal of the Hines
  matrix); $$k$$ (neighbor index, rate constant $$k_{nm}$$); and $$u$$, $$V$$ and $$v$$ for the membrane potential.
  Proposed fix: in a later pass, with the author's approval, name the matrix of the linear system $$J$$, write the
  order of accuracy as $$\varepsilon(\Delta t) = K \Delta t^p$$, the gating term as $$w_i$$, the HH potential as
  $$v$$, and the main diagonal of the Hines matrix as $$D$$, and leave the remaining clashes, which are standard
  notation in their own paragraphs.
- The heading "Fixed Time Stepping Interpolation" names a time integration "interpolation" (the sentence below it now
  says "integration"). Proposed fix: rename the heading to "Fixed Time Step Integration".
- "For an extremely small time step of $$\Delta t=0.0001 ms$$, the system almost achieves equilibrium" is unclear:
  forward Euler with a small step follows the analytic solution closely, and does not bring the system to an
  equilibrium. Proposed fix: after a check of The NEURON book, ch. 4, write "the numerical solution comes close to the
  analytic one, at the cost of a very high computational load".
- The GHK equation ends with "$$\approx -60 mV$$" but the text gives no permeabilities. With the listed concentrations
  at 34 °C and the resting squid axon ratio $$P_K : P_{Na} : P_{Cl} = 1 : 0.04 : 0.45$$ of Hodgkin and Katz (1949), it
  gives about $$-75$$ mV, and the action potential picture shows rest at $$-70$$ mV. Proposed fix: add the ratio and
  its result after the equation ("with $$P_K : P_{Na} : P_{Cl} = 1 : 0.04 : 0.45$$, the resting potential is about
  $$-75$$ mV"), or write "between $$-60$$ and $$-75$$ mV, depending on the permeabilities". Source of the ratio: [The
  enduring legacy of the constant field
  equation](https://rupress.org/jgp/article/149/10/911/43579/The-enduring-legacy-of-the-constant-field-equation), J.
  Gen. Physiol. (2017).
- "NEURON's variant of Crank-Nicolson with staggered time stepping applies the Strang splitting method": Hines and
  Carnevale (1997) describe the staggered scheme (states at half steps, voltage at whole steps) without naming Strang
  splitting, and so does the analysis of Hines' method in [arXiv 1702.05917](https://arxiv.org/abs/1702.05917).
  Proposed fix: check The NEURON book, ch. 4; if it does not name it, write "applies a staggered scheme, similar to
  Strang splitting, to calculate ...".

### Variable Timestep Simulation of the Electrical Activity of Neurons (2017-02-17-neuron-variable-timestep.md)

- NEURON's default tolerance (Local and Global Time Stepping). The text gives NEURON's default error setting for CVODE
  as 10 μV for the membrane potential, i.e. atol 1e-2 in mV. That is the value of the NEURON Book and of the
  documentation of `CVode.atol` ("The default absolute tolerance is 1e-2"). However, NEURON's code sets `atol_ = 1e-3`
  (`src/nrncvode/netcvode.cpp`), i.e. 1 μV. The preprint calls 1e-3 the default and uses it in all its tests (Figs. 5
  to 7 and the benchmark). Proposed fix: add "(the tests below use $$\text{atol} = 10^{-3}$$, the default in NEURON's
  code)" after the sentence, or give only the value of the code.
- The speed-up estimate (last paragraph). The *Overall Runtime Speed-up Estimation* of 228.5-24.6x is the preprint's
  mean of the per-regime speed-ups, weighted by the shares of neurons per regime (31.43%, 38.44%, 27.02%, 3.10% and
  0.01%, Section 4.5). The runtime of a network that mixes the regimes is the weighted sum of the per-regime runtimes.
  So its speed-up is the weighted harmonic mean of the per-regime speed-ups. The moderate regime alone (27.02% of the
  neurons, speed-ups of 7.7 and 1.8) bounds it by 7.7/0.2702 ≈ 28.5 on 1,024 neurons and 1.8/0.2702 ≈ 6.7 on 65,536
  neurons. Proposed fix: add after the estimate "As the runtime of a mixed network is the sum of the runtimes of its
  parts, its speed-up is the weighted harmonic mean of the per-regime speed-ups, at most 28.5 on 1,024 neurons and 6.7
  on 65,536 neurons.", or keep only the preprint's numbers, as now.

### Linear Regression, Matrix Factorization and Loss Functions (2017-08-17-linear-regression-matrix-factorization.md)

- Optional: "dropout" links to a Medium article, in this post and in the deep neural networks post. Proposed fix:
  link the paper of Srivastava et al. (2014), https://jmlr.org/papers/v15/srivastava14a.html, or, in this post, the
  dropout section of the deep neural networks post.
- One symbol for several quantities in the loss sections: $$N$$ is the number of samples and also the negative input
  of the triplet loss and the number of pixels of the Dice loss; $$D$$ is the number of input features and also the
  distance of the contrastive loss; $$\alpha$$ is the margin of the triplet loss and the balancing factor of the focal
  loss, while the contrastive loss names its margin $$m$$. Proposed fix: write the triplet loss with $$x$$, $$x^+$$,
  $$x^-$$ (anchor, positive, negative) and the margin $$m$$, as the contrastive loss; name the contrastive distance
  $$\delta$$ (or $$D_W$$, as Hadsell et al.); use $$M$$ for the number of pixels in the Dice formulas.
- Focal loss: the binary formula $$-\alpha (1 - p_t)^\gamma \log(p_t)$$ uses one constant $$\alpha$$, while the text
  (now) and the paper weight the positive class by $$\alpha$$ and the negative class by $$1-\alpha$$. Proposed fix:
  write the binary formula as $$-\alpha_t (1 - p_t)^\gamma \log(p_t)$$, with $$\alpha_t = \alpha$$ for the positive
  class and $$1-\alpha$$ for the negative class (Lin et al. 2017, section 3.1 and eq. 5), and the multi-class form as
  the same formula with one $$\alpha_t$$ per class; or leave it, as the author's two formulas read.
- Optional, typesetting of formulas: $$min_w$$, $$min_{W,Z}$$, $$max(0,m−D)$$ and $$rank(X)$$ render in italics as
  products of letters, $$R^{D \times K}$$ stands for the real numbers, and $$L_x(θ) = −log p(x | θ)$$ renders as an
  italic "logp". Proposed fix: $$\min_w$$, $$\min_{W,Z}$$, $$\max(0, m-D)$$, $$\text{rank}(X)$$, $$\mathbb{R}^{D
  \times K}$$ and $$L_x(\theta) = -\log p(x \mid \theta)$$ (formula changes that the author reviews).

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- The definition of PCA that the post credits to "Gerstner's lecture notes" is the first sentence of the Wikipedia
  article on PCA. Proposed fix: check the lecture notes, or credit Wikipedia.
- SOM section: the text names the input $$x_u$$ and the winner neuron $$k$$ ($$\| w_k - x_u \| \leq \| w_i - x_u \|$$,
  $$\Lambda (j,k)$$), while `kohonen2.svg`, adapted from Ritter, Martinetz and Schulten (1992), and its caption name
  them $$x_i$$ and $$j^*$$, so the post has two symbols for the winner. Proposed fix: write the winner as $$j^*$$ in
  the SOM text and formulas ("determine which neuron $$j^*$$ is the *winner*", $$\| w_{j^*} - x_u \| \leq \| w_j - x_u
  \| \text{, for all } j$$, $$\Delta w_j = \eta \, \Lambda (j,j^*) \, (x_u - w_j)$$), and, in a picture task, say in
  the caption that $$x_i$$ is the input $$x_u$$ of the text.
- Optional, k-means section: the clusters are $$C_k$$ in the reconstruction error and $$C_i^{(t)}$$, $$C_i$$ in the
  update step and the convergence paragraph, but $$S = \{ S_1, ..., S_k \}$$ and $$S_i$$ in the objective (Wikipedia's
  notation). Proposed fix: "the partition of the data points into $$k$$ clusters $$C_1, ..., C_k$$ that minimizes the
  error above", $$\arg\min_{C_1, ..., C_k} \sum_{i=1}^{k} \sum_{x \in C_i} \| x - w_i \|^2$$, "where $$w_i$$ is the
  mean of the points in $$C_i$$".
- Optional: independent sections reuse symbols. $$k$$ is the kernel function $$k(x_i, x_j)$$ of kernel PCA, the
  kurtosis $$k(y)$$ of ICA, the number of principal components kept, the number of clusters, the winner neuron of the
  SOM and the number of neighbours of UMAP; $$\Lambda$$ is the matrix of eigenvalues of the whitening and the
  neighbourhood function $$\Lambda (j,k)$$ of the SOM; $$U$$ and $$V$$ are the singular vectors of the SVD, the
  rotation and the whitening matrix of ICA, and $$U$$ also the eigenvectors of spectral clustering; $$d$$ is the
  degree of the polynomial kernel and the distance $$d(a,b)$$. Proposed fix: $$\mathrm{kurt}(y)$$ for the kurtosis
  (the notation of Hyvärinen and Oja and of Yokota's slides), so that $$k(\cdot, \cdot)$$ is only the kernel; $$h(j,
  j^*)$$ for the neighbourhood function (Kohonen's notation) with the winner $$j^*$$ (item above); $$H$$ for the
  eigenvectors of spectral clustering.

### Deep Neural Networks, Backpropagation, Dropout, CNNs and Embeddings (2018-03-27-deep-neural-networks.md)

- Deep neural networks: symbols reused across sections. $$K$$ is the size of the hidden layers and the half-width of
  the filter ($$f[k]=0$$ for $$\vert k \vert \ge K$$); $$f$$ is the function of a layer, $$f^{(l)}$$, and the filter
  $$f[n]$$; $$l$$ is the layer index and an index of the 2D filter ($$\sum_{k,l} f[k,l]x^{(0)} [n-k,m-l]$$); $$\phi$$
  is the activation function and the message function $$\phi^{(k)}$$ of the graph formula. Proposed fix: write the
  filter as $$h$$, its half-width as $$r$$ and the 2D indices as $$u, v$$ ($$x^{(1)}[n] = \sum_u h[u]x^{(0)} [n-u]$$,
  $$h[u]=0$$ for $$\vert u \vert \ge r$$, $$x^{(1)}[n,m] = \sum_{u,v} h[u,v]x^{(0)} [n-u,m-v]$$), and the message
  function as $$\psi^{(k)}$$ (two formulas of the CNN section, three inline symbols, and the graph formula with its
  "where" item). Left for the author because it changes the author's formulas and the notation of the cited PyTorch
  Geometric page.
- Deep neural networks, embeddings list: "words: one-hot vectors or Word2vec" lists one-hot vectors as an embedding
  type, while the paragraph above defines embeddings as dense and low-dimensional; one-hot vectors are sparse and as
  long as the vocabulary. Proposed fix: "words: [Word2vec] (*skipgram* or *continuous bag of words*), whose inputs are
  one-hot vectors of the words", or keep the item and add "(sparse, as long as the vocabulary)".
- Deep neural networks, embeddings list: "i.e. the [*information
  bottleneck*](https://en.wikipedia.org/wiki/Information_bottleneck_method)" links Tishby's information bottleneck
  method (an information-theoretic trade-off between compression and prediction), while the U-net item means the
  bottleneck layer, the narrowest part of the network (pass 2 dropped the link for this reason; master kept it).
  Proposed fix: "i.e. the *bottleneck* of the network", without the link, or keep the link if the author means the
  information bottleneck theory of deep learning.

### Mathematics for ML (2018-04-20-algebra-for-ml.md)

- Overlap with the deep neural networks post (2018-03-27-deep-neural-networks.md): both posts explain
  backpropagation as the chain rule and introduce automatic differentiation. Proposed fix: this post keeps the math
  (the chain rule through K layers and the worked example with its computation graph); the deep neural networks
  post keeps the algorithm, its costs and the PyTorch details.
- Optional: independent sections of the post reuse symbols. $$P$$ is the pseudo-inverse
  $$(A^{\intercal}A)^{−1}A^{\intercal}$$ in the least squares derivation, a projection matrix in the Analytic Geometry
  section and the matrix of eigenvectors of the eigendecomposition; $$\sigma^2$$ is the noise variance of the least
  squares derivation, while $$\sigma_i$$ (and the bare $$\sigma$$ of "fewer singular values $$\sigma$$") are the
  singular values. Smaller reuses stay within one section ($$d$$ is the metric, the vector of the iterative solvers,
  an intermediate variable of the autodiff example and the number of variables; $$L$$ is the Cholesky factor and the
  loss). Proposed fix: $$A^{+}$$, the usual symbol of the Moore-Penrose inverse, for the pseudo-inverse in the
  derivation (three formulas, one of them a display), $$P_\pi$$ for the projection matrix (as in the MML book), and
  "fewer singular values $$\sigma_i$$" in the SVD paragraph; or keep the symbols, as each one is defined in its
  section.

### Statistics for ML (2018-06-15-statistics-for-ml.md)

- Notation, one symbol per quantity: the probability of an event is written $$P$$ (basic definitions, cdf, joint
  probability), $$p$$ (Markov's and Chebyshev's inequalities) and $$Pr$$ (sets), and the factorization of the Bayesian
  network has $$P$$ on its left side and $$p$$ on its right side; a density is written $$p$$ in the entropy formula
  and $$f$$ in the conditional and joint entropy formulas; a transpose is written $$^{\intercal}$$ and also $$^T$$
  ($$A \mathbb{Var}[x]A^T$$, $$\eta^T T(x)$$); the Cauchy-Schwarz inequality names its random variables $$A$$ and
  $$B$$, which name events in the basic definitions; $$\alpha$$ is both the level of the quantile and the significance
  level of the p-value. Proposed fix: $$P$$ for the probability of an event and $$p$$ for a density throughout ($$P(X
  \ge \epsilon)$$ in the two inequalities, $$P(F_1 ∪ F_2)$$ in the sets, $$p$$ on both sides of the factorization, as
  in its example $$p(A,B,C,D)$$), $$p$$ in all entropy formulas, $$^{\intercal}$$ for every transpose, $$X$$ and $$Y$$
  in the Cauchy-Schwarz inequality, and optionally another letter for the level of the quantile (e.g. $$u \in
  [0,1]$$); formula changes that the author reviews.
- Exponential family: "For optimization purposes, $$p(x \mid \theta) \propto \exp(\theta^{\intercal} ϕ(x))$$ is used."
  drops $$h(x)$$ and $$A(\theta)$$, and writes $$\theta$$ where the form above has the natural parameter
  $$\eta(\theta)$$. Maximum likelihood needs $$A$$: the gradient of the log-likelihood in the natural parameters is
  $$ϕ(x) - \nabla A(\eta)$$, while $$\eta^{\intercal} ϕ(x)$$ alone is linear in $$\eta$$ and has no maximum. The MML
  book, which this section follows, gives the proportional form (section 6.6, from memory: the PDF could not be read
  here) as an intuitive view of the family. Proposed fix: "As an intuition, the family can be seen as $$p(x \mid \eta)
  \propto \exp(\eta^{\intercal} ϕ(x))$$, ignoring $$h(x)$$ and the normalization $$A(\eta)$$.", or drop the sentence.
- Optional, the table at the end: "Smoothing (KDE)" stands in the column of non-parametric regression models for a
  Gaussian response, but a KDE estimates a density, not a regression function; and the header $$g(x_i^{\intercal}) =
  x_i^{\intercal} \beta$$ applies $$g$$ to a transposed vector. Proposed fix: "Smoothing (e.g. [kernel
  regression](https://en.wikipedia.org/wiki/Kernel_regression))", as pass 2 had, and $$g(x_i) = x_i^{\intercal}
  \beta$$ (a table change that the author reviews).
- Optional, overlap with the exponential family post (2019-03-20-exponential-family.md) and the variational inference
  post (2019-07-01-variational-inference-gmm.md): this post defines the exponential family (canonical form, sufficient
  statistics, Fisher-Neyman theorem, natural form, why it is used), which the exponential family post covers in full,
  and states the ELBO inequality, which the variational inference post derives. Both parts are the author's original
  text, and #69 removed the links between the posts. Proposed fix: keep both parts as a reference sheet, as the author
  wrote them, or shorten the exponential family section of this post to its definition.

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- One symbol for two quantities: $$b$$ is the bias of $$y = mx + b$$ in the introduction and the prior standard
  deviation of the MAP estimation ($$\mathcal{N}(0, b^2 I)$$, $$\frac{\sigma^2}{b^2}I$$, $$\alpha = 1/b^2$$, $$S_0 =
  b^2 I$$); the appendix also uses it for a right-hand side ($$Ax = b$$) and an offset ($$y = Xw + b$$). Proposed fix:
  name the prior standard deviation $$\sigma_w$$ (prior $$\mathcal{N}(0, \sigma_w^2 I)$$, $$w_{MAP} = (X^TX +
  \frac{\sigma^2}{\sigma_w^2}I)^{-1}X^Ty$$, $$\alpha = 1/\sigma_w^2$$, $$S_0 = \sigma_w^2 I$$), about ten formulas of
  the MAP and closed-form sections. It is left to the author because it changes the derivation.
- Appendix, rule 5 (layout of derivatives): rules 5.1 to 5.4 write the derivative of a scalar with respect to a vector
  as a column ($$\frac{d X^Tw}{dw} = X$$, the layout of The Matrix Cookbook, eqs. 69 to 72), while rule 5.5 and every
  derivation of the post write it as a row ($$\frac{d \, w^TAw}{dw} = w^T(A+A^T)$$, $$\frac{1}{\sigma^2}(w^TX^TX -
  y^TX)$$). Proposed fix: write rules 5.1 to 5.4 as rows (e.g. $$\frac{d X^Tw}{dw} = X^T$$), or add one sentence
  saying which layout the derivations use.
- Final remarks (minor): the Gaussian process equivalent to Bayesian linear regression with the prior $$\mathcal{N}(0,
  S_0)$$ has the kernel $$x^TS_0x'$$ (GPML, section 2.2, with $$\Sigma_p = S_0$$); the post's $$K(x, x') = x^Tx'$$
  holds for the prior $$\mathcal{N}(0, I)$$. Proposed fix: "with the kernel $$K(x, x') = x^TS_0x'$$ (i.e. $$x^Tx'$$
  for the prior $$S_0 = I$$)", or leave it.
- Introduction (minor): "the parameters to be learnt would then be all $$\mu$$ and $$\sigma$$": the posterior of $$m$$
  and $$b$$ is a joint Gaussian with a non-diagonal covariance (the tilted ellipses of the online learning pictures).
  Proposed fix: add "(and, in the posterior, the covariance between $$m$$ and $$b$$)", or leave it, since the sentence
  describes the prior.

### Exponential Family of Distributions (2019-03-20-exponential-family.md)

- Detour, Geometric and Negative Binomial, and the table: the text follows the source post (the negative binomial
  counts the failures until $$r$$ successes), while the table row "negative binomial distribution with known number of
  failures $$r$$", with $$f(x; r, p) = \binom{x+r-1}{x} p^x(1-p)^r$$, counts the successes $$x$$ before $$r$$
  failures. Both conventions are in use and swap $$p$$ and $$1-p$$. Proposed fix: add to the bullet "(the table below
  counts the successes before $$r$$ failures instead, which swaps $$p$$ and $$1-p$$)", or leave it.
- One symbol per quantity (minor): $$\mu$$ is the mean of $$x$$ in the Gaussian example, the Bernoulli mean $$\alpha$$
  in the Bernoulli example, and the mean parameter $$\mathbf{E}[T(X)]$$ in the Maximum Likelihood derivation
  ($$\mu_{MLE}$$, a vector $$(\mu, \mu^2 + \sigma^2)$$ for the Gaussian); $$h$$ is the base measure $$h(x)$$ and an
  arbitrary function of $$x$$ in the factorization list; the Bernoulli parameter is $$\alpha$$ in the example and
  $$p$$ in the table. Proposed fix: write the mean parameter of the derivation as $$\boldsymbol{\mu} =
  \mathbf{E}[T(X)]$$ (bold, two formulas), and leave the other two, which follow Wikipedia's notation.

### Variational Inference (ELBO, CAVI) and Gaussian Mixture Models (2019-07-01-variational-inference-gmm.md)

- Optional, Coordinate Ascent Variational Inference, the Lagrange step: the text says that the derivative uses
  Lagrange multipliers, but the equation of the derivative, $$\frac{d\,L}{d\,q(z_j)} = \mathbb{E}_{q_{\neq j}}
  [\ldots] - \log q(z_j) - 1 = 0$$, has no multiplier (as in the CMU 10-708 notes that it follows); the constant is
  absorbed later by the proportionality of $$q^{\star}(z_j)$$. Proposed fix: add the multiplier $$\lambda$$ of the
  constraint $$\int q(z_j) \, dz_j = 1$$ to the equation ("$$- \log q(z_j) - 1 + \lambda = 0$$"), and say after it
  that the factor $$e^{\lambda - 1}$$ is the normalization constant of $$q^{\star}(z_j)$$.
- Optional, one symbol per quantity: (1) in the argmax derivation, $$j$$ is both the index of the updated factor
  $$q_j$$ and the summation index of $$\sum_j^m$$; (2) $$m$$ is the number of latent variables ($$z_{1:m}$$) and the
  vector of variational means ($$ELBO(m, s^2, \varphi)$$, $$m_{\neq k}$$); (3) the MCMC section writes the densities
  as $$P(\cdot)$$ and the rest of the post as $$p(\cdot)$$. Proposed fix: write the sums of the ELBO derivations as
  $$\sum_{l=1}^m$$ with $$z_l$$, $$q_l$$; keep $$m$$ (both sources use it in both meanings, Blei et al. and the CMU
  notes) or call the number of latent variables $$M$$; write $$p$$ in the MCMC section.

### Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) (2019-10-12-vae-gan.md)

- Heading "Challenge: saturation of discriminator loss": the term that saturates is the generator's term of the value
  function, $$\log(1 - D(G(z)))$$, and the problem is the vanishing gradient of the generator (the section's own
  text). Proposed fix: rename the heading to "Challenge: saturation of the generator loss".
- Heading "Generative Adversarial Neural Network (GAN)": the paper, the approved title and now the introduction say
  "Generative Adversarial Network(s)". Proposed fix: rename the heading to "Generative Adversarial Network (GAN)".
  Optional, in the same change: "The reparametrization Trick" to "The reparameterization trick", the spelling of the
  text and of the papers.
- Introduction: "two new methods for non-parametric distributions based on neural networks". VAEs and GANs are
  parametric models (a fixed number of weights); what they model are complex distributions that no standard parametric
  family describes, with an implicit density in the case of the GAN. Proposed fix: "two new methods based on neural
  networks for distributions that no standard parametric family describes", or leave the author's wording.

### Learning from sequences with Encoder-Decoder, Transformers and BERT (2020-02-28-learning-from-sequences.md)

- Heading "BERT: Bidirectional Encoder Representation from Transformers": the model's name is "Bidirectional Encoder
  Representations from Transformers" (BERT paper, abstract). Fix: "## BERT: Bidirectional Encoder Representations from
  Transformers".
- Opening of the Encoder-Decoder section: the link labelled "Sequence to Sequence Learning with Neural Networks
  (Google, arXiv)" points to the NeurIPS proceedings (`papers.nips.cc/paper/5346-...pdf`, a 302 redirect to
  proceedings.neurips.cc), not to arXiv. Fix: link the arXiv abstract page https://arxiv.org/abs/1409.3215 (checked:
  Sutskever, Vinyals and Le), which matches the label, or change the label to "(Google, NeurIPS 2014)".
- Optional, "Other components" of the transformer: the list does not describe the "Add & Norm" blocks that the
  transformer and BERT pictures show. Fix: add one item, e.g. "The *Add & Norm* component adds the input of the
  previous component to its output (a [residual connection](https://en.wikipedia.org/wiki/Residual_neural_network))
  and normalizes the result ([layer
  normalization](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization)), which keeps the
  training of the stacked layers stable;" (Vaswani et al., Section 3.1).

### Building a GPT model in PyTorch from scratch (2023-02-28-gptlite.md)

- Building a GPT model (gptlite), introduction: "The fundamental difference between the encoder and the decoder is the
  *mask* in the decoder's attention matrix" leaves out the second difference, the cross-attention of the decoder over
  the output of the encoder, which the left model of the first picture shows and the attention section mentions.
  Proposed fix: "The fundamental differences between the encoder and the decoder are the *mask* in the decoder's
  attention matrix, which hides ..., and the cross-attention, through which the decoder attends to the output of the
  encoder."
- Building a GPT model (gptlite), section "Batching and dimensionality": "for a given input, the dataset includes all
  sequences from size `1` to size `block_size`", while `get_batch` only takes windows of `block_size` tokens; it is
  the mask of the attention that turns each window into `block_size` training examples. Proposed fix: add, after that
  sentence, "A single input of `block_size` tokens holds all of them: thanks to the mask of the attention (see below),
  the output at position `t` depends only on the tokens `0` to `t`."

### GPT in C++: LibTorch vs. PyTorch, TorchScript and torch.compile (2023-06-27-gptlite-cpp.md)

- Post, TorchScript section, "**Important**" paragraph: `torch.jit.trace` on a model compiled with `torch.compile`
  works in PyTorch 2.1 only. From PyTorch 2.2 on, it raises "Detected that you are using FX to torch.jit.trace a
  dynamo-optimized function. This is not supported at the moment." (`error_on_nested_jit_trace = True` in
  `torch/_dynamo/config.py`). Even in 2.1, the TorchScript file holds the traced eager graph, not the kernels of
  `torch.compile`. Proposed fix, if the author agrees: add "(in PyTorch 2.1; later versions raise an error, so the
  model is traced before it is compiled, and the TorchScript file does not hold the compiled kernels)".
- Overlap with other posts: the paragraph of #66 on CUDA graphs and the static KV cache repeats the section "Static
  shapes for CUDA graphs" of the KV cache post (`2025-06-07-inference-prefill-decode-roofline-cache.md`), and the
  megakernel sentence repeats the related-work paragraph of the kernel fusion tensor parallelism post, which cites the
  same work. Proposed fix, for the author to decide: keep the CUDA graph modes and gpt-fast here, and shorten the KV
  cache sentence to "For text generation, this requires a static KV cache, preallocated to its maximum length.", as
  the KV cache post explains and implements it.

### Data parallelism, sharding and CPU offloading (2023-08-18-data-parallelism.md)

- The post holds a hidden HTML comment, from the mixed precision paragraph to the end of the activation checkpointing
  section, still in the first and second person. The hidden text sets buckets of `4e5` elements, while the configs use
  `5e4`, and calls `to_layers()` `lo_layers()`. The front matter has the tag "activation checkpointing", which only
  the hidden text covers. Proposed fix: delete the comment and the tag, or rewrite the comment in the author's voice
  and publish it.
- Optional, the residual memory list "(activations, normalization layers, etc.)": it holds for the buffers of the
  normalization layers ("FSDP does not shard buffers", `torch/distributed/fsdp/api.py`, v2.0.1), but the LayerNorm
  weights of GPTlite are parameters, which ZeRO-3 and FSDP shard. The ZeRO paper (Sec. 3.2) defines the residual
  states as activations, temporary buffers and unusable fragmented memory. Proposed fix: "(activations, temporary
  buffers, fragmented memory, buffers of normalization layers, etc.)".
- The precision of the benchmark: the text now says fp32, from the configuration files, where the original said mixed
  precision. Proposed fix: confirm that the runs used these configurations, then add "in fp32" to the caption of the
  benchmark (diagram item).

### Pipeline parallelism (1F1B, Zero Bubble, DualPipe) (2023-08-30-pipeline-parallelism.md)

- The DualPipeV row of the summary table does not list the doubled point-to-point communication, which the text
  names, under "Main cost". Proposed fix: add "2× point-to-point communication" to that cell.
- The post keeps the workaround `outputs = outputs.requires_grad_(True)` for the error of the swapaxes lambda. If
  `outputs` has no `grad_fn`, this call lets the backward pass run, but no gradient reaches the layers before it.
  Proposed fix: check that the parameters of the last stage get gradients with the workaround (`p.grad` after
  `train_batch()`), and remove the sentence if they do not.
- Optional, results, memory per GPU: the first GPU used more memory than the others, also with the deep benchmark
  model. The text now says that 1F1B explains only a few MB of this gap for that model, and the cause of the rest is
  not known. Fix: measure the peak memory of every GPU (`torch.cuda.max_memory_allocated()` on each rank, and
  `nvidia-smi` for the CUDA contexts on GPU 0), and give the numbers and the cause in the post.

### Tensor parallelism (2023-09-02-tensor-parallelism.md)

- The last section was named "Final remarks and code", but the repository holds no code for this post: no file
  defines `Megatron_f`, `Megatron_g`, `Megatron_FeedForward` or `Megatron_MHA`. Proposed fix: add these classes
  to `assets/distributed-training` with a small test on 2 GPUs, or keep the code in the post only.
- Section "Final remarks and code": the last two paragraphs are about the DeepSpeed config files and the DeepSpeed
  features of the distributed training series, while this post uses neither DeepSpeed nor a config file (the configs
  are `assets/distributed-training/ds_config.json` and `ds_config_offload.json`, used by the data parallelism post).
  The text now says "the DeepSpeed config files and parallelism settings of the distributed training posts" and "These
  posts only scratched the surface". Proposed fix: move both paragraphs to the end of the data parallelism post, and
  keep here only the paragraph on PyTorch and DeepSpeed model parallelism; or keep them here as the closing remarks of
  the series.

### Mixture-of-Experts and Expert Parallelism (2024-01-19-mixture-of-experts.md)

- The post reuses symbols across sections: $$E$$ is the 1991 error function, the experts $$E_j$$ of the importance
  example, the number of experts in GShard and in the Tutel table, and the embedding size in the code comments
  (`B*S*E`); $$N$$ is the number of experts in the Switch and ST-MoE sections and the number of layers in the GShard
  picture; $$n$$ is the number of experts in the 2017 picture and in the Mixtral formula. Within each section the
  symbols now follow the paper and its picture. Fix: decide whether the post keeps each paper's notation per section
  (as now), or uses one notation (e.g. $$N$$ experts, $$T$$ tokens, $$E$$ only for the embedding size), which also
  needs the GShard and 2017 pictures and the Tutel table relabelled.
- The capacity factor advice of ST-MoE appears twice, in the Switch section (Expert Capacity) and in the ST-MoE
  recommendations of the further reading. Fix: keep it in the ST-MoE note only, and give in the Switch section the
  Switch paper's own finding, that Switch Transformers perform better at lower capacity factors (1.0, 1.25); or keep
  both.
- The main part of the post presents four routing schemes (dense softmax gating, noisy top-k gating, GShard's top-2
  with random routing, Switch's top-1), and the further reading adds expert choice and Soft MoE, without a summary.
  Fix, if wanted: a short table at the end of the Switch Transformers section that compares the experts per token, the
  expert compute per token and the load balancing method of the four schemes (pass 2 had one).
- The heading "Conditional Computing for improved scaling" uses "computing", while the GShard title and the post's own
  text say "conditional computation". Fix: "Conditional Computation for improved scaling".

### Sequence and context parallelism with Ulysses and Ring attention (2024-08-30-sequence-context-parallelism.md)

- The post uses `E` for both the embedding size (introduction, `(B, S, E)`) and the head size (the attention formula,
  the Ulysses view swap `(H, B, S/P, E)` to `(H/P, B, S, E)`, the ring comments, and the pictures
  `context_parallelism.svg` and `ulysses_sequence_parallelism.svg`), and `H` for both the number of heads and the
  frame height of the video shape `(B, C, T, H, W)`. The text keeps `E` for the head size, so that it agrees with the
  pictures and the code. Fix: write the head size as `d` (the code's `d_head`) in the text, in the same change as the
  pass 2 code items (ring comments, `dist_view_swap`) and diagram items (`ulysses_sequence_parallelism.svg`,
  `context_parallelism.svg`) on this symbol; optionally write the video shape as `(B, C, T, X, Y)` for `XxY` pixels
  per frame.

### Diffusion models, from DDPMs to DiTs (2025-02-25-diffusion.md)

- The symbol $$C$$ is the constant of Eqs. 8, 9-10 and 12 ("where $$C$$ is a constant that does not depend on
  $$\theta$$", the notation of DDPM) and the number of channels in the DiT and video sections (and in the caption of
  `align_your_latents.svg`). Fix: write the constant of the three equations and of the "where" sentence as
  $$\text{const}$$, or keep the paper's notation; the text already notes the same double use of $$T$$ ("here $$T$$ is
  the number of frames, not of diffusion steps").
- Movie Gen entry: "and on the use of activation checkpointing for improved memory efficiency" could not be checked.
  Section 3.1.6 does not mention activation checkpointing, and every fetched copy of the paper stops before Appendix
  A.2. Fix: check Appendix A.2 by hand, and remove the clause if the appendix does not mention it.

### Prefill vs decode, roofline, KV cache, batching and disaggregation (2025-06-07-inference-prefill-decode-roofline-cache.md)

- Symbols with two meanings. In "Where does the compute go", $$n$$ is the context length, while the prefill section
  defines $$n$$ as the number of output tokens. The other reuses stay within their own sections:
  - $$P$$ is the performance in the roofline, and the number of prompt or cached tokens in the KV cache code and in
    prefix caching;
  - $$B$$ is the peak bandwidth ($$B_{\text{peak}}$$), an inner matrix dimension, and the batch size of static
    batching;
  - $$N$$ is the vector length, the size of square matrices, and the length of a reused prefix. Fix: write $$T$$ for
    the context length in "Where does the compute go", as pass 1 did. If the author wants more precision there, pass
    1's counts can follow the two terms. With an MLP of hidden size $$4d$$, a token costs about $$24d^2$$ FLOPs in the
    linear layers and $$4Td$$ in attention. So attention dominates beyond $$T = 6d$$ tokens, i.e. 24,576 tokens for
    $$d = 4096$$.

### Faster attention (GQA, FlashAttention, sparse, linear and latent attention) (2025-12-30-inference-attention.md)

- Three symbols still name two quantities in this post, in different sections: $$m$$ and $$m'$$ are the running
  maximum of the online softmax (FlashAttention) and the compression rates of CSA and HCA ($$m = 4$$, $$m' = 128$$,
  also in the code and in the captions of csa.svg and hca.svg); $$P$$ is the softmax matrix in the FlashAttention and
  SageAttention text and the number of cached past tokens in the code comments (`[B, G, P+S, h]`); $$w$$ is the
  sliding-window size in the text, the attention weights `w_t` in a comment of the MLA code, and $$w_{t,j}$$ the
  weights of the indexer heads. The text follows each paper's notation, and each use is local to its section. Fix, if
  the author wants one symbol per quantity: write the running maximum as $$\mu$$ and $$\mu'$$, and the window size as
  $$n_w$$ (DeepSeek-V4 writes $$n_{win}$$); for $$P$$, rename the past tokens of the code comments (for example to
  `T`) in a code task, or keep $$P$$ as in the FlashAttention paper.

### Kernel Fusion for MoE Expert Parallelism (2026-10-09-kernel-fusion-expert-parallelism.md)

- The paper has not measured the decode step (the right panel of its runtime figure, and its decode timeline), the
  partial collective operations (its table of collective and partial collective results) and the number of microbatches in
  flight with the memory each one takes. The post shows the two decode placeholders with a note in their captions,
  and leaves the rest out. Fix: add these results to the post once the paper has them.

### Knowledge distillation, speculative decoding and multi-token decoding (_drafts/2026-03-17-inference-distillation-speculative-decoding.md)

- The usage comment of the Medusa code ("prefill the prompt: [1, S], with the causal_mask of the post on attention")
  relies on `causal_mask` from the post on attention, which the text no longer links (#69). Fix: in the code pass,
  write the mask in the comment as `torch.ones(S, S, dtype=torch.bool).tril()` (True where a token may attend, as in
  `tree_attention_mask`) and drop "of the post on attention", or keep the mention without a link.
- Optional: the LayerSkip text ("exits after the first $$E$$ layers") and `layerskip.svg` ("drafts of the first E = 4
  layers") use E for the number of draft layers, as the LayerSkip paper does, while the shape comments of the code
  blocks, the LayerSkip block included, use E for the embedding size. Fix: keep it, as the prose uses E only for the
  draft layers; or drop the symbol in the text ("the first layers") and in the picture ("drafts of the first 4
  layers") together.

### Pruning and quantization (_drafts/2026-04-22-inference-pruning-quantization.md)

- The draft describes the experiments of `inference/pruning.py` and `inference/quantization.py` on GPTlite (pruning
  with distillation, Wanda, ShortGPT, weight-only and weight-activation quantization, KV cache quantization), but
  shows none of their results. Fix: run the two scripts and add a table of their outputs (parameters, size,
  validation loss, time per token, matching tokens), or present the code without the comparisons.
- Optional: once both drafts are published, "knowledge distillation" in the introduction could link to the
  distillation post instead of Wikipedia.
- The last section, "Which one to use", is a subsection of "Quantization", but its guide now also compares 2:4
  sparsity with structured pruning and distillation. Fix: make it a section of its own ("## Which one to use"), or
  move the pruning sentence to the end of the pruning part.

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
- Optional, DeepSeek-V3.2 and DeepSeek-V3.2-Exp: both entries describe DeepSeek Sparse Attention in nearly the same
  words:
  - the lightning indexer with a few heads, ReLU and FP8;
  - the top-k selection with $$k = 2048$$;
  - $$O(L^2)$$ to $$O(Lk)$$;
  - the dense warm-up with a KL loss. Proposed fix: keep the description in the DeepSeek-V3.2 entry. Shorten the
    second paragraph of the V3.2-Exp entry to what only it says (the KV cache does not shrink, the reads per step
    drop). Or keep both, as the model card came first.
- Optional, Rectified flow (entry 064): the transcription of Sec. 2.1 of the paper stops after Eq. (2), so its "First,
  for a given input coupling ..." has no "Second". Proposed fix: add one sentence on what follows in the paper (the
  rectified flow averages the directions of the lines that cross at a point, so its paths do not cross, and reflow
  makes them straighter), or drop "First,".
- Graph Transformer Networks: one symbol for two quantities.
  - The notes write a meta-path as $$R = t_1 \circ t_2 \circ \cdots \circ t_l$$ between nodes $$v_1$$ and $$v_{l+1}$$,
    so $$l$$ is the length of the meta-path (the notation of Sec. 3.1 of the paper).
  - The caption and the picture use l for the index of the GT layer (𝔸<sup>(l)</sup>, ℚ<sub>1</sub><sup>(l)</sup>). A
    stack of l GT layers learns meta-paths of length up to l + 1.
  - Proposed fix: write the meta-path length as $$k$$ in the notes ($$t_k$$ and $$v_{k+1}$$), or keep the paper's
    notation.

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

### Knowledge distillation, speculative decoding and multi-token decoding (_drafts/2026-03-17-inference-distillation-speculative-decoding.md)

- The title says "multi-token decoding", while the post, its tags and the papers that it cites call the method
  "multi-token prediction" (MTP). Proposed title: "Knowledge distillation, speculative decoding and multi-token
  prediction". With "Efficient Inference: " before it, the home page line measures 686 px in the site's font and 637
  px in Arial (the current title: 681 px), under 750 px, at 92 characters.

### Pruning and quantization (_drafts/2026-04-22-inference-pruning-quantization.md)

- Optional: the other posts of the series name their methods in the title ("Faster attention (GQA, FlashAttention,
  sparse, linear and latent attention)", "Prefill vs decode, roofline, KV cache, batching and disaggregation"). Fix:
  "Pruning and quantization (Wanda, Minitron, GPTQ, AWQ, FP8, FP4)", 84 characters with "Efficient Inference: ".

### Publications bookmark (publications.md)

- Three entry titles do not follow the capitalization of their papers. Proposed fix: write them as the papers do.
  - "Siamese neural networks for one-shot image recognition" should be "Siamese Neural Networks for One-shot Image
    Recognition", as its caption writes it.
  - "Dropout: a simple way to prevent neural networks from overfitting" should be "Dropout: A Simple Way to Prevent
    Neural Networks from Overfitting", as its caption writes it.
  - "Deeply-supervised Nets" should be "Deeply-Supervised Nets", as on arXiv 1409.5185.
