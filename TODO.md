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
  and `diffusion/sora_vae.svg` (#333333) use colors outside the palette; 7 figures have lines thinner than 0.6 px.
  Fix: set them to the values of CLAUDE.md, then run the check again.
- The arrow check of the restyle (CLAUDE.md, Pictures, section 6) still reports 572 findings in 74 draw.io figures:
  285 arrows with less than 6 px of line before the head (90 of them with 4.6 to 5.9 px, as the 6 px heads take more
  of short arrows, for example in the GPT-lite pictures, `gptlite-cpp/torch_compile.svg` and the three
  `distributed-training/imgs/dnn_multiproc*.svg`), 175 lines through the text box of a label, 55 labels that run
  into a box, 29 heads on text, 20 heads that touch and 8 overlapping boxes. Most are in dense pictures from papers:
  `deep-neural-networks/densenet.svg` (50), `matrix-transpose/xcrs_format.svg` (50),
  `distributed-training/imgs/moe_gshard.svg` (36), `publications/helix.svg` (22) and `publications/performers.svg`
  (21); some are text boxes larger than their text. Fix: check each picture at its page size, move the labels and
  arrows that touch, and lengthen the short arrows (move their bend, or the boxes they join).
- The charts drawn from data (the plots, benchmarks and histograms without a draw.io source, for example
  `inference/imgs/tp_results.svg`, `gptlite-cpp/benchmark_*.svg` and the charts of `neuron-variable-timestep` and
  `bayesian-linear-regression`) keep the line weights of their scripts or of their earlier export (`tp_results.py`
  draws 1 px axes and 0.8 px bar outlines), not the weights of CLAUDE.md, Pictures, section 6. Fix: axes 1.6 px, bar
  outlines and plot lines 1.2 px, in each chart's script, or in the SVG of a chart without a script.
- Pictures to redraw, from the user's review of the four line weights (the next step of the restyle):
  - `inference/imgs/continuous_chunked_batching.svg` differs from its original. Fix: redraw it from the original, and
    add continuous batching with chunked prefill.
  - `distributed-training/imgs/workflow_data_parallelism.svg` and `workflow_sharding.svg` changed too much. Fix: the
    layout of the originals in this style, with square inputs, outputs and activations, and round layers and
    parameters.
  - `neuron-electrical-activity/neuron_model_scales.svg`: the compartmental membrane and the reaction-diffusion parts
    look poor. Fix: redraw them from the source file.
  - `distributed-training/imgs/vdnn2.svg` and `vdnn3.svg` are too tall. Fix: a lower layout, with the outlines and
    arrows of CLAUDE.md.
  - `inference/imgs/medusa.svg`, `eagle.svg` and the other pictures of the speculative decoding draft show a 4 x 1
    matrix after the target model, where the originals show a row of 3D cubes. Fix: the row of cubes again.
  - `neuron-variable-timestep/lvardt_exec_time_comparison.svg`. Fix: thick axes and color-coded lines.
  - `inference/imgs/prefill_decode.svg`, `publications/helix.svg` and `distributed-training/imgs/moe_megablocks.svg`
    are crowded. Fix: draw them 880 px wide (100%), with more space between their parts.
  - `distributed-training/imgs/dnn_model_parallelism.svg` is hard to read at its width. Fix: draw it larger.

### Distributed Sorting Algorithms (2014-06-21-distributed-sort.md)

- `distributed-sort/sample_sort.svg`: in step 5, the label "rank 1" is bold (`<b>rank 1</b>` in
  `distributed-sort.drawio`), while "rank 0" and "rank 2" are regular. Fix: the regular weight for the three labels
  (CLAUDE.md, Pictures, section 4), and export the SVG again.

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
- `unsupervised-learning/kohonen2.svg` and its caption name the input $$x_i$$, while the SOM text names it $$x_u$$
  (since iteration 2, the text names the winner $$j^*$$, as the picture does). The picture keeps the labels of Ritter,
  Martinetz and Schulten (1992). Fix: say in the caption that $$x_i$$ is the input $$x_u$$ of the text, e.g. "The
  input $$x_i$$ (the $$x_u$$ of the text) in the input space $$V$$ selects the winner neuron $$j^*$$ ...".
- Optional: `unsupervised-learning/kohonen_1.svg`, `kohonen_2.svg` and `kohonen_3.svg` have no caption; the list
  before them describes the three maps. Fix: one caption under the three pictures, e.g. "The weights of the 6x6
  neurons of the SOM, each drawn as a 28x28 image, for the digits 0 to 3: a tuned SOM (left), a high learning rate
  (center) and a large neighbourhood width (right)." (no credit: the pictures come from the post's own code).

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
- Caption of `bayesian-linear-regression/linear_bayesian_predictive_distribution_*.svg`: the text now names the prior
  standard deviation $$\sigma_w$$ (prior $$\mathcal{N}(0, \sigma_w^2 I)$$, $$\alpha = 1/\sigma_w^2$$), so the caption
  proposed in the existing item, "a prior precision $$\alpha = 1/b^2 = 10^{-3}$$", is stale. Proposed fix: replace the
  proposed caption of that item by "a prior precision $$\alpha = 1/\sigma_w^2 = 10^{-3}$$ and a noise precision
  $$\beta = 1/\sigma^2 = 2$$" (or 16 after the redraw with beta = 16).

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
- `distributed-training/imgs/variable_batch_lr_pipeline.svg` has no caption, unlike the ten other pictures of the
  post. Fix: add one, e.g. "Micro-batches of at most 30 tokens, on 2 processes with 2 gradient accumulation steps, for
  DDP (left) and pipeline parallelism (right). Pipeline parallelism needs the same shape in all micro-batches of a
  batch (here 3x11xE), so it packs fewer samples per micro-batch." (The left micro-batches hold 30, 28, 30 and 26
  tokens, the right ones 24, 25, 22 and 17.)

### Sequence and context parallelism with Ulysses and Ring attention (2024-08-30-sequence-context-parallelism.md)

- `distributed-training/imgs/ulysses_sequence_parallelism.svg`: the label "Local attention shape: B x H/P x S x E"
  and the axis E of the middle cube give the head size as E, the embedding size. An all-to-all keeps the number of
  elements per process, B·(S/P)·E before it and B·(H/P)·S·d after it, with Hd = E, so the label is H times too large.
  Fix: "Local attention shape: B x H/P x S x d", and the axis d on the middle cube.
- Optional, `distributed-training/imgs/context_parallelism.svg` writes the head size as E. Fix: write the head size as
  d (see the content item on E).
- `distributed-training/imgs/ulysses_sequence_parallelism.svg`, `ring_attention.svg` and `ring_attention_qkv.svg`
  color the sequence chunks blue, yellow, red and green (and the ring in this order), while `spdistributedsampler.svg`
  gives the chunks of a sample to process 1 (blue), 2 (green), 3 (red) and 4 (yellow), the colors that
  `sequence_and_data_parallelism.svg` and its caption also use. So the second chunk belongs to process 2 in one
  picture and to process 4 in the others. Fix: swap yellow and green in the three pictures (chunks and ring blue,
  green, red, yellow); the red process of `ring_attention_qkv.svg` then sees red, green, blue and yellow blocks, and
  the text ("the third (red) process") stays as it is.

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

### Kernel Fusion for MoE Expert Parallelism (2026-10-09-kernel-fusion-expert-parallelism.md)

- The four TikZ pictures (`ep_pipeline.svg`, `ep_stages.svg`, `ep_results.svg`, `ep_data_movement.svg`) are copies of
  the figures of the paper (its repository's `imgs/`, commit 7d15d86), not yet redrawn. They have 4 to 6 pt text at
  their natural size, and the post shows them at 75% to 100% of the page, with text of 8 to 13 px. Fix: redraw them
  in the "kernel-fusion-expert-parallelism" tab of `assets/inference/inference.drawio`, from the TikZ sources of the
  paper's latest version, with the sizes and fonts of CLAUDE.md (Pictures, sections 3 and 4).
- `inference/imgs/ep_stages.svg` draws the eight stages on the time axis of the decode step, whose costs are
  placeholders, not measurements, as the caption says. Fix: remove the time axis until the decode step is measured,
  or redraw it with the measured costs.
- `inference/imgs/ep_results.svg`: the right panel plots decode times that are placeholders, not measurements, as the
  caption says. Fix: remove the right panel and its sentence in the caption until the decode step is measured.
- The captions have no credit to the post's paper, whose figures the pictures copy. The paper is the author's own and
  not public yet. Fix: "Source: [MoeInc: an Expert Parallelism MegaKernel with Overlap of Matrix, Vector and
  In-Network Compute](link)" once the paper has a public link, and "Adapted from:" after the redraw; the user
  decides.

### Knowledge distillation, speculative decoding and multi-token prediction (_drafts/2026-03-17-inference-distillation-speculative-decoding.md)

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

### Distributed Matrix Transpose Algorithms (2015-04-21-matrix-transpose.md)

- Optional: the definitions of the concatenations index rows and columns from 1 ("if $$j \leq m$$", "if $$i \leq
  n$$"), while the row interval $$\big[ \sum^{r-1}_{s=0} \vert V_s \vert, \,\, \sum^{r}_{s=0} \vert V_s \vert \big)$$
  and the pictures number rows and columns from 0. Fix: say once, after the two definitions, that the formulas index
  the cells of a matrix from 1, or write the conditions as $$j < m$$ and $$i < n$$ with indices from 0.

### Numerical Simulation of Detailed Neuron Models (2016-08-15-neuron-electrical-activity.md)

- The membrane time constant has two symbols in the post: $$\tau = R_m C_m$$ in the scaling of the cable equation ($$t
  = T \tau$$, end of "Mathematical Basis"), and $$\tau_m = c/g$$ in the rate constant $$k_{nm}$$ ("Fixed Time Step
  Integration"), the same quantity ($$c/g = R_m C_m$$); the paragraph between them also uses $$\tau = RC$$ for a
  generic capacitor. Fix: write $$\tau_m = R_m C_m$$ and $$t = T \tau_m$$ at the end of "Mathematical Basis", and keep
  $$\tau = RC$$ for the capacitor.
- The renames of the item "Symbols that the author uses" bring two letters that the post also uses elsewhere: the
  order of accuracy $$p$$ in $$\varepsilon(\Delta t) = K \Delta t^p$$, also the parent vector $$p$$ and the parent
  index $$p_j$$ of the Hines solver, and the constant $$K$$, also potassium ($$K^+$$, $$g_K$$, $$E_K$$, $$P_K$$). Fix:
  keep them, as each is standard notation in its own section; or write the order as $$q$$ and the constant as
  $$c_\varepsilon$$.

### Variable Timestep Simulation of the Electrical Activity of Neurons (2017-02-17-neuron-variable-timestep.md)

- Optional: the references `[cvode]` (https://aip.scitation.org/doi/pdf/10.1063/1.4822377) and
  `[lytton2005independent]` (on mitpressjournals.org) point to old publisher domains. Proposed fix: link the DOIs,
  https://doi.org/10.1063/1.4822377 and https://doi.org/10.1162/0899766053429453, after a check that both resolve.
- Optional: the step size has two symbols, $$h$$ and $$h_n$$ in the formulas of the CVODE section, and $$\Delta t$$ in
  "The solver then adjusts $$\Delta t$$" and in the fixed-step values of the later sections. Proposed fix: write "The
  solver then adjusts the step size $$h_n$$", and keep $$\Delta t$$ for the fixed step of Backward Euler.

### Linear Regression, Matrix Factorization and Loss Functions (2017-08-17-linear-regression-matrix-factorization.md)

- Optional, other symbols with two meanings: $$\gamma$$ is the step size of gradient descent and the focusing
  parameter of the focal loss; $$\Omega$$ is the regularization term and the set of observed ratings of matrix
  factorization; $$x$$ is the input everywhere, but the class in the multi-class cross-entropy $$H(q,p) = -\sum_x q(x)
  \log(p(x))$$. Proposed fix: name the step size $$\eta$$ in the three update rules of gradient descent (the focal
  loss keeps the $$\gamma$$ of its paper); write the regularizer as $$\mathcal{R}(w)$$ in the section Regularization;
  write the multi-class cross-entropy as $$H(q,p) = -\sum_{c=1}^{C} q_c \log p_c$$, over the $$C$$ classes, as in the
  Dice section.
- Optional, Dice loss: the link on "Dice loss" points to the Generalised Dice Loss of Sudre et al. (2017), which
  weights each class $$l$$ by $$w_l = 1/(\sum_n r_{ln})^2$$, the inverse of the square of its number of ground truth
  pixels, while the multi-class formula of the post is the unweighted mean over the classes. Proposed fix: add after
  the multi-class formula "The Generalised Dice Loss of Sudre et al. also weights each class by the inverse of the
  square of its number of ground truth pixels, so that small structures count as much as large ones."

### Statistics for ML (2018-06-15-statistics-for-ml.md)

- `\mathbb{Var}` and `\mathbb{Cov}` render in MathJax 4 as double-struck letters, lowercase included (𝕍𝕒𝕣, ℂ𝕠𝕧): 42
  times in 18 formulas of this post, 6 times in 2 formulas of the algebra post (2018-04-20-algebra-for-ml.md) and 15
  times in 7 formulas of the Bayesian linear regression post (2018-08-20-bayesian-linear-regression.md); the
  correlation of this post is written $$corr[x,y]$$, in italics like a product of variables. Proposed fix:
  $$\mathrm{Var}$$ and $$\mathrm{Cov}$$ in the three posts (and $$\mathrm{corr}$$ here), keeping $$\mathbb{E}$$ for
  the expectation; or $$\mathbb{V}$$ for the variance, as the MML book writes it; formula changes that the author
  reviews.
- Optional: one symbol for three quantities: $$\phi$$ is the convex function of Jensen's inequality, $$ϕ(x)$$ the
  sufficient statistics of the exponential family, and $$\phi$$ the variational parameters of $$q_{\phi}$$ in the ELBO
  inequality. Proposed fix: $$f$$ for the convex function of Jensen's inequality ($$f(\mathbb{E}[X]) \le
  \mathbb{E}[f(X)]$$), as the MML book writes convex functions (section 7.3); the other two keep their usual symbols.
- Optional: the Gaussian section repeats the mean and the covariance of an affine transformation ($$\mathbb{E}[Ax] =
  A\mu$$, $$\mathbb{Var}[Ax] = A \Sigma A^{\intercal}$$), which the Rules of transformations section gives just
  before. Proposed fix: replace the formula block of the Gaussian section with a reference to those rules ("With the
  rules of transformations above, $$y = Ax$$ has the distribution $$p(y) = \mathcal{N}(y \mid A\mu, A \Sigma
  A^{\intercal})$$."); or keep both, as in the author's original reference sheet.

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- Section Maximum Likelihood Estimator, the coordinate ascent paragraph: "When the prior and posterior distributions
  are computationally tractable, the optimization of the parameters that define them can be performed using the
  coordinate ascent method ... each iteration computes the partial derivative of the prior/posterior with respect to
  one model parameter". The section is about the likelihood (the prior only comes with the MAP section), the MLE of
  the post has a closed form, and coordinate ascent is used when there is no closed-form maximum, not because the
  distributions are tractable. Proposed fix: "When the likelihood (or, with a prior, the posterior) has no closed-form
  maximum, its parameters can be optimized with the coordinate ascent method. ... In practice, each iteration computes
  the partial derivative of the log-likelihood (or log-posterior) with respect to one model parameter ...", or move
  the paragraph to the MAP section.
- Formulas: 14 formulas of the post write `log` without a backslash (26 times, e.g. $$log \,\, p(w \mid X, y)$$,
  $$-\frac{N}{2} log \sigma^2$$), which MathJax sets in italics as the product of $$l$$, $$o$$ and $$g$$, while two
  formulas write $$\log$$. Proposed fix: write `\log` in all formulas of the post (a mechanical change, with no change
  of meaning).

### Exponential Family of Distributions (2019-03-20-exponential-family.md)

- `assets/exponential-family/chapter8.pdf` is a copy of chapter 8 of M. I. Jordan's unpublished book draft, which no
  page links but which the site serves to anyone. The course page that lists the chapters of the same draft (Berkeley
  CS 281A, 2003) asks readers not to distribute the chapters or link to them from public sites. The post names the
  chapter without a link, as the course page asks. Proposed fix: delete the copy from the repository, or keep it
  outside the published folders. The other unlinked file of the folder, `lec-10-5.pdf` (scribe notes of lecture 11 of
  Jordan's Stat 241A course, Fall 2004), can stay or go with it.

### Variational Inference (ELBO, CAVI) and Gaussian Mixture Models (2019-07-01-variational-inference-gmm.md)

- End of the ELBO section: "There are several options to solve the previous optimization problem: coordinate descent
  in all variables $$z_i$$ (...), Stochastic Variational Inference (faster), Mean-field approximation, etc." lists the
  mean-field approximation, which is a choice of the variational family, next to two optimization methods. Proposed
  fix: "... coordinate ascent over the variational factors (slow, ...), as in CAVI below, or stochastic optimization,
  as in Stochastic Variational Inference (faster)", with the mean-field approximation introduced as the family that
  both methods optimize over.
- Start of the Mean-field approximation section: "So far, the focus was on a single posterior defined by a family with
  a single distribution $$q$$, whose posterior $$p(z \mid x)$$ is tractable" contradicts the Variational Inference
  section, which approximates an intractable posterior. Proposed fix: "So far, $$q(z)$$ was a single distribution over
  all latent variables. The analysis now restricts it to a family that makes the optimization tractable."

### Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) (2019-10-12-vae-gan.md)

- Optional, one symbol per quantity: $$\sigma$$ is the standard deviation of the encoder in the reparameterization
  trick ($$z = \mu + \sigma \cdot \epsilon$$), a parameter of the prior in the KL paragraph ($$p(z)=\mathcal{N}(\mu=0,
  \sigma^2=1)$$, where $$\mu$$ is reused too) and the sigmoid of the discriminator in the saturation section
  ($$\sigma(a)$$, $$\log(1 - \sigma(a))$$, $$\log \sigma(a)$$). Proposed fix: write the prior as $$p(z)=\mathcal{N}(0,
  I)$$, as in the paper, and the sigmoid as $$\text{sigmoid}(a)$$ in the three formulas of the saturation section; or
  leave it, since each section defines its symbols.

### Data parallelism, sharding and CPU offloading (2023-08-18-data-parallelism.md)

- Hidden text in the source of the post: about 110 lines inside an HTML comment (`<!--` ... `-->`), between the end of
  the section DeepSpeed implementation and the detour on measuring the memory of the parameters, which the site does
  not show. It is an early draft of three sections: mixed precision (with an `fp16` DeepSpeed config), communication
  quantization with ZeRO++, and activation checkpointing (with its `train.py` and `gptlite.py` code). The draft is in
  the first and second person ("we will store", "if you need to"), sets buckets of `4e5` elements while the configs of
  the post use `5e4`, and calls `to_layers()` `lo_layers()`. The front matter has the tag "activation checkpointing",
  which only this hidden text covers. Proposed fix, for the author to pick in the next iteration: delete the hidden
  text and the tag (git keeps the text), or rewrite the three sections in the author's voice, check them against the
  code, and show them on the page.
- The precision of the benchmark: the post says that all implementations use fp32, as both configuration files of the
  post set `"fp16": {"enabled": false}`, while the original text said mixed precision. The history of the
  configuration does not settle it: `fp16` was enabled in the commit of the first results (4ade5d7, 2023-10-08 02:31),
  disabled at 12:02 (75ac452), and the benchmark pictures were updated again at 17:20 (0beddf7). The author thinks
  that the runs used fp16, but is not sure (2026-10-19). Proposed fix, in the next iteration: confirm the precision of
  the runs (from their logs, or by running one configuration again); if they used fp16, write "mixed precision (fp16)"
  in the text, set `fp16` in the configuration files, and add "in fp16" to the caption of the benchmark (diagram
  item); otherwise add "in fp32" to the caption.

### Pipeline parallelism (1F1B, Zero Bubble, DualPipe) (2023-08-30-pipeline-parallelism.md)

- The post keeps the workaround `outputs = outputs.requires_grad_(True)` for the error of the swapaxes lambda. If
  `outputs` has no `grad_fn`, this call lets the backward pass run, but no gradient reaches the layers before it.
  Proposed fix: check that the parameters of the last stage get gradients with the workaround (`p.grad` after
  `train_batch()`), and remove the sentence if they do not.
- Optional, results, memory per GPU: the first GPU used more memory than the others, also with the deep benchmark
  model. The text now says that 1F1B explains only a few MB of this gap for that model, and the cause of the rest is
  not known. Fix: measure the peak memory of every GPU (`torch.cuda.max_memory_allocated()` on each rank, and
  `nvidia-smi` for the CUDA contexts on GPU 0), and give the numbers and the cause in the post.

### Variable sequence lengths, curriculum learning, adaptive batch size and LR (2024-05-10-training-variable-length-curriculum.md)

- The pipeline parallelism section says that its picture compares DDP and pipeline parallelism "for 2 processes and 2
  gradient accumulation steps (4 micro-batches total)". For DDP, 2 processes with 2 accumulation steps give 4
  micro-batches. For pipeline parallelism, 2 processes are the 2 stages of one pipeline, which processes every
  micro-batch of the batch, so 4 micro-batches need 4 accumulation steps, or 2 data-parallel pipelines of several GPUs
  each. Fix: "for 2 data-parallel ranks (each a whole pipeline in the right panel) and 2 gradient accumulation steps
  (4 micro-batches total)", or "for a batch of 4 micro-batches" without the processes.

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

- The Animate Anyone entry of the further reading repeats, with the same shapes, the definition of the spatial and
  temporal attentions given in the first item of the section "Video diffusion and 3D attention" ("a spatial attention
  that converts an input $$B \times T \times H \times W \times C$$ into ..."). Fix: shorten it to "Spatio-temporal
  attention is achieved by a spatial attention followed by a temporal attention, the first approach of the section
  Video diffusion and 3D attention.", or keep it so that the entry reads on its own.

### Prefill vs decode, roofline, KV cache, batching and disaggregation (2025-06-07-inference-prefill-decode-roofline-cache.md)

- Optional: the heading "Vector Sum (real use case)" names a worked example whose execution time is assumed ("If the
  GPU runs the vector loop above in exactly 0.0001 seconds"), not measured. Fix: "Vector sum (example)".

### Kernel Fusion for MoE Expert Parallelism (2026-10-09-kernel-fusion-expert-parallelism.md)

- The paper has not measured the decode step (the right panel of its runtime figure, and its decode timeline), the
  partial collective operations (its table of collective and partial collective results) and the number of microbatches in
  flight with the memory each one takes. The post shows the two decode placeholders with a note in their captions,
  and leaves the rest out. Fix: add these results to the post once the paper has them.
- The SHArP reference of the section In-network computing links to a PDF on the website of a network vendor
  (network.nvidia.com), in one of the two posts that follow "The author's papers". The text names no vendor, but the
  link does. Fix: link the paper's page in IEEE Xplore instead (Graham et al., "Scalable Hierarchical Aggregation
  Protocol (SHArP): A Hardware Architecture for Efficient Data Reduction", COMHPC 2016), after checking its address;
  or keep the link if the user accepts a vendor address for cited work.
- Results, the AllGather-ReduceScatter backend: "the results have the same shape and are slightly smaller:
  MegaKernel+INC improves on the Baseline by 8.9% to 18.5%, and on MegaKernel by up to 10.6%". The gain over the
  Baseline is about that of All-to-All (8.7% to 19.2% from the values of `ep_results.svg`), and the gain over
  MegaKernel is larger than with All-to-All (up to 7.8%), so "slightly smaller" does not say what is smaller. Fix:
  check the numbers in the paper and write, e.g., "the results have the same shape: MegaKernel+INC improves on the
  Baseline by 8.9% to 18.5%, about as much as with All-to-All, and on MegaKernel by up to 10.6%, more than with
  All-to-All".
- A new version of the paper is in the paper's repository. Proposed fix, in iteration 3, after the merge of iteration
  2: update the text of the post where the new version of the paper changes it. The new pictures of the paper come
  with the picture work, which owns the figure files.

### Knowledge distillation, speculative decoding and multi-token prediction (_drafts/2026-03-17-inference-distillation-speculative-decoding.md)

- The usage comment of the Medusa code ("prefill the prompt: [1, S], with the causal_mask of the post on attention")
  relies on `causal_mask` from the post on attention, which the text no longer links (#69). Fix: in the code pass,
  write the mask in the comment as `torch.ones(S, S, dtype=torch.bool).tril()` (True where a token may attend, as in
  `tree_attention_mask`) and drop "of the post on attention", or keep the mention without a link.

### Pruning and quantization (_drafts/2026-04-22-inference-pruning-quantization.md)

- The draft describes the experiments of `inference/pruning.py` and `inference/quantization.py` on GPTlite (pruning
  with distillation, Wanda, ShortGPT, weight-only and weight-activation quantization, KV cache quantization), but
  shows none of their results. Fix: run the two scripts and add a table of their outputs (parameters, size,
  validation loss, time per token, matching tokens), or present the code without the comparisons.
- Optional: once both drafts are published, "knowledge distillation" in the introduction could link to the
  distillation post instead of Wikipedia.

### Publications bookmark (publications.md)

- DeepSeek-V2 (entry 038): the last sentence of item 2 of DeepSeekMoE, "Finely segmenting the experts into $$mN$$ ones
  and activating $$mK$$ from them allows for a more flexible combination of activated experts", restates the sentence
  before it ("far more *combinations* of experts per token"). Proposed fix: remove the sentence (a text pass that
  removes math must report it as a formula change, so iteration 2 left it).
- Optional, ZeRO-Offload++ (entry 052): the entry title links the blog on the branch `offloadpp-news` of
  microsoft/DeepSpeed, and the caption links it on `master`; both copies exist today, and the repository has moved to
  deepspeedai/DeepSpeed. Proposed fix: link
  https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-offloadpp in both places, after a check that it
  opens.
- RoFormer (entry 076) and LoRA (entry 079): the formulas write the same quantities with Unicode mathematical italic
  letters ($$𝜃_𝑝$$, $$Δ𝑊$$, $$𝑊$$) and with TeX letters ($$\theta$$, $$ΔW$$, $$W$$). In RoFormer, $$R(x)$$ is the
  rotated pair while $$R_m$$ is the rotation matrix of position $$m$$, and the angle $$𝜃_𝑝 = p \, w_i$$ depends on the
  pair $$i$$, which its symbol does not show. Proposed fix: write the letters in TeX ($$\theta$$, $$W$$, $$\Delta
  W$$), write the angle of the $$i$$-th pair at position $$p$$ as $$\theta_{p,i} = p \, w_i$$, and the rotation of a
  pair as $$R_p x$$, so that $$R_p$$ is the rotation matrix of position $$p$$ in both formulas.

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

### Pruning and quantization (_drafts/2026-04-22-inference-pruning-quantization.md)

- Optional: the other posts of the series name their methods in the title ("Faster attention (GQA, FlashAttention,
  sparse, linear and latent attention)", "Prefill vs decode, roofline, KV cache, batching and disaggregation"). Fix:
  "Pruning and quantization (Wanda, Minitron, GPTQ, AWQ, FP8, FP4)", 84 characters with "Efficient Inference: ".
