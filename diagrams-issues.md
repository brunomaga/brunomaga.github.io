# Picture issues

Pictures that are wrong or disagree with the post text, found by the text rewrite, which does not change pictures.
Each item names the picture (under `assets/`), the problem and a proposed fix. Items marked "Optional" improve a
picture that is not wrong.

## Distributed Matrix Multiplication and Database Joins (2014-04-02-distributed-matmul-dbs.md)

- `distributed-matmul-dbs/joins.svg`, `layout_columns.svg`: the `NULL` cells are filled with `#f5f5f5`, which reads
  as white on the page, while the caption of `joins.svg` says "Gray cells are `NULL`". Fix: a visible gray from the
  palette for `NULL` cells in all join pictures.

## Distributed Sorting Algorithms (2014-06-21-distributed-sort.md)

- `distributed-sort/odd_even_sort.svg`: the ranks hold 3, 5, 3, 4 and 5 elements. Odd-even sort is only guaranteed
  to finish in p phases when all ranks hold the same number of elements (a random test with unequal counts fails
  in about a third of the cases); this example happens to finish in 5 phases. Fix: 4 elements per rank.
- `distributed-sort/odd_even_sort.svg`, `sample_sort.svg`: the labels say "processor", "CPU", "cpu" and "root node",
  and "N" for the number of processors, while the text says "rank" and "p". Fix: "rank", "rank 0" and "p" in
  both pictures.
- `distributed-sort/sample_sort.svg`, step 3: "takes N samples" while only the first N-1 split the value range (the
  last is the maximum). Fix: "takes every N-th sample as a splitter".

## Distributed Matrix Transpose Algorithms (2015-04-21-matrix-transpose.md)

- `matrix-transpose/xcrs_transpose_3.svg`: the brackets under the arrays say "CPU 0", "CPU 1", "CPU 2", while every
  other picture of the post says "rank". Fix: "rank 0", "rank 1", "rank 2".
- `matrix-transpose/crs_format.svg`, `xcrs_format.svg`: the array of column indices is labeled `disps`, MPI's name
  for displacements, which reads as offsets. Optional fix: `col_idx`, as in the matmul post.

## Distributed Load Balancing and Spatial Slicing (2015-12-01-slicing.md)

- `slicing/balance.svg`: the observation says "2 collective operations (MPI_Alltoall, MPI_Alltoallv)", but step 1
  (every rank sends its count to all ranks) is an `MPI_Allgather`. The code (`DistributedLoadBalancing.cxx`) calls
  `MPI_Allgather`, `MPI_Alltoall` and `MPI_Alltoallv`. Fix: "2 collective operations (MPI_Allgather,
  MPI_Alltoallv)", since the receive counts follow from the gathered counts, or list the three calls of the code.
- `slicing/balance.svg`: the extra element goes to the last rank (6, 6, 7), while the code gives it to the first
  ranks (7, 6, 6). Optional fix: 7, 6, 6.
- `slicing/split.svg`: the data is unsorted, with 2, 2, 3, 2, 3, 2, 1 and 3 elements per rank. In SBS, the split
  follows the sort and the balance, so the data is sorted and the counts differ by at most one. The K=3 case splits
  8 ranks into 3, 3 and 2, which SBS does not allow: the number of subnetworks must divide the number of ranks (the
  code warns otherwise). Fix: sorted data with 2 or 3 elements per rank, and K=2 and K=4 only.
- `slicing/str_steps.svg`: the key numbers the ranks 1 to 4, while `balance.svg`, `split.svg` and `sbs.svg` number
  them from 0. Fix: ranks 0 to 3.
- `slicing/sbs.svg`: the labels say "CPU" and "N" for the number of CPUs, while the text says "rank" and "p". Fix:
  "rank" and "p". All data and cut points of the 18-object example are correct.
- `slicing/mini-*.svg`: the four layouts have 10 (single axis), 35 (NUG, 5 x 7), 16 (ORB) and 35 (STR, 5 x 7)
  slices. Optional fix: 16 slices in all four (4 x 4 for NUG and STR), so that they compare on equal terms.

## The Leaky Integrate-and-Fire Neuron Model and The Brunel Network (2016-02-08-lif-brunel.md)

- `lif-brunel/raster_plot_g4.5_f0.9.svg`, `raster_plot_g5_f2.svg`, `raster_plot_g3_f2.svg`: the plots do not name
  their state. The post used to call them "slow", "regular" and "fast" dynamics, but their parameters are those of
  Brunel's states SI slow (g = 4.5, eta = 0.9), AI (g = 5, eta = 2) and SR (g = 3, eta = 2); the text now uses these
  names. Fix: a title per plot, e.g. "SI slow (g = 4.5, eta = 0.9)".
- Same three plots: the label "time interval (ms)" overlaps the bars at the right end of the lower panel, and
  "spikes count" reads better as "spike count (per ms)". Fix: move the label below the axis.
- `lif-brunel/neuron_plot_equations.svg`: the legend says "Analytic", the text says "exact"; the legend text is too
  small to read at page width. Optional fix: "Exact (fixed dt = 0.1 ms)", "Exact (variable dt)", and a larger font.
- `lif-brunel/action_potential.gif`: a "MakeAGIF.com" watermark in the bottom right corner. Optional fix: crop it.

## Numerical Resolution of the Electrical Activity of Detailed Neuron Models (2016-08-15-neuron-electrical-activity.md)

- `neuron-electrical-activity/action_potential.svg`: the caption says "Adapted from: unknown". Fix: name the source of
  the figure, or redraw it in the site style (rest at -70 mV, threshold at -55 mV, peak at +30 mV, as now).
- `neuron-electrical-activity/neuron_model_scales.svg`, reaction-diffusion panel: the arrow from PDE to cAMP reads as
  "PDE produces cAMP", but phosphodiesterase (PDE) breaks cAMP down. Optional fix: an inhibition arrow (flat head)
  from PDE to cAMP.

## Variable Timestep Simulation of the Electrical Activity of Neurons (2017-02-17-neuron-variable-timestep.md)

- `neuron-variable-timestep/cvode_pulse_currents_table.svg`, `cvode_dependency_on_variation_solution.svg` and
  `cvode_dependency_on_events_arrival.svg`: the currents are in mA and µA (2.7 mA, "threshold current (0.206mA)",
  "0.1µA" to "1mA"), as in the preprint. NEURON's `IClamp` amplitude is in nA, and the threshold current of a cortical
  pyramidal cell model is of the order of 0.1 to 1 nA, so 0.206 mA is likely 0.206 nA. The text gives no current
  values. Fix, after a check of the simulation scripts: mA to nA and µA to pA in the labels.
- Same pictures, and `l5_neuron_pulse1_3ma_100ms_cvode_vs_backward_euler.svg`, `l5_neuron_pulse_1000ms_results.svg`
  and the six `neuron_*.svg`: a dot is the thousands separator ("48.001 steps", "9.750 steps", "1.512 steps",
  "100.001 steps", "200.001 steps", "6.240 steps", axis ticks "1.000" and "1.100"). English text reads these as
  decimals. Fix: a comma or no separator ("48,001", "1,000").
- `neuron_208223.svg` and `neuron_87784.svg`: the histogram axis says "interval events (ms)"; the other four
  `neuron_*.svg` say "events interval (ms)". Fix: "interval between events (ms)" in all six.
- `lvardt_exec_time_comparison.svg`: the curve labels "dt=0.0025" and "dt=0.025" have no unit. Fix: "dt = 0.0025 ms"
  and "dt = 0.025 ms". Its caption "source: Lytton et al. 2005" names a two-author paper. Fix: "Source: Lytton and
  Hines (2005)".
- `benchmark_two_rows_2.svg`: the legend entry "(2a) BSP Deriv. Implicit†" has no key for the dagger. The preprint's
  caption defines it: Newton iterations solve each mechanism implicitly, but the coupling between mechanisms stays
  explicit. Fix: a key line in the legend, "† implicit per mechanism, explicit across mechanisms".
- `local_variable_timestep_six_cell_example.svg`: the caption cites the NEURON book; the figure matches Fig. 1 of
  Lytton and Hines (2005). Optional fix: cite the paper in the caption.

## Closed-form Linear Regression and Matrix Factorization, and loss functions (2017-08-17-linear-regression-matrix-factorization.md)

- `linear-regression-matrix-factorization/matrix-factorization.svg`: W and Z^T have swapped orientations. W is drawn as
  a horizontal strip with the movie index d along it, and Z^T as a vertical strip with the user index n along it. In
  X, d indexes the rows (movies) and n the columns (users), so W (D x K) is a tall block left of X, with row d
  aligned to movie d, and Z^T (K x N) a wide block above X, with column n aligned to user n. Fix: redraw W as a 7 x K
  block left of X and Z^T as a K x 7 block above X, with K = 2 or 3 and K marked on both. Optional: "x_dn" in lower
  case, as in the text.
- `linear-regression-matrix-factorization/coordinate_descent.svg`: the steps stop short of the minimum along each
  coordinate. From (-0.5, -1), the minimum along y is at y = 0.6x = -0.3, but the path stops at y = -0.4; the next
  minimum along x is at x = 0.6y, but the path stops at x = -0.34. The text describes exact minimization along each
  coordinate. Fix: draw the exact path (-0.5, -1), (-0.5, -0.3), (-0.18, -0.3), (-0.18, -0.108), (-0.065, -0.108),
  (-0.065, -0.039), ..., where each sweep scales the point by 0.36.

## Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- `unsupervised-learning/ica_whitening.svg`: the panel label is "(d) whitening (z1, z2)", after "(a) source" and
  "(b) observed"; no panel is labelled (c). Fix: "(c) whitening (z1, z2)".
- `unsupervised-learning/kohonen2.svg` and its caption: the picture uses v for the input, s for the winner neuron and
  w_s for its weights; the text uses x_i, j* and w_j. The caption defines its own symbols, so the picture is not
  wrong. Optional fix: relabel v, s and w_s as x, j* and w_j*.

## Deep Neural Networks, backpropagation, autodiff, dropout, CNNs and embeddings (2018-03-27-deep-neural-networks.md)

- `deep-neural-networks/fully_sparse_connectivity.svg`: the picture has no caption, and it reproduces Figure 9.2 of
  Goodfellow, Bengio and Courville, Deep Learning (2016). It uses s for the outputs; the text uses x^(1). Fix: a
  caption "Sparse (top) and full (bottom) connectivity; the outputs that depend on x3 are highlighted. Adapted from:
  Goodfellow et al., Deep Learning (2016), Fig. 9.2". Optional: relabel s as x^(1).
- `deep-neural-networks/densenet.svg`: the caption says "An example of a CNN with skip connections (DenseNet)", but
  the picture shows one dense block of 5 layers with a growth rate of 4, not a full CNN. Fix: "A dense block of
  DenseNet, with 5 layers and a growth rate of 4: each layer takes the feature maps of all preceding layers as input.
  Adapted from: Densely Connected Convolutional Networks".
- `deep-neural-networks/dropout.svg`: the picture has no caption. Optional fix: "A network without dropout (left) and
  with a dropout rate of 0.5 (right); crossed units are dropped."

## Mathematics for ML (2018-04-20-algebra-for-ml.md)

- `algebra-for-ml/newton_method.gif`: the legend is in German ("Funktion", "Tangente"). Fix: "function" and
  "tangent".
- `algebra-for-ml/svd_example.svg`: the right half is the "Lenna" test image, which IEEE no longer allows in its
  publications since April 1, 2024. Optional fix: the same experiment (all 256, then 20, 50 and 85 singular values)
  on another test image.
- `algebra-for-ml/mml_eigenvalues_and_eigenvectors.svg`: the right third of the picture is the book's caption text,
  in a small font at 65% of the page width. Optional fix: remove the text from the picture, widen the plots, and put
  a short description in the page caption, e.g. "Five linear mappings of 400 points (left to right: input,
  eigenvectors scaled by their eigenvalues, output)".
