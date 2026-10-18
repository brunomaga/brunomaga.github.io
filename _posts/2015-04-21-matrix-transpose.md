---
layout: post
title:  "Distributed Matrix Transpose Algorithms"
categories: ["Distributed Algorithms"]
tags: ["MPI", "linear algebra", "matrix transpose", "all-to-all"]
---

The [transpose][wiki-transpose] of a matrix $$M$$ swaps its rows and columns, $$M^T_{ij} = M_{ji}$$, and it is an important operation in large-scale linear solvers and graph algorithms. For a graph, the transpose reverses the direction of every edge, and it gives the orthogonal view of the graph's connectivity: instead of the edges that leave each vertex, the edges that arrive at it. On a single compute node, the transpose may seem trivial, as it only moves data around in memory. It becomes a hard problem when the matrix is distributed across many compute nodes, because every node then has to exchange data with every other node, and the matrix is usually sparse, with a different number of values in each row. The method described below solves this problem for distributed sparse matrices, even with several values per cell, with only three to five collective calls of the [Message Passing Interface][mpi] (MPI).

[wiki-transpose]: https://en.wikipedia.org/wiki/Transpose
[mpi]: https://en.wikipedia.org/wiki/Message_Passing_Interface

## Graphs and adjacency matrices

A [graph][graph] $$G = (V, E)$$ is a data structure made of a set of nodes, or **vertices**, $$V$$, and a set of connections between them, the **edges** $$E$$. An edge $$(i, j, e) \in V \times V \times T$$ connects the vertices $$i$$ and $$j$$, and carries a value $$e$$ of some type $$T$$, such as a weight or a label. In a directed graph, the pair $$(i, j)$$ is ordered and the edge goes from $$i$$ to $$j$$, while in an undirected graph it goes both ways. The connectivity of the graph is described by its [adjacency matrix][adjacency] $$M$$, a square matrix of size $$\vert V \vert \times \vert V \vert$$ with one row and one column per vertex, where $$M_{ij} = e$$ for every edge $$(i, j, e)$$, and also $$M_{ji} = e$$ if the graph is undirected. All other cells are empty.

On a single compute node, the transpose follows directly from its mathematical definition, $$M^T_{ij} = M_{ji}$$: every cell moves from position $$(i, j)$$ to position $$(j, i)$$. The matrix $$M^T$$ is then the adjacency matrix of the [transpose graph][transpose-graph] $$G^{\star} = (V, E^{\star})$$, which has the same vertices and all its edges reversed: $$E^{\star} = \{ (j, i, e) : (i, j, e) \in E \}$$.

[graph]: https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)
[adjacency]: https://en.wikipedia.org/wiki/Adjacency_matrix
[transpose-graph]: https://en.wikipedia.org/wiki/Transpose_graph

## Distributed layout

In a distributed memory system, the vertices are split across $$R$$ compute nodes, or **ranks**, numbered from 0, and each rank can directly access only the data in its own memory. Rank $$r$$ holds the subgraph $$G_r = (V_r, E_r)$$: a subset of the vertices, $$V_r$$, and only their outgoing edges, $$E_r = \{ (i, j, e) \in E : i \in V_r \}$$. In matrix terms, rank $$r$$ holds the rows of $$M$$ whose indices are in $$V_r$$. Every vertex belongs to exactly one rank, i.e. the sets $$V_r$$ are disjoint and together cover $$V$$, and since every edge belongs to the rank of the vertex it leaves, the same holds for the sets of edges $$E_r$$.

The vertices do not move during the transpose: they keep their rank and their place in memory, so $$V$$ is the same for $$G$$ and $$G^{\star}$$. Only the edges move. After the transpose, rank $$r$$ must hold the rows of $$M^T$$ with indices in $$V_r$$, i.e. the edges $$E^{\star}_r$$ that arrive at its vertices.

For a dense matrix, this is a well-known problem with a ready-made solution, the `MPI_Alltoall` collective call. If every rank holds the same number of rows, each rank cuts its rows into $$R$$ blocks of columns and sends block $$s$$ to rank $$s$$, and a local transpose of every received block then completes the operation. The call takes as input (output) an array of elements to send (receive), the number of elements sent to (received from) each rank, and their datatype.

{: .picture}
<img width="71%" height="71%" src="{{ site.assets }}/matrix-transpose/all_to_all.svg">

{: .caption}
`MPI_Alltoall` on 6 processes: block $$j$$ of process $$i$$ goes to process $$j$$, as its block $$i$$. The blocks of process A (red frame) become the first block of every process. Adapted from: [MPI Tutorial, Shao-Ching Huang, IDRE High Performance Computing Workshop, 2013](https://www.slideshare.net/slideshow/mpi-49244468/49244468)

For sparse matrices, the problem is harder: every row holds a different number of values, so a rank does not know in advance how many values it will receive from each other rank. A common format for the distributed storage of a sparse matrix is the [compressed sparse row][csr] (CSR) format, which keeps only the nonzero cells. Each rank serializes its rows into four arrays: `row_count`, the number of rows; `counts`, the number of nonzeros in each row; `disps`, the column index of each nonzero; and `cells`, their values. This variant of CSR stores the number of nonzeros of each row instead of the usual row offsets. The following picture shows a sample graph with 6 vertices and 12 edges, labeled with letters from A to T, its adjacency matrix, and its CSR arrays on 3 ranks:

[csr]: https://en.wikipedia.org/wiki/Sparse_matrix

{: .picture}
<img width="99.1%" height="99.1%" src="{{ site.assets }}/matrix-transpose/crs_format.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

## Local transpose and view swap

The key idea of the method is to split the transpose into two simpler operations: a **local transpose**, which every rank performs on its own data without any communication, and a **view swap**, which exchanges data between the ranks without changing the matrix. Together, the two operations move a distributed matrix through the four layouts illustrated below.

{: .picture}
<img width="96.3%" height="96.3%" src="{{ site.assets }}/matrix-transpose/layouts.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

In layout (a), the ranks hold $$M$$ by rows. The local transpose turns it into layout (b), where the ranks hold $$M^T$$ by columns, and the view swap turns it into layout (c), where they hold $$M$$ by columns. Applying the other operation to (b) or to (c) gives layout (d), $$M^T$$ by rows, which is the distributed transpose. To see why this works, the two operations need a precise definition, and for that, two operations on matrices come first. The **horizontal concatenation** $$M \| N$$ of the matrices $$M_{n \times m}$$ and $$N_{n \times m'}$$ places them side by side, in a matrix of size $${n \times (m+m')}$$:

$$
(M \| N)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    M_{i \,\, j} & \text{, if } j \leq m\\
    N_{i \,\, j-m} & \text{, otherwise}
  \end{array}\right.
$$

Similarly, the **vertical concatenation** $$M_{n \times m} // N_{n' \times m}$$ stacks them, in a matrix of size $${(n+n') \times m}$$:

$$
(M // N)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    M_{i \,\, j} & \text{, if } i \leq n\\
    N_{i-n \,\, j} & \text{, otherwise}
  \end{array}\right.
$$

Transposing a concatenation turns it into the other kind, with every block transposed: $$(M \| N)^T = M^T // N^T$$ and $$(M // N)^T = M^T \| N^T$$. This simple identity is at the heart of the method.

A **view** describes how a matrix is split across the ranks. In the **row view**, rank $$r$$ holds a block of rows $$M_r$$, and the matrix is the vertical concatenation of the blocks, $$M = M_0 // M_1 // \cdots // M_{R-1}$$. In the **column view**, rank $$r$$ holds the block of columns $$M_r$$ with the same indices, and $$M = M_0 \| M_1 \| \cdots \| M_{R-1}$$.

The **local transpose** transposes the block of every rank. Thanks to the identity above, this transposes the whole matrix and flips its view at the same time: for a matrix in row view, $$(M_0 // \cdots // M_{R-1})^T = M_0^T \| \cdots \| M_{R-1}^T$$, so after the local transpose the ranks hold $$M^T$$ in column view. In the same way, from the column view, the local transpose gives $$M^T$$ in row view. Applying it twice gives back the original blocks, so the local transpose is an [involution][involution], i.e. a function that is its own inverse.

The **view swap** does the opposite: it keeps the matrix and changes its view, from the row view to the column view or back. This requires communication, since every rank must send each part of its rows to the rank that owns the corresponding columns. Swapping the view twice gives back the initial layout, so the view swap is an involution too.

[involution]: https://en.wikipedia.org/wiki/Involution_(mathematics)

## The view swap with MPI

The view swap is where all the communication happens, and it needs some care. From the point of view of a single rank, the layout of the matrix after the swap is unknown, because no rank knows which rows the other ranks hold. The first step is therefore an `MPI_Allgather` of the number of rows of each rank. From these counts, every rank computes the interval of row indices held by each rank $$r$$:

$$
\big[ \sum^{r-1}_{s=0} \vert V_s \vert, \,\, \sum^{r}_{s=0} \vert V_s \vert \big)
$$

With these intervals, a rank can map every column index of its data to the rank that will own it after the swap. The swap itself then takes two communication steps:
1. a dense transpose of the counts: each rank knows how many elements it sends to each rank, and an `MPI_Alltoall` of this dense $$R \times R$$ matrix of counts tells each rank how many elements it receives from each rank;
2. a selective scatter-gather, `MPI_Alltoallv`, which sends every element, together with its row and column indices, to its new rank, into receive buffers sized by the counts of the first step.

{: .picture}
<img width="98%" height="98%" src="{{ site.assets }}/matrix-transpose/matrix-transposition-crs.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

A final sort of the received elements by their row and column indices rebuilds the CSR arrays. The distributed transpose of a CSR matrix therefore takes only three collective calls: one `MPI_Allgather` for the row intervals, and one `MPI_Alltoall` and one `MPI_Alltoallv` in the view swap.

## Correctness

Is the method guaranteed to work for any matrix? With the definitions above, the proof is short. The distributed transpose is the composition of the two operations. Starting from $$M$$ in row view, the local transpose gives $$M^T$$ in column view, and the view swap then gives $$M^T$$ in row view, which is the distributed transpose of $$M$$. In the other order, the view swap gives $$M$$ in column view, and the local transpose then gives $$M^T$$ in row view, the same result. The two operations therefore commute:

$$
Transpose = ViewSwap \cdot LocalTranspose = LocalTranspose \cdot ViewSwap
$$

A composition of two involutions that commute is itself an involution ([Elements of Operator Theory, Carlos S. Kubrusly, Springer 2011][kubrusly2011elements]). With $$S$$ the view swap, $$L$$ the local transpose and $$I$$ the identity, the proof takes one line: $$(S \cdot L) \cdot (S \cdot L) = S \cdot (L \cdot S) \cdot L = S \cdot (S \cdot L) \cdot L = (S \cdot S) \cdot (L \cdot L) = I$$. This property matters in practice. The transpose can be applied any number of times, and a second transpose always returns the original matrix in its original layout, so the CSR representation of a graph keeps its integrity.

[kubrusly2011elements]: https://link.springer.com/content/pdf/10.1007/978-0-8176-4998-2.pdf

## Matrices with several values per cell

In real-life problems, a pair of vertices is often connected by more than one edge, which makes the graph a [multigraph][multigraph]. One example is a matrix of the common interests of pairs of users in a social network: some pairs of users share hundreds of interests, while others share barely any. Another example is a map of cross-references between web pages, which may store not only the number of hyperlinks between two pages, but also information about every link. Each cell of the adjacency matrix then holds a list of values, and the lists have different lengths.

[multigraph]: https://en.wikipedia.org/wiki/Multigraph

Such matrices need a data format that can hold many values per cell. The **extended CSR** format, or XCSR, adds one array to CSR, `cell_counts`, which holds the number of values in each cell, i.e. the number of edges between each pair of vertices:

{: .picture}
<img width="98.8%" height="98.8%" src="{{ site.assets }}/matrix-transpose/xcrs_format.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

The transpose then takes three steps. The first one is the **local transpose**. Each rank collects the metadata of its own block, one triplet (row index, column index, number of values) per cell, which gives the position and the size of every cell. It then transposes its data locally, by sorting the triplets, and the values with them, by column and then by row index:

{: .picture}
<img width="99.1%" height="99.1%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_1.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

The second step starts the **view swap**, with the metadata. The ranks exchange their triplets in an all-to-all communication, which sends each triplet to the rank that owns its cell in the transposed matrix. The local transpose has already sorted each block by columns, i.e. by rows of the transposed matrix, so the triplets are already grouped by destination rank. More importantly, the triplets all have the same size, so together they form a distributed sparse matrix with one value per cell, and the CSR view swap of the previous section can exchange them: an `MPI_Alltoall` of the triplet counts, then an `MPI_Alltoallv` of the triplets. After the exchange, every rank holds the metadata of its part of the transposed matrix, but not the values yet:

{: .picture}
<img width="97.7%" height="97.7%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_2.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

The third step completes the view swap by sending the values, guided by the metadata. Each rank counts the number of values it sends to each rank, and an `MPI_Alltoall` transposes this dense matrix of counts, so that every rank learns how many values it receives from each rank. These counts give the size of the receive buffers, and an `MPI_Alltoallv` delivers the values. Finally, each rank sorts the values it received, which arrive in column order, by their row and column indices in $$M^T$$, which gives the final matrix:

{: .picture}
<img width="99%" height="99%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_3.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

With the `MPI_Allgather` of the row counts, the XCSR transpose takes five collective calls in total: one `MPI_Allgather`, two `MPI_Alltoall` and two `MPI_Alltoallv`. It keeps the involution property of the CSR transpose, and since XCSR can hold any number of values per cell, the same method transposes any sparse matrix, any number of times, without losing data.

## Further details

More details, the implementation and the benchmarks are in the <a href="{{ site.assets }}/matrix-transpose/matrix-transposer.pdf">original publication (preprint)</a>. The <a href="{{ site.assets }}/matrix-transpose/matrix-transposer.zip">source code</a> is also available for download.
