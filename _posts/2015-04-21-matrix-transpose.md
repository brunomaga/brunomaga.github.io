---
layout: post
title:  "Distributed Matrix Transpose Algorithms"
categories: ["Distributed Algorithms"]
tags: ["MPI", "linear algebra", "matrix transpose", "all-to-all"]
---

The [transpose][wiki-transpose] $$M^T$$ of a matrix $$M$$, with $$M^T_{ij} = M_{ji}$$, is a core operation of linear solvers and graph algorithms. On one compute node, it only reorders data in memory. On distributed memory, every rank exchanges data with every other rank. The method below transposes distributed sparse matrices, including matrices with several values per cell, with three to five [MPI][mpi] collective calls.

[wiki-transpose]: https://en.wikipedia.org/wiki/Transpose
[mpi]: https://en.wikipedia.org/wiki/Message_Passing_Interface

## Graphs and adjacency matrices

A [graph][graph] $$G = (V, E)$$ has a set of vertices $$V$$ and a set of edges $$E$$. An edge $$(i, j, e) \in V \times V \times T$$ connects vertices $$i$$ and $$j$$, and holds a value $$e$$ of type $$T$$. In a directed graph, the pair $$(i, j)$$ is ordered. The [adjacency matrix][adjacency] $$M$$ of $$G$$ is the $$\vert V \vert \times \vert V \vert$$ matrix with $$M_{ij} = e$$ for every edge $$(i, j, e)$$, and also $$M_{ji} = e$$ if $$G$$ is undirected. All other cells are empty.

The transpose $$M^T$$ is the adjacency matrix of the [transpose graph][transpose-graph] $$G^{\star} = (V, E^{\star})$$, with all edges reversed: $$E^{\star} = \{ (j, i, e) : (i, j, e) \in E \}$$. The vertices do not change.

[graph]: https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)
[adjacency]: https://en.wikipedia.org/wiki/Adjacency_matrix
[transpose-graph]: https://en.wikipedia.org/wiki/Transpose_graph

## Distributed layout

The vertices are split across $$R$$ ranks, numbered from 0. Rank $$r$$ holds the vertices $$V_r$$ and their outgoing edges $$E_r = \{ (i, j, e) \in E : i \in V_r \}$$, i.e. the rows of $$M$$ with indices in $$V_r$$. The sets $$V_r$$ are disjoint and cover $$V$$, so the sets $$E_r$$ are disjoint and cover $$E$$. A rank can only access its own rows.

If every rank holds the same number of rows of a dense matrix, `MPI_Alltoall` performs the distributed transpose. Each rank cuts its rows into $$R$$ blocks of columns, and sends block $$s$$ to rank $$s$$. A local transpose of every received block completes the operation.

{: .picture}
<img width="71%" height="71%" src="{{ site.assets }}/matrix-transpose/all_to_all.svg">

{: .caption}
`MPI_Alltoall` on 6 processes: block $$j$$ of process $$i$$ goes to process $$j$$, as its block $$i$$. The blocks of process A (red frame) become the first block of every process. Adapted from: [MPI Tutorial, Shao-Ching Huang, IDRE High Performance Computing Workshop, 2013](https://www.slideshare.net/slideshow/mpi-49244468/49244468)

A sparse matrix is stored in a [compressed sparse row][csr] (CSR) format. Each rank stores its rows in four arrays: `row_count`, the number of rows; `counts`, the number of nonzeros per row; `disps`, the column index of each nonzero; and `cells`, the values. This variant of CSR stores the number of nonzeros per row instead of the row offsets. The figure shows a graph with 6 vertices and 12 edges, labeled with letters from A to T, its adjacency matrix, and the CSR arrays on 3 ranks:

[csr]: https://en.wikipedia.org/wiki/Sparse_matrix

{: .picture}
<img width="99.1%" height="99.1%" src="{{ site.assets }}/matrix-transpose/crs_format.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

## Local transpose and view swap

$$M \| N$$ denotes the **horizontal concatenation** of the matrices $$M_{n \times m}$$ and $$N_{n \times m'}$$, a matrix of size $${n \times (m+m')}$$:

$$
(M \| N)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    M_{i \,\, j} & \text{, if } j \leq m\\
    N_{i \,\, j-m} & \text{, otherwise}
  \end{array}\right.
$$

Similarly, the **vertical concatenation** $$M_{n \times m} // N_{n' \times m}$$ is a matrix of size $${(n+n') \times m}$$:

$$
(M // N)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    M_{i \,\, j} & \text{, if } i \leq n\\
    N_{i-n \,\, j} & \text{, otherwise}
  \end{array}\right.
$$

The transpose turns one concatenation into the other: $$(M \| N)^T = M^T // N^T$$ and $$(M // N)^T = M^T \| N^T$$.

A **view** is the way a matrix is split across ranks. In the **row view**, rank $$r$$ holds a block of rows $$M_r$$, and $$M = M_0 // M_1 // \cdots // M_{R-1}$$. In the **column view**, rank $$r$$ holds the block of columns $$M_r$$ with the same indices, and $$M = M_0 \| M_1 \| \cdots \| M_{R-1}$$. Two operations change the layout of a distributed matrix:
- the **local transpose**: every rank transposes its own block, without communication. For a matrix in row view, $$(M_0 // \cdots // M_{R-1})^T = M_0^T \| \cdots \| M_{R-1}^T$$, so the ranks then hold $$M^T$$ in column view. From the column view, the local transpose gives $$M^T$$ in row view;
- the **view swap**: the ranks exchange data, so that the same matrix moves from the row view to the column view, or back.

{: .picture}
<img width="96.3%" height="96.3%" src="{{ site.assets }}/matrix-transpose/layouts.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

The four layouts are: (a) $$M$$ in row view; (b) its local transpose, $$M^T$$ in column view; (c) its view swap, $$M$$ in column view; (d) $$M^T$$ in row view, the distributed transpose.

## The view swap with MPI

A rank does not know which rows the other ranks hold. So an `MPI_Allgather` first collects the row counts, from which every rank computes the row interval of each rank $$r$$:

$$
\big[ \sum^{r-1}_{s=0} \vert V_s \vert, \,\, \sum^{r}_{s=0} \vert V_s \vert \big)
$$

These intervals map every column index to its rank after the swap. The swap then takes two communication steps:
1. an `MPI_Alltoall` transposes the dense $$R \times R$$ matrix of element counts, so that each rank knows how many elements it receives from each rank;
2. an `MPI_Alltoallv` sends every element, with its row and column indices, to its new rank.

{: .picture}
<img width="98%" height="98%" src="{{ site.assets }}/matrix-transpose/matrix-transposition-crs.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

A final sort of the received elements by their row and column index in $$M^T$$ builds the CSR arrays of $$M^T$$. The distributed transpose of a CSR matrix thus takes three collective calls: one `MPI_Allgather`, one `MPI_Alltoall` and one `MPI_Alltoallv`.

## Correctness

The distributed transpose is the composition of the two operations. Starting from $$M$$ in row view, the local transpose gives $$M^T$$ in column view, and the view swap then gives $$M^T$$ in row view. In the other order, the view swap gives $$M$$ in column view, and the local transpose then gives $$M^T$$ in row view. So the two operations commute:

$$
Transpose = ViewSwap \cdot LocalTranspose = LocalTranspose \cdot ViewSwap.
$$

Both operations are [involutions][involution]: applied twice, each returns its input. With $$V$$ the view swap, $$L$$ the local transpose and $$I$$ the identity, the transpose is an involution too: $$(V \cdot L) \cdot (V \cdot L) = V \cdot (L \cdot V) \cdot L = (V \cdot V) \cdot (L \cdot L) = I$$. So a second transpose returns the original matrix, in its original layout.

[involution]: https://en.wikipedia.org/wiki/Involution_(mathematics)

## Matrices with several values per cell

In a [multigraph][multigraph], two vertices can be connected by several edges. Examples are the common interests of two users of a social network, or the hyperlinks between two web pages, each with its own data. Each cell of the adjacency matrix then holds a list of values of variable length. The **extended CSR** format (XCSR) stores such matrices: it adds to CSR the array `cell_counts`, the number of values per cell.

[multigraph]: https://en.wikipedia.org/wiki/Multigraph

{: .picture}
<img width="98.8%" height="98.8%" src="{{ site.assets }}/matrix-transpose/xcrs_format.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

The transpose takes three steps. First, the **local transpose**: each rank builds the metadata of its cells, one triplet (row index, column index, number of values) per cell. It sorts the triplets and the values by column and row index.

{: .picture}
<img width="99.1%" height="99.1%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_1.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

Second, the **view swap of the metadata**. The triplets have a fixed size, so they form a distributed sparse matrix with one value per cell, which the CSR view swap above exchanges: an `MPI_Alltoall` of the triplet counts, then an `MPI_Alltoallv` of the triplets. The local sort has already grouped the triplets by destination rank. Every rank then holds the metadata of its part of the transposed matrix, but not the values yet.

{: .picture}
<img width="97.7%" height="97.7%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_2.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

Third, the **view swap of the values**. Each rank computes the number of values it sends to each rank, and an `MPI_Alltoall` transposes this dense matrix of counts. The counts give the size of the receive buffers, and an `MPI_Alltoallv` delivers the values. Each rank finally sorts the values by their row and column index in $$M^T$$.

{: .picture}
<img width="99%" height="99%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_3.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

With the `MPI_Allgather` of the row counts, the XCSR transpose takes five collective calls: one `MPI_Allgather`, two `MPI_Alltoall` and two `MPI_Alltoallv`. Like the CSR transpose, it is an involution.

## Further details

Details, the implementation and benchmarks are in the <a href="{{ site.assets }}/matrix-transpose/matrix-transposer.pdf">original publication (preprint)</a>. The <a href="{{ site.assets }}/matrix-transpose/matrix-transposer.zip">source code</a> is available for download.
