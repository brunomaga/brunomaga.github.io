---
layout: post
title:  "Distributed Matrix Transpose Algorithms"
categories: ["Distributed Algorithms"]
tags: ["MPI", "linear algebra", "matrix transpose", "all-to-all"]
---

[Matrix transpose][wiki-transpose] is a problem of high importance, especially in fields such as large-scale algebraic solvers and graph-based algorithms. The [transpose of a graph](https://en.wikipedia.org/wiki/Transpose_graph) provides the converse of its edge connectivity and the orthogonal view of its connectivity matrix. Although the transpose of an algebraic table (or, equivalently, of the connectivity matrix of a graph) may seem trivial when executed on a single compute node, it becomes a hard task when the dataset is distributed across multiple compute nodes.

This post details an efficient method to solve that problem and shows how to implement it using the [Message Passing Interface](https://en.wikipedia.org/wiki/Message_Passing_Interface) (`MPI`) runtime for distributed execution.

[wiki-transpose]: https://en.wikipedia.org/wiki/Transpose

## The basics

A graph $$G = (V, E)$$ is a data structure consisting of a set of nodes or **vertices** $$V$$, and a set of inter-node connections or **edges** $$E$$. An edge $$\{i,j,e\}$$ in $$E$$, with $$i,j \in \mathbb{N}$$ and $$e \in T$$, represents a connection between vertices $$i$$ and $$j$$ in $$V$$ with edge information of type $$T$$, and can be either ordered or unordered, depending on whether the graph is directed or undirected.
A matrix $$M$$ represents the connectivity $$E$$ in $$G$$, iff for all $$\{i,j,e\} \in E$$, if $$G$$ is ordered then $$M_{ij}=e$$, and if it is unordered then $$M_{ij} = M_{ji} = e$$. Since $$M$$ holds the possible connectivity between nodes in $$V$$, $$M$$ is a square matrix of size $$\vert V \vert \times \vert V \vert$$, also known as the [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) of the graph.

When a matrix is stored in a single compute node, the transpose is easy and follows the mathematical definition: the transpose of a matrix $$M$$ is given by $$M^T_{ij} = M_{ji}$$, and swaps the row and column placement of the cells of the initial matrix. The graph whose connectivity is described by the matrix $$M^T$$ can then be described simply by $$G^{\star} = (V, E^{\star})$$, where $$E^{\star} = \big\{ \{j,i,e\}$$ for all $$\{i,j,e\} \in E \big\}$$.

Given a dense matrix representation of a graph connectivity $$E$$, the algorithm to compute $$G^{\star}_{r} = (V_{r}, E^{\star}_{r})$$ for every rank $$r$$ requires only the computation of the matrix $$M^{T}_{r}$$ for the connectivity $$E^{\star}_{r}$$. Note that the vertex information is local to a compute node and stays in the same memory location after the transpose,
 so $$V$$ is the same for $$G$$ and $$G^{\star}$$. The implementation of the distributed transpose is well known for a **dense** connectivity matrix, and is available via the `MPI_Alltoall` collective call, which takes as input (output) an array of elements, the number of elements per rank, and the datatype of the elements to be sent (received).

{: .picture}
<img width="71.1%" height="71.1%" src="{{ site.assets }}/matrix-transpose/all_to_all.svg">

{: .caption}
`MPI_Alltoall` on 6 processes: block $$j$$ of process $$i$$ goes to process $$j$$, as its block $$i$$. The blocks of process A (red frame) become the first block of every process. Adapted from: [MPI Tutorial, Shao-Ching Huang, IDRE High Performance Computing Workshop, 2013](https://www.slideshare.net/slideshow/mpi-49244468/49244468)

For [sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix), the problem is not so trivial. A distributed memory data layout assumes that the vertices $$V$$ are distributed across $$R$$ compute nodes (**ranks**) and that only the rows local to each memory region are directly accessible to a rank. Let $$G_{r} = (V_{r}, E_{r})$$ be the subset of $$G$$ that is stored in rank $$r$$, with vertices $$V_{r}$$ and edges $$E_{r}$$. Each rank holds a disjoint subset of rows of the initial graph $$G$$, such that the cover ($$\bigcup\limits_{r} G_{r} = G$$) and disjoint ($$G_{r} \bigcap G_{s} = \emptyset, \forall r \neq s$$) properties hold. Ranks only hold information about outgoing connectivity (edges from vertices in this rank to other vertices), i.e. $$E_{r} = \Big\{ \{i,j,e\} \in E $$ such that $$i \in V_r \Big\}$$. Thus, the same cover and disjoint properties also hold for edges.

A common format for the distributed storage of a sparse matrix is the Compressed Sparse Row (CSR) format, where each submatrix stored on a compute node is represented (serialized) as three arrays, holding (1) the number of populated columns per row, (2) the id of each populated column, and (3) their respective values. For clarity, the following picture illustrates a sample graph with vertices 0-5 and 12 edges labeled A-T, its representative sparse matrix, and the CSR data structure on 3 ranks:

{: .picture}
<img width="99.2%" height="99.2%" src="{{ site.assets }}/matrix-transpose/crs_format.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

## How to transpose it?

A small mental trick to understand this is to notice that a sparse matrix transpose is nothing more than a composition of two operations: a local matrix transpose and a view swap. So these two operations can be used to describe a distributed sparse matrix in four different data layouts, illustrated below:

{: .picture}
<img width="96.4%" height="96.4%" src="{{ site.assets }}/matrix-transpose/layouts.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

A local transpose of the original dataset (a) yields layout (b), a view swap yields layout (c), and both operations together yield the distributed transpose (d).

The formulation of the solution starts with the mathematical formalism underlying the distributed matrix transpose operations. A **horizontal concatenation** of two matrices $$M_{n \times m}$$ and $$N_{n \times m'}$$ is represented by $$M \| N$$ and defined as the operation that joins two sub-matrices horizontally into a matrix of dimensionality $${n \times (m+m')}$$, such that: 

$$
(M \| N)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    M_{i \,\, j} & \text{, if } j \leq m\\
    N_{i \,\, j-m} & \text{, otherwise}
  \end{array}\right.
$$

Analogously, **a vertical concatenation** $$M_{n \times m} // N_{n' \times m}$$ joins two sub-matrices vertically into a matrix of dimensionality $${(n+n') \times m}$$, such that: 

$$
(M // N)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    M_{i \,\, j} & \text{, if } i \leq n\\
    N_{i-n \,\, j} & \text{, otherwise}
  \end{array}\right.
$$

A **view** refers to the perspective of data storage: the row view describes the matrix as the vertical concatenation of the subsets of rows stored on each rank, and the column view as the horizontal concatenation of the subsets of columns stored on each rank. It follows that $$(M \| N)^T = (M^T // N^T)$$ and $$(M // N)^T = (M^T \| N^T)$$, as both concatenations provide the same dataset described by two alternative views, and the transpose of a concatenated dataset yields the orthogonally concatenated dataset. 

The **local transpose** of a matrix $$M$$ represented by the concatenation of $$R$$ submatrices in either view is defined as the concatenation of the transposes of the individual matrices in the orthogonal view:

$$
LocalTranspose(M)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    \big( M^T_1 \| M^T_2 \| ... \| M^T_R \big)_{i j}& \text{, if } M \text{ is in row view }\\
    \big( M^T_1 // M^T_2 // ... // M^T_R \big)_{i j}& \text{, otherwise}
  \end{array}\right.
$$

It is worth emphasizing that the operation yields a transposed version of the original rank matrices that formed the initial dataset, so no communication between ranks is necessary, as the data $$M_r$$ of every rank is locally transposed into $$M^T_r$$. Moreover, it is an [involutory function](https://en.wikipedia.org/wiki/Involution_(mathematics)), as $$(LocalTranspose \cdot LocalTranspose) (M) = M$$. 

The **view swap** of a distributed matrix, which alternates between a data representation in one view and its orthogonal one (from row- to column-accessible and vice versa) while maintaining the same matrix contents, is defined by:

$$
ViewSwap(M)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    \big( M_1 \| M_2 \| ... \| M_R \big)_{i j}& \text{, if } M \text{ is in row view}\\
    \big( M_1 //  M_2 //  ... //  M_R \big)_{i j}& \text{, otherwise}
  \end{array}\right.
$$


At the level of a rank, the matrix data layout after a view swap is unknown until the swap is performed, as ranks do not hold information about the matrix structure across other ranks. Therefore, a communication step must be executed beforehand to gather the number of rows held by individual ranks and compute the row interval of every other rank $$r$$ as 
$$
\big[ \sum^{r-1}_{s=1} \vert V_s \vert, \,\, \sum^{r}_{s=1} \vert V_s \vert \big)
$$.

This information is required to match each column/row id to its target rank in the sparse transpose steps that follow. The view swap algorithm then follows in two communication steps. The first step performs a dense matrix transpose to send/receive the number of elements to/from other ranks. The second step performs a selective scatter-gather (`MPI_Alltoallv`) that delivers the elements to the final rank.

{: .picture}
<img width="98%" height="98%" src="{{ site.assets }}/matrix-transpose/matrix-transposition-crs.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

A memory realignment based on column id and row id leads to the final distributed matrix dataset. Finally, the view swap method is also an involutory function, as each view swap performs two involutory transpose operations, and two consecutive view swaps yield the initial dataset.


The full transpose algorithm performs only three collective communication calls: an `MPI_Allgather` to compute the row offsets of ranks, and one `MPI_Alltoall` and one `MPI_Alltoallv` for the metadata and cell values transpose in the view swap. Moreover, the involutory property of this transpose method is of high importance, as it guarantees that the distributed transpose can be validly executed any number of times while respecting the data integrity of a CSR-based graph problem representation. 

## Algorithmic verification

Is this algorithm guaranteed to work regardless of the dataset? The proof below shows that it is indeed mathematically sound.

The **distributed sparse transpose** is defined as the composition of the local transpose and view swap methods. The previous section showed that both the local transpose and the view swap are CSR compatible. It remains to show that the distributed sparse transpose is also mathematically sound. Take a matrix $$M$$ in row view, represented by the vertical concatenation of partial matrices $$M_1 // M_2 // ... // M_R$$. A local transpose of $$M$$ leads to $$M^T_{i j} = ( M^T_1 \| M^T_2 \| ... \| M^T_R)_{ij}$$.
Applying a view swap yields $$M^T_{i j} = ( M^T_1 // M^T_2 // ... // M^T_R)_{ij}$$, which is the definition of the distributed transpose of $$M$$ in the original view. In brief, it follows that $$Transpose (M) = (ViewSwap \cdot LocalTranspose) (M)$$.

The verification with the commuted involutory functions, $$Transpose (M) = (LocalTranspose \cdot ViewSwap) (M)$$, also holds: a view swap over the original dataset leads to $$M_{i j} = ( M_1 \| M_2 \| ... \| M_R)_{i j}$$, and a subsequent local transpose yields the final result $$M^T_{i j} = ( M^T_1 // M^T_2 // ... // M^T_R)_{ij}$$. As $$Transpose$$ is a composition of two involutory functions, and because both functions commute, i.e. $$(ViewSwap \cdot LocalTranspose) = (LocalTranspose \cdot ViewSwap)$$, $$Transpose$$ is by definition involutory ([Elements of Operator Theory, Carlos S. Kubrusly, Springer 2011][kubrusly2011elements]).

[kubrusly2011elements]: https://link.springer.com/content/pdf/10.1007/978-0-8176-4998-2.pdf

## Higher Cardinality Matrices

Having a single edge between each pair of nodes is often not the case in real-life situations, which then require [multigraphs](https://en.wikipedia.org/wiki/Multigraph). Take as an example the matrix representing information about common interests between pairs of users in a social network: some pairs of users may be connected by hundreds of common interests, while others may have barely any interests in common; or 
a URL cross-referral map, where one may be required to store not only the number of hyperlinks between pages, but also additional information about every link. How to implement the transpose in this case?

The solution starts by defining a data format that can hold high-cardinality elements per pair of graph nodes. It is called the extended CSR, or XCSR for short, and is defined as the previous CSR data structure with an added dimension that holds the number of elements per matrix cell (or edges between pairs of nodes). The layout is as follows:

{: .picture}
<img width="98.8%" height="98.8%" src="{{ site.assets }}/matrix-transpose/xcrs_format.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

The transpose algorithm then follows in three steps. First, each compute node (rank) collects the metadata of its own sub-matrix, composed of the set of triplets of the form [ row id, column id, number of elements], indicating the row, column and dimensionality of each cell. Then the ranks perform a **local transpose** of their own data, by reordering their triplets by column and row id. This sequence illustrates the first step:
 
{: .picture}
<img width="99.2%" height="99.2%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_1.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

The second step starts the **view swap** method. All ranks pack their triplets and exchange them (i.e. an all-to-all communication), sending each triplet to the recipient of its cell in the final transposed matrix. In practice, the previous local transpose operation already sorted each sub-matrix by columns (i.e. by rows of the final transposed dataset), so the triplets are already grouped by destination rank. More importantly, the matrix composed of all triplets across all ranks is a distributed matrix with a fixed element size per cell (a tuple of fixed size). Therefore, this exchange is performed by the same distributed transpose presented above in the section *How to transpose it?*. After the exchange, all ranks will hold the triplets of the metadata (*but not the data yet*) of the final matrix structure:   

{: .picture}
<img width="97.7%" height="97.7%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_2.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

Finally, the view swap is completed by communicating the data, based on the metadata of the transposed matrix. First, each rank counts the number of elements to be sent to each rank. The matrix of distributed counts of all elements to be sent and received per rank is a dense matrix that can be transposed easily (`MPI_Alltoall`). These counts are then used to allocate the buffers for the incoming data and to perform an `MPI_Alltoallv` that sends/delivers the elements of the final matrix across all compute nodes. Upon completion, the (previously column-oriented) elements are sorted by row and column, which yields the final dataset: 

{: .picture}
<img width="99.2%" height="99.2%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_3.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)


To sum up, the full transpose algorithm requires five collective communication calls: an `MPI_Allgather` to compute the row offsets of ranks, and two `MPI_Alltoall` and two `MPI_Alltoallv` for the metadata and cell values transpose. Moreover, the general applicability of the XCSR and the involutory property of this transpose method are two properties of high importance, as they guarantee that the distributed transpose can be validly executed any number of times while respecting the data integrity of an XCSR-based graph problem representation.

## Further details

For more details, implementation and benchmark information, the <a href="{{ site.assets }}/matrix-transpose/matrix-transposer.pdf">original publication (preprint)</a> and the <a href="{{ site.assets }}/matrix-transpose/matrix-transposer.zip">source code</a> are available for download.

