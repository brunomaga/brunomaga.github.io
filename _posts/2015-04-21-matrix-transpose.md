---
layout: post
title:  "Distributed Matrix Transpose Algorithms"
categories: ["Distributed Algorithms"]
tags: ["MPI", "linear algebra", "matrix transpose", "all-to-all"]
---

[Matrix transpose][wiki-transpose] is an important operation, particularly in large-scale algebraic solvers and graph-based algorithms. The transpose of a graph reverses the direction of its edges and provides the orthogonal view of its connectivity matrix. The transpose of an algebraic table (or, equivalently, of the connectivity matrix of a graph) may seem trivial on a single compute node, but it becomes a hard task when the dataset is distributed across multiple compute nodes.

This post describes an efficient method to solve this problem and its implementation with the Message Passing Interface (`MPI`) runtime for distributed execution.

[wiki-transpose]: https://en.wikipedia.org/wiki/Transpose

## The basics

A graph $$G = (V, E)$$ is a data structure made of a set of nodes or **vertices** $$V$$, and a set of connections between nodes, or **edges** $$E$$. An edge $$\{i,j,e\} \subset \{\mathbb{N}, \mathbb{N}, T\}$$ in $$E$$ represents a connection between vertices $$i$$ and $$j$$ in $$V$$ with edge information of type $$T$$. Edges are ordered or unordered, depending on whether the graph is directed or undirected.
A matrix $$M$$ represents the connectivity $$E$$ in $$G$$ iff, for all $$\{i,j,e\} \in E$$, $$M_{ij}=e$$ if $$G$$ is ordered, and $$M_{ij} = M_{ji} = e$$ if it is unordered. Since $$M$$ holds the possible connectivity between nodes in $$V$$, $$M$$ is a square matrix of size $$|V| \times |V|$$.

When a matrix is stored on a single compute node, the transpose is simple and follows the mathematical definition: the transpose of a matrix $$M$$ is $$M^T_{ij} = M_{ji}$$, i.e. it swaps the row and column positions of the cells of the initial matrix. The graph whose connectivity is described by the matrix $$M^T$$ is then simply $$G^{\star} = (V, E^{\star})$$, where $$E^{\star} = \big\{ \{j,i,e\}$$ for all $$\{i,j,e\} \in E \big\}$$. In practice, elements follow a row-column swap:

{: style="text-align:center; font-size: small;"}
<img width="15%" height="15%" src="https://upload.wikimedia.org/wikipedia/commons/e/e4/Matrix_transpose.gif">

{: style="text-align:center; font-size: small;"}
(image source: Wikipedia)

Given a dense matrix representation of a graph connectivity $$E$$, computing $$G^{\star}_{r} = (V_{r}, E^{\star}_{r})$$ for every rank $$r$$ requires only the computation of the matrix $$M^{T}_{r}$$ for the connectivity $$E^{\star}_{r}$$. Vertex information is local to a compute node and stays in the same memory location after the transpose,
 so $$V$$ is the same for $$G$$ and $$G^{\star}$$. The distributed transpose of a **dense** connectivity matrix is well known and is available via the `MPI_Alltoall` collective call. This call takes as input (output) an array of elements, the size of the array, and the datatype of the elements to be sent (received).

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="{{ site.assets }}/matrix-transpose/all_to_all.svg"><span class="text-secondary">

{: style="text-align:center; font-size: small;"}
(image adapted from: MPI Tutorial, Shao-Ching Huang, IDRE High Performance Computing Workshop, 2013-02-13)</span>

For sparse matrices, the problem is harder. A distributed memory data layout assumes that the vertices $$V$$ are distributed across $$R$$ compute nodes (**ranks**), and that each rank can directly access only the rows stored in its own memory region. The notation $$G_{r} = (V_{r}, E_{r})$$ refers to the subset of $$G$$ stored in rank $$r$$, with vertices $$V_{r}$$ and edges $$E_{r}$$. Each rank holds a disjoint subset of rows of the initial graph $$G$$, such that the cover ($$\bigcup\limits_{r} G_{r} = G$$) and disjoint ($$G_{r} \bigcap G_{s} = \emptyset, \forall r \neq s$$) properties hold. Ranks only hold information about outgoing connectivity (edges from vertices in this rank to other vertices), i.e. $$E_{r} = \Big\{ \{i,j,e\} \in E $$ such that $$i \in V_r \Big\}$$. Therefore, the cover and disjoint properties also hold for edges.

A common format for the distributed storage of a sparse matrix is the Compressed Sparse Row (CSR) format. Each submatrix stored on a compute node is represented (serialized) as three arrays, holding (1) the number of populated columns per row, (2) the id of each populated column, and (3) their values. The following picture illustrates a sample graph with vertices 0-5 and edges A-T, its sparse matrix, and the CSR data structure on 3 ranks:

{: style="text-align:center; font-size: small;"}
<img width="89%" height="89%" src="{{ site.assets }}/matrix-transpose/crs_format.svg">

{: style="text-align:center; font-size: small;"}
Based on the [original publication]({{ site.assets }}/matrix-transpose/matrix-transposer.pdf).

## How to transpose it?

A sparse matrix transpose can be seen as the composition of two operations: a local matrix transpose and a view swap. These two operations describe a distributed sparse matrix in four different data layouts, illustrated below:

{: style="text-align:center; font-size: small;"}
<img width="94%" height="94%" src="{{ site.assets }}/matrix-transpose/layouts.svg">

{: style="text-align:center; font-size: small;"}
Based on the [original publication]({{ site.assets }}/matrix-transpose/matrix-transposer.pdf).

The formulation starts with the mathematical formalism of the distributed matrix transpose operations. A **horizontal concatenation** of two matrices $$M_{n \times m}$$ and $$N_{n \times m'}$$ is represented by $$M \| N$$. It joins two sub-matrices horizontally into a matrix of dimensionality $${n \times (m+m')}$$, such that: 

$$
(M \| N)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    M_{i \,\, j} & \text{, if } j \leq m\\
    N_{i \,\, j-m} & \text{, otherwise}
  \end{array}\right.
$$

Similarly, a **vertical concatenation** $$M_{n \times m} // N_{n' \times m}$$ joins two sub-matrices vertically into a matrix of dimensionality $${(n+n') \times m}$$, such that: 

$$
(M // N)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    M_{i \,\, j} & \text{, if } i \leq n\\
    N_{i-n \,\, j} & \text{, otherwise}
  \end{array}\right.
$$

A **view** is the perspective of data storage: the row view describes the matrix as the vertical concatenation of the subsets of rows stored on each rank, and the column view describes it as the horizontal concatenation of the subsets of columns stored on each rank. It follows that $$(M \| N)^T = (M^T // N^T)$$ and $$(M // N)^T = (M^T \| N^T)$$: both concatenations describe the same dataset in two alternative views, and the transpose of a concatenated dataset yields the orthogonally concatenated dataset. 

The **local transpose** of a matrix $$M$$, represented by the concatenation of $$R$$ submatrices in either view, is the concatenation of the transposes of the individual matrices in the orthogonal view:

$$
LocalTranspose(M)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    \big( M^T_1 \| M^T_2 \| ... \| M^T_R \big)_{j i}& \text{, if } M \text{ is in row view }\\
    \big( M^T_1 // M^T_2 // ... // M^T_R \big)_{j i}& \text{, otherwise}
  \end{array}\right.
$$

This operation yields the transposes of the original rank matrices that form the initial dataset. No communication between ranks is needed, as the data $$M_r$$ of every rank is locally transposed into $$M^T_r$$. Moreover, it is an involutory function, as $$(LocalTranspose \cdot LocalTranspose) (M) = M$$. 

The **view swap** of a distributed matrix switches the data representation from one view to the orthogonal one (from row- to column-accessible and vice versa), while keeping the same matrix contents. It is defined by:

$$
ViewSwap(M)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    \big( M_1 \| M_2 \| ... \| M_R \big)_{i j}& \text{, if } M \text{ is in row view}\\
    \big( M_1 //  M_2 //  ... //  M_R \big)_{i j}& \text{, otherwise}
  \end{array}\right.
$$


From the perspective of a single rank, the matrix data layout after a view swap is unknown until the swap is performed, as ranks do not hold information about the matrix structure on other ranks. Therefore, a communication step must first gather the number of rows held by each rank, to compute the row interval of every other rank $$r$$ as 
$$
\big[ \sum^{r-1}_{s=1} \vert V_s \vert, \,\, \sum^{r}_{s=1} \vert V_s \vert \big)
$$.

This information is needed to map each column/row id to its target rank in the sparse transpose steps that follow. The view swap then performs two communication steps. The first step performs a dense matrix transpose to send/receive the number of elements to/from other ranks. The second step performs a selective scatter-gather (`MPI_Alltoallv`) that delivers the elements to their final rank.

{: style="text-align:center; font-size: small;"}
<img width="99%" height="99%" src="{{ site.assets }}/matrix-transpose/matrix-transposition-crs.svg">

{: style="text-align:center; font-size: small;"}
Based on the [original publication]({{ site.assets }}/matrix-transpose/matrix-transposer.pdf).

A memory realignment based on column id and row id produces the final distributed matrix. The view swap is also an involutory function: each view swap performs two involutory transpose operations, so two consecutive view swaps yield the initial dataset.


The full transpose algorithm performs only three collective communication calls: an `MPI_Allgather` to compute the row offsets of ranks, and one `MPI_Alltoall` and one `MPI_Alltoallv` for the metadata and cell values transpose in the view swap. Moreover, the involutory property of this transpose method is important, as it guarantees that the distributed transpose can be executed any number of times while preserving the data integrity of a CSR-based graph representation. 

## Algorithmic verification

Is this algorithm guaranteed to work for any dataset? This section shows that it is mathematically sound.

The **distributed sparse transpose** is defined as the composition of the local transpose and view swap methods. Both the local transpose and the view swap are CSR compatible, and the distributed sparse transpose is also mathematically sound. Take a matrix $$M$$ in row view, represented by the vertical concatenation of partial matrices $$M_1 // M_2 // ... // M_R$$. A local transpose of $$M$$ leads to $$M^T_{i j} = ( M^T_1 \| M^T_2 \| ... \| M^T_R)_{ji}$$.
Applying a view swap gives $$M^T_{i j} = ( M^T_1 // M^T_2 // ... // M^T_R)_{ji}$$, which is the definition of the distributed transpose of $$M$$ in the original view. In brief, $$Transpose (M) = (ViewSwap \cdot LocalTranspose) (M)$$.

The commuted composition, $$Transpose (M) = (LocalTranspose \cdot ViewSwap) (M)$$, also holds: a view swap over the original dataset leads to $$M_{i j} = ( M_1 \| M_2 \| ... \| M_R)_{i j}$$, and a subsequent local transpose yields the final result $$M^T_{i j} = ( M^T_1 // M^T_2 // ... // M^T_R)_{ji}$$. Since $$Transpose$$ is a composition of two involutory functions that commute, i.e. $$(ViewSwap \cdot LocalTranspose) = (LocalTranspose \cdot ViewSwap)$$, $$Transpose$$ is by definition involutory ([Elements of Operator Theory, Carlos S. Kubrusly, Springer 2011][kubrusly2011elements]).

[kubrusly2011elements]: https://link.springer.com/content/pdf/10.1007/978-0-8176-4998-2.pdf

## Higher Cardinality Matrices

In real-life problems, a pair of nodes is often connected by more than one edge. One example is a matrix of common interests between pairs of users in a social network: some pairs of users share hundreds of interests, while others share barely any. Another example is 
a URL cross-referral map, which may store not only the number of hyperlinks between pages, but also additional information about every link. This section describes the transpose for this case.

The first step is a data format that holds high-cardinality elements per graph node. This format is called the extended CSR, or XCSR, and is defined as the previous CSR data structure with an added dimension that holds the number of elements per matrix cell (i.e. edges between pairs of nodes). The layout is as follows:

{: style="text-align:center; font-size: small;"}
<img width="97%" height="97%" src="{{ site.assets }}/matrix-transpose/xcrs_format.svg">

{: style="text-align:center; font-size: small;"}
Based on the [original publication]({{ site.assets }}/matrix-transpose/matrix-transposer.pdf).

The transpose algorithm has three steps. First, each compute node (rank) collects the metadata of its own sub-matrix: a set of triplets of the form [ row id, column id, number of elements], indicating the row, column and dimensionality of each cell. The ranks then perform a **local transpose** of their own data, by reordering their triplets by column and row id. The following sequence illustrates this first step:
 
{: style="text-align:center; font-size: small;"}
<img width="97%" height="97%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_1.svg">

{: style="text-align:center; font-size: small;"}
Based on the [original publication]({{ site.assets }}/matrix-transpose/matrix-transposer.pdf).

The second step starts the **view swap** method. All ranks pack their triplets and exchange them in an all-to-all communication, sending each triplet to the owner of its cell in the final transposed matrix. The previous local transpose already sorted each sub-matrix by columns (i.e. by rows of the final transposed dataset), so the triplets are already grouped by destination rank. More importantly, the matrix of all triplets across all ranks is a distributed matrix with a fixed element size per cell (a tuple of fixed size). Therefore, this exchange uses the same distributed transpose presented in the section *How to transpose it?*. After the exchange, all ranks hold the triplets of the metadata (*but not the data yet*) of the final matrix structure:   

{: style="text-align:center; font-size: small;"}
<img width="99%" height="99%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_2.svg">

{: style="text-align:center; font-size: small;"}
Based on the [original publication]({{ site.assets }}/matrix-transpose/matrix-transposer.pdf).

Finally, the view swap is completed by communicating the data, based on the metadata of the transposed matrix. First, each rank counts the number of elements to be received. The distributed matrix of counts of elements sent and received per rank is a dense matrix, so it is easily transposed with `MPI_Alltoall`. These counts are then used to allocate the buffers for the incoming data and to perform an `MPI_Alltoallv` that delivers the elements of the final matrix across all compute nodes. Upon completion, the (previously column-oriented) elements are sorted by row and column, which yields the final dataset: 

{: style="text-align:center; font-size: small;"}
<img width="99%" height="99%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_3.svg">

{: style="text-align:center; font-size: small;"}
Based on the [original publication]({{ site.assets }}/matrix-transpose/matrix-transposer.pdf).


In summary, the full transpose algorithm requires five collective communication calls: an `MPI_Allgather` to compute the row offsets of ranks, and two `MPI_Alltoall` and two `MPI_Alltoallv` for the metadata and cell values transpose. Moreover, the general applicability of the XCSR format and the involutory property of the transpose method are important, as they guarantee that the distributed transpose can be executed any number of times while preserving the data integrity of an XCSR-based graph representation.

## Further details

More details, implementation and benchmark information are available in the <a href="{{ site.assets }}/matrix-transpose/matrix-transposer.pdf">original publication (preprint)</a>. The <a href="{{ site.assets }}/matrix-transpose/matrix-transposer.zip">source code</a> is also available for download.

