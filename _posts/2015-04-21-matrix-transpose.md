---
layout: post
title:  "Distributed Matrix Transpose Algorithms"
categories: ["Distributed Algorithms"]
tags: ["MPI", "linear algebra", "matrix transpose", "all-to-all", "sparse matrix", "CSR", "multigraph"]
description: "The distributed transpose of a sparse matrix with MPI, as a local transpose and a view swap: the all-to-all exchange of counts and cells in CSR, and the extended CSR (XCSR) for multigraphs."
---

[Matrix transpose][wiki-transpose] is a problem of high importance, especially in fields such as large-scale algebraic solvers and graph-based algorithms. As an example, a web crawler stores the links that leave each page (its out-links), one row of the connectivity matrix per page, while the pages that link to a given page (its in-links) form a column of that matrix. The [transpose of a graph](https://en.wikipedia.org/wiki/Transpose_graph) reverses every edge: it provides the converse of its edge connectivity and the orthogonal view of its connectivity matrix, where the in-links of each page form a row.

Although the transpose of an algebraic table (or, equivalently, of the connectivity matrix of a graph) may seem trivial when executed on a single compute node, it becomes a hard task when the dataset is distributed across multiple compute nodes. A graph of a billion vertices with 100 edges each has $$10^{11}$$ edges, and the column ids of its matrix alone take 400 GB as 32-bit integers. When its rows are split across 100 compute nodes, each node holds 10 million vertices and 4 GB of column ids, and the edges that reach a vertex may start on any node.

This post details an efficient method to solve that problem for sparse matrices, including matrices with several values per cell, and shows how to implement it using the [Message Passing Interface](https://en.wikipedia.org/wiki/Message_Passing_Interface) (`MPI`) runtime for distributed execution.

[wiki-transpose]: https://en.wikipedia.org/wiki/Transpose

## The basics

A [graph](https://en.wikipedia.org/wiki/Graph_(abstract_data_type)) $$G = (V, E)$$ is a data structure consisting of a set of nodes or **vertices** $$V$$, and a set of inter-node connections or **edges** $$E$$. An edge $$(i,j,e)$$ in $$E$$, with $$i,j \in \mathbb{N}$$ and $$e \in T$$, represents a connection between vertices $$i$$ and $$j$$ in $$V$$ with edge information of type $$T$$. It can be either ordered or unordered, depending on whether the graph is directed or undirected.

A matrix $$M$$ represents the connectivity $$E$$ in $$G$$, iff for all $$(i,j,e) \in E$$, if $$G$$ is directed then $$M_{ij}=e$$, and if it is undirected then $$M_{ij} = M_{ji} = e$$. Since $$M$$ holds the possible connectivity between nodes in $$V$$, $$M$$ is a square matrix of size $$\vert V \vert \times \vert V \vert$$, also known as the [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) of the graph. The cells of the pairs of vertices without an edge are empty.

When a matrix is stored in a single compute node, the transpose is easy and follows the mathematical definition: the transpose of a matrix $$M$$ is given by $$(M^T)_{ij} = M_{ji}$$. It swaps the row and column placement of the cells of the initial matrix. The graph whose connectivity is described by the matrix $$M^T$$ can then be described by $$G^{\star} = (V, E^{\star})$$, where $$E^{\star} = \{ (j,i,e) : (i,j,e) \in E \}$$ holds the reversed edges. For an undirected graph, $$M$$ is symmetric and equal to its transpose, so the rest of this post considers directed graphs.

A distributed memory data layout assumes that the vertices $$V$$ are distributed across $$R$$ compute nodes (**ranks**, numbered from 0) and that only the rows local to each memory region are directly accessible to a rank. Let $$G_{r} = (V_{r}, E_{r})$$ be the subset of $$G$$ that is stored in rank $$r$$, with vertices $$V_{r}$$ and edges $$E_{r}$$.

Each rank holds a disjoint range of consecutive rows of the initial graph $$G$$ (a **1D partitioning** of the matrix by rows), such that the cover ($$\bigcup\limits_{r} G_{r} = G$$) and disjoint ($$G_{r} \cap G_{s} = \emptyset, \forall r \neq s$$) properties hold. Ranks only hold information about outgoing connectivity (edges from vertices in this rank to other vertices), i.e. $$E_{r} = \{ (i,j,e) \in E : i \in V_r \}$$. Thus, the same cover and disjoint properties also hold for edges, and the rows $$V_r$$ of $$M$$ form the block $$M_r$$ of rank $$r$$, of size $$\vert V_r \vert \times \vert V \vert$$.

Given a dense matrix representation of a graph connectivity $$E$$, the algorithm to compute $$G^{\star}_{r} = (V_{r}, E^{\star}_{r})$$ for every rank $$r$$ requires only the computation of the block $$(M^{T})_{r}$$ of the rows $$V_r$$ of $$M^T$$, for the connectivity $$E^{\star}_{r}$$. Note that the vertex information is local to a compute node and stays in the same memory location after the transpose, so $$V$$ is the same for $$G$$ and $$G^{\star}$$.

The implementation of the distributed transpose is well known for a **dense** connectivity matrix, and is available via the `MPI_Alltoall` [collective](https://en.wikipedia.org/wiki/Collective_operation) call. This call takes as input (output) an array of elements, the number of elements per rank, and the datatype of the elements to be sent (received). Each rank sends a different block of its array to every rank, an exchange known as the **all-to-all personalized** communication, or total exchange. In practice, if each rank holds one row of the matrix, the all-to-all is the transpose itself; if it holds several rows, each block is also transposed locally. The following picture illustrates the all-to-all, with the buffer of each process in a row, before (left) and after (right) the exchange:

{: .picture}
<img width="71.1%" height="71.1%" src="{{ site.assets }}/matrix-transpose/all_to_all.svg">

{: .caption}
`MPI_Alltoall` on 6 processes: block $$j$$ of process $$i$$ goes to process $$j$$, as its block $$i$$. The blocks of process A (red frame) become the first block of every process. Adapted from: [MPI Tutorial, Shao-Ching Huang, IDRE High Performance Computing Workshop, 2013](https://www.slideshare.net/slideshow/mpi-49244468/49244468)

For [sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix), the problem is not so trivial. Most cells are empty, so each rank stores only its nonempty cells, each with its column id. The number of cells that a rank sends to each other rank then depends on the data: no rank knows in advance how many cells it receives, nor where they go in its rows, so the cells are sent with their row and column ids.

A common format for the distributed storage of a sparse matrix is the **Compressed Sparse Row** (CSR) format. Each submatrix stored on a compute node is represented (serialized) as three arrays, holding (1) the number of populated columns per row (`counts` in the picture), (2) the id of each populated column (`disps`), and (3) their respective values (`cells`). Note that the standard CSR stores the offset of each row instead of its number of populated columns. For clarity, the following picture illustrates a sample graph with vertices 0-5 and 12 edges labeled A-T, its representative sparse matrix, and the CSR data structure on 3 ranks:

{: .picture}
<img width="92.2%" height="92.2%" src="{{ site.assets }}/matrix-transpose/crs_format.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

In the picture, rank 0 holds the rows 0 and 1 (its `row_count` is 2). Row 0 has 3 cells, A, B and D, in the columns 1, 2 and 5, and row 1 has 1 cell, C, in column 4: the three arrays of rank 0 hold 3 and 1, then 1, 2, 5 and 4, then A, B, D and C.

## How to transpose it?

After the transpose, rank $$r$$ must hold the rows $$V_r$$ of $$M^T$$, i.e. the columns $$V_r$$ of $$M$$, which have cells on every rank. A small mental trick to understand this is to notice that a sparse matrix transpose is nothing more than a composition of two operations: a local matrix transpose and a view swap. The local transpose transposes the block of each rank in its own memory, with no communication. The view swap keeps the matrix as it is but changes the part that each rank holds, from the rows $$V_r$$ to the columns $$V_r$$, and performs all the communication. These two operations describe a distributed sparse matrix in four different data layouts, illustrated below:

{: .picture}
<img width="53.0%" height="53.0%" src="{{ site.assets }}/matrix-transpose/layouts.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

A local transpose of the original dataset (a) yields layout (b), a view swap yields layout (c), and both operations together yield the distributed transpose (d), whichever comes first: a view swap of (b) and a local transpose of (c) both lead to (d).

Take cell A, in row 0 and column 1 on rank 0. The local transpose moves it to row 1 and column 0 without leaving rank 0, which now holds the columns 0 and 1 of $$M^T$$. The view swap keeps A in row 0 and column 1, but rank 0 now holds the columns 0 and 1 of $$M$$: A, G, J, O and R, which come from all three ranks. In the distributed transpose (d), rank 0 holds the rows 0 and 1 of $$M^T$$: J and R in row 0, and A, G and O in row 1.

The formulation of the solution starts with the mathematical formalism underlying the distributed matrix transpose operations, with rows and columns indexed from 0, as in the pictures. A **horizontal concatenation** of two matrices $$M_{n \times m}$$ and $$N_{n \times m'}$$ is represented by $$M \| N$$ and defined as the operation that joins two sub-matrices horizontally into a matrix of dimensionality $${n \times (m+m')}$$, such that:

$$
(M \| N)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    M_{i \,\, j} & \text{, if } j < m\\
    N_{i \,\, j-m} & \text{, otherwise}
  \end{array}\right.
$$

Analogously, a **vertical concatenation** $$M_{n \times m} // N_{n' \times m}$$ joins two sub-matrices vertically into a matrix of dimensionality $${(n+n') \times m}$$, such that: 

$$
(M // N)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    M_{i \,\, j} & \text{, if } i < n\\
    N_{i-n \,\, j} & \text{, otherwise}
  \end{array}\right.
$$

A **view** refers to the perspective of data storage. The **row view** describes the matrix as the vertical concatenation of the subsets of rows stored on each rank, the blocks $$M_r$$: $$M = M_0 // M_1 // ... // M_{R-1}$$. The **column view** describes it as the horizontal concatenation of the subsets of columns stored on each rank, $$M = \tilde{M}_0 \| \tilde{M}_1 \| ... \| \tilde{M}_{R-1}$$, where the block $$\tilde{M}_r$$, of size $$\vert V \vert \times \vert V_r \vert$$, holds the columns $$V_r$$ of $$M$$. It follows that $$(M \| N)^T = M^T // N^T$$ and $$(M // N)^T = M^T \| N^T$$, as both concatenations provide the same dataset described by two alternative views, and the transpose of a concatenated dataset yields the orthogonally concatenated dataset.

The **local transpose** of a matrix $$M$$ represented by the concatenation of $$R$$ submatrices in either view is defined as the concatenation of the transposes of the individual matrices in the orthogonal view:

$$
\mathrm{LocalTranspose}(M)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    \big( M^T_0 \| M^T_1 \| ... \| M^T_{R-1} \big)_{i j}& \text{, if } M \text{ is in row view }\\
    \big( \tilde{M}^T_0 // \tilde{M}^T_1 // ... // \tilde{M}^T_{R-1} \big)_{i j}& \text{, otherwise}
  \end{array}\right.
$$

Note that the operation yields a transposed version of the original rank matrices that formed the initial dataset, so no communication between ranks is necessary, as the data $$M_r$$ of every rank is locally transposed into $$M^T_r = (M_r)^T$$. This block holds the columns $$V_r$$ of $$M^T$$; it is not the block $$(M^T)_r$$ of its rows $$V_r$$. In the CSR format, the local transpose reorders the cells of each rank by column id, then by row id. Moreover, it is an [involutory function](https://en.wikipedia.org/wiki/Involution_(mathematics)), as $$(\mathrm{LocalTranspose} \circ \mathrm{LocalTranspose})(M) = M$$, where $$\circ$$ is the [composition](https://en.wikipedia.org/wiki/Function_composition) of functions, which applies the right function first.

The **view swap** of a distributed matrix, which alternates between a data representation in one view and its orthogonal one (from row- to column-accessible and vice versa) while maintaining the same matrix contents, is defined by:

$$
\mathrm{ViewSwap}(M)_{i j} =\left\{
  \begin{array}{@{}ll@{}}
    \big( \tilde{M}_0 \| \tilde{M}_1 \| ... \| \tilde{M}_{R-1} \big)_{i j}& \text{, if } M \text{ is in row view}\\
    \big( M_0 //  M_1 //  ... //  M_{R-1} \big)_{i j}& \text{, otherwise}
  \end{array}\right.
$$

Unlike the local transpose, the view swap needs communication: the block $$\tilde{M}_r$$ holds the columns $$V_r$$ of all the rows, and so cells of every rank, as layout (c) shows. Moreover, the view swap is also an involutory function, as two consecutive view swaps yield the initial dataset.

## Algorithmic verification

Is this algorithm guaranteed to work regardless of the dataset? The proof below shows that it is indeed mathematically sound.

The **distributed sparse transpose** is defined as the composition of the local transpose and view swap methods. Both methods take a distributed matrix in the CSR format and yield one in the CSR format: the local transpose reorders the cells of each rank, and the view swap sends them to their new ranks, as the next section shows. It remains to show that the distributed sparse transpose is also mathematically sound, i.e. that it yields the transpose of any matrix.

Take a matrix $$M$$ in row view, represented by the vertical concatenation of partial matrices $$M_0 // M_1 // ... // M_{R-1}$$. A local transpose of $$M$$ leads to $$M^T_0 \| M^T_1 \| ... \| M^T_{R-1}$$, which is $$M^T$$ in column view, as the transpose of a vertical concatenation is the horizontal concatenation of the transposes. Applying a view swap yields $$(M^T)_0 // (M^T)_1 // ... // (M^T)_{R-1}$$, with the rows $$V_r$$ of $$M^T$$ on each rank $$r$$, which is the definition of the distributed transpose of $$M$$ in the original view. In brief, it follows that $$\mathrm{Transpose}(M) = (\mathrm{ViewSwap} \circ \mathrm{LocalTranspose})(M)$$.

The verification with the commuted functions, $$\mathrm{Transpose}(M) = (\mathrm{LocalTranspose} \circ \mathrm{ViewSwap})(M)$$, also holds: a view swap over the original dataset leads to $$\tilde{M}_0 \| \tilde{M}_1 \| ... \| \tilde{M}_{R-1}$$, and a subsequent local transpose yields the final result $$\tilde{M}^T_0 // \tilde{M}^T_1 // ... // \tilde{M}^T_{R-1}$$. This is the same matrix, as the transpose $$\tilde{M}^T_r$$ of the columns $$V_r$$ of $$M$$ is the block $$(M^T)_r$$ of the rows $$V_r$$ of $$M^T$$.

As $$\mathrm{Transpose}$$ is a composition of two involutory functions, and because both functions commute, i.e. $$\mathrm{ViewSwap} \circ \mathrm{LocalTranspose} = \mathrm{LocalTranspose} \circ \mathrm{ViewSwap}$$, $$\mathrm{Transpose}$$ is also involutory ([Elements of Operator Theory, Carlos S. Kubrusly, Springer 2011][kubrusly2011elements]). Two transposes in a row swap the two functions in the middle, and then each function cancels itself:

$$
\begin{aligned}
\mathrm{Transpose} \circ \mathrm{Transpose} &= \mathrm{ViewSwap} \circ (\mathrm{LocalTranspose} \circ \mathrm{ViewSwap}) \circ \mathrm{LocalTranspose} \\
&= (\mathrm{ViewSwap} \circ \mathrm{ViewSwap}) \circ (\mathrm{LocalTranspose} \circ \mathrm{LocalTranspose}),
\end{aligned}
$$

which is the identity, as both compositions in parentheses are.

[kubrusly2011elements]: https://link.springer.com/content/pdf/10.1007/978-0-8176-4998-2.pdf

## The view swap in MPI

At the level of a rank, the matrix data layout after a view swap is unknown until the swap is performed, as ranks do not hold information about the matrix structure across other ranks. Therefore, a communication step (`MPI_Allgather`) must be executed beforehand to gather the number of rows held by individual ranks and compute the row interval of every rank $$r$$ as:

$$
\left[ \sum^{r-1}_{s=0} \vert V_s \vert, \,\, \sum^{r}_{s=0} \vert V_s \vert \right)
$$

where $$\vert V_s \vert$$ is the number of rows of rank $$s$$. The start of each interval is the [prefix sum](https://en.wikipedia.org/wiki/Prefix_sum) of the row counts of the previous ranks (an exclusive scan). In the example, each rank holds 2 rows, so the intervals are $$[0, 2)$$, $$[2, 4)$$ and $$[4, 6)$$.

This information is required to match each column/row id to its target rank in the sparse transpose steps that follow: the cells of column 4, for instance, go to rank 2, whose interval $$[4, 6)$$ holds the row 4 of $$M^T$$. The view swap algorithm then follows in two communication steps.

The first step performs a dense matrix transpose (`MPI_Alltoall`) to send/receive the number of elements to/from other ranks. The second step performs a selective scatter-gather (`MPI_Alltoallv`) that delivers the elements to the final rank. Both steps follow the local transpose, which sorts the elements of each rank by column id, so that the elements for each destination rank form one contiguous segment of the send buffer. The following picture illustrates both steps on the sample matrix above: the transpose of the element counts (left), and the exchange of the elements with their row and column ids (right):

{: .picture}
<img width="92.7%" height="92.7%" src="{{ site.assets }}/matrix-transpose/matrix-transposition-crs.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

In the picture, rank 0 holds A, B, C and D, in the columns 1, 2, 4 and 5, which fall in the row intervals of ranks 0, 1, 2 and 2: it sends 1, 1 and 2 elements to ranks 0, 1 and 2, the first row of the outgoing counts. Ranks 1 and 2 send 2, 1 and 1 elements each. The transpose of these counts gives the number of elements that each rank receives from each rank: rank 0 receives 1, 2 and 2 elements, i.e. A, then J and G, then R and O.

A memory realignment based on column id and row id leads to the final distributed matrix dataset: rank 0 orders its elements into J and R, the row 0 of $$M^T$$, then A, G and O, its row 1, as in layout (d).

The full transpose algorithm performs only three collective communication calls: an `MPI_Allgather` to compute the row offsets of ranks, and one `MPI_Alltoall` and one `MPI_Alltoallv` to transpose the element counts and the cell values in the view swap. The first two calls move $$R$$ integers per rank, and the `MPI_Allgather` can run once, at the onset of execution, as the rows of each rank do not change. The `MPI_Alltoallv` sends every element with its row and column ids, so it carries most of the data: 9 of the 12 elements of the example change rank. Moreover, the involutory property of this transpose method is of high importance, as it guarantees that the distributed transpose can be validly executed any number of times while respecting the data integrity of a CSR-based graph problem representation.

## Higher Cardinality Matrices

Having a single edge between each pair of nodes is often not the case in real-life situations, which then require [multigraphs](https://en.wikipedia.org/wiki/Multigraph). Take as an example the matrix representing information about common interests between pairs of users in a social network: some pairs of users may be connected by hundreds of common interests, while others may have barely any interests in common. Another example is a URL cross-referral map, where one may be required to store not only the number of hyperlinks between pages, but also additional information about every link. Each cell of such a matrix holds a list of values, whose number, the **cardinality** of the cell, differs from cell to cell: these are **high-cardinality** matrices. How to implement the transpose in this case?

The solution starts by defining a data format that can hold high-cardinality cells. It is called the **extended CSR**, or XCSR for short, and is defined as the previous CSR data structure with an added dimension that holds the number of values per matrix cell (or edges between pairs of nodes), `cell_counts` in the picture. The layout is as follows:

{: .picture}
<img width="92.3%" height="92.3%" src="{{ site.assets }}/matrix-transpose/xcrs_format.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

Rank 0 now holds 6 values in the same 4 cells: A in cell (0, 1), B and C in (0, 2), D and E in (0, 5), and F in (1, 4). Its `cell_counts` are thus 1, 2, 2 and 1, and its other arrays are those of the CSR picture.

The transpose algorithm then follows in three steps. First, each compute node (rank) collects the metadata of its own sub-matrix, composed of the set of triplets of the form [row id, column id, number of values], indicating the row, column and cardinality of each cell. Then the ranks perform a local transpose of their own data, by reordering their triplets by column and row id. This sequence illustrates the first step:

{: .picture}
<img width="89.0%" height="89.0%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_1.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

The second step starts the view swap method. All ranks pack their triplets and exchange them (i.e. an all-to-all communication), sending each triplet to the recipient of its cell in the final transposed matrix. In practice, the previous local transpose operation already sorted each sub-matrix by columns (i.e. by rows of the final transposed dataset), so the triplets are already grouped by destination rank.

More importantly, the matrix composed of all triplets across all ranks is a distributed matrix with a fixed element size per cell (a tuple of fixed size). Therefore, this exchange is performed by the same view swap as the CSR matrix, presented in the section *The view swap in MPI*, with the triplets as elements. As the two matrices have the same 12 cells, the counts are also the same: rank 0 sends 1, 1 and 2 triplets to ranks 0, 1 and 2, and ranks 1 and 2 send 2, 1 and 1 each. After the exchange, all ranks will hold the triplets of the metadata (*but not the data yet*) of the final matrix structure:

{: .picture}
<img width="92.7%" height="92.7%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_2.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

Finally, the view swap is completed by communicating the data, based on the metadata of the transposed matrix. First, each rank counts the number of values to be sent to each rank: rank 0 sends 1 value to itself (A), 2 to rank 1 (B and C) and 3 to rank 2 (F, D and E). The matrix of distributed counts of all values to be sent and received per rank is a dense matrix that can be transposed easily (`MPI_Alltoall`): rank 0 receives 1, 3 and 3 values from ranks 0, 1 and 2, rank 1 receives 2, 2 and 1, and rank 2 receives 3, 3 and 2.

These counts are then used to allocate the buffers for the incoming data and to perform an `MPI_Alltoallv` that sends/delivers the values of the final matrix across all compute nodes. Upon completion, the (previously column-oriented) values are sorted by row and column, which yields the final dataset:

{: .picture}
<img width="93.1%" height="93.1%" src="{{ site.assets }}/matrix-transpose/xcrs_transpose_3.svg">

{: .caption}
Adapted from: [Efficient Distributed Transposition Of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012)

In the picture, rank 0 receives A from itself, J, K and G from rank 1, and R, S and O from rank 2. The triplets received in the second step give the cell and the cardinality of each group of values, so the sort by row and column id puts J and K, then R and S, in row 0 of $$M^T$$, and A, G and O in row 1.

To sum up, the full transpose algorithm requires five collective communication calls: an `MPI_Allgather` to compute the row offsets of ranks, and two `MPI_Alltoall` and two `MPI_Alltoallv` for the metadata and cell values transpose. These are two calls more than for a CSR matrix, as the metadata and the values are exchanged separately, so the CSR transpose is preferred for matrices with a single value per cell. The same holds when every cell has the same number of values, as a cell can then be sent as one element of fixed size (an MPI derived datatype). Moreover, the general applicability of the XCSR and the involutory property of this transpose method are two properties of high importance. They guarantee that the distributed transpose can be validly executed any number of times while respecting the data integrity of an XCSR-based graph problem representation.

## Further details

For more details, implementation and benchmark information, the <a href="{{ site.assets }}/matrix-transpose/matrix-transposer.pdf">original publication (preprint)</a> and the <a href="{{ site.assets }}/matrix-transpose/matrix-transposer.zip">source code</a> are available for download.

To find more about the topic, the [slides of chapter 4](https://www.cs.purdue.edu/homes/ayg/book/Slides/chap4_slides.pdf) of the book Introduction to Parallel Computing by Grama, Gupta, Karypis and Kumar cover the all-to-all personalized communication, with the transpose of a dense matrix as an example.

