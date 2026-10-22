---
layout: post
title:  "Distributed Matrix Multiplication and Database Joins"
categories: ["Distributed Algorithms"]
tags: ["MPI", "linear algebra", "sparse matrices", "CSR", "databases", "joins"]
---

The matrix and the table are probably the most common data structures in computing: matrices hold the data of scientific simulations and machine learning models, and tables hold the records of databases. On a single compute node, their main operations, the matrix multiplication and the database join, are well studied, with fast algorithms and libraries (e.g. [BLAS][blas] for matrices). However, when the data is spread across a network of compute nodes (i.e. a [distributed memory][dist-mem] environment), each node reads only its own memory, and any other data must arrive as a message, which is much slower.

This post describes how to split the data of both operations so that each node works *mostly alone*, for dense matrices, sparse matrices and database tables. The algorithms use the terminology of [MPI][mpi]: they run on $$p$$ processes, called **ranks** (usually one per compute node), or **servers** in the database part.

[blas]: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
[dist-mem]: https://en.wikipedia.org/wiki/Distributed_memory
[mpi]: https://en.wikipedia.org/wiki/Message_Passing_Interface

## Part 1: dense matrices

The product $$C = AB$$ of two $$n \times n$$ matrices is given by:

$$
C_{ij} = \sum_{k=0}^{n-1} A_{ik} \, B_{kj},
$$

where each element takes $$n$$ [multiply-adds][mac], i.e. $$n^3$$ operations for the whole matrix, while the matrix-vector product $$y = Ax$$ takes $$n^2$$ operations.

[mac]: https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation

### Warm-up: the matrix-vector product

Each rank holds $$n/p$$ consecutive rows of $$A$$, and the matching entries of $$x$$ and $$y$$. To compute $$y_i$$, a rank needs row $$i$$ of $$A$$, which it holds, and *all* of $$x$$, which it does not. Therefore, the algorithm runs in two steps, illustrated below on 4 ranks:
1. an **[all-gather][collective]** (`MPI_Allgather`) sends the entries of $$x$$ of each rank to all other ranks;
2. each rank multiplies its rows of $$A$$ by $$x$$.

{: .picture}
<img width="93.2%" height="93.2%" src="{{ site.assets }}/distributed-matmul-dbs/matvec_row1d.svg">

{: .caption}
Matrix-vector product on 4 ranks, colored by rank. **Left:** the initial layout. **Right:** rank 1 after the all-gather, ready to compute $$y_2$$ and $$y_3$$.

Each rank performs $$n^2/p$$ operations and receives about $$n$$ entries of $$x$$. Note that $$y$$ ends up split like $$x$$, ready for the next product of an iterative solver, such as the [conjugate gradient method][cg].

[collective]: https://en.wikipedia.org/wiki/Collective_operation
[cg]: https://en.wikipedia.org/wiki/Conjugate_gradient_method

### The matrix-matrix product: shifting tiles

The same idea fails for $$C = AB$$: each rank would need all of $$B$$, and an all-gather would store the $$n^2$$ elements of $$B$$ on every rank, which defeats the purpose of splitting the data. Instead, each rank holds only one piece of each matrix. At every step, it multiplies the pieces that it holds, and then **shifts** some of them to a neighbor, cyclically (i.e. the first rank sends to the last). The shifted pieces belong to one matrix, or to two.

### Shifting one matrix: the ring algorithm

The ranks form a ring, and rank $$i$$ holds row block $$i$$ of $$A$$, $$B$$ and $$C$$. Its block of $$A$$ is cut into $$p$$ tiles $$A_{i,k}$$, one for each row block $$B_k$$ of $$B$$, so that row block $$i$$ of $$C$$ is computed as:

$$
C_i = \sum_{k=0}^{p-1} A_{i,k} \, B_k,
$$

where each tile $$A_{i,k}$$ has $$n/p \times n/p$$ elements, and each block $$B_k$$ has $$n/p \times n$$ elements. $$A$$ and $$C$$ stay in place, and only $$B$$ moves. At each step, every rank multiplies the tile of $$A$$ that matches its current block of $$B$$, adds the result to $$C_i$$, and passes the block of $$B$$ to the previous rank. After $$p$$ steps, $$C_i$$ is complete, as shown below on 3 ranks:

{: .picture}
<img width="88.7%" height="88.7%" src="{{ site.assets }}/distributed-matmul-dbs/ring.svg">

{: .caption}
The ring algorithm on 3 ranks. Blocks of $$B$$ shift up, and rank 0 sends to rank 2. At each step, a rank multiplies the tile of $$A$$ that has the color of its block of $$B$$.

The output $$C$$ is split by rows, like $$A$$. In the *mirror* case, $$B$$ stays in place, the column blocks of $$A$$ move, and $$C$$ is split by columns. Either way, each rank stores $$3n^2/p$$ elements, but it receives $$p-1$$ blocks, i.e. almost $$n^2$$ elements, however many ranks are added.

### Shifting two matrices: Cannon's algorithm

To send less data, the matrices are cut in both dimensions, which is the idea of [Cannon's algorithm][cannon], first described by Lynn Elliot Cannon in 1969. The ranks form a $$q \times q$$ grid (with $$q = \sqrt{p}$$), and each matrix is cut into $$q \times q$$ tiles. Rank $$(i,j)$$ starts with $$A_{i,j}$$ and $$B_{i,j}$$, and computes:

$$
C_{i,j} = \sum_{k=0}^{q-1} A_{i,k} \, B_{k,j}.
$$

As an example, rank $$(1,2)$$ of a $$3 \times 3$$ grid computes $$C_{1,2} = A_{1,0} B_{0,2} + A_{1,1} B_{1,2} + A_{1,2} B_{2,2}$$. It needs the tiles of row 1 of $$A$$ and of column 2 of $$B$$, but only *one* matching pair at a time. Cannon's algorithm delivers these pairs in three steps:
1. row $$i$$ of $$A$$ is shifted left by $$i$$, and column $$j$$ of $$B$$ up by $$j$$ (the **alignment**), so that rank $$(i,j)$$ holds the matching pair $$A_{i,k}$$ and $$B_{k,j}$$, with $$k = (i+j) \bmod q$$;
2. every rank adds $$A_{i,k} B_{k,j}$$ to $$C_{i,j}$$;
3. $$A$$ is shifted left by one and $$B$$ up by one, so that rank $$(i,j)$$ holds the next matching pair, $$A_{i,k+1}$$ and $$B_{k+1,j}$$ (indices modulo $$q$$).

Steps 2 and 3 are repeated until each rank has done $$q$$ multiplications, as illustrated below:

{: .picture}
<img width="89.5%" height="89.5%" src="{{ site.assets }}/distributed-matmul-dbs/cannon.svg">

{: .caption}
Cannon's algorithm on a $$3 \times 3$$ grid of ranks. Tiles of $$A$$ (top) shift left, and tiles of $$B$$ (bottom) shift up. The thick frame marks rank $$(1,2)$$, and colors show its three pairs: blue for $$k=0$$, green for $$k=1$$ and red for $$k=2$$.

In brief, $$A$$ moves along the grid rows, $$B$$ along the grid columns, and $$C$$ stays, so that tile $$C_{i,j}$$ ends on rank $$(i,j)$$, in the same layout as the inputs. Note that, on a 2D grid, two of the three matrices must move. Each rank still stores $$3n^2/p$$ elements, but receives at most $$2q$$ tiles, or $$2n^2/\sqrt{p}$$ elements. Unlike in the ring algorithm, this volume shrinks as ranks are added, and it is the smaller of the two from 9 ranks on.

Cannon's algorithm needs a square grid of ranks. To find more about the topic, [SUMMA][summa] (van de Geijn and Watts, 1995) replaces the shifts with broadcasts along the grid rows and columns, and works on rectangular grids. The [2.5D algorithm][solomonik] (Solomonik and Demmel, 2011) stores extra copies of the matrices to send even less data.

[cannon]: https://en.wikipedia.org/wiki/Cannon%27s_algorithm
[summa]: https://www.netlib.org/lapack/lawnspdf/lawn96.pdf
[solomonik]: https://doi.org/10.1007/978-3-642-23397-5_10

### Overlapping computation and communication

The tiles of the next step do not depend on the current product, so a rank can receive them while it multiplies:
1. the rank starts a non-blocking send of the current tiles, and a non-blocking receive of the next ones into a second buffer (`MPI_Isend`, `MPI_Irecv`);
2. it multiplies the current tiles;
3. it waits for both messages (`MPI_Waitall`), and swaps the buffers.

The following picture compares the timelines with and without this overlap:

{: .picture}
<img width="91.6%" height="91.6%" src="{{ site.assets }}/distributed-matmul-dbs/overlap.svg">

{: .caption}
Timeline of one rank in Cannon's algorithm on a $$3 \times 3$$ grid. **Top:** shifts and multiplications take turns. **Bottom:** each shift runs during a multiplication.

This technique, called **[double buffering][buffering]**, costs one extra pair of tiles, and works for the ring algorithm too. A shift is hidden whenever the multiplication takes longer. In practice, a multiplication costs $$(n/q)^3$$ operations and a shift moves $$2(n/q)^2$$ elements, so large tiles hide the communication easily.

[buffering]: https://en.wikipedia.org/wiki/Multiple_buffering

## Part 2: sparse matrices

Many matrices are mostly zeros, e.g. in [finite element][fem] simulations, where each point of a mesh interacts only with a few neighbors. Such matrices are called **[sparse][sparse]**, and a sparse format stores only their nonzeros.

[fem]: https://en.wikipedia.org/wiki/Finite_element_method
[sparse]: https://en.wikipedia.org/wiki/Sparse_matrix

### The compressed sparse row format

The **compressed sparse row** format (CSR, also called the Yale format), in use since the mid-1960s, stores a matrix in three arrays:
- `values`: the nonzeros, row by row;
- `col_idx`: the column of each nonzero;
- `row_ptr`: where each row starts, i.e. row $$i$$ is at positions `row_ptr[i]` to `row_ptr[i+1]-1` of the other two arrays.

An example is displayed below:

{: .picture}
<img width="69.8%" height="69.8%" src="{{ site.assets }}/distributed-matmul-dbs/csr_format.svg">

{: .caption}
A sparse matrix (left) and its CSR arrays (right), colored by row. Row 3 is empty, so `row_ptr[3]` and `row_ptr[4]` are both 6.

For $$n$$ rows and $$nnz$$ nonzeros, CSR stores $$2 \, nnz + n + 1$$ numbers (22 in the example above, instead of 30 for the dense layout). Reading a row is easy, but reading a column requires a search through all rows, which makes the distributed transpose hard. The product $$y = Ax$$ visits each row and each nonzero once, in $$O(n + nnz)$$ operations:

```cpp
for (int i = 0; i < n; i++) {
  y[i] = 0;
  for (int k = row_ptr[i]; k < row_ptr[i+1]; k++)
    y[i] += values[k] * x[col_idx[k]];
}
```

### The sparse matrix-vector product

As before, $$A$$ is split by rows. An all-gather of $$x$$ would work, but a rank needs $$x_j$$ only for the columns $$j$$ where its rows have a nonzero, which is usually a small set. So only those entries are sent:
1. once, in a setup step, each rank tells the owners of $$x$$ which of their entries it needs;
2. before every product, an exchange (`MPI_Alltoallv`) sends these entries to the ranks that need them. The received copies are called **ghost** entries;
3. each rank runs the CSR loop over its own and ghost entries.

The following picture illustrates the exchange on 3 ranks:

{: .picture}
<img width="64.4%" height="64.4%" src="{{ site.assets }}/distributed-matmul-dbs/spmv.svg">

{: .caption}
Sparse matrix-vector product on 3 ranks. **Left:** nonzeros colored by rank. A dot marks a nonzero whose entry of $$x$$ is on another rank. **Right:** the entries of $$x$$ on each rank after the exchange. Dashed boxes are ghost entries.

In the picture above, rank 0 needs $$x_2$$ and $$x_5$$ from the other ranks, and ranks 1 and 2 need one entry each. The communication now grows with the number of ghost entries, and *not* with $$n$$. For a matrix from a mesh, these are only the entries at the border of the region of each rank. In practice, a [graph partitioning][partition] tool assigns the rows to the ranks, so that the regions have equal sizes and short borders.

[partition]: https://en.wikipedia.org/wiki/Graph_partition

### The sparse matrix-matrix product

Row $$i$$ of $$C = AB$$ is given by a weighted sum of rows of $$B$$:

$$
C_{i,:} = \sum_{k \, : \, A_{ik} \neq 0} A_{ik} \, B_{k,:},
$$

where $$B_{k,:}$$ is row $$k$$ of $$B$$. So a rank needs row $$k$$ of $$B$$ whenever its rows of $$A$$ have a nonzero in column $$k$$, i.e. the same indices as the ghost entries, but whole rows instead of single entries. This row-by-row formulation ([Gustavson, 1978][gustavson]) suits CSR, which stores both matrices by rows, and leads to the following algorithm:
1. each rank requests the remote rows of $$B$$ it needs;
2. the owners send them in CSR, with two `MPI_Alltoallv` calls: first the row sizes, so that the receiver can allocate memory, then the indices and values;
3. each rank builds its rows of $$C$$ in a dense buffer, and compresses them into CSR.

The number of nonzeros of $$C$$ is *not known in advance*, so step 3 often runs twice: a **symbolic** pass counts the nonzeros and allocates memory, and a **numeric** pass computes the values. The output $$C$$ is split by rows, like $$A$$.

[gustavson]: https://doi.org/10.1145/355791.355796

To find more about the topic, [Çatalyürek and Aykanat][hypergraph-spmv] assign the rows of a sparse matrix to the ranks with a [hypergraph][hypergraph] instead of a graph, as the hypergraph counts the ghost entries of the matrix-vector product exactly. [Buluç and Gilbert][sparse-summa] note that the row split of the sparse matrix-matrix product does not scale to thousands of ranks, and split both matrices on a 2D grid of ranks instead, as SUMMA does for dense matrices.

[hypergraph-spmv]: https://doi.org/10.1109/71.780863
[hypergraph]: https://en.wikipedia.org/wiki/Hypergraph
[sparse-summa]: https://arxiv.org/abs/1109.3739

## Part 3: database joins

In 1970, Edgar F. Codd proposed the [relational model][relational], where the data is stored in tables, with one record per row. A **key** is a column that identifies a record, and links tables. In the example below, `Person` has one row per person, with key `id`, and `Messages` has one row per message, whose `id` is the key of its author (a [foreign key][fkey]).

[relational]: https://en.wikipedia.org/wiki/Relational_model
[fkey]: https://en.wikipedia.org/wiki/Foreign_key

### Inner and outer joins

A **[join][join]** pairs the rows of two tables whose keys match, as in this [SQL][sql] query:

```sql
SELECT Person.id, name, age, msg, year
FROM Person INNER JOIN Messages ON Person.id = Messages.id;
```

There are four kinds of inner and outer joins:
- **inner join**: only the matching pairs. In the example below, Kim wrote two messages, so Kim appears twice, while Mary, Joe and the message with key 9 have no match, and are dropped;
- **left outer join**: also the unmatched rows of the left table (`Person`), with `NULL` in the other columns;
- **right outer join**: also the unmatched rows of the right table (`Messages`);
- **full outer join**: also the unmatched rows of both tables.

[join]: https://en.wikipedia.org/wiki/Join_(SQL)
[sql]: https://en.wikipedia.org/wiki/SQL

The following picture shows the two tables and three of these joins:

{: .picture}
<img width="90.5%" height="90.5%" src="{{ site.assets }}/distributed-matmul-dbs/joins.svg">

{: .caption}
**Top:** the tables `Person` (blue) and `Messages` (green). **Bottom:** three joins on `id`. Dark gray cells are `NULL`.

On one node, there are three classic algorithms to join tables of $$N$$ and $$M$$ rows (their costs exclude the size of the output):
- **[nested loop][nlj]**: compares all pairs of rows, in $$O(N \cdot M)$$;
- **[sort-merge][smj]**: sorts both tables by key and scans them together, in $$O(N \log N + M \log M)$$, or $$O(N + M)$$ if already sorted;
- **[hash join][hj]**: hashes the keys of the smaller table and looks up the rows of the other, in $$O(N + M)$$ on average.

[nlj]: https://en.wikipedia.org/wiki/Nested_loop_join
[smj]: https://en.wikipedia.org/wiki/Sort-merge_join
[hj]: https://en.wikipedia.org/wiki/Hash_join

In practice, the hash join is the usual choice for large tables, and the sort-merge join is preferred when both tables are already sorted by key. The nested loop suits small tables, or join conditions other than an equality of keys. For outer joins, each row also records whether it found a match, and the rows without one are returned with `NULL`. However, all these algorithms assume that rows with the same key are on the same node, and the following sections show how to ensure this for three layouts.

### Layout 1: each table on a different server

Here, `Person` is on server 1, and `Messages` is on server 2.

**Inner join.** Sending a whole table works, but a **semi-join** ([Bernstein and Chiu, 1981][semijoin]) sends less:
1. server 1 sends its keys to server 2;
2. server 2 returns only its rows with one of these keys;
3. server 1 joins them with its table, locally.

**Outer joins.** In step 3, server 1 also returns its unmatched persons, with `NULL` (the left outer join). Server 2 knows which of its messages had no match, as their keys were not in the list of step 1, so it returns them with `NULL` in a step 4. The inner join and step 4 give the right outer join, and steps 3 and 4 the full outer join, as displayed below:

{: .picture}
<img width="92.6%" height="92.6%" src="{{ site.assets }}/distributed-matmul-dbs/layout_tables.svg">

{: .caption}
A semi-join between two servers: the keys (1), the matching rows (2), the left outer join on server 1 (3), and the unmatched messages of server 2 (4). The first three rows of (3) are the inner join.

For a long list of keys, a [Bloom filter][bloom] is a smaller substitute. It lets a few wrong rows (*false positives*) through in step 2, and server 1 treats them as unmatched messages.

[semijoin]: https://doi.org/10.1145/322234.322238
[bloom]: https://en.wikipedia.org/wiki/Bloom_filter

### Layout 2: tables split by rows

Each server holds some rows of each table, which is called **horizontal partitioning**, or **[sharding][shard]**. Matching rows may then sit on different servers: below, John is on server 2, but his message "Bye" is on server 1. A local join misses this pair, and a local left outer join is even *wrong*, as it returns John with `NULL`.

[shard]: https://en.wikipedia.org/wiki/Shard_(database_architecture)

**Inner join.** Each row is placed by its key, with the same rule for both tables (e.g. a hash of the key). Below, even keys go to server 1, and odd keys to server 2. Matching rows then share a server, and every server joins locally. If the tables are not placed this way, a **shuffle** first sends each row to its server, with an all-to-all exchange (`MPI_Alltoallv`), as illustrated below:

{: .picture}
<img width="87.5%" height="87.5%" src="{{ site.assets }}/distributed-matmul-dbs/layout_rows.svg">

{: .caption}
Two tables split by rows across 2 servers, colored by target server: blue for server 1 (even keys), and orange for server 2 (odd keys). **Left:** before the shuffle. **Right:** after it.

**Outer joins.** After a shuffle, outer joins are local too, as a row without a match on its server has no match anywhere. Above, server 2 returns Mary and Joe for the left outer join, and "Hey" for the right outer join. However, a shuffle may cause a [load imbalance][balance], as all rows with a frequent key go to the same server.

[balance]: https://en.wikipedia.org/wiki/Load_balancing_(computing)

**Broadcast join.** If `Person` is small, it can be copied to every server instead, while `Messages` stays in place, so that a frequent key does not overload any server. Each message then sits with all persons, so the inner and right outer joins are local.

The left outer join is not: John matches on server 1 but not on server 2, so no server can tell alone that a person has no match. Therefore, each server flags its local matches, and a logical-or all-reduce (`MPI_Allreduce` with `MPI_LOR`) combines the flags. Server 1 then returns the persons with a final flag of 0, with `NULL`, as presented below:

{: .picture}
<img width="69.6%" height="69.6%" src="{{ site.assets }}/distributed-matmul-dbs/broadcast.svg">

{: .caption}
A broadcast join, for a left outer join. After the all-reduce of the match flags, only Mary and Joe have no match.

### Layout 3: tables split by columns

Each server holds some columns, for all rows. This is called **vertical partitioning**, as in [column-oriented databases][column-db], and the rows are aligned by position, i.e. row $$r$$ on every server belongs to the same record. Below, server 1 holds the keys, server 2 the other columns of `Person`, and server 3 the other columns of `Messages`.

[column-db]: https://en.wikipedia.org/wiki/Column-oriented_DBMS

**Inner join.** Only the keys decide which rows match, so the keys are matched first, and the other columns move last:
1. server 1 matches the keys into a **join index** ([Valduriez, 1987][joinindex]): a list of pairs of row numbers, one from each table;
2. server 1 sends the `Person` row numbers to server 2, and the `Messages` row numbers to server 3;
3. servers 2 and 3 gather their rows in that order.

In practice, only *row numbers* cross the network, and the result stays split by columns. Building the output rows only at the end, and only for the rows that match, is called **late materialization**.

[joinindex]: https://doi.org/10.1145/22952.22955

**Outer joins.** The join index also holds pairs with one side missing, such as Mary (row 1) and Joe (row 3) in the left outer join, and the servers write `NULL` where a row number is missing, as illustrated below:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-matmul-dbs/layout_columns.svg">

{: .caption}
A left outer join of tables split by columns across 3 servers. Column `#` is the implicit row number. Server 1 builds the join index (1) and sends row numbers (2), and servers 2 and 3 gather their rows (3). The first three rows are the inner join.

To find more about the topic, [Blanas et al.][blanas] compare the shuffle join (called *repartition join* there), the broadcast join and the semi-join on a cluster of 100 nodes running MapReduce, and say when to use each. The survey of [Abadi et al.][column-stores] on column-oriented databases details late materialization (section 4.4) and the joins of column stores (section 4.5).

[blanas]: https://doi.org/10.1145/1807167.1807273
[column-stores]: https://doi.org/10.1561/1900000024

## Summary

To sum up, both problems follow one rule: the data that is used together should be kept *together*. When the layout does not fit the operation, a few communication steps move only the data that is needed, overlapped with the computation where possible.

In practice, the ring algorithm works for any number of ranks, while Cannon's algorithm sends less data from 9 ranks on, but needs a square grid. A semi-join pays off when few rows match, a shuffle when both tables are large, a broadcast join when one table is small or a few keys are frequent, and a join index when the tables are stored by columns. The following tables compare the methods:

| Matrix algorithm | What moves | Output split by | Received per rank |
|---|---|---|---|
| matrix-vector | all of $$x$$ | rows | $$\approx n$$ |
| ring | row blocks of $$B$$ | rows | $$\approx n^2$$ |
| Cannon | tiles of $$A$$ and $$B$$ | tiles | $$2n^2/\sqrt{p}$$ |
| sparse matrix-vector | ghost entries of $$x$$ | rows | the ghost entries |
| sparse matrix-matrix | rows of $$B$$ at the ghost indices | rows | these rows |

| Join layout | Inner join | Outer joins | Sent over the network |
|---|---|---|---|
| each table on a server | semi-join | each server adds its unmatched rows | keys, matching rows |
| split by rows | shuffle by key, or broadcast a small table | local after a shuffle, all-reduce of flags after a broadcast | moved rows, or the small table |
| split by columns | join index | pairs with a missing side | row numbers |
