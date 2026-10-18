---
layout: post
title:  "Distributed Matrix Multiplication and Database Joins"
categories: ["Distributed Algorithms"]
tags: ["MPI", "linear algebra", "sparse matrices", "CSR", "databases", "joins"]
---

Scientific computing and data processing rely on two data structures: the matrix and the table. When the data is too large for one machine, it is split across many compute nodes. Each node can read only its own memory, and any other data must arrive as a message over the network, which is slow. Two operations, the matrix multiplication and the database join, therefore raise the same question: how should the data be split so that each node can work mostly alone?

The algorithms are described in the terms of the [Message Passing Interface][mpi] (MPI): a program runs as $$p$$ processes, called **ranks**, usually one per node. In the database part, the ranks are called **servers**.

[mpi]: https://en.wikipedia.org/wiki/Message_Passing_Interface

## Part 1: dense matrices

The product $$C = AB$$ of two $$n \times n$$ matrices, with $$C_{ij} = \sum_k A_{ik} B_{kj}$$, costs $$n^3$$ [multiply-adds][mac], since each of the $$n^2$$ entries of $$C$$ is a sum of $$n$$ products. The matrix-vector product $$y = Ax$$ costs $$n^2$$.

[mac]: https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation

### The matrix-vector product

The matrix-vector product is a good warm-up, because it already shows the central problem of a distributed algorithm: a rank does not hold all the data it needs. Each rank holds $$n/p$$ consecutive rows of $$A$$, and the matching entries of $$x$$ and $$y$$. To compute $$y_i$$, a rank needs row $$i$$ of $$A$$, which it holds, and all of $$x$$, which it does not. Two steps solve this:
1. **[all-gather][collective]** (`MPI_Allgather`): every rank sends its entries of $$x$$ to all other ranks;
2. **local product:** every rank multiplies its rows of $$A$$ by $$x$$.

[collective]: https://en.wikipedia.org/wiki/Collective_operation

{: .picture}
<img width="93.8%" height="93.8%" src="{{ site.assets }}/distributed-matmul-dbs/matvec_row1d.svg">

{: .caption}
Matrix-vector product on 4 ranks, colored by rank. **Left:** the initial layout. **Right:** rank 1 after the all-gather, ready to compute $$y_2$$ and $$y_3$$.

Each rank performs $$n^2/p$$ multiply-adds, and receives the $$n - n/p$$ entries of $$x$$ that it does not hold. The result $$y$$ ends up split like $$x$$, so it is ready for the next product.

### The matrix-matrix product: shifting tiles

The same approach does not carry over to $$C = AB$$. Each rank would need all of $$B$$, and an all-gather would store all $$n^2$$ elements of $$B$$ on every rank, which defeats the purpose of distributing the matrix. Instead, each rank holds only one piece of each matrix at a time. At every step, it multiplies the pieces it holds, and then **shifts** some of them to a neighbor. The shifts wrap around, i.e. the first rank sends to the last one. Which pieces move? One matrix, or two.

### Shifting one matrix: the ring algorithm

In the simplest scheme, the ranks form a ring. Rank $$i$$ holds row block $$i$$ of $$A$$, $$B$$ and $$C$$, of $$n/p$$ rows each, and its block of $$A$$ is cut into $$p$$ tiles $$A_{i,k}$$, one for each block $$B_k$$ of $$B$$, so that

$$
C_i = \sum_{k=0}^{p-1} A_{i,k} \, B_k.
$$

Each term needs one tile of $$A$$, which the rank holds, and one block of $$B$$, which travels around the ring. So $$A$$ and $$C$$ stay in place, and only $$B$$ moves. At each step, every rank multiplies the tile of $$A$$ that matches its current block of $$B$$, adds the result to $$C_i$$, and passes the block of $$B$$ to the previous rank. After $$p$$ steps, every block of $$B$$ has visited every rank, and $$C_i$$ is complete.

{: .picture}
<img width="88.7%" height="88.7%" src="{{ site.assets }}/distributed-matmul-dbs/ring.svg">

{: .caption}
The ring algorithm on 3 ranks. Blocks of $$B$$ shift up, and rank 0 sends to rank 2. At each step, a rank multiplies the tile of $$A$$ that has the color of its block of $$B$$.

The output $$C$$ is split by rows, like $$A$$. In the mirror case, $$B$$ stays, the column blocks of $$A$$ move, and $$C$$ is split by columns. Each rank stores $$3n^2/p$$ elements, one block of each matrix, but it receives $$p-1$$ blocks of $$n^2/p$$ elements, i.e. $$(p-1) \, n^2/p \approx n^2$$ elements, regardless of $$p$$. Adding ranks divides the work, but not the communication.

### Shifting two matrices: Cannon's algorithm

To send less, the matrices are cut in both dimensions. The ranks form a $$q \times q$$ grid, with $$q = \sqrt{p}$$, and each matrix is cut into $$q \times q$$ tiles. Rank $$(i,j)$$ starts with $$A_{i,j}$$ and $$B_{i,j}$$, and computes

$$
C_{i,j} = \sum_{k=0}^{q-1} A_{i,k} \, B_{k,j}.
$$

For example, rank $$(1,2)$$ of a $$3 \times 3$$ grid computes $$C_{1,2} = A_{1,0} B_{0,2} + A_{1,1} B_{1,2} + A_{1,2} B_{2,2}$$. It needs the tiles of row 1 of $$A$$ and of column 2 of $$B$$, but only one matching pair at a time. [Cannon's algorithm][cannon] delivers these pairs:
1. **align:** shift row $$i$$ of $$A$$ left by $$i$$, and column $$j$$ of $$B$$ up by $$j$$. Rank $$(i,j)$$ now holds the matching pair $$A_{i,k}$$ and $$B_{k,j}$$, with $$k = (i+j) \bmod q$$;
2. **multiply:** add $$A_{i,k} B_{k,j}$$ to $$C_{i,j}$$;
3. **shift:** shift $$A$$ left by one and $$B$$ up by one. Rank $$(i,j)$$ receives the tile of $$A$$ of its right neighbor and the tile of $$B$$ of the neighbor below, which form the pair for $$k+1$$ (modulo $$q$$), so the new pair matches again.

Steps 2 and 3 repeat until each rank has done $$q$$ multiplications.

{: .picture}
<img width="89.2%" height="89.2%" src="{{ site.assets }}/distributed-matmul-dbs/cannon.svg">

{: .caption}
Cannon's algorithm on a $$3 \times 3$$ grid of ranks. Tiles of $$A$$ (top) shift left, and tiles of $$B$$ (bottom) shift up. The thick frame marks rank $$(1,2)$$, and colors show its three pairs: blue for $$k=0$$, green for $$k=1$$ and red for $$k=2$$.

$$A$$ moves along the grid rows, $$B$$ along the grid columns, and $$C$$ stays in place. Tile $$C_{i,j}$$ ends on rank $$(i,j)$$, in the same layout as the inputs. On a 2D grid, two of the three matrices must move. Each rank still stores $$3n^2/p$$ elements, but it receives at most $$2q$$ tiles, two in the alignment and two at each of the $$q-1$$ shifts, or $$2n^2/\sqrt{p}$$ elements. Unlike the ring's, this volume decreases as ranks are added, and from 9 ranks on it is the smaller of the two: on a $$3 \times 3$$ grid, a rank receives $$2n^2/3$$ elements instead of $$8n^2/9$$, while on 4 ranks the ring algorithm moves less data, $$3n^2/4$$ elements per rank instead of $$n^2$$.

[cannon]: https://en.wikipedia.org/wiki/Cannon%27s_algorithm

### Overlapping computation and communication

In both algorithms, a rank waits during each shift. This wait can be hidden: the tiles of the next step do not depend on the current product, so a rank can receive them while it multiplies:
1. start a non-blocking send of the current tiles, and a non-blocking receive of the next ones into a second buffer (`MPI_Isend`, `MPI_Irecv`);
2. multiply the current tiles;
3. wait for both messages (`MPI_Waitall`), and swap the buffers.

{: .picture}
<img width="91.6%" height="91.6%" src="{{ site.assets }}/distributed-matmul-dbs/overlap.svg">

{: .caption}
Timeline of one rank in Cannon's algorithm on a $$3 \times 3$$ grid. **Top:** shifts and multiplications take turns. **Bottom:** each shift runs during a multiplication.

This technique, **[double buffering][buffering]**, gives each moving matrix a second buffer, which receives the next tiles while the first one holds the tiles in use. It costs one extra buffer per moving matrix, and it applies to the ring algorithm too. In the figure, it saves the time of the two shifts. A shift is hidden when the multiplication takes longer than the shift. A multiplication costs $$(n/q)^3$$ multiply-adds, while a shift moves $$2(n/q)^2$$ elements, so the ratio of computation to communication grows with the tile size $$n/q$$, and large tiles hide the communication easily.

[buffering]: https://en.wikipedia.org/wiki/Multiple_buffering

## Part 2: sparse matrices

Many matrices are mostly zeros. In a simulation on a mesh, for example, each point interacts only with a few neighbors, so each row has only a few nonzeros, however large the mesh. A **[sparse][sparse]** format stores only the nonzeros, which saves both memory and multiply-adds.

[sparse]: https://en.wikipedia.org/wiki/Sparse_matrix

### The compressed sparse row format

A widely used sparse format is the **compressed sparse row** (CSR) format, which uses three arrays:
- `values`: the nonzeros, row by row;
- `col_idx`: the column of each nonzero;
- `row_ptr`: where each row starts. Row $$i$$ is at positions `row_ptr[i]` to `row_ptr[i+1]-1` of the other two arrays.

{: .picture}
<img width="69.7%" height="69.7%" src="{{ site.assets }}/distributed-matmul-dbs/csr_format.svg">

{: .caption}
A sparse matrix (left) and its CSR arrays (right), colored by row. Row 3 is empty, so `row_ptr[3]` and `row_ptr[4]` are both 6.

For $$n$$ rows and $$nnz$$ nonzeros, CSR stores $$2 \, nnz + n + 1$$ numbers. Reading a row is fast, but reading a column requires a search through all rows, which makes the distributed transpose hard, as the [matrix transpose post]({{ site.baseurl }}{% post_url 2015-04-21-matrix-transpose %}) shows. The product $$y = Ax$$ reads the matrix by rows, and visits each row and each nonzero once, in $$O(n + nnz)$$ operations:

```cpp
for (int i = 0; i < n; i++) {
  y[i] = 0;
  for (int k = row_ptr[i]; k < row_ptr[i+1]; k++)
    y[i] += values[k] * x[col_idx[k]];
}
```

### The sparse matrix-vector product

$$A$$ is split by rows again. An all-gather of $$x$$ works, but a rank needs $$x_j$$ only if its rows have a nonzero in column $$j$$, and this set of columns is usually small. So only these entries are sent:
1. **setup** (once): each rank tells the owners of $$x$$ which of their entries it needs;
2. **exchange** (before every product): the owners send these entries (`MPI_Alltoallv`). The received copies are called **ghost** entries;
3. **local product:** each rank runs the CSR loop over its own and ghost entries.

The setup runs only once, because the positions of the nonzeros stay the same from one product to the next, for example in the iterations of a solver.

{: .picture}
<img width="64.4%" height="64.4%" src="{{ site.assets }}/distributed-matmul-dbs/spmv.svg">

{: .caption}
Sparse matrix-vector product on 3 ranks. **Left:** nonzeros colored by rank. A dot marks a nonzero whose entry of $$x$$ is on another rank. **Right:** the entries of $$x$$ on each rank after the exchange. Dashed boxes are ghost entries.

The communication now grows with the number of ghost entries, not with $$n$$. For a matrix from a mesh, the ghost entries are the points on the other side of the border of each rank's region, so the communication grows with the size of the border, while the work grows with the size of the region.

### The sparse matrix-matrix product

Row $$i$$ of $$C = AB$$ is a weighted sum of rows of $$B$$:

$$
C_{i,:} = \sum_{k \, : \, A_{ik} \neq 0} A_{ik} \, B_{k,:}.
$$

This row-by-row formulation, from [Gustavson (1978)][gustavson], reads both matrices by rows, as CSR stores them. With $$B$$ split by rows too, a rank needs row $$k$$ of $$B$$ whenever its rows of $$A$$ have a nonzero in column $$k$$. These are the indices of the ghost entries, but the rank now needs whole rows instead of single entries. The steps are:
1. each rank requests the remote rows of $$B$$ it needs;
2. the owners send these rows in CSR, in two rounds of `MPI_Alltoallv`: first the row sizes, so that the receiver can allocate memory, then the column indices and values;
3. each rank accumulates each of its rows of $$C$$ in a dense buffer, and compresses it into CSR.

[gustavson]: https://doi.org/10.1145/355791.355796

Unlike in the dense case, the number of nonzeros of $$C$$ is not known in advance. So step 3 often runs twice: a **symbolic** pass counts the nonzeros and allocates memory, and a **numeric** pass computes the values. $$C$$ is split by rows, like $$A$$.

## Part 3: database joins

A database table holds one record per row. A **key** is a column that identifies a record and links tables. In the example below, `Person` has one row per person, with key `id`, and `Messages` has one row per message, whose `id` column holds the key of the author, a [foreign key][fkey]. Here too, the rows that belong together may sit on different nodes.

[fkey]: https://en.wikipedia.org/wiki/Foreign_key

### Inner and outer joins

A **[join][join]** pairs the rows of two tables whose keys match:

```sql
SELECT Person.id, name, age, msg, year
FROM Person INNER JOIN Messages ON Person.id = Messages.id;
```

There are four kinds of join, which differ in what they do with the rows that have no match:
- **inner join:** only the matching pairs. Kim wrote two messages, so Kim appears twice. Mary, Joe and the message with key 9 have no match, and are dropped;
- **left outer join:** also the unmatched rows of the left table (`Person`), with `NULL` in the other columns;
- **right outer join:** also the unmatched rows of the right table (`Messages`);
- **full outer join:** also the unmatched rows of both tables.

[join]: https://en.wikipedia.org/wiki/Join_(SQL)

{: .picture}
<img width="90.4%" height="90.4%" src="{{ site.assets }}/distributed-matmul-dbs/joins.svg">

{: .caption}
**Top:** the tables `Person` (blue) and `Messages` (green). **Bottom:** three joins on `id`. Dark gray cells are `NULL`.

On one node, three classic algorithms join tables of $$N$$ and $$M$$ rows. Their costs exclude the size of the output:
- **[nested loop][nlj]:** compare all pairs of rows, in $$O(N \cdot M)$$;
- **[sort-merge][smj]:** sort both tables by key and scan them together, in $$O(N \log N + M \log M)$$, or $$O(N + M)$$ if already sorted;
- **[hash join][hj]:** build a hash table of the keys of one table, and probe it with the rows of the other, in $$O(N + M)$$ on average.

[nlj]: https://en.wikipedia.org/wiki/Nested_loop_join
[smj]: https://en.wikipedia.org/wiki/Sort-merge_join
[hj]: https://en.wikipedia.org/wiki/Hash_join

For outer joins, each row also records whether it found a match, and the rows without one are returned with `NULL`. All these algorithms assume that rows with the same key are on the same node. The next sections show how to ensure this for three layouts of the tables.

### Layout 1: each table on a different server

**Problem.** `Person` is on server 1, and `Messages` is on server 2.

**Inner join.** Sending a whole table to the other server works, but it also sends the rows that have no match. A **semi-join** sends less, because it matches the keys first:
1. server 1 sends its keys to server 2;
2. server 2 returns only its rows with one of these keys;
3. server 1 joins locally.

**Outer joins.** In step 3, server 1 also returns its unmatched persons, with `NULL`: this is the left outer join. Server 2 knows which of its messages had no match, since their keys were not in the list of step 1, so in a step 4 it returns them, with `NULL`. The inner join plus step 4 is the right outer join, and steps 3 and 4 together are the full outer join.

{: .picture}
<img width="92.4%" height="92.4%" src="{{ site.assets }}/distributed-matmul-dbs/layout_tables.svg">

{: .caption}
A semi-join between two servers: the keys (1), the matching rows (2), the left outer join on server 1 (3), and the unmatched messages of server 2 (4). The first three rows of (3) are the inner join.

For a long list of keys, a [Bloom filter][bloom] is a smaller substitute: a compact bit array that never rejects a key of the list, but accepts a few keys that are not in it. These false positives let a few unmatched messages through in step 2. Server 1 finds no match for them, so it drops them from the inner join, and returns them with `NULL` in the right and full outer joins.

[bloom]: https://en.wikipedia.org/wiki/Bloom_filter

### Layout 2: tables split by rows

**Problem.** Each server holds some rows of each table. This is called **horizontal partitioning**, or **[sharding][shard]**. Matching rows may then sit on different servers. In the figure below, John is on server 2, but his message "Bye" is on server 1. A local join misses this pair, and a local left outer join wrongly returns John with `NULL`.

[shard]: https://en.wikipedia.org/wiki/Shard_(database_architecture)

**Inner join.** The solution is to place each row by its key, with the same rule for both tables, such as a hash of the key. In the figure, even keys go to server 1, and odd keys to server 2. Matching rows then share a server, and every server joins locally. If the tables are not placed this way, a **shuffle** first sends each row to its server, with an all-to-all exchange (`MPI_Alltoallv`).

{: .picture}
<img width="87.4%" height="87.4%" src="{{ site.assets }}/distributed-matmul-dbs/layout_rows.svg">

{: .caption}
Two tables split by rows across 2 servers, colored by target server: blue for server 1 (even keys), and orange for server 2 (odd keys). **Left:** before the shuffle. **Right:** after it.

**Outer joins.** After a shuffle, the outer joins are local too, because a row without a match on its server has no match anywhere. In the figure, server 2 returns Mary and Joe for the left outer join, and "Hey" for the right outer join.

A shuffle has one weakness: all the rows with the same key go to the same server, so a frequent key gives that server more work than the others, and the join ends only when the slowest server is done.

**Broadcast join.** If one table is small, such as `Person`, it can be copied to every server instead, while `Messages` stays in place. Each message then sits with all persons, so the inner and right outer joins are local. The left outer join is not: John matches on server 1 but not on server 2, so no server alone can tell that a person has no match. Each server therefore flags its local matches, and a logical-or all-reduce (`MPI_Allreduce` with `MPI_LOR`) combines the flags. Server 1 then returns the persons with a final flag of 0, with `NULL`. Since the large table does not move, a frequent key does not overload any server.

{: .picture}
<img width="69.5%" height="69.5%" src="{{ site.assets }}/distributed-matmul-dbs/broadcast.svg">

{: .caption}
A broadcast join, for a left outer join. After the all-reduce of the match flags, only Mary and Joe have no match.

### Layout 3: tables split by columns

**Problem.** Each server holds some columns, for all rows. This is called **vertical partitioning**, as in [column-oriented databases][column-db], which store each column separately so that a query reads only the columns it needs. Row $$r$$ on every server belongs to the same record. Here, server 1 holds the keys, server 2 the other columns of `Person`, and server 3 the other columns of `Messages`.

**Inner join.** Only the keys decide which rows match, so the keys are matched first, and the other columns move last:
1. server 1 matches the keys into a **join index**: a list of pairs of row numbers, one per table;
2. server 1 sends the `Person` row numbers to server 2, and the `Messages` row numbers to server 3;
3. servers 2 and 3 gather their rows in that order.

Only row numbers cross the network, and the result stays split by columns. Building the output rows at the end, and only for the rows that match, is called **late materialization**.

**Outer joins.** The join index also gets pairs with one side missing, such as Mary (row 1) and Joe (row 3) in the left outer join. Where a row number is missing, the server writes `NULL`.

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/distributed-matmul-dbs/layout_columns.svg">

{: .caption}
A left outer join of tables split by columns across 3 servers. Column `#` is the implicit row number. Server 1 builds the join index (1) and sends row numbers (2), and servers 2 and 3 gather their rows (3). The first three rows are the inner join.

[column-db]: https://en.wikipedia.org/wiki/Column-oriented_DBMS

## Summary

The matrix products and the joins follow one rule: keep together the data that is used together. When the layout does not fit the operation, the data moves in a few collective steps, only the data that is needed is sent, and the communication overlaps with the computation where possible.

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
