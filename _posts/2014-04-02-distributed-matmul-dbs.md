---
layout: post
title:  "Distributed Matrix Multiplication and Database Joins"
categories: ["Distributed Algorithms"]
tags: ["MPI", "linear algebra", "sparse matrices", "CSR", "databases", "joins"]
---

Scientific computing and data processing rely on two data structures: the matrix and the table. Data too large for one machine is split across compute nodes. A node accesses only its own memory. Any other data arrives as a message over the network, which is slow. Matrix multiplication and database joins raise the same question: how to split the data so that each node computes mostly on local data.

The terms follow [MPI][mpi]: a program runs as $$p$$ processes, called **ranks**, usually one per node. In the database part, ranks are called **servers**.

[mpi]: https://en.wikipedia.org/wiki/Message_Passing_Interface

## Part 1: dense matrices

The product $$C = AB$$ of two $$n \times n$$ matrices, with $$C_{ij} = \sum_k A_{ik} B_{kj}$$, costs $$n^3$$ [multiply-adds][mac]. The matrix-vector product $$y = Ax$$ costs $$n^2$$.

[mac]: https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation

### The matrix-vector product

Each rank holds $$n/p$$ consecutive rows of $$A$$, and the matching entries of $$x$$ and $$y$$. To compute $$y_i$$, a rank needs row $$i$$ of $$A$$, which it holds, and all of $$x$$, which it does not. Two steps solve this:
1. **[all-gather][collective]** (`MPI_Allgather`): every rank sends its entries of $$x$$ to all other ranks;
2. **local product:** every rank multiplies its rows of $$A$$ by $$x$$.

[collective]: https://en.wikipedia.org/wiki/Collective_operation

{: .picture}
<img width="94%" height="94%" src="{{ site.assets }}/distributed-matmul-dbs/matvec_row1d.svg">

{: .caption}
Matrix-vector product on 4 ranks, colored by rank. **Left:** the initial layout. **Right:** rank 1 after the all-gather, ready to compute $$y_2$$ and $$y_3$$.

Each rank performs $$n^2/p$$ multiply-adds, and $$y$$ ends up split like $$x$$, ready for the next product.

### The matrix-matrix product: shifting tiles

The same approach fails for $$C = AB$$. Each rank would need all of $$B$$, and an all-gather would store $$n^2$$ elements on every rank. Instead, each rank holds one piece of each matrix. At every step, it multiplies the pieces it holds, then **shifts** some of them to a neighbor. Shifts wrap around: the first rank sends to the last.

### Shifting one matrix: the ring algorithm

The ranks form a ring. Rank $$i$$ holds row block $$i$$ of $$A$$, $$B$$ and $$C$$. Its block of $$A$$ is cut into $$p$$ tiles, so that

$$
C_i = \sum_{k=0}^{p-1} A_{i,k} \, B_k.
$$

$$A$$ and $$C$$ stay in place, and $$B$$ moves. At each step, every rank multiplies the tile of $$A$$ that matches its current block of $$B$$, adds the result to $$C_i$$, and passes the block of $$B$$ to the previous rank. After $$p$$ steps, $$C_i$$ is complete.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-matmul-dbs/ring.svg">

{: .caption}
The ring algorithm on 3 ranks. Blocks of $$B$$ shift up, and rank 0 sends to rank 2. At each step, a rank multiplies the tile of $$A$$ that has the color of its block of $$B$$.

The output $$C$$ is split by rows, like $$A$$. In the mirror case, $$B$$ stays, column blocks of $$A$$ move, and $$C$$ is split by columns. Each rank stores $$3n^2/p$$ elements, but receives $$(p-1) \, n^2/p \approx n^2$$ elements, regardless of $$p$$.

### Shifting two matrices: Cannon's algorithm

Cutting the matrices in both dimensions reduces the communication. The ranks form a $$q \times q$$ grid, with $$q = \sqrt{p}$$, and each matrix is cut into $$q \times q$$ tiles. Rank $$(i,j)$$ starts with $$A_{i,j}$$ and $$B_{i,j}$$, and computes

$$
C_{i,j} = \sum_{k=0}^{q-1} A_{i,k} \, B_{k,j}.
$$

For example, rank $$(1,2)$$ of a $$3 \times 3$$ grid computes $$C_{1,2} = A_{1,0} B_{0,2} + A_{1,1} B_{1,2} + A_{1,2} B_{2,2}$$. It needs the tiles of row 1 of $$A$$ and of column 2 of $$B$$, one matching pair at a time. [Cannon's algorithm][cannon] delivers these pairs:
1. **align:** shift row $$i$$ of $$A$$ left by $$i$$, and column $$j$$ of $$B$$ up by $$j$$. Rank $$(i,j)$$ now holds the matching pair $$A_{i,k}$$ and $$B_{k,j}$$, with $$k = (i+j) \bmod q$$;
2. **multiply:** add $$A_{i,k} B_{k,j}$$ to $$C_{i,j}$$;
3. **shift:** shift $$A$$ left by one and $$B$$ up by one. The new pair matches again.

Steps 2 and 3 repeat until each rank has done $$q$$ multiplications.

{: .picture}
<img width="98%" height="98%" src="{{ site.assets }}/distributed-matmul-dbs/cannon.svg">

{: .caption}
Cannon's algorithm on a $$3 \times 3$$ grid of ranks. Tiles of $$A$$ (top) shift left, and tiles of $$B$$ (bottom) shift up. The thick frame marks rank $$(1,2)$$, and colors show its three pairs: blue for $$k=0$$, green for $$k=1$$ and red for $$k=2$$.

$$A$$ moves along the grid rows, $$B$$ along the grid columns, and $$C$$ stays. Tile $$C_{i,j}$$ ends on rank $$(i,j)$$, in the same layout as the inputs. On a 2D grid, two of the three matrices must move. Each rank still stores $$3n^2/p$$ elements, but receives at most $$2q$$ tiles, or $$2n^2/\sqrt{p}$$ elements. This volume decreases as $$p$$ grows, and is smaller than the ring's for $$p \geq 9$$.

[cannon]: https://en.wikipedia.org/wiki/Cannon%27s_algorithm

### Overlapping computation and communication

The tiles of the next step do not depend on the current product. So a rank receives them during the multiplication:
1. start a non-blocking send of the current tiles, and a non-blocking receive of the next ones into a second buffer (`MPI_Isend`, `MPI_Irecv`);
2. multiply the current tiles;
3. wait for both messages (`MPI_Waitall`), and swap the buffers.

{: .picture}
<img width="89%" height="89%" src="{{ site.assets }}/distributed-matmul-dbs/overlap.svg">

{: .caption}
Timeline of one rank in Cannon's algorithm on a $$3 \times 3$$ grid. **Top:** shifts and multiplications take turns. **Bottom:** each shift runs during a multiplication.

This **[double buffering][buffering]** costs one extra buffer per moving matrix, and also applies to the ring algorithm. It hides a shift when the multiplication takes longer than the shift. A multiplication costs $$(n/q)^3$$ multiply-adds, and a shift moves $$2(n/q)^2$$ elements. The ratio grows with the tile size $$n/q$$, so large tiles hide the communication.

[buffering]: https://en.wikipedia.org/wiki/Multiple_buffering

## Part 2: sparse matrices

Many matrices are mostly zeros. In a simulation on a mesh, for example, each point interacts only with a few neighbors. A **[sparse][sparse]** format stores only the nonzeros.

[sparse]: https://en.wikipedia.org/wiki/Sparse_matrix

### The compressed sparse row format

The **compressed sparse row** (CSR) format uses three arrays:
- `values`: the nonzeros, row by row;
- `col_idx`: the column of each nonzero;
- `row_ptr`: where each row starts. Row $$i$$ is at positions `row_ptr[i]` to `row_ptr[i+1]-1` of the other two arrays.

{: .picture}
<img width="66%" height="66%" src="{{ site.assets }}/distributed-matmul-dbs/csr_format.svg">

{: .caption}
A sparse matrix (left) and its CSR arrays (right), colored by row. Row 3 is empty, so `row_ptr[3]` and `row_ptr[4]` are both 6.

For $$n$$ rows and $$nnz$$ nonzeros, CSR stores $$2 \, nnz + n + 1$$ numbers. Reading a row is fast. Reading a column requires a search through all rows, which makes the distributed transpose hard. The product $$y = Ax$$ takes $$O(n + nnz)$$ operations:

```cpp
for (int i = 0; i < n; i++) {
  y[i] = 0;
  for (int k = row_ptr[i]; k < row_ptr[i+1]; k++)
    y[i] += values[k] * x[col_idx[k]];
}
```

### The sparse matrix-vector product

$$A$$ is split by rows again. An all-gather of $$x$$ works, but a rank needs $$x_j$$ only if its rows have a nonzero in column $$j$$, and this set is usually small. So only these entries are sent:
1. **setup** (once): each rank tells the owners of $$x$$ which of their entries it needs;
2. **exchange** (before every product): the owners send these entries (`MPI_Alltoallv`). The received copies are called **ghost** entries;
3. **local product:** each rank runs the CSR loop over its own and ghost entries.

{: .picture}
<img width="78%" height="78%" src="{{ site.assets }}/distributed-matmul-dbs/spmv.svg">

{: .caption}
Sparse matrix-vector product on 3 ranks. **Left:** nonzeros colored by rank. A dot marks a nonzero whose entry of $$x$$ is on another rank. **Right:** the entries of $$x$$ on each rank after the exchange. Dashed boxes are ghost entries.

The communication now scales with the number of ghost entries, not with $$n$$. For a matrix from a mesh, the ghost entries are the points at the border of each rank's region.

### The sparse matrix-matrix product

Row $$i$$ of $$C = AB$$ is a weighted sum of rows of $$B$$:

$$
C_{i,:} = \sum_{k \, : \, A_{ik} \neq 0} A_{ik} \, B_{k,:}.
$$

A rank thus needs row $$k$$ of $$B$$ whenever its rows of $$A$$ have a nonzero in column $$k$$. These are the indices of the ghost entries, but the rank needs whole rows instead of single entries. The steps are:
1. each rank requests the remote rows of $$B$$ it needs;
2. the owners send these rows in CSR, in two rounds of `MPI_Alltoallv`: first the row sizes, so that the receiver can allocate memory, then the column indices and values;
3. each rank accumulates each of its rows of $$C$$ in a dense buffer, and compresses it into CSR.

The number of nonzeros of $$C$$ is not known in advance. So step 3 often runs twice: a **symbolic** pass counts the nonzeros and allocates memory, and a **numeric** pass computes the values. $$C$$ is split by rows, like $$A$$.

## Part 3: database joins

A database table holds one record per row. A **key** is a column that identifies a record and links tables. In the example below, `Person` has one row per person, with key `id`. `Messages` has one row per message, and its `id` column holds the key of the author, a [foreign key][fkey].

[fkey]: https://en.wikipedia.org/wiki/Foreign_key

### Inner and outer joins

A **[join][join]** pairs the rows of two tables whose keys match:

```sql
SELECT Person.id, name, age, msg, year
FROM Person INNER JOIN Messages ON Person.id = Messages.id;
```

There are four kinds of join:
- **inner join:** only the matching pairs. Kim wrote two messages, so Kim appears twice. Mary, Joe and the message with key 9 have no match, and are dropped;
- **left outer join:** also the unmatched rows of the left table (`Person`), with `NULL` in the other columns;
- **right outer join:** also the unmatched rows of the right table (`Messages`);
- **full outer join:** also the unmatched rows of both tables.

[join]: https://en.wikipedia.org/wiki/Join_(SQL)

{: .picture}
<img width="85%" height="85%" src="{{ site.assets }}/distributed-matmul-dbs/joins.svg">

{: .caption}
**Top:** the tables `Person` (blue) and `Messages` (green). **Bottom:** three joins on `id`. Gray cells are `NULL`.

On one node, three classic algorithms join tables of $$N$$ and $$M$$ rows. Their costs exclude the size of the output:
- **[nested loop][nlj]:** compare all pairs of rows, in $$O(N \cdot M)$$;
- **[sort-merge][smj]:** sort both tables by key and scan them together, in $$O(N \log N + M \log M)$$, or $$O(N + M)$$ if already sorted;
- **[hash join][hj]:** build a hash table of the keys of one table, and probe it with the rows of the other, in $$O(N + M)$$ on average.

[nlj]: https://en.wikipedia.org/wiki/Nested_loop_join
[smj]: https://en.wikipedia.org/wiki/Sort-merge_join
[hj]: https://en.wikipedia.org/wiki/Hash_join

For outer joins, each row also records whether it found a match, and the rows without one are returned with `NULL`. All these algorithms assume that rows with the same key are on the same node. The next sections ensure this for three layouts.

### Layout 1: each table on a different server

**Problem.** `Person` is on server 1, and `Messages` is on server 2.

**Inner join.** Sending a whole table works, but a **semi-join** sends less:
1. server 1 sends its keys to server 2;
2. server 2 returns only its rows with one of these keys;
3. server 1 joins locally.

**Outer joins.** In step 3, server 1 also returns its unmatched persons, with `NULL`: this is the left outer join. Server 2 knows which of its messages had no match, since their keys were not in the list of step 1. In step 4, it returns them, with `NULL`. The inner join plus step 4 is the right outer join, and steps 3 and 4 together are the full outer join.

{: .picture}
<img width="77%" height="77%" src="{{ site.assets }}/distributed-matmul-dbs/layout_tables.svg">

{: .caption}
A semi-join between two servers: the keys (1), the matching rows (2), the left outer join on server 1 (3), and the unmatched messages of server 2 (4). The first three rows of (3) are the inner join.

For a long list of keys, a [Bloom filter][bloom] is a smaller substitute. Its false positives let a few unmatched rows through in step 2. Server 1 finds no match for them: it drops them from the inner join, and returns them with `NULL` in the right and full outer joins.

[bloom]: https://en.wikipedia.org/wiki/Bloom_filter

### Layout 2: tables split by rows

**Problem.** Each server holds some rows of each table. This is **horizontal partitioning**, or **[sharding][shard]**. Matching rows may sit on different servers. Below, John is on server 2, but his message "Bye" is on server 1. A local join misses this pair, and a local left outer join wrongly returns John with `NULL`.

[shard]: https://en.wikipedia.org/wiki/Shard_(database_architecture)

**Inner join.** Each row is placed by its key, with the same rule for both tables, such as a hash of the key. Below, even keys go to server 1, and odd keys to server 2. Matching rows then share a server, and every server joins locally. If the tables are not placed this way, a **shuffle** first sends each row to its server with an all-to-all exchange (`MPI_Alltoallv`).

{: .picture}
<img width="82%" height="82%" src="{{ site.assets }}/distributed-matmul-dbs/layout_rows.svg">

{: .caption}
Two tables split by rows across 2 servers, colored by target server: blue for server 1 (even keys), and orange for server 2 (odd keys). **Left:** before the shuffle. **Right:** after it.

**Outer joins.** After a shuffle, outer joins are local too: a row without a match on its server has no match anywhere. In the figure, server 2 returns Mary and Joe for the left outer join, and "Hey" for the right outer join.

A frequent key sends many rows to the same server, which then has more work than the others.

**Broadcast join.** If `Person` is small, it is copied to every server instead, and `Messages` stays in place. Each message then sits with all persons, so the inner and right outer joins are local. The left outer join is not: John matches on server 1 but not on server 2, so no server alone can tell that a person has no match. Each server flags its local matches, and a logical-or all-reduce (`MPI_Allreduce` with `MPI_LOR`) combines the flags. Server 1 then returns the persons with a final flag of 0, with `NULL`.

{: .picture}
<img width="65%" height="65%" src="{{ site.assets }}/distributed-matmul-dbs/broadcast.svg">

{: .caption}
A broadcast join, for a left outer join. After the all-reduce of the match flags, only Mary and Joe have no match.

### Layout 3: tables split by columns

**Problem.** Each server holds some columns, for all rows. This is **vertical partitioning**, as in [column-oriented databases][column-db]. Row $$r$$ on every server belongs to the same record. Here, server 1 holds the keys, server 2 the other columns of `Person`, and server 3 the other columns of `Messages`.

**Inner join.** Only the keys decide what matches. So the keys are matched first, and the other columns move last:
1. server 1 matches the keys into a **join index**: a list of pairs of row numbers, one per table;
2. server 1 sends the `Person` row numbers to server 2, and the `Messages` row numbers to server 3;
3. servers 2 and 3 gather their rows in that order.

Only row numbers cross the network, and the result stays split by columns. Building the output rows at the end, and only for the rows that match, is called **late materialization**.

**Outer joins.** The join index also gets pairs with one side missing, such as Mary (row 1) and Joe (row 3) in the left outer join. Where a row number is missing, the server writes `NULL`.

{: .picture}
<img width="95%" height="95%" src="{{ site.assets }}/distributed-matmul-dbs/layout_columns.svg">

{: .caption}
A left outer join of tables split by columns across 3 servers. Column `#` is the implicit row number. Server 1 builds the join index (1) and sends row numbers (2), and servers 2 and 3 gather their rows (3). The first three rows are the inner join.

[column-db]: https://en.wikipedia.org/wiki/Column-oriented_DBMS

## Summary

Both problems follow one rule: keep together the data that is used together. When the layout does not fit the operation, the data moves in a few collective steps, only the needed data is sent, and communication overlaps with computation.

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
