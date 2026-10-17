---
layout: post
title:  "Distributed Matrix Multiplication and Database Joins"
categories: ["Distributed Algorithms"]
tags: ["MPI", "linear algebra", "sparse matrices", "CSR", "databases", "joins"]
---

Two data structures are behind most of scientific computing and data processing: the matrix and the table. A matrix holds numbers on a grid. A database table holds records, one per row, with one named field per column.

Sometimes the data does not fit in the memory of one machine, or the computation is too slow. We then split the data across many compute nodes connected by a network. Each node reads only its own memory. If it needs data from another node, that data must arrive as a message.

In this lecture we study two operations in this setting: the matrix multiplication and the database join. For both, we ask the same question. How do we split the data so that each node does most of its work locally, with little communication?

We use the vocabulary of [MPI][mpi]. A compute node is a **rank**, and $$p$$ is the number of ranks. In the database part, we call the ranks **servers**.

[mpi]: https://en.wikipedia.org/wiki/Message_Passing_Interface

## Part 1: dense matrix multiplication

Let $$A$$ be an $$n \times m$$ matrix and $$B$$ an $$m \times k$$ matrix. Their product $$C = AB$$ is the $$n \times k$$ matrix with entries

$$
C_{ij} = \sum_{l=1}^{m} A_{il} B_{lj}.
$$

For square $$n \times n$$ matrices, this costs $$n^3$$ multiply-adds. The matrix-vector product $$y = Ax$$ is the case where $$B$$ has one column. It costs $$n^2$$ operations.

### Warm-up: matrix-vector multiplication

We start with the simplest layout. We split $$A$$ by rows: each rank holds $$n/p$$ consecutive rows. We split $$x$$ and $$y$$ the same way.

To compute $$y_i$$, a rank needs row $$i$$ of $$A$$ and all of $$x$$. It has the row, but only $$n/p$$ entries of $$x$$. So the algorithm has two steps:
1. **all-gather:** every rank sends its entries of $$x$$ to all other ranks (`MPI_Allgather`). Now every rank holds the full $$x$$;
2. **local product:** every rank multiplies its rows of $$A$$ by $$x$$, and gets its entries of $$y$$.

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="{{ site.assets }}/distributed-matmul-dbs/matvec_row1d.svg">

{: style="text-align:center; font-size: small;"}
Row-wise matrix-vector multiplication on 4 ranks. Colors show which rank holds each element. **Left:** the initial layout. **Right:** rank 1 after the all-gather of $$x$$. It now computes $$y_2$$ and $$y_3$$ alone.

Each rank does $$n^2/p$$ operations. The output $$y$$ has the same layout as the input $$x$$. This is convenient for iterative methods, such as the [power method][power], where the output of one product is the input of the next.

[power]: https://en.wikipedia.org/wiki/Power_iteration

### Matrix-matrix multiplication: why not the same trick?

Let us try the same idea for $$C = AB$$, with all matrices split by rows. Rank $$r$$ needs all of $$B$$ to compute its rows of $$C$$. An all-gather of $$B$$ copies the whole matrix to every rank. Each rank then stores $$n^2$$ elements, and splitting the data was pointless.

The fix is to never hold the whole matrix. Instead, we keep a small tile of each matrix on each rank, and we **rotate the tiles** between ranks. At each step, a rank multiplies the tiles it holds now, then passes them to a neighbor.

### Rotating both matrices: Cannon's algorithm

We arrange the $$p$$ ranks on a $$q \times q$$ grid, with $$q = \sqrt{p}$$. We cut each matrix into $$q \times q$$ tiles of size $$n/q \times n/q$$. Rank $$(i,j)$$ starts with tiles $$A_{i,j}$$ and $$B_{i,j}$$, and it must compute

$$
C_{i,j} = \sum_{k=0}^{q-1} A_{i,k} \, B_{k,j}.
$$

For example, on a $$3 \times 3$$ grid, rank $$(1,2)$$ computes

$$
C_{1,2} = A_{1,0} B_{0,2} + A_{1,1} B_{1,2} + A_{1,2} B_{2,2}.
$$

It needs all tiles of row 1 of $$A$$, and all tiles of column 2 of $$B$$. But it needs them one pair at a time. [Cannon's algorithm][cannon] brings each pair to the rank at the right moment:
1. **alignment (or skewing):** row $$i$$ of $$A$$ rotates left by $$i$$ positions. Column $$j$$ of $$B$$ rotates up by $$j$$ positions. Rotations are cyclic: a tile that leaves on one side comes back on the other. Rank $$(i,j)$$ now holds $$A_{i,k}$$ and $$B_{k,j}$$, with $$k = (i+j) \bmod q$$. The two tiles match;
2. **multiply and shift, $$q$$ times:** each rank multiplies its two tiles and adds the result to $$C_{i,j}$$. Then all tiles of $$A$$ rotate one position left, and all tiles of $$B$$ rotate one position up. The new pair matches again, with the next value of $$k$$.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-matmul-dbs/cannon.svg">

{: style="text-align:center; font-size: small;"}
Cannon's algorithm on a $$3 \times 3$$ grid. Each cell is a rank. Tiles of $$A$$ (top) rotate left, and tiles of $$B$$ (bottom) rotate up. The thick frame marks rank $$(1,2)$$. Colors show the three products it needs: blue for $$k=0$$, green for $$k=1$$ and red for $$k=2$$. After the alignment, and after each shift, the rank holds a matching pair.

So which tiles rotate? Both: the tiles of $$A$$ move along the rows of the grid, and the tiles of $$B$$ move along the columns. The output $$C$$ never moves. At the end, rank $$(i,j)$$ holds tile $$C_{i,j}$$, so $$C$$ has the same 2D tile layout as $$A$$ and $$B$$.

What does it cost? A rank stores three tiles, or $$3n^2/p$$ elements. Each shift sends two tiles of $$n^2/p$$ elements, so a rank sends $$2n^2/\sqrt{p}$$ elements in total. All messages go to direct neighbors on the grid.

[cannon]: https://en.wikipedia.org/wiki/Cannon%27s_algorithm

### Overlapping computation and communication

In the algorithm above, a rank waits for each shift to finish before it multiplies again. We can hide this wait. The tiles for the next step do not depend on the current product. So a rank can receive the next tiles while it multiplies the current ones:
1. post a non-blocking send of the current tiles, and a non-blocking receive of the next tiles into a second buffer (`MPI_Isend`, `MPI_Irecv`);
2. multiply the current tiles;
3. wait for the messages to complete (`MPI_Waitall`), and swap the two buffers.

{: style="text-align:center; font-size: small;"}
<img width="85%" height="85%" src="{{ site.assets }}/distributed-matmul-dbs/overlap.svg">

{: style="text-align:center; font-size: small;"}
Timeline of one rank in Cannon's algorithm on a $$3 \times 3$$ grid. **Top:** each shift waits for the previous multiplication, and the next multiplication waits for the shift. **Bottom:** each shift runs during the previous multiplication, so its cost disappears from the total time.

This is called **double buffering**. It costs one extra pair of tiles in memory. It hides the communication fully when a multiplication takes longer than a shift. Multiplying two tiles costs $$(n/q)^3$$ operations, but a shift moves only $$2(n/q)^2$$ elements. So the larger the tiles, the easier it is to hide the communication.

### Rotating only one matrix: the ring algorithm

Do we have to rotate both matrices? No. We can keep $$A$$ in place and rotate only $$B$$. The ranks then form a 1D ring instead of a 2D grid:
- rank $$i$$ holds a block of rows of $$A$$, which we cut into $$p$$ tiles $$A_{i,0}, \dots, A_{i,p-1}$$. It never sends them;
- rank $$i$$ also holds one block of rows of $$B$$. At step $$t$$, it holds block $$B_k$$, with $$k = (i+t) \bmod p$$;
- at each step, the rank adds $$A_{i,k} B_k$$ to its rows of $$C$$. Then it passes $$B_k$$ to its neighbor on the ring.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-matmul-dbs/ring.svg">

{: style="text-align:center; font-size: small;"}
The ring algorithm on 3 ranks. Tiles of $$A$$ stay in place. Row blocks of $$B$$ rotate up the ring, and rank 0 passes its block to rank 2. At each step, a rank multiplies the colored tile of $$A$$ by the block of $$B$$ of the same color.

The output $$C$$ is split by rows, like $$A$$. We can also do the mirror case: keep $$B$$ in place, split by columns, and rotate column blocks of $$A$$. The output $$C$$ is then split by columns, like $$B$$.

The ring algorithm is simpler, and double buffering works the same way. But each rank now sends $$p-1$$ blocks of $$n^2/p$$ elements, close to $$n^2$$ elements in total. Cannon's algorithm sends only $$2n^2/\sqrt{p}$$. From 9 ranks on, rotating both matrices sends less data.

## Part 2: sparse matrix multiplication

Many matrices are mostly zeros. Think of the connectivity matrix of a graph: a social network, or the synapses of a neural network. A matrix is **sparse** when it pays to store only its nonzeros.

### The compressed sparse row format

The **compressed sparse row** (CSR) format stores a matrix with $$n$$ rows and $$nnz$$ nonzeros in three arrays:
- `values` ($$nnz$$ entries): the nonzeros, row by row, from left to right;
- `col_idx` ($$nnz$$ entries): the column of each nonzero;
- `row_ptr` ($$n+1$$ entries): row $$i$$ is stored at positions `row_ptr[i]` to `row_ptr[i+1]-1` of the two arrays above. The last entry is $$nnz$$.

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="{{ site.assets }}/distributed-matmul-dbs/csr_format.svg">

{: style="text-align:center; font-size: small;"}
A $$5 \times 6$$ sparse matrix with 8 nonzeros (left) and its CSR format (right). Colors show the row of each nonzero. Row 3 is empty, so `row_ptr[3]` and `row_ptr[4]` are both 6.

CSR uses $$2 \, nnz + n + 1$$ entries instead of $$n \times m$$. Reading a row is direct. Reading a column requires a search through all rows. This is why transposing a distributed CSR matrix is hard, as we show [in a different post]({{ site.baseurl }}{% post_url 2015-04-21-matrix-transpose %}). The product $$y = Ax$$ is a short loop with $$O(nnz)$$ operations:

```cpp
for (int i = 0; i < n; i++) {
  y[i] = 0;
  for (int k = row_ptr[i]; k < row_ptr[i+1]; k++)
    y[i] += values[k] * x[col_idx[k]];
}
```

### Distributed sparse matrix-vector multiplication

We split $$A$$ by rows, as in the dense case. Each rank stores its rows in CSR. Should we start with an all-gather of $$x$$ again? That would be wasteful. A rank only needs $$x_j$$ for the columns $$j$$ where its rows have a nonzero. Usually, that is a small set. The algorithm is:
1. **setup, once per matrix:** each rank scans its `col_idx` and lists the entries of $$x$$ it needs from other ranks. It sends each owner the list of entries it needs. Now every rank knows what to send, and to whom;
2. **exchange, before every product:** ranks send only the requested entries (`MPI_Alltoallv`). A rank stores the received entries, called **ghost** entries, after its own;
3. **local product:** each rank runs the CSR loop, with `col_idx` renumbered to point into its own and ghost entries.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="{{ site.assets }}/distributed-matmul-dbs/spmv.svg">

{: style="text-align:center; font-size: small;"}
Sparse matrix-vector multiplication on 3 ranks. **Left:** nonzeros of $$A$$, colored by rank. A dot marks a nonzero whose entry of $$x$$ is on another rank. **Right:** entries of $$x$$ on each rank after the exchange. Dashed boxes are ghost entries, colored by the rank they came from.

The communication now grows with the number of ghost entries, not with $$n$$. For a matrix that comes from a mesh, these are only the entries on the border of each rank's region.

### Distributed sparse matrix-matrix multiplication

Now take $$C = AB$$, with $$A$$ and $$B$$ in CSR, split by rows. Row $$i$$ of $$C$$ is a weighted sum of rows of $$B$$:

$$
C_{i,:} = \sum_{k \, : \, A_{ik} \neq 0} A_{ik} \, B_{k,:}.
$$

So a rank needs row $$k$$ of $$B$$ for every column $$k$$ where its rows of $$A$$ have a nonzero. These are the same indices as the ghost entries above. The difference is that we now fetch whole rows of $$B$$, not single entries of $$x$$. The algorithm is:
1. each rank lists the remote rows of $$B$$ it needs, and requests them;
2. the owners send these rows in CSR with two `MPI_Alltoallv` calls. The first sends the size of each row, so that the receiver can allocate memory. The second sends the column indices and values;
3. each rank computes its rows of $$C$$. For each row, it adds the scaled rows of $$B$$ into a dense array, then compresses that array into CSR.

We do not know the number of nonzeros of $$C$$ in advance. So step 3 often runs twice: a symbolic pass counts the nonzeros per row and allocates memory, and a numeric pass computes the values. The output $$C$$ is split by rows, like $$A$$.

## Part 3: distributed database joins

A relational database stores data in tables. A **key** is a column that identifies a record, and links tables together. In our example, the table `Person` has one row per person, with key `id`. The table `Messages` has one row per message. Its column `id` holds the key of the person who wrote it.

### Inner and outer joins

A **join** combines the rows of two tables whose keys match. In SQL:

```sql
SELECT Person.id, name, age, msg, year
FROM Person INNER JOIN Messages ON Person.id = Messages.id;
```

There are four kinds of join:
- the **inner join** returns one row per pair of matching rows. Kim wrote two messages, so Kim appears twice. Mary and Joe wrote nothing, and nobody has key 9, so these rows are dropped;
- the **left outer join** returns the inner join, plus the rows of the left table (`Person`) that have no match. Their `Messages` columns are `NULL`;
- the **right outer join** does the same for the right table (`Messages`);
- the **full outer join** keeps the unmatched rows of both tables.

{: style="text-align:center; font-size: small;"}
<img width="85%" height="85%" src="{{ site.assets }}/distributed-matmul-dbs/joins.svg">

{: style="text-align:center; font-size: small;"}
**Top:** the tables `Person` (blue) and `Messages` (green). **Bottom:** their inner join, left outer join and full outer join on `id`. Gray cells are `NULL`.

How does one node compute a join of $$N$$ and $$M$$ rows? There are three classic algorithms:
- **nested loop:** compare every pair of rows. This costs $$O(N \cdot M)$$;
- **sort-merge:** sort both tables by key, then walk through both with two pointers, as in merge sort. This costs $$O(N \log N + M \log M)$$, or $$O(N + M)$$ if the tables are already sorted;
- **hash join:** put the keys of the smaller table in a hash table, then look up each key of the other table. This costs $$O(N + M)$$ on average.

For outer joins, each row also keeps a flag that says whether it found a match. At the end, the rows without a match are returned with `NULL` columns.

All three algorithms assume one thing: rows with the same key are on the same node. In a distributed database, this is not true anymore. Let us see how to fix it for three ways of splitting the data.

### Layout 1: different tables on different servers

**The problem.** `Person` is on server 1, and `Messages` is on server 2. No server holds both tables.

**The simple solution.** Send the smaller table to the other server, and join there. The cost is one whole table.

**A better solution: the semi-join.** Often, only a few rows match. Then we send less data:
1. server 1 sends the keys of `Person` to server 2;
2. server 2 keeps only the rows of `Messages` with one of these keys, and sends them back;
3. server 1 joins `Person` with these rows. This gives the inner join.

**Outer joins.** For the left outer join, server 1 already has every row of `Person`. In step 3, it simply returns the unmatched ones with `NULL`. For the right and full outer joins, we also need the rows of `Messages` without a match, such as key 9. Server 2 can find them alone: they are the rows whose key was not in the list of step 1. So, in a step 4, server 2 returns them with `NULL` in the `Person` columns. The full outer join is the result of server 1 plus the result of server 2.

{: style="text-align:center; font-size: small;"}
<img width="85%" height="85%" src="{{ site.assets }}/distributed-matmul-dbs/layout_tables.svg">

{: style="text-align:center; font-size: small;"}
A semi-join of two tables held by two servers. Server 1 sends its keys (1). Server 2 replies with the matching rows only (2). Server 1 computes the left outer join (3), whose first three rows are the inner join. Server 2 returns its unmatched rows (4), which complete the full outer join.

If the list of keys is long, we can send a [Bloom filter][bloom] instead. It is much smaller. It lets a few non-matching rows through, but step 3 drops them.

[bloom]: https://en.wikipedia.org/wiki/Bloom_filter

### Layout 2: the same table split by rows

**The problem.** Each server holds some rows of each table. This is called **horizontal partitioning**, or **sharding**. Each server can join its own rows, but matching rows may sit on different servers. In the figure below, John is on server 2, but his message "Bye" is on server 1. A local inner join misses this pair. A local left outer join is even wrong: server 2 returns John with `NULL` columns.

**The solution: partition by key.** We place rows with the same function of the key on the same server. For example, the row with key $$id$$ goes to server $$id \bmod p$$. Then all rows with the same key live on the same server. Every server joins its rows locally, with no communication.

If the tables are not yet partitioned this way, we **shuffle** them first. Each server computes the target server of each of its rows. An all-to-all exchange (`MPI_Alltoallv`) sends every row to its target. This is the same pattern as in the distributed sample sort, described [in a different post]({{ site.baseurl }}{% post_url 2014-06-21-distributed-sort %}).

{: style="text-align:center; font-size: small;"}
<img width="85%" height="85%" src="{{ site.assets }}/distributed-matmul-dbs/layout_rows.svg">

{: style="text-align:center; font-size: small;"}
Both tables split by rows across 2 servers. Colors show the target server of each row, given by $$id \bmod 2$$: blue for server 1 (even keys) and orange for server 2 (odd keys). **Left:** before the shuffle. **Right:** after the shuffle, rows with the same key are on the same server.

**Outer joins.** After the shuffle, outer joins are local too. If a row has no match on its server, it has no match anywhere. In the figure, server 1 returns the three inner join rows. Server 2 returns Mary and Joe for the left outer join, and "Hey" for the right outer join.

**When one table is small: the broadcast join.** If `Person` is small, we can skip the shuffle. We copy `Person` to every server, and each server joins it with its local rows of `Messages`. The large table never moves. The inner join is correct, but outer joins need one more step. A row of `Person` may match on one server and not on another. In the figure below, John matches on server 1 but not on server 2. So no server can decide alone that a row has no match. Each server keeps a match flag per row of `Person`. A logical-or all-reduce (`MPI_Allreduce` with `MPI_LOR`) combines the flags. Then one server returns the rows whose final flag is 0, with `NULL` columns.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="{{ site.assets }}/distributed-matmul-dbs/broadcast.svg">

{: style="text-align:center; font-size: small;"}
A broadcast join for a left outer join. Every server holds a copy of `Person` and some rows of `Messages`, and flags the rows of `Person` that match locally. The all-reduce combines the flags. Only Mary and Joe have no match on any server.

A word of caution about load balance. If one key is very frequent, such as a person with thousands of messages, its server gets much more work than the others.

### Layout 3: the same table split by columns

**The problem.** Each server holds some columns of a table, for all rows. This is called **vertical partitioning**, and is the basis of [column-oriented databases][column-db]. Rows are aligned by position: row $$r$$ on every server is the same record. In our example, server 1 holds the key columns of both tables. Server 2 holds the other columns of `Person`, and server 3 the other columns of `Messages`.

**The idea.** To decide which rows match, we only need the keys. The other columns are just copied to the output. So we never move them until we know which rows we need.

**The solution: the join index.**
1. server 1 matches the keys, with any algorithm from above. The result is a **join index**: a list of pairs of row numbers, one in `Person` and one in `Messages`;
2. server 1 sends the `Person` row numbers to server 2, and the `Messages` row numbers to server 3;
3. servers 2 and 3 gather their rows in that order.

The output is split by columns too, and its rows are aligned by position. The large columns never cross the network. Building output rows only at the end, and only for the rows we need, is called **late materialization**. If the two key columns are on different servers, we first send one of them to the other. This is cheap: a key column holds one value per row.

**Outer joins.** The join index simply gets extra pairs with a missing side. For the left outer join, we add Mary (row 1) and Joe (row 3) with no row of `Messages`. Where the row number is missing, the server writes `NULL`.

{: style="text-align:center; font-size: small;"}
<img width="95%" height="95%" src="{{ site.assets }}/distributed-matmul-dbs/layout_columns.svg">

{: style="text-align:center; font-size: small;"}
A left outer join of two tables split by columns across 3 servers. The column `#` is the row number, implicit in the storage. Server 1 matches the keys and builds the join index (1). Servers 2 and 3 receive row numbers (2) and gather their rows (3). The first three rows are the inner join. The last two have no message, so server 3 writes `NULL`.

[column-db]: https://en.wikipedia.org/wiki/Column-oriented_DBMS

## Summary

Matrix multiplication and database joins follow the same rule: keep together the data that is used together. For a matrix product, these are the tiles of $$A$$ and $$B$$ that meet in a product. For a join, these are the rows with the same key.

When the layout already matches the operation, every rank works alone. When it does not, we move data with a few collective steps: an all-gather, an all-to-all, or rotations between neighbors. Two habits keep this cost low. First, send only what is needed: ghost entries, keys, or row numbers. Second, overlap the communication with useful computation.
