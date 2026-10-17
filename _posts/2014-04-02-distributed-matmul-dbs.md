---
layout: post
title:  "Distributed Matrix Multiplication and Database Joins"
categories: ["Distributed Algorithms"]
tags: ["MPI", "linear algebra", "sparse matrices", "CSR", "databases", "joins"]
---

Two data structures sit behind a large share of scientific computing and data processing: the matrix and the table. A matrix holds numbers on a grid of rows and columns. A database table holds records, one per row, with named fields, one per column. When the data does not fit on one machine, or the computation takes too long, we split the data across several compute nodes connected by a network. Each node can only read its own memory. Any data it needs from another node must arrive as a message.

This post covers two operations in this setting: the matrix multiplication and the database join. In both cases the same question comes up. How do we split the data across nodes so that each node does most of its work locally, with as little communication as possible?

As in [MPI][mpi], a compute node is called a **rank**, and $$p$$ is the number of ranks. In the database sections, ranks are called **servers**.

[mpi]: https://en.wikipedia.org/wiki/Message_Passing_Interface

## Dense matrix multiplication

For a matrix $$A$$ of size $$n \times m$$ and a matrix $$B$$ of size $$m \times k$$, the product $$C = AB$$ is the $$n \times k$$ matrix with entries

$$
C_{ij} = \sum_{l=1}^{m} A_{il} B_{lj}.
$$

For square $$n \times n$$ matrices, this takes $$n^3$$ multiply-add operations. The matrix-vector product $$y = Ax$$ is the special case where $$B$$ has a single column, and takes $$n^2$$ operations.

### Matrix-vector multiplication

The simplest layout splits $$A$$ row-wise: each rank holds a block of $$n/p$$ consecutive rows. The vectors $$x$$ and $$y$$ are split the same way, so rank $$r$$ holds the entries of $$x$$ and $$y$$ that match its rows.

To compute an entry $$y_i$$, a rank needs row $$i$$ of $$A$$, which it holds, and the full vector $$x$$, which it does not. Each rank holds only $$n/p$$ entries of $$x$$. The algorithm has two steps:
1. an **all-gather** of $$x$$: every rank sends its entries of $$x$$ to all other ranks (`MPI_Allgather`). After this step, every rank holds the full vector;
2. a local matrix-vector product of the $$n/p$$ rows of $$A$$ with $$x$$, which gives the $$n/p$$ entries of $$y$$ of that rank.

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="{{ site.assets }}/distributed-matmul-dbs/matvec_row1d.svg">

{: style="text-align:center; font-size: small;"}
Row-wise matrix-vector multiplication on 4 ranks. Colors show the rank that holds each element. **Left:** the initial layout. **Right:** the data on rank 1 after the all-gather of $$x$$. Rank 1 computes $$y_2$$ and $$y_3$$ without further communication.

Each rank performs $$n^2/p$$ operations and receives $$n (p-1)/p$$ entries of $$x$$. The result $$y$$ has the same layout as $$x$$. This matters for iterative methods, such as the [power method][power] or the [conjugate gradient][cg], which use the output of one product as the input of the next.

[power]: https://en.wikipedia.org/wiki/Power_iteration
[cg]: https://en.wikipedia.org/wiki/Conjugate_gradient_method

### Matrix-matrix multiplication: Cannon's algorithm

We could apply the same idea to $$C = AB$$, with $$A$$, $$B$$ and $$C$$ split row-wise. Rank $$r$$ would then need every row of $$B$$. An all-gather of $$B$$ would copy the full matrix to every rank, so each rank would store $$n^2$$ elements, and we would lose the point of splitting the data.

A better option is a **2D block layout**. We arrange the $$p$$ ranks on a $$q \times q$$ grid, with $$q = \sqrt{p}$$, and split each matrix into $$q \times q$$ blocks of size $$n/q \times n/q$$. Rank $$(i,j)$$ holds the blocks $$A_{ij}$$ and $$B_{ij}$$ and must compute

$$
C_{ij} = \sum_{k=0}^{q-1} A_{ik} B_{kj}.
$$

This needs all blocks of row $$i$$ of $$A$$ and all blocks of column $$j$$ of $$B$$, but not at the same time. [Cannon's algorithm][cannon] moves the blocks around the grid so that, at every step, each rank holds a matching pair $$A_{ik}$$ and $$B_{kj}$$:
1. **alignment:** row $$i$$ of $$A$$ shifts left by $$i$$ positions, and column $$j$$ of $$B$$ shifts up by $$j$$ positions. Shifts are cyclic: a block that leaves the grid on one side enters it on the other. Rank $$(i,j)$$ now holds $$A_{ik}$$ and $$B_{kj}$$, with $$k = (i+j) \bmod q$$;
2. **repeat $$q$$ times:** each rank multiplies its two blocks and adds the product to $$C_{ij}$$. Then $$A$$ shifts one position left, and $$B$$ shifts one position up.

After $$q$$ steps, each rank has seen every value of $$k$$ exactly once, and $$C_{ij}$$ is complete.

{: style="text-align:center; font-size: small;"}
<img width="95%" height="95%" src="{{ site.assets }}/distributed-matmul-dbs/cannon.svg">

{: style="text-align:center; font-size: small;"}
Cannon's algorithm on a $$3 \times 3$$ grid of ranks. Each gray box is a rank, holding one block of $$A$$ and one block of $$B$$. **Left:** the initial layout. **Center:** after the alignment, every rank holds a pair $$A_{ik}$$, $$B_{kj}$$ with matching $$k$$. **Right:** after the first shift of $$A$$ to the left and of $$B$$ upwards, every rank holds a new matching pair.

At any time, a rank stores only three blocks, or $$3n^2/p$$ elements. At each step, it sends two blocks of $$n^2/p$$ elements, so the total communication per rank is $$2 q \, n^2 / p = 2 n^2 / \sqrt{p}$$ elements. All messages go to direct neighbors on the grid, which suits networks with a mesh or torus topology. The [SUMMA][summa] algorithm is a popular variant: it replaces the shifts with broadcasts along grid rows and columns, and also works on rectangular grids.

[cannon]: https://en.wikipedia.org/wiki/Cannon%27s_algorithm
[summa]: https://www.netlib.org/lapack/lawnspdf/lawn96.pdf

## Sparse matrix multiplication

Many matrices in practice are mostly zeros. Examples are the connectivity matrix of a graph, such as a social network or the synapses of a neural network, and the matrices of finite-element simulations. A matrix is **sparse** when it is worth storing only its nonzero elements.

### The compressed sparse row format

The **compressed sparse row** (CSR) format stores a sparse matrix with $$n$$ rows and $$nnz$$ nonzeros in three arrays:
- `values`, of size $$nnz$$: the nonzero values, row by row, from left to right;
- `col_idx`, of size $$nnz$$: the column index of each value;
- `row_ptr`, of size $$n+1$$: the values of row $$i$$ are at positions `row_ptr[i]` to `row_ptr[i+1]-1` of the two previous arrays. The last entry is $$nnz$$.

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="{{ site.assets }}/distributed-matmul-dbs/csr_format.svg">

{: style="text-align:center; font-size: small;"}
A $$5 \times 6$$ sparse matrix with 8 nonzeros (left) and its CSR representation (right). Colors show the row of each nonzero. Row 3 is empty, so `row_ptr[3]` and `row_ptr[4]` are both 6.

The format needs $$2 \, nnz + n + 1$$ entries instead of the $$n \times m$$ of the dense matrix. It gives direct access to any row, but finding the elements of a column requires a search over all rows. This is why the transpose of a distributed CSR matrix is a hard problem, covered [in a different post]({{ site.baseurl }}{% post_url 2015-04-21-matrix-transpose %}). The sparse matrix-vector product $$y = Ax$$ is a short loop with $$O(nnz)$$ operations:

```cpp
for (int i = 0; i < n; i++) {
  y[i] = 0;
  for (int k = row_ptr[i]; k < row_ptr[i+1]; k++)
    y[i] += values[k] * x[col_idx[k]];
}
```

### Distributed sparse matrix-vector multiplication

We use the same row-wise layout as in the dense case. Each rank holds a block of rows in CSR format, and the matching entries of $$x$$ and $$y$$. The dense algorithm started with an all-gather of $$x$$. For a sparse matrix this is wasteful: a rank only needs the entries $$x_j$$ for the columns $$j$$ where its rows have a nonzero, and these are usually few. The algorithm has a setup phase, run once per matrix, and a multiplication phase:
1. **setup:** each rank scans its `col_idx` array and lists the columns whose entry of $$x$$ is held by another rank. The owner of entry $$x_j$$ follows from the row split, since $$x$$ is split like the rows. Ranks send these lists to the owners, so that each rank knows which of its entries to send, and to whom;
2. **exchange:** before each product, ranks send only the requested entries of $$x$$, with an `MPI_Alltoallv` or with point-to-point messages. A rank stores the received entries, called **ghost** or **halo** entries, after its own entries;
3. **local product:** each rank runs the CSR loop above, with `col_idx` renumbered during the setup to point into the array of own and ghost entries.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="{{ site.assets }}/distributed-matmul-dbs/spmv.svg">

{: style="text-align:center; font-size: small;"}
Sparse matrix-vector multiplication on 3 ranks. **Left:** nonzeros of $$A$$, colored by the rank that holds them. A dot marks a nonzero whose entry of $$x$$ is held by another rank. **Right:** entries of $$x$$ on each rank after the exchange. Dashed boxes are received copies, colored by the rank they came from.

The communication volume now depends on the number of distinct remote columns, not on $$n$$. For a matrix that comes from a mesh, and a good split of its rows, a rank only needs the entries on the border of its region of the mesh.

### Distributed sparse matrix-matrix multiplication

Consider now $$C = AB$$, with $$A$$ and $$B$$ in CSR format and split row-wise. Row $$i$$ of $$C$$ is a weighted sum of rows of $$B$$:

$$
C_{i,:} = \sum_{k \, : \, A_{ik} \neq 0} A_{ik} \, B_{k,:}.
$$

A rank therefore needs the rows of $$B$$ whose indices are the column indices of its nonzeros of $$A$$. This is the same set of indices as the ghost entries of the matrix-vector product, but we now fetch whole rows of $$B$$ instead of single entries of $$x$$. The algorithm is:
1. each rank lists the remote rows of $$B$$ that it needs, and requests them from their owners;
2. the owners send these rows in CSR format, with two `MPI_Alltoallv` calls: the first sends the number of nonzeros of each row, so that receivers can allocate memory, and the second sends the column indices and values;
3. each rank computes its rows of $$C$$ locally. For each row $$i$$, it adds the scaled rows of $$B$$ into a dense array of size $$k$$, the number of columns of $$B$$, and then compresses the nonzeros of that array into CSR format.

The number of nonzeros of $$C$$ is not known in advance. Implementations therefore often run step 3 twice: a first, symbolic pass counts the nonzeros of each row of $$C$$ and allocates memory, and a second, numeric pass computes the values. The result $$C$$ has the same row split as $$A$$, so it is ready to be the input of the next product.

## Distributed database joins

A relational database stores data in tables. Each row is a record, and each column is a field. A **key** is a column that identifies a record, and is used to link tables. In our example, the table `Person` has one row per person, identified by the key `id`. The table `Messages` has one row per message, and its column `id` holds the key of the person who wrote it.

### Inner and outer joins

A **join** combines the rows of two tables whose keys match. In SQL, the join of `Person` and `Messages` is written as:

```sql
SELECT Person.id, name, age, msg, year
FROM Person INNER JOIN Messages ON Person.id = Messages.id;
```

There are four types of join:
- the **inner join** returns one row for each pair of rows with matching keys. Kim wrote two messages, so Kim appears twice. Mary and Joe wrote no messages, and nobody has the key 9, so these rows do not appear;
- the **left outer join** returns the inner join, plus every row of the left table (`Person`) that has no match. The columns of the right table are then filled with `NULL`;
- the **right outer join** is the mirror case: it keeps every row of the right table (`Messages`);
- the **full outer join** keeps the unmatched rows of both tables.

{: style="text-align:center; font-size: small;"}
<img width="85%" height="85%" src="{{ site.assets }}/distributed-matmul-dbs/joins.svg">

{: style="text-align:center; font-size: small;"}
**Top:** the tables `Person` (blue) and `Messages` (green). **Bottom:** their inner join, left outer join and full outer join on the column `id`. Gray cells hold `NULL`.

On a single node, with $$N$$ rows in one table and $$M$$ in the other, there are three classic join algorithms:
- the **nested loop join** compares every pair of rows, and takes $$O(N \cdot M)$$ operations;
- the **sort-merge join** sorts both tables by key, and then walks through both with two pointers, as in the merge step of the merge sort. It takes $$O(N \log N + M \log M)$$ operations, or $$O(N+M)$$ if the tables are already sorted;
- the **hash join** builds a hash table with the keys of the smaller table, then scans the other table and looks up each key. It takes $$O(N+M)$$ operations on average.

Outer joins use the same algorithms. Each row keeps a flag that records whether it found a match, and at the end the rows without a match are returned with `NULL` columns.

All these algorithms rely on one fact: rows with the same key are on the same node. When the tables are distributed, this is no longer true, and the data layout decides how much data must move. We now look at three layouts.

### Layout 1: different tables on different servers

In the first layout, `Person` is on server 1 and `Messages` is on server 2. A simple solution is to send the smaller table to the other server and join locally. The cost is the size of the smaller table.

When only a few rows match, a **semi-join** sends less data:
1. server 1 sends the distinct keys of `Person` to server 2;
2. server 2 keeps only the rows of `Messages` whose key is in that list, and sends them back;
3. server 1 joins `Person` with the received rows locally.

{: style="text-align:center; font-size: small;"}
<img width="68%" height="68%" src="{{ site.assets }}/distributed-matmul-dbs/layout_tables.svg">

{: style="text-align:center; font-size: small;"}
A semi-join of two tables held by two servers. Server 1 sends its keys, and server 2 replies with its matching rows only. The message with key 9 never leaves server 2.

The network now carries one key per row of `Person` and the matching rows of `Messages`, instead of a whole table. When the list of keys is long, a [Bloom filter][bloom] can replace it. The filter is much smaller, but it lets a few non-matching rows through, which step 3 then discards.

A left outer join works unchanged, because server 1 holds every row of `Person` and fills the unmatched ones with `NULL`. A right or full outer join also needs the rows of `Messages` without a match, such as key 9. The semi-join filters these rows out, but server 2 can find them locally: they are the rows whose key is not in the received list. Server 2 returns them with `NULL` in the columns of `Person`, and the result is the union of the outputs of both servers.

[bloom]: https://en.wikipedia.org/wiki/Bloom_filter

### Layout 2: the same table split row-wise

In the second layout, each table is split row-wise: every server holds a subset of the rows of each table. This is also called **horizontal partitioning**, or **sharding**. Each server can join its local rows, but a row of `Person` and a matching row of `Messages` may be on different servers. In the example below, John is on server 2, and John's message "Bye" is on server 1. Local joins would miss this match. Worse, a local left outer join on server 2 would wrongly return John with `NULL` columns.

The solution is to **partition both tables by the join key**, with the same function. For example, the row with key $$id$$ goes to server $$id \bmod p$$, or each server receives a range of keys. All rows with the same key then live on the same server, and every server joins its rows locally, with no communication. This also holds for outer joins: a row without a local match has no match on any other server.

When the tables are not partitioned by the join key, we first repartition them with a **shuffle**. Each server computes the target server of each of its rows, and an all-to-all exchange (`MPI_Alltoallv`) sends every row to its target. This is the same communication pattern as in the distributed sample sort, described [in a different post]({{ site.baseurl }}{% post_url 2014-06-21-distributed-sort %}).

{: style="text-align:center; font-size: small;"}
<img width="85%" height="85%" src="{{ site.assets }}/distributed-matmul-dbs/layout_rows.svg">

{: style="text-align:center; font-size: small;"}
Both tables split row-wise across 2 servers. Colors show the target server of each row, given by $$id \bmod 2$$: blue for server 1 (even keys) and orange for server 2 (odd keys). **Left:** the initial layout. **Right:** after the all-to-all shuffle, all rows with the same key are on the same server, and each server joins its rows locally.

When one table is much smaller than the other, a **broadcast join** avoids moving the large table. Every server receives a full copy of the small table and joins it with its local rows of the large table. Outer joins need care here. Suppose `Person` is the small, copied table and we want a left outer join. A server cannot tell whether a row of `Person` without a local match found a match on another server. An extra step fixes this: the servers combine their match flags with a logical-or all-reduce, and a single server returns the rows of `Person` that matched nowhere.

The shuffle can also cause load imbalance. If a key is very frequent, for example a person who wrote thousands of messages, its server receives much more data than the others.

### Layout 3: the same table split column-wise

In the third layout, each table is split column-wise: every server holds some columns of a table, for all its rows. This is also called **vertical partitioning**, and is the basis of [column-oriented databases][column-db], which read only the columns a query needs. Rows are aligned by position: row number $$r$$ on every server belongs to the same record.

The join needs only the key columns to decide which rows match. The other columns are just copied to the output. Assume server 1 holds the key columns of both tables, server 2 the other columns of `Person`, and server 3 the other columns of `Messages`. The join then works as follows:
1. server 1 matches the keys with any of the single-node algorithms above. The result is a **join index**: the list of pairs of row numbers, one in `Person` and one in `Messages`, whose keys match;
2. server 1 sends the row numbers of `Person` to server 2, and the row numbers of `Messages` to server 3;
3. servers 2 and 3 gather their rows in the order of the received lists.

{: style="text-align:center; font-size: small;"}
<img width="95%" height="95%" src="{{ site.assets }}/distributed-matmul-dbs/layout_columns.svg">

{: style="text-align:center; font-size: small;"}
An inner join of two tables split column-wise across 3 servers. The column `#` is the row number, which is implicit in the storage. Server 1 holds the keys and computes the join index. Servers 2 and 3 receive row numbers and gather their rows. The result is split column-wise as well, with its rows aligned by position.

The other columns never cross the network, and the result keeps the column-wise layout of the input. Building the output rows only at the end, and only for the rows that match, is called **late materialization**. If the two key columns are on different servers, one of them is sent to the other server first. This is cheap, as a key column holds a single value per row.

Outer joins use the same steps. The join index then also holds pairs with a missing side, such as Mary's row 1 with no row of `Messages`, and the servers write `NULL` where the row number is missing.

[column-db]: https://en.wikipedia.org/wiki/Column-oriented_DBMS

## Summary

Matrix multiplications and database joins reward the same strategy: place the data that is used together on the same rank. For a matrix product, this means the blocks of $$A$$ and $$B$$ that meet in a product. For a join, it means the rows with the same key. When the layout matches the operation, each rank works locally. When it does not, a single collective operation, such as an all-gather, an all-to-all or a series of shifts between neighbors, rearranges the data first. Sending only what is needed, such as the ghost entries of $$x$$, the keys of a semi-join or the row numbers of a join index, keeps that communication small.
