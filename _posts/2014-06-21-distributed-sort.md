---
layout: post
title:  "Distributed Sorting Algorithms"
categories: ["Distributed Algorithms"]
tags: ["MPI", "sorting", "sample sort", "odd-even sort"]
---

Sorting is one of the most common operations in computer science, and the cost of [sorting algorithms][sorting_algs], measured with the [Big-O notation][bigo], is one of the first topics of an algorithms course. On a single compute node, sorting is a well-studied problem, with dozens of algorithms to choose from. For a list of $$n$$ elements, some of the best known are:
- [quicksort][quick] picks a pivot element, splits the list into the elements smaller and larger than the pivot, and sorts both parts recursively. It takes $$O(n \log n)$$ operations on average, but $$O(n^2)$$ in the worst case, for example for a sorted input with the first or last element as pivot, since every split then removes only one element;
- [insertion sort][insertion] takes one element at a time and inserts it at its place in a sorted list. It takes $$O(n^2)$$ operations, but only $$O(n)$$ for a sorted input, which makes it fast for small or almost sorted lists;
- [selection sort][selection] repeatedly finds the smallest unsorted element and swaps it with the first unsorted one. It needs $$O(n^2)$$ comparisons, but at most $$n-1$$ swaps, which helps when writing to memory is expensive;
- [merge sort][merge], invented by John von Neumann in 1945, splits the list into halves recursively, down to single elements, and merges the sorted halves. It takes $$O(n \log n)$$ operations in all cases, and $$O(n)$$ extra memory for the merges;
- [heapsort][heap] improves on selection sort, as it finds the largest element with a [heap][heapds] instead of a linear search. It takes $$O(n \log n)$$ operations in the worst case, and $$O(n)$$ when all elements are equal;
- [bubble sort][bubble] passes over the list again and again, and swaps adjacent elements that are in the wrong order. It takes $$O(n^2)$$ operations, but a sorted input needs a single pass, in $$O(n)$$;
- [counting sort][counting] sorts integers in a small range of $$k$$ values. One pass over the list counts the occurrences of each value, and a [prefix sum][prefix] of the counts gives the output position of each value, in $$O(n+k)$$ operations in total;
- [radix sort][radix] extends counting sort to larger ranges. It sorts integers digit by digit, from the least to the most significant digit, with a counting sort per digit, in $$O(d(n+k))$$ operations for $$d$$ digits of $$k$$ possible values, e.g. $$d=3$$ and $$k=10$$ for the decimal number 991. The counting sort must be stable, i.e. keep elements with equal digits in their order, so that it does not undo the order set by the previous digits;
- [bucket sort][bucket] distributes the values into $$b$$ buckets of disjoint, consecutive value ranges, sorts each bucket, and concatenates the buckets in order. The distribution and the concatenation take $$O(n)$$ operations, and for uniformly distributed values each bucket holds about $$n/b$$ elements. The buckets are independent, so they can be sorted in parallel, e.g. on the cores of a multicore processor.

[bigo]: https://en.wikipedia.org/wiki/Big_O_notation
[quick]: https://en.wikipedia.org/wiki/Quicksort
[insertion]: https://en.wikipedia.org/wiki/Insertion_sort
[selection]: https://en.wikipedia.org/wiki/Selection_sort
[merge]: https://en.wikipedia.org/wiki/Merge_sort
[heap]: https://en.wikipedia.org/wiki/Heapsort
[heapds]: https://en.wikipedia.org/wiki/Heap_(data_structure)
[bubble]: https://en.wikipedia.org/wiki/Bubble_sort
[counting]: https://en.wikipedia.org/wiki/Counting_sort
[prefix]: https://en.wikipedia.org/wiki/Prefix_sum
[bucket]: https://en.wikipedia.org/wiki/Bucket_sort
[radix]: https://en.wikipedia.org/wiki/Radix_sort
[sorting_algs]: https://en.wikipedia.org/wiki/Sorting_algorithm

Most of these algorithms also have parallel versions for a single machine. The two recursive calls of quicksort and merge sort, for example, work on separate parts of the list, so they can run in parallel on [shared memory][shm]. The problem is harder when the list is spread over the compute nodes of a [distributed memory][dm] system. Each node then holds only part of the list, and the elements that sit on the wrong node must move over the network, which is slow. The two algorithms below take opposite approaches: the first one moves the elements in small steps between neighbors, and the second one sends each element directly to its final node. In both, the $$n$$ elements are spread over $$p$$ processes, called **ranks** in [MPI][mpi], usually one per node.

[shm]: https://en.wikipedia.org/wiki/Shared_memory
[dm]: https://en.wikipedia.org/wiki/Distributed_memory
[mpi]: https://en.wikipedia.org/wiki/Message_Passing_Interface

## Distributed odd-even sort

The first algorithm, [odd-even sort][oddeven], also called odd-even transposition sort, is a parallel version of bubble sort that works on blocks of elements instead of single elements. The ranks are numbered in a line, so each rank has two neighbors, the previous and the next rank, and it exchanges data only with them:
1. every rank sorts its local elements;
2. the ranks then go through a series of phases. In even phases, ranks $$2i$$ and $$2i+1$$ form pairs, and in odd phases, ranks $$2i+1$$ and $$2i+2$$, so each rank alternates between its two neighbors. The two ranks of a pair exchange all their elements and merge them with their own, in an operation called **compare-split**: the lower rank keeps the smallest elements, and the higher rank keeps the largest. Each rank keeps its number of elements;
3. if all ranks hold the same number of elements, the list is sorted after $$p$$ phases.

[oddeven]: https://en.wikipedia.org/wiki/Odd%E2%80%93even_sort

{: .picture}
<img width="65.8%" height="65.8%" src="{{ site.assets }}/distributed-sort/odd_even_sort.svg">

{: .caption}
Odd-even sort of 20 elements on $$p = 5$$ ranks, colored by rank. Every rank holds 4 elements, so the list is sorted after $$p$$ phases. Step 2.2 shows the compare-split of ranks 0 and 1 in phase 0.

Since every rank keeps its number of elements, the memory use is known in advance, and a balanced load stays balanced. The price is the number of phases. A compare-split moves elements by one rank only, so the largest elements, if they start on rank 0, need $$p-1$$ phases to reach the last rank. Every phase sends all the local elements, so with $$n/p$$ elements per rank, the $$p$$ phases move $$n$$ elements per rank and pay $$p$$ message latencies, which makes the algorithm slow on many ranks. With different numbers of elements per rank, $$p$$ phases are not always enough, since a rank with few elements lets only a few through in each phase, and the ranks must continue until an even and an odd phase in a row change nothing.

## Distributed sample sort

A faster alternative is [sample sort][samplesort], a distributed version of bucket sort in which every rank is a bucket. The difficulty is to choose the value range of each bucket so that all ranks receive similar numbers of elements, without knowing the distribution of the values in advance. Sample sort estimates this distribution from a small sample of the data:
1. every rank sorts its local elements and picks $$p-1$$ equidistant samples;
2. rank 0 gathers the samples, sorts them, and picks every $$p$$-th one as a **splitter**;
3. rank 0 broadcasts the $$p-1$$ splitters, which divide the value range into $$p$$ intervals, one per rank;
4. an all-to-all exchange sends each element to the rank of its interval: an `MPI_Alltoall` of the counts tells every rank how many elements it receives, and an `MPI_Alltoallv` moves the elements;
5. every rank sorts the elements it received. The list is now sorted across the ranks.

[samplesort]: https://en.wikipedia.org/wiki/Samplesort

{: .picture}
<img width="57%" height="57%" src="{{ site.assets }}/distributed-sort/sample_sort.svg">

{: .caption}
Sample sort of 21 elements on $$p = 3$$ ranks, colored by rank. Each rank sends $$p-1 = 2$$ samples to rank 0, which sorts them and picks every $$p$$-th one as a splitter (yellow): 4 and 10. The last interval is open: rank 2 receives every element from 10 upwards.

Sample sort needs a constant number of collective operations, whatever $$n$$ and $$p$$: a gather of the samples, a broadcast of the splitters and the all-to-all exchange, in which each element moves at most once. Only the sampling grows quickly with $$p$$, since rank 0 receives $$p(p-1)$$ samples, so for more than 4,096 ranks the implementation gathers them in two levels, first within groups of 128 ranks.

The weakness of sample sort is the balance. In the figure, the ranks end with 9, 4 and 8 elements, since six samples describe 21 values only roughly, and such differences unbalance both the computation and the memory. Regular sampling, with equidistant samples from every rank as here, bounds this imbalance: with distinct keys, no rank receives more than about $$2n/p$$ elements ([Harsh et al., 2019][hss], Theorem 3.2). If an even load is needed, a final exchange can rebalance the elements. Overall, sample sort is the better choice on many ranks, while odd-even sort suits a few ranks, or a program that needs a fixed number of elements on every rank.

[hss]: https://arxiv.org/abs/1803.01237

The `C++` implementation of both algorithms is available in <a href="https://github.com/{{ site.repository }}/blob/master/assets/distributed-sort/distributedmemorysorter.cxx">DistributedMemorySorter.cxx</a> and <a href="https://github.com/{{ site.repository }}/blob/master/assets/distributed-sort/distributedmemorysorter.h">DistributedMemorySorter.h</a>.
