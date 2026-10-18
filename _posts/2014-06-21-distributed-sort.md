---
layout: post
title:  "Distributed Sorting Algorithms"
categories: ["Distributed Algorithms"]
tags: ["MPI", "sorting", "sample sort", "odd-even sort"]
---

[Sorting algorithms][sorting_algs] on a single node are well studied. Their costs for a list of $$n$$ elements, in [Big-O notation][bigo], are:
- [quicksort][quick]: picks a pivot, splits the list into the elements smaller and larger than the pivot, and sorts both parts recursively. $$O(n \log n)$$ on average, and $$O(n^2)$$ in the worst case, e.g. a sorted input with the first or last element as pivot;
- [insertion sort][insertion]: inserts each element, one at a time, into a sorted list. $$O(n^2)$$, and $$O(n)$$ for a sorted input. Fast for small lists;
- [selection sort][selection]: repeatedly finds the smallest unsorted element and swaps it with the first unsorted one. $$O(n^2)$$ comparisons, but at most $$n-1$$ swaps;
- [merge sort][merge]: splits the list into halves recursively, down to single elements, and merges the sorted halves. $$O(n \log n)$$ in all cases, with $$O(n)$$ extra memory;
- [heapsort][heap]: a selection sort that finds the largest element with a [heap][heapds] instead of a linear search. $$O(n \log n)$$ in the worst case;
- [bubble sort][bubble]: repeatedly passes over the list and swaps adjacent elements in the wrong order. $$O(n^2)$$, and $$O(n)$$ for a sorted input, which needs a single pass;
- [counting sort][counting]: sorts integers in a range of size $$k$$. One pass counts the occurrences of each value, and a [prefix sum][prefix] of the counts gives the output position of each value. $$O(n+k)$$;
- [radix sort][radix]: sorts integers digit by digit, from the least to the most significant digit, with a stable counting sort per digit. $$O(d(n+k))$$ for $$d$$ digits of $$k$$ possible values, e.g. $$d=3$$ and $$k=10$$ for the decimal number 991;
- [bucket sort][bucket]: distributes the values into $$b$$ buckets of disjoint, consecutive value ranges, sorts each bucket, and concatenates the buckets. The distribution and the concatenation take $$O(n)$$. For uniformly distributed values, each bucket holds about $$n/b$$ elements, and the buckets can be sorted in parallel.

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

The two recursive calls of quicksort and merge sort are independent, so these algorithms parallelize well on [shared memory][shm]. On [distributed memory][dm], each compute node holds part of the list, and elements move between nodes over the network. Below, $$p$$ ranks hold $$n$$ elements in total.

[shm]: https://en.wikipedia.org/wiki/Shared_memory
[dm]: https://en.wikipedia.org/wiki/Distributed_memory

## Distributed odd-even sort

[Odd-even sort][oddeven], or odd-even transposition sort, is a parallel bubble sort. The ranks form a line, and each rank exchanges data only with its two neighbors:
1. every rank sorts its local elements;
2. in even phases, ranks $$2i$$ and $$2i+1$$ form pairs, and in odd phases, ranks $$2i+1$$ and $$2i+2$$. The two ranks of a pair exchange their elements and merge them (**compare-split**): the lower rank keeps the smallest elements, and the higher rank the largest. Each rank keeps its number of elements;
3. if all ranks hold the same number of elements, the list is sorted after $$p$$ phases. With different numbers, $$p$$ phases are not always enough.

[oddeven]: https://en.wikipedia.org/wiki/Odd%E2%80%93even_sort

{: .picture}
<img width="65.8%" height="65.8%" src="{{ site.assets }}/distributed-sort/odd_even_sort.svg">

{: .caption}
Odd-even sort of 20 elements on $$p = 5$$ ranks, colored by rank. Every rank holds 4 elements, so the list is sorted after $$p$$ phases. Step 2.2 shows the compare-split of ranks 0 and 1 in phase 0.

The memory use is therefore known in advance, and a balanced load stays balanced. However, every phase sends all local elements: the $$p$$ phases move $$n$$ elements per rank, with $$p$$ message latencies. The sort is slow for large $$p$$.

## Distributed sample sort

[Sample sort][samplesort] is a distributed bucket sort. Each rank is a bucket, and the bucket ranges come from samples of the data:
1. every rank sorts its local elements and picks $$p$$ equidistant samples;
2. rank 0 gathers the samples, sorts them, and picks every $$p$$-th one as a **splitter**;
3. rank 0 broadcasts the splitters. They divide the value range into $$p$$ intervals, one per rank;
4. an all-to-all exchange (`MPI_Alltoallv`) sends each element to the rank of its interval;
5. every rank sorts the elements it received. The list is now sorted across the ranks.

[samplesort]: https://en.wikipedia.org/wiki/Samplesort

{: .picture}
<img width="57%" height="57%" src="{{ site.assets }}/distributed-sort/sample_sort.svg">

{: .caption}
Sample sort of 21 elements on $$p = 3$$ ranks, colored by rank. Each rank sends $$p-1 = 2$$ samples to rank 0, which sorts them and picks every $$p$$-th one as a splitter (yellow): 4 and 10. The last interval is open: rank 2 receives every element from 10 upwards.

Sample sort uses three collective operations, whatever $$n$$ and $$p$$: a gather of the samples, a broadcast of the splitters and an all-to-all of the elements. Each element moves once. However, the ranks may end with different numbers of elements, which unbalances computation and memory. Regular sampling, as here, bounds this imbalance: with distinct keys, no rank receives more than about $$2n/p$$ elements. If needed, a final exchange rebalances the elements.

The `C++` implementation of both algorithms is available in <a href="https://github.com/{{ site.repository }}/blob/master/assets/distributed-sort/distributedmemorysorter.cxx">DistributedMemorySorter.cxx</a> and <a href="https://github.com/{{ site.repository }}/blob/master/assets/distributed-sort/distributedmemorysorter.h">DistributedMemorySorter.h</a>.
