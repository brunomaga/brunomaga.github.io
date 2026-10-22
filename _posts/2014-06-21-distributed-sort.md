---
layout: post
title:  "Distributed Sorting Algorithms"
categories: ["Distributed Algorithms"]
tags: ["MPI", "sorting", "sample sort", "odd-even sort"]
---

Sorting is probably the most common algorithm in computer science. At some point in life, every CS student has studied the computational complexity of sorting algorithms, as measured by the [Big-O notation][bigo]. Serial sorting on a single compute node is a solved problem, with dozens of algorithms already available. To name a few:
- [quicksort][quick]: probably the most famous of all, relies on recursively comparing the remaining elements of the list to a pivot element. Performance ranges from $$O(n^2)$$ in the worst-case scenario (e.g. an already sorted input when the first or last element is the pivot) to $$O(n \log n)$$;
- [insertion sort][insertion]: iteratively takes one element from a list and inserts it into a final list of sorted elements. Efficient for small datasets, with a complexity ranging from $$O (n^2) $$ to $$O (n) $$ if the insertion position of each element is found immediately in the final list;
- [selection sort][selection]: a swapping algorithm that iteratively finds the smallest element in an unsorted list and swaps it with the leftmost unsorted element, decreasing the unsorted list size by 1. It is computationally inefficient as it requires $$ O(n^2)$$ comparisons, yet memory efficient with $$O(n)$$ memory swaps;
- [merge sort][merge]: probably the most general-purpose sorting algorithm. Created by John von Neumann, it follows a divide-and-conquer approach that divides a list into $$ n $$ sublists, which are recursively merged into larger lists until the initial set is sorted. Both the best and the worst case run with complexity $$ O (n \log n) $$;
- [heap sort][heap]: similar to the selection sort mentioned previously, yet improved, as it uses a [heap data structure][heapds] rather than a linear-time search to find the elements in the unsorted list, leading to a complexity of $$ O(n \log n)$$ (and $$O(n)$$ in the extreme case where all elements of the list are equal);
- [bubble sort][bubble]: performs sequential swaps of neighboring elements until sorting is completed. It thus requires *half* the number of operations of an all-to-all comparison, leading to a complexity of $$O(n^2)$$. The best-case complexity is $$O(n)$$, which is possible only if the elements are already sorted;
- [counting sort][counting]: highly efficient for integer values distributed across a small range of values. Navigates the list of unsorted elements once and increments the value counts in an array of counts (initialized to zero and of the same size as the value range). Complexity $$O(n+k)$$ for navigating the initial list of size $$n$$ and the count array of range $$k$$;  
- [radix sort][radix]: an adaptation of the previous one to larger intervals, which sequentially orders integer values by their key (given by the digit in a given position of the number). Iterations go from the least to the most significant digit, and perform a counting sort (with the i-th digit as key) at every iteration. Complexity: $$O(d(n+k))$$, where $$d$$ is the number of iterations, given by the number of digits (e.g. $$d=3$$ for value $$991$$), and $$k$$ is now the range of a digit (10 for decimal digits);    
- [bucket sort][bucket]: efficient on multicore architectures. Distributes a set of $$n$$ values into $$b$$ buckets, then sorts each bucket independently. The buckets hold disjoint and consecutive ranges of values, so the next operation concatenates the sorted buckets in order, which yields the sorted list. Complexity $$O(n)$$ for iterating through the initial dataset in the first scatter phase, plus the complexity of sorting $$n/b$$ elements in each bucket, plus a final iteration to concatenate the sorted buckets in the gather phase ($$O(n)$$);
- and [others][sorting_algs]...
    
[bigo]: https://en.wikipedia.org/wiki/Big_O_notation
[quick]: https://en.wikipedia.org/wiki/Quicksort
[insertion]: https://en.wikipedia.org/wiki/Insertion_sort
[selection]: https://en.wikipedia.org/wiki/Selection_sort
[merge]: https://en.wikipedia.org/wiki/Merge_sort
[heap]: https://en.wikipedia.org/wiki/Heapsort
[heapds]: https://en.wikipedia.org/wiki/Heap_(data_structure)
[bubble]: https://en.wikipedia.org/wiki/Bubble_sort
[counting]: https://en.wikipedia.org/wiki/Counting_sort
[bucket]: https://en.wikipedia.org/wiki/Bucket_sort
[radix]: https://en.wikipedia.org/wiki/Radix_sort
[sorting_algs]: https://en.wikipedia.org/wiki/Sorting_algorithm

Most of these algorithms allow for a multicore implementation: the underlying recursive- or tree-based nature of the problem makes them well suited for parallel execution on a single memory region.

However, if the list of elements is spread across a network of compute nodes (i.e. a [distributed memory](https://en.wikipedia.org/wiki/Distributed_memory) environment), the solution is not trivial. With that in mind, this post presents two algorithms for sorting distributed lists: the odd-even sort and the sample sort.

## Distributed Odd-Even Sort

The first one is the **[Odd-Even sort](https://en.wikipedia.org/wiki/Odd%E2%80%93even_sort)**, a distributed swap-based sorting algorithm. It was presented by Habermann in 1972 for a line of processors that hold one element each, and extended to several elements per processor by [Baudet and Stevenson](https://doi.org/10.1109/TC.1978.1674957) in 1978. Its rationale is the following:
1. compute nodes (ranks) are numbered sequentially. Each node has two neighbors: the next and the previous rank;
2. every node sorts its elements. Then, at every iteration (or *phase*), it exchanges all its elements with one of its neighbors (alternately the next and the previous rank), merges them with its own, and keeps the smallest elements if it is the lower rank of the pair, or the largest ones otherwise (a **compare-split** operation);
3. if all nodes hold the same number of elements, the operation is complete after a number of phases equal to the number of compute nodes (otherwise, it continues until two consecutive phases change no element).

The algorithm is illustrated with a sample dataset in the following workflow: 

{: .picture}
<img width="62.1%" height="62.1%" src="{{ site.assets }}/distributed-sort/odd_even_sort.svg">

{: .caption}
Odd-even sort of 20 elements on $$p = 5$$ ranks, colored by rank. Every rank holds 4 elements, so the list is sorted after $$p$$ phases. Step 2.2 shows the compare-split of ranks 0 and 1 in phase 0. In an MPI implementation, each exchange of step 2.1 is one `MPI_Sendrecv` between the two ranks of a pair.

The algorithm has the advantage of being very *memory-stable*, i.e. one can predict the worst-case memory consumption in advance. Moreover, each compute node ends up with the same number of elements it started with, so the initial load balancing across nodes is preserved. However, it is inefficient for a very large network of nodes, as it requires a high number of phases, and each phase sends all the elements of a node to its neighbor.

## Distributed Sample Sort

A faster alternative is based on a distributed implementation of the **[sample sort](https://en.wikipedia.org/wiki/Samplesort)**, a serial sorting algorithm introduced by [Frazer and McKellar](https://doi.org/10.1145/321592.321600) in 1970. The algorithm is as follows:
1. Each compute node sorts its local dataset and collects a few equidistant samples from it;
2. Samples are sent to the master rank, which sorts them and picks $$p-1$$ equidistant samples (the **splitters**, for $$p$$ compute nodes) from the set of received samples;
3. The master rank broadcasts the splitters to the network. The $$p-1$$ splitters delimit the range/interval of values to be sent to each of the $$p$$ nodes; 
4. Based on those ranges, the compute nodes perform a [collective](https://en.wikipedia.org/wiki/Collective_operation) scatter-gather (or a selective all-to-all) in which each element is sent to the appropriate target rank;
5. A final sort of the data in each memory region yields a globally sorted dataset, distributed across the network.

Here is an illustrative workflow:

{: .picture}
<img width="57%" height="57%" src="{{ site.assets }}/distributed-sort/sample_sort.svg">

{: .caption}
Sample sort of 21 elements on $$p = 3$$ ranks, colored by rank. Each rank sends $$p-1 = 2$$ samples to rank 0, which sorts them and picks every $$p$$-th one as a splitter (yellow): 4 and 10. The last interval is open: rank 2 receives every element from 10 upwards. In an MPI implementation, step 2 is one `MPI_Gather` of the samples, step 3 one `MPI_Bcast` of the splitters, and step 4 one `MPI_Alltoall` of the counts followed by one `MPI_Alltoallv` of the elements.

This method is computationally very efficient, as the number of communication operations is constant, regardless of the input size or network size, and each element is sent at most once. However, it may lead to a highly heterogeneous number of elements across nodes (from 4 to 9 in the example above), which leads to a computational and memory imbalance. If necessary, a network balancing operation may follow the sorting to balance the dataset across the network. In brief, the sample sort is preferred on large networks, and the odd-even sort on small networks or when each node must keep its number of elements.

The `C++` implementation of both algorithms, with [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface), is available in <a href="https://github.com/{{ site.repository }}/blob/master/assets/distributed-sort/distributedmemorysorter.cxx">DistributedMemorySorter.cxx</a> and <a href="https://github.com/{{ site.repository }}/blob/master/assets/distributed-sort/distributedmemorysorter.h">DistributedMemorySorter.h</a>.

To find more about the topic, the [slides of chapter 9](https://www.cs.purdue.edu/homes/ayg/book/Slides/chap9_slides.pdf) of the book Introduction to Parallel Computing by Grama, Gupta, Karypis and Kumar cover both algorithms, and the parallel versions of the quicksort and the bitonic sort.
