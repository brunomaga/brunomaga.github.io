---
layout: post
title:  "Distributed Sorting Algorithms"
categories: ["Distributed Algorithms"]
tags: ["MPI", "sorting", "sample sort", "odd-even sort"]
---

Sorting is probably the most common algorithm in computer science. Every computer science student studies the computational complexity of sorting algorithms, measured with the [Big-O notation][bigo]. Serial sorting on a single compute node is a solved problem, with dozens of algorithms available. Some examples are:
- [quicksort][quick]: probably the best known. It recursively compares the remaining elements of the list to a pivot element. Complexity ranges from $$O(n^2)$$ in the worst case (e.g. an already sorted input when the first or last element is the pivot) to $$O(n \log n)$$;
- [insertion sort][insertion]: takes one element at a time from a list and inserts it into a final list of sorted elements. Efficient for small datasets. Complexity ranges from $$O (n^2) $$ to $$O (n) $$, the latter when the insertion position of each element is found immediately in the final list;
- [selection sort][selection]: a swapping algorithm that repeatedly finds the smallest element in the unsorted list and swaps it with the leftmost unsorted element, reducing the unsorted list size by 1. It is computationally inefficient, requiring $$ O(n^2)$$ comparisons, but memory efficient, with $$O(n)$$ memory swaps;
- [merge sort][merge]: probably the most general-purpose sorting algorithm. Invented by John von Neumann, it follows a divide-and-conquer approach: the list is divided into $$ n $$ sublists, which are recursively merged into larger lists until the whole set is sorted. Both the best and the worst case have complexity $$ O (n \log n) $$;
- [heap sort][heap]: an improved selection sort. It uses a heap data structure instead of a linear-time search to find elements in the unsorted list, leading to a complexity of $$ O(n \log n)$$ (and $$O(n)$$ in the extreme case where all elements of the list are equal);
- [bubble sort][bubble]: performs sequential swaps of neighboring elements until the list is sorted. It requires *half* the number of operations of an all-to-all comparison, leading to a complexity of $$O(n^2)$$. The best-case complexity is $$O(n)$$, reached only when the elements are already sorted;
- [counting sort][counting]: highly efficient for integer values in a small range. It traverses the unsorted list once and increments the value counts in an array of counts (initialized to zero, with the same size as the value range). Complexity is $$O(n+k)$$, for traversing the initial list of size $$n$$ and the count array of range $$k$$;  
- [radix sort][radix]: an extension of counting sort to larger ranges. It orders integer values by their key, given by the digit at a given position of the number. Iterations go from the least to the most significant digit, and each iteration performs a counting sort with the i-th digit as key. Complexity: $$O(d(n+k))$$, where $$d$$ is the number of iterations, given by the number of digits (e.g. $$d=3$$ for value $$991$$);    
- [bucket sort][bucket]: efficient on multicore architectures. It distributes a set of $$n$$ values into $$b$$ buckets and sorts each bucket independently. The buckets cover disjoint, consecutive value ranges, so concatenating the sorted buckets in order yields the sorted list. Complexity is $$O(n)$$ for iterating through the initial dataset in the scatter phase, plus the complexity of sorting $$n/b$$ elements in each bucket, plus a final iteration that concatenates the sorted buckets in the gather phase ($$O(n)$$);
- and [others][sorting_algs].
    
[bigo]: https://en.wikipedia.org/wiki/Big_O_notation
[quick]: https://en.wikipedia.org/wiki/Quicksort
[insertion]: https://en.wikipedia.org/wiki/Insertion_sort
[selection]: https://en.wikipedia.org/wiki/Selection_sort
[merge]: https://en.wikipedia.org/wiki/Merge_sort
[heap]: https://en.wikipedia.org/wiki/Heapsort
[bubble]: https://en.wikipedia.org/wiki/Bubble_sort
[counting]: https://en.wikipedia.org/wiki/Counting_sort
[bucket]: https://en.wikipedia.org/wiki/Bucket_sort
[radix]: https://en.wikipedia.org/wiki/Radix_sort
[sorting_algs]: https://en.wikipedia.org/wiki/Sorting_algorithm

Most of these algorithms have multicore implementations: their recursive or tree-based structure makes them well suited for parallel execution on a single memory region.

However, when the list of elements is spread across a network of compute nodes --- i.e. a distributed memory environment --- the solution is not trivial. This post presents two algorithms for sorting distributed lists.

## Distributed Odd-Even Sort

The first is the **Odd-Even sort**, a distributed swap-based sorting algorithm. It works as follows:
1. compute nodes (ranks) are numbered sequentially, so each node has two neighbors: the previous and the next rank;
2. every node sorts its local elements. Then, at every iteration, each node exchanges all its elements with one of its neighbors (alternating between the next and the previous rank) and merges them with its own. The lower rank of the pair keeps the smallest elements, and the higher rank keeps the largest;
3. the sort completes after a number of iterations equal to the number of compute nodes.

The following workflow illustrates the algorithm on a sample dataset: 

{: .picture}
<img width="63.1%" height="63.1%" src="{{ site.assets }}/distributed-sort/odd_even_sort.svg">

{: .caption}
Odd-even sort of 20 elements on $$p = 5$$ ranks, colored by rank. Every rank holds 4 elements, so the list is sorted after $$p$$ phases. Step 2.2 shows the compare-split of ranks 0 and 1 in phase 0.

The algorithm is very *memory-stable*, i.e. its worst-case memory consumption is known in advance. Moreover, each compute node ends up with the same number of elements it started with, so the load is balanced across nodes. However, it is inefficient on very large networks, as it requires a high number of iterations.

## Distributed Sample Sort

A faster alternative is a distributed implementation of the **sample sort**, with the following steps:
1. Each compute node sorts its local dataset and picks a few equidistant samples from it;
2. All nodes send their samples to the master rank, which sorts them and picks $$c-1$$ equidistant samples (splitters, for $$c$$ compute nodes) from the received set;
3. The master rank broadcasts these samples to the network. The $$c-1$$ splitters divide the value range into $$c$$ intervals, one per node; 
4. Based on these ranges, the compute nodes perform a collective scatter-gather (or a selective all-to-all) that sends each element to its target rank;
5. A final local sort on each node yields a globally sorted dataset, distributed across the network.

The following workflow illustrates the method:

{: .picture}
<img width="57%" height="57%" src="{{ site.assets }}/distributed-sort/sample_sort.svg">

{: .caption}
Sample sort of 21 elements on $$p = 3$$ ranks, colored by rank. Each rank sends $$p-1 = 2$$ samples to rank 0, which sorts them and picks every $$p$$-th one as a splitter (yellow): 4 and 10. The last interval is open: rank 2 receives every element from 10 upwards.

This method is computationally very efficient, as the number of communication steps is constant, regardless of the input size or network size. However, it may lead to very different numbers of elements across nodes, causing computational and memory imbalance. If needed, a balancing step can follow the sort to redistribute the dataset evenly across the network.

The `C++` implementation of both algorithms is available in <a href="https://github.com/{{ site.repository }}/blob/master/assets/distributed-sort/distributedmemorysorter.cxx">DistributedMemorySorter.cxx</a> and <a href="https://github.com/{{ site.repository }}/blob/master/assets/distributed-sort/distributedmemorysorter.h">DistributedMemorySorter.h</a>. 
