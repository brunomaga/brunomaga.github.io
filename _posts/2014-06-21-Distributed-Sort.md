---
layout: post
title:  "Distributed Sorting Algorithms"
date:   2014-06-21 12:01:42 +0100
categories: [distributed algorithms, parallel algorithms]
tags: [algorithms]
---

Sorting is probably the most common computer science algorithm. In fact, at some point in life, every CS student studied computational complexity based on the number of operations of different sorting algorithms, measured by the [Big-O notation][bigo]. Sorting of serial algorithms in single compute nodes is a solved problem with dozens of algorithms analyzed. To name a few:
- [quicksort][quick]: probably the most famous of all, relies on comparing a pivot term and to the remainder elements of the list recursively. Performance ranges from $$O (n^2) $$ for worst-case scenario with sorted lists to $n log (n) $$;
- [insertion sort][insertion]: iteratively takes one element from a list and inserts it into a final list of sorted elements. Efficient for small datasets, with an efficient ranging from $$O (n^2) $$ to $$O (n) $$ if element to be inserted is found immediately in the final list;
- [selection list][selection]: a swapping algorithm that iteratively finds the smallest element in an unsorted list, and swaps itself with the leftmost unsorted, decreasing the unsorted list size by 1. It is computationally inefficient as it requires $$ O(n^2) $$ comparisons yet memory efficient with $$ O(n) $$ memory swaps;
- [merge sort][merge]: probably the most general-purpose sorting algorithm. Created by John Von Neumann, follows a divide-and-conquer algorithm that divides a list into $$ n $$ sublists that recursively merge larger lists until the initial set is sorted. Best and worst case scenario run with complexity $$ O (n log n) $$;
- [heap sort][heap]: similar to the selection sort mentioned previously, yet improved as it uses a heap data structure rather than a linear-time search to find the elements in the unsorted list, leading to a complexity of $$ O(n log n)$$ (with $$O(n)$$ complexity possible in the extreme case of having equal elements on the list);
- [bubble sort][bubble]: performs sequential swaps of neighboring elements until sorting is completed. Thus requires an *halved* all-to-all number of operations, leading to a complexity of $$O(n^2)$$. A best case scenarion complexity of $$O(n)$$ is possible if elements are sorted beforehand;
- [counting sort][counting]: highly efficient for integer values on small number intervals, ideally with several repeated elements. Navigates the list of unsorted elements once and increments the value count on a key values map (as indexes into an array). $$O(n+k)$$ efficiency for navigating initial list of size $$n$$ and key value map of range of values $$k$$;  
- [bucket sort][bucket]: efficient on multicore architectures. Distributes a set o $$n$$ values into $$b$$ buckets, sorting each bucket individually. A final operation iterates through buckets and retrieves sorted list. Complexity $$O(n)$$ for iterating through inital dataset in the first scatter phase, plus the complexity of sorting $$n/b$$ elements on each bucket, plus final iteration to merge sorted buckets in the gather phase ($$O(n)$$) --- disclaimer: different implementations yield different efficiencies;  
- and [others][sorting_algs]...
    
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

Most algorithms described allow for a multicore implementation and the underlying recursive- or tree-based nature of the problem is well suitable for parallel execution in the same memory region.

However, if the list of elements is spread across a network of compute nodes --- i.e. a distributed memmory environment --- the resolution is not trivial. With that in mind, we present two algorithms for the sorting of distributed lists.

The first one is the *Odd-Even sort*. Its rationale is similar to any swapping-based sort algorithm. to the afforementioned selec  

<p align="center"><img width="60%" height="60%" src="/assets/2014-Distributed-Sort/odd_even_sort.png"></p>

<p align="center"><img width="60%" height="60%" src="/assets/2014-Distributed-Sort/sample_sort.png"></p>

