/**
 * @file DistributedMemorySorter.h
 * @brief Class responsible for algorithms used on distributed memmory
 * (eg: load balancing, sorters, etc)
 * @date 2012-07-18
 */

#ifndef BBP_DISTRIBUTEDMEMORYSORTER_H
#define	BBP_DISTRIBUTEDMEMORYSORTER_H

#include "mpi.h"

//For some situations we dont need to sort using the real data size/accuracy,
//e.g. when we perform slicing, the sorting operation is just for the slicing,
// values wont be used later on.
//These datatypes below define the types used for the sorting: even though the
//original datatype is double/MPI_DOUBLE, we use float to speed up things a bit,
//and to save RAM (as BG is limited to 1GB per CPU)

#define t_sorting_data double //float
#define t_sorting_data_MPI MPI_DOUBLE //MPI_FLOAT

/*
 * Defines the size of each group used for sub sampling in the Sample Sort
 * algorithm.\n\n
 * Example: for 16384 CPUs (full machine), if SUB_SAMPLING_GROUP_SIZE
 * is set as 2048, then we do sampling for [0..2043, 2044..4095, 4096.. 8172, etc].
 * After that, all 8 masters (1 for each group) will do the same procedure with
 * a unique master (rank 0);
 */
#define SUB_SAMPLING_GROUP_SIZE 128

/*
 * Determines the minumum number of CPUs used for subsampling in Sample Sort
 * algorithm.
 */
#define SUB_SAMPLING_MINIMUM_CPUS 4096

/**
 * Class with static methods to handle distributed memmory sort.
 * Two algorithms available: Odd Even sort and Sample sort
 */
class DistributedMemorySorter{

public:
     //TEST METHOD TO BE DELETED
     static void test_OddEvenSort();
     static void test_SampleSort();

     /**
      * Performs an Odd Even sorting parallel algorithm.\n
      * This methods is slower than the parallel Sample Sort, but it has the
      * advantadge of each CPU holding always the same amount of coordinates
      * that were inputed.\n
      * Approximate memory usage is 3 times the size of input coordinates array.
      * @param segmentsCount (IN) number of segments that will be input
      * - by definition, its also the ammount of elements to be output.
      * @param coordinates (IN/OUT) original and sorted coordinates
      * @param sortDimension (IN) sorting dimensions
      * @return returns the number of segments in the total system, i.e. sum of the segments of all CPUs.
      */
     static long long oddEvenSort_MPI( long long segmentsCount, t_sorting_data *coordinates, int sortDimension, MPI_Comm mpiComm, bool outputWarnings = false);

     /**
      * Performs a Compare Split ordering: \n
      * Receives two ALREADY SORTED arrays (coordinates and recvCoords), with
      * segmentsCount and recvSegmentsCount segments respectively.
      * Uses tempCoords as temporary array for the calculation (same size of
      * segmentsCount).\n
      * Output is returned as the coordinates array, sorted by dimension dim.
      * If keepsmall is 0, it will contains the biggest of the comparison
      * Otherwise it will return the smallest elements of the comparison\n
      * EG: basic compare split for [1,5,8,10,11] (coords) and [1,3,9] (recv):\n
      * if keepsmall returns [1,1,3,5,8]\n
      * if !keepSmall returns [9,10,11].
      * This algorithm does the same but takes coordinates instead of numbers.
      * \n
      * It validades the results by performing the 3 follwing checks
      * \n - checks if total number of elements in the system are the same (before and after)
      * \n - checks if sum of all elements in the system are the same (before and after)
      * \n - checks if they are sorted - ie. max(rank)<min(rank+1)<max(rank+1)<min(rank+2)<... , for all ranks
      * @param segmentsCount (IN) number of segments/coordinates in coordinates array
      * @param coordinates (IN/OUT) coordinates array (format [x0,y0,z0, x1,y1,z1, x2,y2,z2, ...]
      * @param recvSegmentsCount (IN) number of segments/coordinates in recvCoords array
      * @param recvCoords (IN) second coordinates array (same formate as coordinates)
      * @param tempCoords (IN) an array with same size ad type as coordinates - to be used as intermediatte data
      * structure for the calculation. Data to be allocated before calling this functions, and to be freed after.
      * @param dim (IN) dimension for sorting
      * @param keepSmall (IN) Zero or Not, whether we keep bigger or smaller portion of elements (as description)
      */
     static void compareSplit(long long segmentsCount, t_sorting_data *coordinates, long long recvSegmentsCount, t_sorting_data *recvCoords, t_sorting_data* tempCoords, int dim, int keepSmall);

     /**
      * Performs a parallel Sample Sort.\n
      * This algorithm is a very fast implementation of a distributed memory
      * sort, but results on a different number of segments per CPU. Use
      * OddEvenSort_MPI instead, if you want the same number of segments per CPU.\n
      * \n
      * \nIt validades the results by performing the 3 follwing checks
      * \n - checks if total number of elements in the system are the same (before and after)
      * \n - checks if sum of all elements in the system are the same (before and after)
      * \n - checks if they are sorted - ie. max(rank)<min(rank+1)<max(rank+1)<min(rank+2)<... , for all ranks
      * @param segmentsCount_ptr (IN/OUT) number of coordinates input.
      * After execution will be updated with the number of segments on the new
      * sorted coordinates. I.e. its a pointer to the count;
      * @param coordinates_ptr (IN/OUT) contains all coordinates: a pointer to the array containing all coordinates
      * After execution it holds an arrays set of (probably differente size), holding sorted coordinates
      * @param (IN) sortDimension sorting dimension
      * @return returns the number of segments in the total system, i.e. sum of the segments of all CPUs.
      */
     static long long sampleSort_MPI( long long * segmentsCount_ptr, t_sorting_data **coordinates_ptr, int sortDimension, MPI_Comm mpiComm);


     /**
      * Balances the ammount of data accross all the CPUs in the system.
      * Ie: if started by eg 4 CPUs with arrays size 3,1,5 and 7 (ie total 16),
      * after the execution they will all contains 4 elements each (4 x 4 = 16).
      * This algorithm keeps the order, ie. the final elements of an array will
      * be appended to the beginning of next rank's array. Therefore, if given
      * a set of ordered arrays, the output will still be ordered.\n
      * \n
      * The following validations are performed:\n
      * - Sum of all segments before and after is the same\n
      * - Count of all elements before and after is the same\n
      * - Gap between highest and lowest element count is either 0 or 1;
      * @param mySegmentsCount_ptr (IN/OUT) pointer to number of segments in the array
      * @param coordinates_ptr (IN/OUT) pointer to the array containing the coordinates
      * @param mpiComm MPI communicatoryy
      * @return number of segments in the whole system (after balancing)
      */
     static int loadBalancing_MPI(long long * mySegmentsCount_ptr, t_sorting_data ** coordinates_ptr, MPI_Comm mpiComm, bool outputWarnings = false);

private:
     /**
      * populates preCheck_globalValuesSum and preCheck_globalValuesCount variables with the
      * sum and count of all elements in the system
      * @param segmentsCount (IN) number of Segments
      * @param coordinates (IN) array of elements
      * @param preCheck_globalValuesCount (OUT) count of all elements in the whole system - to be populated
      * @param preCheck_globalValuesSum (OUT) sum of all elements in the whole system - to be populated
      */
     static void getGlobalInformation(long long segmentsCount, t_sorting_data * coordinates, long long *check_globalValuesCount, long double *check_globalValuesSum, MPI_Comm mpiComm);

     /**
      * Performs all checkings that validate the sorting, i.e. it validades the results by performing the 3 follwing checks
      * \n - checks if total number of elements in the system are the same (before and after)
      * \n - checks if sum of all elements in the system are the same (before and after)
      * \n - checks if they are sorted - ie. max(rank)<min(rank+1)<max(rank+1)<min(rank+2)<... , for all ranks
      *
      * Usage: \n
      * - 1st: call getGlobalInformation(...) before the sorting to get preCheck variables
      * - 2nd: call getGlobalInformation(...) after the sorting to get postCheck variables
      * - 3rd: call checkSortingResults(...) with previous variables (to validate the first 2 conditions mentioned above)
      * and the coordinates array, the number of coordinates, and the sort dimension (to check the 3d condition)
      * @param preCheck_globalValuesCount (IN) count of all elements in the whole system - before sorting
      * @param preCheck_globalValuesSum (IN) sum of all elements in the whole system - before sorting
      * @param postCheck_globalValuesCount (IN) count of all elements in the whole system - after sorting
      * @param postCheck_globalValuesSum (IN) sum of all elements in the whole system - after sorting
      * @param coordinates (IN) coordinates array AFTER the sorting
      * @param segmentsCount (IN) number of segments in coordinates. Coordinates size = segments*3
      * @param sortDimension (IN) dimension to consider
      * @param mpiComm MPI communicator
      */
     static void checkSortingResults(
	long long preCheck_globalValuesCount,
	long double preCheck_globalValuesSum,
	long long postCheck_globalValuesCount,
	long double postCheck_globalValuesSum,
	t_sorting_data * coordinates,
	long long segmentsCount,
	int sortDimension,
	MPI_Comm mpiComm,
        bool outputWarnings = false
	);

     /**
      * Performs all checking necessary to validade a load balancing operation:\n
      * - Sum of all elements before and after load balancing are the same;\n
      * - Count of all elements before and after load balancing are the same;\n
      * - Gap between highest and lowest number of elements in the system is either 0 or 1;\n
      * @param preCheck_globalValuesCount (IN) count of all elements in the whole system - before sorting
      * @param preCheck_globalValuesSum (IN) sum of all elements in the whole system - before sorting
      * @param postCheck_globalValuesCount (IN) count of all elements in the whole system - after sorting
      * @param postCheck_globalValuesSum (IN) sum of all elements in the whole system - after sorting
      * @param mySegmentsCount (IN) number of elements in this cpu
      * @param mpiComm
      */
     static void checkLoadBalancingResults(
	long long preCheck_globalValuesCount,
	long double preCheck_globalValuesSum,
	long long postCheck_globalValuesCount,
	long double postCheck_globalValuesSum,
	long long myElementsCount,
	MPI_Comm mpiComm,
        bool outputWarnings = false
	);

     template <class T> static inline const T& max ( const T& a, const T& b ) { return (b<a)?a:b; }
     template <class T> static inline const T& min ( const T& a, const T& b ) { return (b>a)?a:b; }
};

#endif	/* DISTRIBUTEDMEMORYSORTER_H */

