/**
 * @file DistributedMemorySorter.cxx
 * @brief See header file 
 * @author bmagalha
 */

#include "DistributedMemorySorter.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include <limits>

#include "OutputHandler.h"

int __lessOrEqual3D_var_dimension = 0; //this is the var past to qsort. qsort_r supports this var, but BG has no qsort_r...
int lessOrEqual3D(const void * a, const void * b);

int lessOrEqual(const void * a, const void * b);

/******************************************************************/
/*               Sorting Validation functions                     */

/******************************************************************/

void DistributedMemorySorter::getGlobalInformation(
        long long segmentsCount,
        t_sorting_data * coordinates,
        long long *preCheck_globalValuesCount,
        long double *preCheck_globalValuesSum,
	MPI_Comm mpiComm) {
    long double preCheck_myValuesSum = 0;

    /* Calculates preCheck Variables*/
    for (long long i = 0; i < segmentsCount * 3; i++) preCheck_myValuesSum += (long double) coordinates[i];

    MPI_Allreduce(&preCheck_myValuesSum, preCheck_globalValuesSum, 1, MPI_LONG_DOUBLE, MPI_SUM, mpiComm);
    MPI_Allreduce(&segmentsCount, preCheck_globalValuesCount, 1, MPI_LONG_LONG, MPI_SUM, mpiComm);
}

void DistributedMemorySorter::checkLoadBalancingResults(
	long long preCheck_globalValuesCount,
	long double preCheck_globalValuesSum,
	long long postCheck_globalValuesCount,
	long double postCheck_globalValuesSum,
	long long mySegmentsCount,
	MPI_Comm mpiComm,
        bool outputWarnings
	)
{
    int mpiRank=-1;
    long long minCount=-1, maxCount=-1;
    MPI_Comm_rank(mpiComm, &mpiRank);

    //basic checksum
    if (outputWarnings && mpiRank == 0 && fabs(preCheck_globalValuesSum - postCheck_globalValuesSum) < 0.000001 )
        OutputHandler::showWarning("Load checksum balancing failed: "
            "the sum of all elements in the system before (%.15llf) and after the sorting/balancing"
            "(%.15llf) differ - possibly due to numeric overflow or system dependent bug? (FLAG11)", preCheck_globalValuesSum, postCheck_globalValuesSum);

    if (mpiRank == 0 && preCheck_globalValuesCount != postCheck_globalValuesCount)
        OutputHandler::throwMpiError("Load balancing failed: "
            "the number of elements in the system before (%lld) and after the sorting/balancing"
            "(%lld) differ - possibly due to numeric overflow or system dependent bug? (FLAG12)", preCheck_globalValuesCount, postCheck_globalValuesCount);

    //Root gets the minimum and maximum number of elements
    MPI_Reduce(&mySegmentsCount, &minCount, 1, MPI_LONG_LONG, MPI_MIN, 0, mpiComm);
    MPI_Reduce(&mySegmentsCount, &maxCount, 1, MPI_LONG_LONG, MPI_MAX, 0, mpiComm);

//  OutputHandler::show("- rank %d, segmentsCount=%lld", mpiRank, mySegmentsCount);
    if (mpiRank == 0 && maxCount - minCount >1 )
        OutputHandler::showWarning("Load balancing not accurate: the gap between the highest and lowest number of elements is %lld (expected 0 or 1). (FLAG10)", maxCount - minCount);
    fflush(stdout);
}

void DistributedMemorySorter::checkSortingResults(
        long long preCheck_globalValuesCount,
        long double preCheck_globalValuesSum,
        long long postCheck_globalValuesCount,
        long double postCheck_globalValuesSum,
        t_sorting_data * coordinates,
        long long segmentsCount,
        int sortDimension,
	MPI_Comm mpiComm,
        bool outputWarnings
        ) {
    int mpiRank = -1, mpiSize = -1;

    t_sorting_data *postCheck_globalMins = NULL, *postCheck_globalMaxs = NULL;

    MPI_Comm_rank(mpiComm, &mpiRank);
    MPI_Comm_size(mpiComm, &mpiSize);

    //basic checksum
    if (outputWarnings && mpiRank == 0 && fabs(preCheck_globalValuesSum - postCheck_globalValuesSum) < 0.000001 )
        OutputHandler::showWarning("Data checksum validation failed: "
            "the sum of all elements' coordinates in the system before (%.15llf) and after the sorting"
            "(%.15llf) differ - possibly due to numeric overflow or system dependent bug? (FLAG13)", preCheck_globalValuesSum, postCheck_globalValuesSum);

    if (mpiRank == 0 && preCheck_globalValuesCount != postCheck_globalValuesCount)
        OutputHandler::throwMpiError("Data validation failed: "
            "the number of elements in the system before (%lld) and after the sorting"
            "(%lld) differ - possibly due to numeric overflow or system dependent bug? (FLAG14)", preCheck_globalValuesCount, postCheck_globalValuesCount);

    if (mpiRank == 0) {
        postCheck_globalMins = new t_sorting_data[mpiSize];
        postCheck_globalMaxs = new t_sorting_data[mpiSize];
    }

    //All processes send to processor zero its min and max value (for checking)
    MPI_Gather(&(coordinates[0 + sortDimension]), 1, t_sorting_data_MPI, postCheck_globalMins, 1, t_sorting_data_MPI, 0, mpiComm);
    MPI_Gather(&(coordinates[(segmentsCount - 1)*3 + sortDimension]), 1, t_sorting_data_MPI, postCheck_globalMaxs, 1, t_sorting_data_MPI, 0, mpiComm);

    if (mpiRank == 0) {
        for (int i = 0; i < mpiSize; i++) {
            if (postCheck_globalMins[i] > postCheck_globalMaxs[i])
                OutputHandler::throwMpiError("Sorting failed: rank %d has min value %f and max %f. (FLAG15)", i, postCheck_globalMins[i], postCheck_globalMaxs[i]);

            if (i < mpiSize - 1 && postCheck_globalMaxs[i] > postCheck_globalMins[i + 1])
                OutputHandler::throwMpiError("Sorting failed: rank %d has max value %f and rank %d has min value %f. (FLAG16)", i, postCheck_globalMaxs[i], i + 1, postCheck_globalMins[i]);
        }
    }
    fflush(stdout);

    if (mpiRank == 0) {
        delete [] postCheck_globalMins; postCheck_globalMins=NULL;
        delete [] postCheck_globalMaxs; postCheck_globalMaxs=NULL;
    }
}

/******************************************************************/
/*               Work load balancing method                       */
/******************************************************************/

int DistributedMemorySorter::loadBalancing_MPI(long long * mySegmentsCount_ptr, t_sorting_data ** coordinates_ptr, MPI_Comm mpiComm, bool outputWarnings) {
    //Initialization of MPI variabls
    int mpiSize = -1, mpiRank = -1;
    MPI_Comm_size(mpiComm, &mpiSize);
    MPI_Comm_rank(mpiComm, &mpiRank);

    long long preCheck_globalValuesCount, postCheck_globalValuesCount;
    long double preCheck_globalValuesSum,  postCheck_globalValuesSum;
    getGlobalInformation(*mySegmentsCount_ptr, *coordinates_ptr, &preCheck_globalValuesCount, &preCheck_globalValuesSum, mpiComm);

    long long globalSegmentsCount = preCheck_globalValuesCount;

    //each CPU gets the current number of elements of all the others
    long long * allSegmentsCount = new long long[mpiSize];
    MPI_Allgather(mySegmentsCount_ptr, 1, MPI_LONG_LONG, allSegmentsCount, 1, MPI_LONG_LONG, mpiComm);

    //calculates the number of segments to be sent to each processor
    int * sentElemsSize = new int [mpiSize];
    for (int i = 0; i < mpiSize; i++) 
	sentElemsSize[i] = 0;

    //global index of beginning & end of my data (inclusive)
    long long myStartPos = 0, myEndPos=-1;
    for (int i = 1; i <= mpiRank; i++)
	myStartPos+= allSegmentsCount[i-1];
    myEndPos=myStartPos+allSegmentsCount[mpiRank]-1;

    delete [] allSegmentsCount; allSegmentsCount=NULL;
    
/*  for (int i=0; i<mpiSize; i++ )
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i!= mpiRank) continue;
            OutputHandler::show("mpiRank %d :: my %lld segments are between %lld and %lld",mpiRank, *mySegmentsCount_ptr, myStartPos, myEndPos);
            fflush(stdout);
        }
*/

    //number of final elements per cpu
    long long * elementsCountPerCpu = new long long [mpiSize];
    for (int i=0; i<mpiSize; i++)
        elementsCountPerCpu[i] = floor ((long double) globalSegmentsCount / mpiSize) + ( i < globalSegmentsCount % mpiSize ? 1 : 0);

    //global index of beginning & end of future slices (inclusive)
    long long indexStart = 0;
    for (int i = 0; i < mpiSize; i++)
    {
        long long indexEnd = i==(mpiSize-1) ? globalSegmentsCount-1 : indexStart + elementsCountPerCpu[i] -1; //inclusive

        //if (mpiRank==0) OutputHandler::show("slice %d indexStart=%lld indexEnd=%lld (size=%lld)", i, indexStart, indexEnd, indexEnd - indexStart +1 );

	sentElemsSize[i] = 0;

	//if the two intervals have regions in commmon (ie intersect)
	if (!(indexStart > myEndPos || indexEnd < myStartPos))
	{
	    //if they intersect, the data to be sent is the size of the intersection
	    //(+1 because beginning and end positions are inclusive
	    sentElemsSize[i] = (int) (min(indexEnd, myEndPos) - max(indexStart, myStartPos) +1);
	    sentElemsSize[i] *=3; //because we send coordinates

	    //OutputHandler::show("rank %d: slice %d intersect with size %d", mpiRank, i, sentElemsSize[i]/3);
	}
	//else
	//    OutputHandler::show("slice %i doesnt intersect", i);

	indexStart += elementsCountPerCpu[i];
    }
 /*      for (int i=0; i<mpiSize; i++ )
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i!= mpiRank) continue;
            printf("sentElemssize %d: ", mpiRank);
            for (int j=0; j<mpiSize; j++)
                printf(" %07d ", sentElemsSize[j]/3);
            printf("\n");
            fflush(stdout);
        }
*/
    delete [] elementsCountPerCpu; elementsCountPerCpu = NULL;

    //calculates the offset on his data, for the received data from the other ranks.
    int * sentElemsOffset = new int[mpiSize];
    sentElemsOffset[0] = 0;
    for (int i = 1; i < mpiSize; i++)
        sentElemsOffset[i] = sentElemsOffset[i - 1] + sentElemsSize[i - 1];

/*     for (int i=0; i<mpiSize; i++ )
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i!= mpiRank) continue;
            printf("sentElemsOffset %d: ", mpiRank);
            for (int j=0; j<mpiSize; j++)
                printf(" %07d ", sentElemsOffset[j]/3);
            printf("\n");
            fflush(stdout);
        }*/
    
    //calculate the ammounf of data received from each rank
    int * recvElemsSize = new int [mpiSize];
    MPI_Alltoall(sentElemsSize, 1, MPI_INT, recvElemsSize, 1, MPI_INT, mpiComm);

    //calculate the offset of the data received
    int * recvElemsOffset = new int [mpiSize];
    recvElemsOffset[0] = 0;
    for (int i = 1; i < mpiSize; i++)
        recvElemsOffset[i] = recvElemsOffset[i - 1] + recvElemsSize[i - 1];

/*    for (int i=0; i<mpiSize; i++ )
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i!= mpiRank) continue;
            printf("recvElemsOffset %d: ", mpiRank);
            for (int j=0; j<mpiSize; j++)
                printf(" %07d ", recvElemsOffset[j]/3);
            printf("\n");
            fflush(stdout);
        }*/

    //calculate final size for data received and allocates memory
    int myRecvElemsSize = recvElemsOffset[mpiSize - 1] + recvElemsSize[mpiSize - 1];
    t_sorting_data * recvCoordinates = new t_sorting_data[myRecvElemsSize];

    MPI_Alltoallv(*coordinates_ptr, sentElemsSize, sentElemsOffset, t_sorting_data_MPI, recvCoordinates, recvElemsSize, recvElemsOffset, t_sorting_data_MPI, mpiComm);

    delete [] (*coordinates_ptr);
    *coordinates_ptr = recvCoordinates;
    *mySegmentsCount_ptr = myRecvElemsSize / 3;

    getGlobalInformation(*mySegmentsCount_ptr, *coordinates_ptr, &postCheck_globalValuesCount, &postCheck_globalValuesSum, mpiComm);
    checkLoadBalancingResults(preCheck_globalValuesCount, preCheck_globalValuesSum, postCheck_globalValuesCount, postCheck_globalValuesSum, *mySegmentsCount_ptr, mpiComm, outputWarnings);


/*      for (int i=0; i<mpiSize; i++ )
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i!= mpiRank) continue;
            printf("myElems %d: ", mpiRank);
            for (int j=0; j<12; j+=3)
                printf(" (%f,%f,%f)", (*coordinates_ptr)[j], (*coordinates_ptr)[j+1], (*coordinates_ptr)[j+2]);
	    printf("\n");
            for (int j=(*mySegmentsCount_ptr-1)*3; j>(*mySegmentsCount_ptr-4)*3; j-=3)
                printf(" (%f,%f,%f)", (*coordinates_ptr)[j], (*coordinates_ptr)[j+1], (*coordinates_ptr)[j+2]);
            printf("\n");
            fflush(stdout);
          }*/

    delete [] recvElemsOffset; recvElemsOffset=NULL;
    delete [] recvElemsSize; recvElemsSize=NULL;
    delete [] sentElemsOffset; sentElemsOffset=NULL;
    delete [] sentElemsSize; sentElemsSize=NULL;
    
    return postCheck_globalValuesCount;

}


/******************************************************************/
/*    Odd-Even Transposition Sort - MPI implementation            */
/******************************************************************/

/**
 * function used by stdlib::qsort_r() to determine order of elements
 * a = (Segment *) segmentA
 * b = (Segment *) segmentB
 * d = (int *) dimension
 */

int lessOrEqual(const void * a, const void * b)
{
    t_sorting_data coordA = *(t_sorting_data*) a;
    t_sorting_data coordB = *(t_sorting_data*) b;
    if (coordA > coordB) return 1;
    if (coordB > coordA) return -1;
    return 0;
}

int lessOrEqual3D(const void * a, const void * b)
{
    t_sorting_data* coordsA = (t_sorting_data*) a;
    t_sorting_data* coordsB = (t_sorting_data*) b;

    int dim = __lessOrEqual3D_var_dimension;

    if (coordsA[dim] > coordsB[dim]) return 1;
    if (coordsB[dim] > coordsA[dim]) return -1;

//  int mpiRank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
//    if (mpiRank == 0)
//        printf("\t Quicksort %04d between %f and %f in dim %d = %f\n", mpiRank, coordsA[dim], coordsB[dim], dim, comparison);

    if (dim==2) return 0;

    __lessOrEqual3D_var_dimension++;
    int result = lessOrEqual3D(a, b); //sorts them by the next dimension
    __lessOrEqual3D_var_dimension--;
    return result;
}

void DistributedMemorySorter::test_OddEvenSort() {
    int mpiSize, mpiRank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    long long segmentCount = 1000000 + mpiRank*3;
    t_sorting_data * coords;
    coords = (t_sorting_data *) malloc(segmentCount * 3 * sizeof (t_sorting_data));
    int dim = 0;

    unsigned int iseed = (unsigned int) time(NULL);
    srand(iseed);
    
    for (int i = 0; i < segmentCount * 3; i += 3) {
        coords[i + 0] = (segmentCount*3 -i ) / (3.233 + mpiRank) * pow(-1,i);//((float) rand()+pow(-1, rand())) / 3;
        coords[i + 1] = ((float) rand()+pow(-1, rand())) / 2;
        coords[i + 2] = ((float) rand()+pow(-1, rand())) / 4;
    }
   
     for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != mpiRank) continue;
        //for (int j = 0; j < segmentCount * 3; j += 3)
        for (int j = 0; j < 3 * 3; j += 3)
            printf("A %04d %010d (%.4f, %.4f, %.4f)\n", mpiRank, j / 3, coords[j], coords[j + 1], coords[j + 2]);
        printf("\n");
        fflush(stdout);
    }

    oddEvenSort_MPI(segmentCount, coords, dim, MPI_COMM_WORLD);

    for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != mpiRank) continue;
        //for (int j = 0; j < segmentCount * 3; j += 3)
        for (int j = 0; j < 3 * 3; j += 3)
            printf("Z %04d %06d (%.4f, %.4f, %.4f)\n", mpiRank, j / 3, coords[j], coords[j + 1], coords[j + 2]);
        printf("\n");
        fflush(stdout);
    }

    //OutputHandler::throwMpiError("FINITO");
    MPI_Barrier(MPI_COMM_WORLD);
    if (!mpiRank) printf("DONE");
    MPI_Finalize();
    exit(1);
}

void DistributedMemorySorter::test_SampleSort() {
    int mpiSize, mpiRank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    long long segmentsCount = 1000000 + mpiRank;
    t_sorting_data * coords;
    coords = (t_sorting_data *) malloc(segmentsCount * 3 * sizeof (t_sorting_data));
    int dim = 0;

    unsigned int iseed = (unsigned int) time(NULL);
    srand(iseed);

    printf("A %04d OLD SEGMENTS count: %lld\n", mpiRank, segmentsCount);
    for (int i = 0; i < segmentsCount * 3; i += 3) {
        coords[i + 0] = (segmentsCount*3 -i ) / (3.233 + mpiRank) * pow(-1,i);
        coords[i + 1] = rand() % 5 - 5;
        coords[i + 2] = rand() % 2 - 2;
    }

/*    for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != mpiRank) continue;
        //for (int j = 0; j < segmentsCount * 3; j += 3)
        for (int j = 0; j < 3 * 3; j += 3)
            printf("A %04d %010d (%.4f, %.4f, %.4f)\n", mpiRank, j / 3, coords[j], coords[j + 1], coords[j + 2]);
        //printf("\n");
        fflush(stdout);
    }
*/
    sampleSort_MPI(&segmentsCount, &coords, dim, MPI_COMM_WORLD);
    printf("Z %04d NEW SEGMENTS count: %lld\n", mpiRank, segmentsCount);

/*  for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != mpiRank) continue;
        //for (int j = 0; j < segmentsCount * 3; j += 3)
        for (int j = 0; j < 3 * 3; j += 3)
            printf("Z %04d %010d (%.4f, %.4f, %.4f)\n", mpiRank, j / 3, coords[j], coords[j + 1], coords[j + 2]);
        //printf("\n");
        fflush(stdout);
    }
*/
    //OutputHandler::throwMpiError("FINITO");
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

/** NOTE:
 * From my calculations, 1GB RAM will limit the ammount of segments to about 50Million.
 * If we need to increase this probably we will have to use file system, and
 * the temp data structs in a file.
 */
long long DistributedMemorySorter::oddEvenSort_MPI(long long segmentsCount, t_sorting_data *coordinates, int sortDimension, MPI_Comm mpiComm, bool outputWarnings) {

    //Variables used for the pre and post check of the sorting (results validation)
    long double preCheck_globalValuesSum = 0, postCheck_globalValuesSum = 0;
    long long preCheck_globalValuesCount = 0, postCheck_globalValuesCount = 0;

    //OutputHandler::show("rank %d: Performing Odd Evern Sorting (MPI), dimmension %d, %lld segments.",_rank, sortDimension, segmentsCount);

    //Communication vars
    int mpiSize, mpiRank, oddRank, evenRank;
    MPI_Status status;

    //received data
    t_sorting_data *recvCoordinates;
    long long recvSegmentsCount_odd = 0, recvSegmentsCount_even = 0;
    long long maxSegmentsCount;

    /* Get communicator-related information */
    MPI_Comm_size(mpiComm, &mpiSize);
    MPI_Comm_rank(mpiComm, &mpiRank);

    if (segmentsCount == 0)
        OutputHandler::throwMpiError("Cant continue sorting operation : rank %d has no segments at all", mpiRank);

    getGlobalInformation(segmentsCount, coordinates, &preCheck_globalValuesCount, &preCheck_globalValuesSum, mpiComm);

    //Array that will hold data for compare-split operations
    t_sorting_data * tempCoordinates = (t_sorting_data*) malloc(segmentsCount * 3 * sizeof (t_sorting_data));
    if (tempCoordinates == NULL) OutputHandler::throwMpiError("Failed to allocate memmory for the slicing temporary datatypes. DistributedMemorySorter.cxx :: FLAG1");

    //allocates memory for temporary array (size is the max that can receive)
    MPI_Allreduce(&segmentsCount, &maxSegmentsCount, 1, MPI_LONG_LONG, MPI_MAX, mpiComm);

    recvCoordinates = (t_sorting_data *) malloc(maxSegmentsCount * 3 * sizeof (t_sorting_data));
    if (recvCoordinates == NULL) OutputHandler::throwMpiError("Failed to allocate memmory for the slicing temporary datatypes. DistributedMemorySorter.cxx :: FLAG2");

    //sorts the segments locally (BG1 doesnt support qsort_r);
    //qsort_r(coordinates, segmentsCount, sizeof(TYPE_COORDS_SORTING)*3, lessOrEqual, &sortDimension);
    __lessOrEqual3D_var_dimension = sortDimension;
    qsort(coordinates, segmentsCount, sizeof (t_sorting_data)*3, lessOrEqual3D);
    /*
       for (int i=0; i<mpiSize; i++ )
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i!= mpiRank) continue;
            printf("local quickSort %d: ", mpiRank);
            for (int j=0; j<segmentsCount*3; j+=3)
                printf("(%.1f, %.1f, %.1f) ", coordinates[j], coordinates[j+1], coordinates[j+2]);
            printf("\n");
            fflush(stdout);
        }
     */

    //determines the rank of the processors with which to communicate
    if (mpiRank % 2 == 0) {
        oddRank = mpiRank - 1;
        evenRank = mpiRank + 1;
    } else {
        oddRank = mpiRank + 1;
        evenRank = mpiRank - 1;
    }

    //sets the ranks of the boundary processors
    if ((oddRank == -1) || (oddRank == mpiSize)) oddRank = MPI_PROC_NULL;
    if ((evenRank == -1) || (evenRank == mpiSize)) evenRank = MPI_PROC_NULL;

    //gets the size of the data of the odd and even processors near by:
    MPI_Sendrecv(
            &segmentsCount, 1, MPI_LONG_LONG, oddRank, 1,
            &recvSegmentsCount_odd, 1, MPI_LONG_LONG, oddRank, 1,
            mpiComm, &status);

    MPI_Sendrecv(
            &segmentsCount, 1, MPI_LONG_LONG, evenRank, 2,
            &recvSegmentsCount_even, 1, MPI_LONG_LONG, evenRank, 2,
            mpiComm, &status);

    /* the main sorting loop */
    for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(mpiComm); //TODO REMOVE
        //TODO put safeguards on the mallocs in the sorting process
        if (i % 2 == 1) /* odd phase */ {
            //Sends/receives data itself
            MPI_Sendrecv(
                    coordinates, segmentsCount * 3, t_sorting_data_MPI, oddRank, i + 3,
                    recvCoordinates, recvSegmentsCount_odd * 3, t_sorting_data_MPI, oddRank, i + 3,
                    mpiComm, &status);

            //	    if (oddRank!=MPI_PROC_NULL)
            //	    	printf("Iter %d odd:: ME %d and %d\n", i, mpiRank, oddRank);
            /*
                        if (oddRank!=MPI_PROC_NULL)
                        {
                            printf("Iter %d  odd:: ME %d [size %lld] and %d [size %lld]:", i, mpiRank, segmentsCount, oddRank, recvSegmentsCount_odd);
                            for (int j=0; j<recvSegmentsCount_odd*3; j+=3)
                                printf("(%.1f, %.1f, %.1f) ", recvCoordinates[j], recvCoordinates[j+1], recvCoordinates[j+2]);
                            printf("\n");
                            fflush(stdout);
                        }
             */
            compareSplit(segmentsCount, coordinates, recvSegmentsCount_odd, recvCoordinates, tempCoordinates, sortDimension, mpiRank < status.MPI_SOURCE);
        } else /* even phase */ {
            MPI_Sendrecv(
                    coordinates, segmentsCount * 3, t_sorting_data_MPI, evenRank, i + 4,
                    recvCoordinates, recvSegmentsCount_even * 3, t_sorting_data_MPI, evenRank, i + 4,
                    mpiComm, &status);

            //if (oddRank!=MPI_PROC_NULL)
            //	printf("Iter %d odd:: ME %d and %d\n", i, mpiRank, evenRank);
            /*
                            if (evenRank!=MPI_PROC_NULL)
                            {
                                    printf("Iter %d even:: ME %d [size %lld] and %d [size %lld]:", i, mpiRank, segmentsCount, evenRank, recvSegmentsCount_even);
                                    for (int j=0; j<recvSegmentsCount_even*3; j+=3)
                                            printf("(%.1f, %.1f, %.1f) ", recvCoordinates[j], recvCoordinates[j+1], recvCoordinates[j+2]);
                                    printf("\n");
                                    fflush(stdout);
                            }
             */
            compareSplit(segmentsCount, coordinates, recvSegmentsCount_even, recvCoordinates, tempCoordinates, sortDimension, mpiRank < status.MPI_SOURCE);
        }
    }

    free(recvCoordinates);
    free(tempCoordinates);

    getGlobalInformation(segmentsCount, coordinates, &postCheck_globalValuesCount, &postCheck_globalValuesSum, mpiComm);
    checkSortingResults(preCheck_globalValuesCount, preCheck_globalValuesSum,
            postCheck_globalValuesCount, postCheck_globalValuesSum, coordinates,
            segmentsCount, sortDimension, mpiComm, outputWarnings);


    return postCheck_globalValuesCount;
}

/* Implementation of Compare and Split:
 * Looks at list coordinates of size segmentsCount*3
 * and list recvCoords of size recvSegmentsCount*3
 * and:
 * if keepSmall : puts the smallest elements into coordinates array
 * if !keepSmall: puts the greatest elements into coordinates array
 *
 * used tempCoords (size segmentsCount*3) as temporary data structure;
 */
void DistributedMemorySorter::compareSplit(
        long long segmentsCount,
        t_sorting_data *coordinates,
        long long recvSegmentsCount,
        t_sorting_data *recvCoords,
        t_sorting_data *tempCoords,
        int dim,
        int keepSmall) {

    //temp array becomes coordinates, and coordinates will be the result vector
    //    double * swapPtr = coordinates;
    //    coordinates = tempCoords;
    //    tempCoords = swapPtr;
    //TODO improve this with the swap pointer
    memcpy(tempCoords, coordinates, segmentsCount * 3 * sizeof (t_sorting_data));

    //performs maximum of O(n+m) operations:
    //as both lists are sorted, we go through both lists at same time and
    //compare the current position on each, for the next element to add
    if (keepSmall) { /* keep the n smaller elements */
        long long tempPos = 0, recvPos = 0;
        for (long long coordPos = 0; coordPos < segmentsCount * 3; coordPos += 3) {
            //if we reached the limit of the recv array
            //or if current element of tempPos < current element in recvCoord
            if (recvPos == recvSegmentsCount * 3 || tempCoords[tempPos + dim] < recvCoords[recvPos + dim]) {
                memcpy(&(coordinates[coordPos]), &(tempCoords[tempPos]), 3 * sizeof (t_sorting_data));
                tempPos += 3;
            }                //we dont need the "if reached limit of coords/temp" condition
                //as it is implicit on the for loop (temp has same size as coords)
            else {
                memcpy(&(coordinates[coordPos]), &(recvCoords[recvPos]), 3 * sizeof (t_sorting_data));
                recvPos += 3;
            }
        }
    } else {
        long long tempPos = (segmentsCount - 1)*3, recvPos = (recvSegmentsCount - 1)*3;
        for (long long coordPos = (segmentsCount - 1)*3; coordPos >= 0; coordPos -= 3) {
            if (recvPos < 0 || tempCoords[tempPos + dim] > recvCoords[recvPos + dim]) {
                memcpy(&(coordinates[coordPos]), &(tempCoords[tempPos]), 3 * sizeof (t_sorting_data));
                tempPos -= 3;
            } else {
                memcpy(&(coordinates[coordPos]), &(recvCoords[recvPos]), 3 * sizeof (t_sorting_data));
                recvPos -= 3;
            }
        }
    }
    /*      int mpiRank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            for (int i=0; i<size; i++)
            {
                if (i==mpiRank)
                {
                    printf("CompareSplit %d (keepsmall=%d):", mpiRank, keepSmall);
                    for (int j=0; j<segmentsCount*3; j+=3)
                            printf("(%.1f, %.1f, %.1f) ", coordinates[j], coordinates[j+1], coordinates[j+2]);
                    printf("\n");
                    fflush(stdout);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
     */
}

/******************************************************************/
/*                 Sample Sort - MPI implementation               */

/******************************************************************/



long long DistributedMemorySorter::sampleSort_MPI(long long * segmentsCount_ptr, t_sorting_data **coordinates_ptr, int sortDimension, MPI_Comm mpiComm)
{
    //Variables used for the pre and post check of the sorting (results validation)
    long double preCheck_globalValuesSum = 0, postCheck_globalValuesSum = 0;
    long long preCheck_globalValuesCount = 0, postCheck_globalValuesCount = 0;

    //coordinates
    t_sorting_data *coordinates = *coordinates_ptr;

    int bucketIndex, mpiSize, mpiRank;
    long long i;
    t_sorting_data *sortedData, *samples, *recvSamples = NULL;
    //this would ideally be long long, but is int as MPI_AllToAllv takes int* as parameters....
    int *bucketElemsSize, *recvElemsSize, *bucketOffsets, *recvElemsOffset;
    long long myRecvElemsSize, myRecvElemsCount;

    MPI_Comm_size(mpiComm, &mpiSize);
    MPI_Comm_rank(mpiComm, &mpiRank);

    if (*segmentsCount_ptr == 0)
        OutputHandler::throwMpiError("Cant continue sorting operation : rank %d has no segments at all", mpiRank);

    getGlobalInformation(*segmentsCount_ptr, coordinates, &preCheck_globalValuesCount, &preCheck_globalValuesSum, mpiComm);

    /* sort local array */
    __lessOrEqual3D_var_dimension = sortDimension;
    qsort(coordinates, *segmentsCount_ptr, sizeof (t_sorting_data)*3, lessOrEqual3D);

    /* allocate memory for the arrays that will store the samples */
    samples = (t_sorting_data *) malloc(mpiSize * sizeof (t_sorting_data));

    /* select local p-1 equally spaced elements (samples only for the dimension we consider)*/
    for (i = 1; i <= mpiSize; i++) {
        unsigned int pos = (int) ((double) (*segmentsCount_ptr - 1) * i/mpiSize) *3 + sortDimension;
        samples[i - 1] = coordinates[pos];
    }

    //bug fix: as we couldnt allocate memory for N*N samples in only 1 CPU,
    //this method prevents that by using groups for sampling, i.e. a group of
    //CPU does sampling to a group master, and then group masters do sampling,
    //and send it to root 0 (master of all masters).
    if (mpiSize > SUB_SAMPLING_MINIMUM_CPUS)
    {
	if (mpiRank==0)
		OutputHandler::show("Using sampling by sub-groups, as MPI size is greater than %d...", SUB_SAMPLING_MINIMUM_CPUS);

        MPI_Comm newComm;
        int newMpiSize = -1;
        int newMpiRank = mpiRank % SUB_SAMPLING_GROUP_SIZE; //[0..255];
        int newMpiGroup = floor((float) mpiRank / SUB_SAMPLING_GROUP_SIZE); //[0..63] for ranks [0..1638]
        int numberOfGroups = ceil((float)mpiSize / SUB_SAMPLING_GROUP_SIZE);

        MPI_Comm_split(mpiComm, newMpiGroup, newMpiRank, &newComm);
        MPI_Comm_size(newComm, &newMpiSize);

	//printf("mpiRank %d, number groups = %d, newMpiGroup = %d, newMpiRank = %d, newMpiSize = %d;\n", mpiRank, numberOfGroups, newMpiGroup, newMpiRank, newMpiSize);

        if (newMpiRank == 0) {
            //in each group, receives from max 256 cpus, an ammount of mpiSize-1 samples
            recvSamples = (t_sorting_data *) malloc(newMpiSize * (mpiSize - 1) * sizeof (t_sorting_data));
            if (recvSamples == NULL) OutputHandler::throwMpiError("Failed to allocate memmory for the slicing temporary datatypes. DistributedMemorySorter.cxx :: FLAG3.5");
        }

        /* gathers the samples into the master of his group */
        MPI_Gather(samples, mpiSize - 1, t_sorting_data_MPI, recvSamples, mpiSize - 1, t_sorting_data_MPI, 0, newComm);

        /* masters select the samples among all received samples, in order to broadcast them */
        if (newMpiRank == 0) {

	    qsort(recvSamples, newMpiSize * (mpiSize - 1), sizeof (t_sorting_data), lessOrEqual);

            for (i = 1; i < mpiSize; i++)
                samples[i - 1] = recvSamples[i * newMpiSize - 1];
            samples[mpiSize - 1] = std::numeric_limits<t_sorting_data>::max();

            free(recvSamples);
	    recvSamples=NULL;

            /*
	    for (int j = 0; j < mpiSize; j++)
	    {
               	printf("sub sample rank %d, j=%d (%f)\n", mpiRank, j, samples[j]);
            	fflush(stdout);
	    }*/

        }
        MPI_Comm_free(&newComm);

        //sub sampling is now complete, so masters of sub groups will do the same
        //step with master of mpiComm group...
        //group number will be the rank in this group of masters
        MPI_Comm_split(mpiComm, newMpiRank, newMpiGroup, &newComm);

        if (newMpiRank == 0) //if it was a master before...
        {
            //Rank 0 will collect all samples from previous masters, and take samples of their samples!
            if (mpiRank == 0)
	    {
                recvSamples = (t_sorting_data *) malloc(numberOfGroups * (mpiSize - 1) * sizeof (t_sorting_data));
                if (recvSamples == NULL) OutputHandler::throwMpiError("Failed to allocate memmory for the sorting temporary datatypes. DistributedMemorySorter.cxx :: FLAG3.8");
	    }

            MPI_Gather(samples, mpiSize - 1, t_sorting_data_MPI, recvSamples, mpiSize - 1, t_sorting_data_MPI, 0, newComm);

	    if (mpiRank ==0)
	    {
       	    	//master takes samples of received samples...
	        qsort(recvSamples, numberOfGroups * (mpiSize - 1), sizeof (t_sorting_data), lessOrEqual);

       	    	for (i = 1; i < mpiSize; i++)
	            samples[i - 1] = recvSamples[i * numberOfGroups - 1];
	        samples[mpiSize - 1] = std::numeric_limits<t_sorting_data>::max();

		/*
            	for (int i = 0; i<mpiSize; i++)
	    	{
	    	    printf("FINAL SAMPLE %d == %f\n", i, samples[i]);
            		fflush(stdout);
	    	}*/
		free(recvSamples);
		recvSamples=NULL;
	    }
        }
        MPI_Comm_free(&newComm);
    }
    else //basic version: simple and fast
    {
	if (mpiRank == 0) {
            recvSamples = (t_sorting_data *) malloc(mpiSize * (mpiSize - 1) * sizeof (t_sorting_data));
            if (recvSamples == NULL)
		OutputHandler::throwMpiError("Failed to allocate memmory for the slicing temporary datatypes. DistributedMemorySorter.cxx :: FLAG3");
	}

        /* gather the samples into the master */
        MPI_Gather(samples, mpiSize - 1, t_sorting_data_MPI, recvSamples, mpiSize - 1, t_sorting_data_MPI, 0, mpiComm);

        /* master selects the samples among all received samples, in order to broadcast them */
        if (mpiRank == 0) 
	{
        /* for (int j = 0; j < mpiSize * (mpiSize - 1); j++)
              printf("recvSample before %04d %06d (%.1f)\n", mpiRank, j, recvSamples[j]);
           printf("\n");
       */
            qsort(recvSamples, mpiSize * (mpiSize - 1), sizeof (t_sorting_data), lessOrEqual);

         /* for (int j = 0; j < mpiSize * (mpiSize - 1); j++)
                printf("recvSample quicksort %04d %06d (%.1f)\n", mpiRank, j, recvSamples[j]);
            printf("\n");
         */
            for (i = 1; i < mpiSize; i++)
                samples[i - 1] = recvSamples[i * mpiSize - 1];
            samples[mpiSize - 1] = std::numeric_limits<t_sorting_data>::max();

	    free(recvSamples);
	    recvSamples=NULL;
	}
    }

    /* now the samples array contains the global samples */
    MPI_Bcast(samples, mpiSize, t_sorting_data_MPI, 0, mpiComm);

    /* compute the number of elements that belong to each bucket */
    bucketElemsSize = (int *) malloc(mpiSize * sizeof (int));
    for (i = 0; i < mpiSize; i++) bucketElemsSize[i] = 0;

    bucketIndex = 0;
    while (coordinates[0 + sortDimension] >= samples[bucketIndex]) bucketIndex++;
    //ie gets the index of the bucket that contains our 1st coordinate

    for (i = 0; i < (*segmentsCount_ptr)*3; i += 3)
        if (coordinates[i + sortDimension] < samples[bucketIndex])
            bucketElemsSize[bucketIndex] += 3;
        else {
            while (coordinates[i + sortDimension] >= samples[bucketIndex]) bucketIndex++;
            bucketElemsSize[bucketIndex] += 3;
        }
/*
    for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != mpiRank) continue;
        for (int j = 0; j < mpiSize; j++)
            printf("bucket elements size %04d %06d (%.d), %d coordinates\n", mpiRank, j, bucketElemsSize[j], bucketElemsSize[j] / 3);
        printf("\n");
        fflush(stdout);
    }
*/
    /* determine the starting location of each bucket's elements in the data array */
    bucketOffsets = (int *) malloc(mpiSize * sizeof (int));
    bucketOffsets[0] = 0;
    for (i = 1; i < mpiSize; i++)
        bucketOffsets[i] = bucketOffsets[i - 1] + bucketElemsSize[i - 1];
    // *3 in order to include the XYZ gap

/*
    for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != mpiRank) continue;
        for (int j = 0; j < mpiSize; j++)
            printf("bucket offset %04d %06d (%.d)\n", mpiRank, j, bucketOffsets[j]);
        printf("\n");
        fflush(stdout);
    }
*/
    /* Perform an all2all communication to inform the corresponding processes */
    /* of the number of elements they are going to receive. */
    /* This information is stored in bucketElemsCount array */
    recvElemsSize = (int *) malloc(mpiSize * sizeof (int));
    MPI_Alltoall(bucketElemsSize, 1, MPI_INT, recvElemsSize, 1, MPI_INT, mpiComm);

/*    for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != mpiRank) continue;
        for (int j = 0; j < mpiSize; j++)
            printf("recv size %04d %06d (%d), %d coordinates\n", mpiRank, j, recvElemsSize[j], recvElemsSize[j] / 3);
        printf("\n");
        fflush(stdout);
    }*/

    /* Based on recvElemsCount determines where in the local array the data from each processor */
    /* will be stored. This array will store the received elements as well as the final */
    /* sorted sequence.*/
    recvElemsOffset = (int *) malloc(mpiSize * sizeof (int));
    recvElemsOffset[0] = 0;
    for (i = 1; i < mpiSize; i++)
        recvElemsOffset[i] = recvElemsOffset[i - 1] + recvElemsSize[i - 1];

/*    for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != mpiRank) continue;
        for (int j = 0; j < mpiSize; j++)
            printf("recv elems offset %04d %06d (%d)\n", mpiRank, j, recvElemsOffset[j]);
        printf("\n");
        fflush(stdout);
    }*/

    /* how many elements I will get */
    myRecvElemsSize = recvElemsOffset[mpiSize - 1] + recvElemsSize[mpiSize - 1];
    myRecvElemsCount = myRecvElemsSize / 3;
    sortedData = (t_sorting_data *) malloc(myRecvElemsSize * sizeof (t_sorting_data));
    if (sortedData == NULL) OutputHandler::throwMpiError("Failed to allocate memmory for the slicing temporary datatypes. DistributedMemorySorter.cxx :: FLAG9");

/*    for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != mpiRank) continue;
        printf("sortedData %04d Ill receive size %lld doubles (%lld coordinates)\n", mpiRank, myRecvElemsSize, myRecvElemsCount);
        fflush(stdout);
    }
*/
    /* Each process sends and receives the corresponding elements, using the MPI__Alltoallv */
    /* operation. The arrays bucketElemsCount and bucketOffsets are used to specify the number of elements */
    /* to be sent and where these elements are stored, respectively. The arrays recvElemsCount */
    /* and recvElemsOffset are used to specify the number of elements to be received, and where these */
    /* elements will be stored, respectively. */
    MPI_Alltoallv(coordinates, bucketElemsSize, bucketOffsets, t_sorting_data_MPI, sortedData, recvElemsSize, recvElemsOffset, t_sorting_data_MPI, mpiComm);


/*    for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != mpiRank) continue;
        for (int j = 0; j < myRecvElemsCount * 3; j += 3)
            printf("recv b4 quick sort %04d %06d (%.1f, %.1f, %.1f)\n", mpiRank, j / 3, sortedData[j], sortedData[j + 1], sortedData[j + 2]);
        printf("\n");
        fflush(stdout);
    }
*/

    /* perform the final local sort */
    __lessOrEqual3D_var_dimension = sortDimension;
    qsort(sortedData, myRecvElemsCount, sizeof (t_sorting_data)*3, lessOrEqual3D);

/*  for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != mpiRank) continue;
        for (int j = 0; j < myRecvElemsCount * 3; j += 3)
            printf("recv after quick sort %04d %06d (%.1f, %.1f, %.1f)\n", mpiRank, j / 3, sortedData[j], sortedData[j + 1], sortedData[j + 2]);
        printf("\n");
        fflush(stdout);
    }*/

    /* cleanup */
    free(samples);
    free(bucketElemsSize);
    free(bucketOffsets);
    free(recvElemsSize);
    free(recvElemsOffset);
//    free(coordinates);

    *coordinates_ptr = sortedData;
    *segmentsCount_ptr = myRecvElemsCount;

/*  for (int i = 0; i < mpiSize; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != mpiRank) continue;
        for (int j = 0; j < myRecvElemsCount * 3; j += 3)
            printf("before function exit %04d %06d (%.1f, %.1f, %.1f)\n", mpiRank, j / 3, (*coordinates_ptr)[j], (*coordinates_ptr)[j + 1], (*coordinates_ptr)[j + 2]);
        printf("\n");
        fflush(stdout);
    }*/

    /* Validates sorting*/
    getGlobalInformation(*segmentsCount_ptr, *coordinates_ptr, &postCheck_globalValuesCount, &postCheck_globalValuesSum, mpiComm);
    checkSortingResults(preCheck_globalValuesCount, preCheck_globalValuesSum,
            postCheck_globalValuesCount, postCheck_globalValuesSum,
            *coordinates_ptr, *segmentsCount_ptr, sortDimension, mpiComm);

    return postCheck_globalValuesCount;
}

