/**
 * @file DistributedMemorySorter.cxx
 * @brief 
 * @author bmagalha
 * @date 2012-07-18
 * @remark Copyright © BBP/EPFL 2005-2011; All rights reserved. Do not distribute without further notice.
 */

#include "DistributedMatrixOperations.h"

//#define OUTPUT_DEBUG_INFO
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include <limits>
#include <vector>
#include <set>

#include "OutputHandler.h"

//Testing purposes only

int generateRandomNumber (int min, int max)
{    return rand() % (max-min) + min; }

int main_(int argv, char** argc)
{
    MPI_Init(&argv, &argc);
    
    //MPI variables
    int mpiSize=-1, mpiRank=-1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    //TEST CASE 2
    srand (time(NULL)/(mpiRank+1));
    
    unsigned int numberOfRows=0;
    unsigned int * endRowCpus;
    unsigned int * colsPerRow;
    unsigned int * colsIndex;
    unsigned int * cellsPerCol;
    int * cells;
    
    unsigned long long colId=0, cellId=0;
 
    (void) colId; (void) cellId; //clear Unused Var warning;
    
    endRowCpus = new unsigned int[mpiSize];

    ///numberOfRows
    //numberOfRows = generateRandomNumber (10000, 100000); //1M and 10M;
    numberOfRows = generateRandomNumber (3, 4); //1M and 10M;
   
    //endRowCpus
    unsigned int * allRowsCount = new unsigned int[mpiSize];
    MPI_Allgather(&numberOfRows, 1, MPI_UNSIGNED, allRowsCount, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
    for (int i=0; i<mpiSize; i++)
	endRowCpus[i] = i==0 ? allRowsCount[i]-1 : allRowsCount[i] + endRowCpus[i-1];

    unsigned int totalRowsCount = 0;
    MPI_Allreduce ( &numberOfRows, &totalRowsCount, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

    //colsPerRow
    colsPerRow = new unsigned int [numberOfRows];

    long long numberOfColumns=0, totalColumnsCount=0;;
    for (unsigned int i=0; i<numberOfRows; i++)
    {
	//colsPerRow[i] = generateRandomNumber (100, 1000);
	colsPerRow[i] = 2;
        numberOfColumns += colsPerRow[i];
    }

    long long myCellsCount=0, totalCellsCount = 0;

    colsIndex = new unsigned int[numberOfColumns];
    cellsPerCol = new unsigned int [numberOfColumns];
    colId=0;
    for (unsigned int i=0; i<numberOfRows; i++)
    {
        for (unsigned int j=0; j<colsPerRow[i]; j++)
        {
            colsIndex[colId]= totalRowsCount*j/colsPerRow[i];
	    assert(colsIndex[colId]<totalRowsCount);
            cellsPerCol[colId]= 1;
            //cellsPerCol[colId]= generateRandomNumber(2,20);
            myCellsCount+=cellsPerCol[colId];
            colId++;
        }
    }

    MPI_Allreduce ( &myCellsCount, &totalCellsCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce ( &numberOfColumns, &totalColumnsCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (mpiRank==0) printf ("TOTAL COLUMNS COUNT %lld\n", totalColumnsCount);
    if (mpiRank==0) printf ("TOTAL CELLS COUNT %lld\n", totalCellsCount);

    cells = new int[myCellsCount];
    for (unsigned int i=0; i<myCellsCount; i++)
        cells[i]=generateRandomNumber(0,8);

/*    //TEST CASE 1
 
        unsigned int _endRowCpus[4] = {3,8,11,16};
        endRowCpus = _endRowCpus;

      if (mpiRank % 2 == 0)
      {
        numberOfRows=3;
        unsigned int _colsPerRow[3] = {4,4,0};
        unsigned int _colsIndex[8] =  {0, 3, 9, 13, 3, 8, 12, 15};
        unsigned int _cellsPerCol[8] = {1,4,1,1,3,1,4,1};
        int _cells[16] = {0,1,2,3,4,2,3,4,5,6,5,6,7,8,9,7};
    
        colsPerRow = _colsPerRow;
        colsIndex = _colsIndex;
        cellsPerCol = _cellsPerCol;
        cells = _cells;
      }
      else
      {
        numberOfRows=5;
        unsigned int _colsPerRow[5] = {4,3,3,0,1};
        unsigned int _colsIndex[11] =  {1, 4, 10, 13, 1, 5, 9, 3, 8, 12, 15};
        unsigned int _cellsPerCol[11] = {1,4,1,1,3,1,4,1,1,2,3};
        int _cells[22] = {0,1,2,3,4,2,3,4,5,6,5,6,7,8,9,7,8,9,0,0,1,2};

        colsPerRow = _colsPerRow;
        colsIndex = _colsIndex;
        cellsPerCol = _cellsPerCol;
        cells = _cells;
     }
     */

#ifdef OUTPUT_DEBUG_INFO
    colId=0, cellId=0;
    for (int i=0; i<mpiSize; i++)
    {
        if (mpiRank==i)
        {
           fprintf(stderr,"RANK %d\n", mpiRank);
    for (unsigned int r=0; r<numberOfRows; r++)
    {
        fprintf(stderr,"ROW %d (%d cols) ::", r+myFirstRowIndex, colsPerRow[r]);
        for (unsigned int c=0; c<colsPerRow[r]; c++)
        {
            fprintf(stderr,"[id %d, %d cell] ", colsIndex[colId], cellsPerCol[colId]);
            for (unsigned int e=0; e<cellsPerCol[colId]; e++)
                fprintf(stderr, "%d ", cells[cellId++]);
            colId++;
            fprintf(stderr,", ");
        }
        fprintf(stderr,"\n");
    }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        fprintf (stderr,"\n");
    }
    if (mpiRank==0) fprintf(stderr,"Start\n");
#endif

    DistributedMatrixOperations<int>::sparseMatrixTranspose(endRowCpus, colsPerRow, colsIndex, cellsPerCol, cells, MPI_COMM_WORLD);

#ifdef OUTPUT_DEBUG_INFO
    colId=0, cellId=0;
    for (int i=0; i<mpiSize; i++)
    {
        if (mpiRank==i)
        {
           fprintf(stderr,"RANK %d\n", mpiRank);
    for (unsigned int r=0; r<numberOfRows; r++)
    {
        fprintf(stderr,"ROW %d (%d cols) ::", r+myFirstRowIndex, colsPerRow[r]);
        for (unsigned int c=0; c<colsPerRow[r]; c++)
        {
            fprintf(stderr,"[id %d, %d cell] ", colsIndex[colId], cellsPerCol[colId]);
            for (unsigned int e=0; e<cellsPerCol[colId]; e++)
                fprintf(stderr, "%d ", cells[cellId++]);
            colId++;
            fprintf(stderr,", ");
        }   
        fprintf(stderr,"\n");
    }     
        }
        MPI_Barrier(MPI_COMM_WORLD);
        fprintf (stderr,"\n");
    }
    if (mpiRank==0) fprintf(stderr,"Finish\n");
#endif
    
    return 0;
}
