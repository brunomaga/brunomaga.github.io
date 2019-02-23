/**
 * @file DistributedMemorySorter.h
 * @brief Class responsible for algorithms used on distributed memmory
 * (eg: load balancing, sorters, etc)
 * @author bmagalha
 * @date 2012-07-18
 * @remark Copyright © BBP/EPFL 2005-2011; All rights reserved. Do not distribute without further notice.
 */

#ifndef BBP_DISTRIBUTEDMATRIXOPERATIONS_H
#define	BBP_DISTRIBUTEDMATRIXOPERATIONS_H

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <cstdlib>

template<class T>
class DistributedMatrixOperations
{
   
private:
  struct QSortWrapper
  {
    void * cells;
    int cpu; 
    unsigned int row; 
    unsigned int col; 
    unsigned int cellsCount;
  };

  static int qsortCompare (const void * _a, const void * _b)
  {
    QSortWrapper * a = (QSortWrapper*) _a;
    QSortWrapper * b = (QSortWrapper*) _b;
    if (a->cpu != b->cpu) return ( a->cpu - b->cpu );
    if (a->row != b->row) return ( a->row - b->row );
    assert(a->col != b->col);
    return ( a->col - b->col );
  }

  static void sortCellsByCpuRowColumn(unsigned int myColumnsCount, unsigned int totalCellsCount, QSortWrapper * qsortWrapper, T*& cells)
  {
    qsort(qsortWrapper,myColumnsCount, sizeof(QSortWrapper), qsortCompare);
 
    //now that the qsort ir ordered correctly, we will shuffle the elements 
    //of cells to follow the same order (by looking at the pointers of the structure data)
    unsigned long long cellId=0;
    T * cells_temp = new T[totalCellsCount];
    for (unsigned int c=0; c<myColumnsCount; c++)
    {
        //copy column's cells
        void * firstCell = qsortWrapper[c].cells;
    	memcpy(&(cells_temp[cellId]), firstCell, sizeof(T)*qsortWrapper[c].cellsCount); 
        
        cellId += qsortWrapper[c].cellsCount;
    }

    delete [] cells; cells=NULL;
    cells = cells_temp;
  }  
  
public:

  static void sparseMatrixTranspose(unsigned int * endRowIndexPerCpu, unsigned int *& numberOfColumnsPerRow, unsigned int *& columnsIndex, unsigned int *& numberOfCellsPerColumn, T *& cells, MPI_Comm mpiComm)
  {
    //MPI variables
    int mpiSize=-1, mpiRank=-1;;
    MPI_Comm_rank(mpiComm, &mpiRank);
    MPI_Comm_size(mpiComm, &mpiSize);

    unsigned int myFirstRowIndex = mpiRank==0 ? 0 : endRowIndexPerCpu[mpiRank-1]+1;
    unsigned int myRowsCount = endRowIndexPerCpu[mpiRank] - (mpiRank==0 ? -1 : endRowIndexPerCpu[mpiRank-1]);
    
    //get total number of cells and columns
    unsigned int myColumnsCount=0;
    for (unsigned int r=0; r<myRowsCount; r++)
	myColumnsCount+=numberOfColumnsPerRow[r];

    //zero step: replace a matrix into 'mpiSize' sub-matrices transposed
    QSortWrapper * qsortWrapper = new QSortWrapper[myColumnsCount];
    
    //we will divide this L lines in 'mpiSize' matrices, and transpose each
    //one of them. For that we perform a sorting based on destinationCPU, then
    //column Id (since its the row for transposed matrix), and then on row
    //(transposed matrix column)
 
    //keeps the map of which cpu takes each neuron
    unsigned int totalRowsCount=0;
    MPI_Allreduce( &myRowsCount, &totalRowsCount, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    
    unsigned int * cpuPerRowIndex = new unsigned int[totalRowsCount];
    unsigned int cpuId=0;
    for (unsigned int n=0; n< (unsigned int) totalRowsCount; n++)
    {
	assert(cpuId < (unsigned int) mpiSize);
	
   	cpuPerRowIndex[n] = cpuId;
	while (endRowIndexPerCpu[cpuId] <= n) cpuId++;
    }
    
    //converting the Sparse matrix arrays into a qsort wrapper structure
    unsigned int cellId=0, colId=0, totalCellsCount=0;
    for (unsigned int rowId=0; rowId<myRowsCount; rowId++)
    {
	for (unsigned int c=0; c<numberOfColumnsPerRow[rowId]; c++)
	{
	  qsortWrapper[colId].cells = &(cells[cellId]);
          qsortWrapper[colId].cellsCount = numberOfCellsPerColumn[colId];

	  totalCellsCount += numberOfCellsPerColumn[colId];
	  
          //we intentionally swapped row and column, to force
          //the qsort to sort them like this (therefor transposing it)
	  unsigned int & columnId = columnsIndex[colId]; 
	  qsortWrapper[colId].row = columnId; 
	  qsortWrapper[colId].col = myFirstRowIndex + rowId;
          qsortWrapper[colId].cpu = cpuPerRowIndex[columnId]; 

	  //increment cell and column Id
  	  cellId += numberOfCellsPerColumn[colId];
          colId++;
	}
    }

    delete [] cpuPerRowIndex; cpuPerRowIndex=NULL;
    delete [] columnsIndex; columnsIndex=NULL;
    delete [] numberOfCellsPerColumn; numberOfCellsPerColumn=NULL;
    delete [] numberOfColumnsPerRow; numberOfColumnsPerRow=NULL;

    sortCellsByCpuRowColumn(myColumnsCount, totalCellsCount, qsortWrapper, cells);
    
   //2nd step: now we will send the respective table info to each CPU (qsortStruct)
    int * sentElemsSize = new int[mpiSize];
    int * recvElemsSize = new int [mpiSize];
    int * sentElemsOffset = new int[mpiSize];
    int * recvElemsOffset = new int[mpiSize];

    for (int cpu=0; cpu<mpiSize; cpu++)
	sentElemsSize[cpu]=0; 

    for (unsigned int c=0; c<myColumnsCount; c++)
    {
	int & cpu = qsortWrapper[c].cpu;
        sentElemsSize[cpu] += sizeof(QSortWrapper);
    }

    //share the amount of data to be received by each other cpu
    MPI_Alltoall(sentElemsSize, 1, MPI_INT, recvElemsSize, 1, MPI_INT, mpiComm);

    //calculate offset for the data sent/received
    for (int cpu=0; cpu<mpiSize; cpu++)
	sentElemsOffset[cpu] = cpu==0 ? 0 : sentElemsOffset[cpu-1] + sentElemsSize[cpu-1];  

    for (int cpu=0; cpu < mpiSize; cpu++)
        recvElemsOffset[cpu] = cpu==0 ? 0 : recvElemsOffset[cpu-1] + recvElemsSize[cpu - 1];
    
    //calculate total data size to be received
    unsigned int myColumnsCount_T = (recvElemsOffset[mpiSize - 1] + recvElemsSize[mpiSize - 1])/sizeof(QSortWrapper);
    QSortWrapper * qsortWrapper_T = new QSortWrapper[myColumnsCount_T];

    //send around the table structure of the elements to be received
    MPI_Alltoallv(qsortWrapper, sentElemsSize, sentElemsOffset, MPI_BYTE, qsortWrapper_T, recvElemsSize, recvElemsOffset, MPI_BYTE, mpiComm);
    
    //3rd step: now we will send the respective cells to each CPU 
    for (int cpu=0; cpu<mpiSize; cpu++)
	sentElemsSize[cpu]=0;
    
    for (unsigned int c=0; c<myColumnsCount; c++)
    {
	int & cpu = qsortWrapper[c].cpu;
        sentElemsSize[cpu] += qsortWrapper[c].cellsCount * sizeof(T);
    }

    delete [] qsortWrapper; qsortWrapper=NULL;

    //share the amount of data to be received by each other cpu
    MPI_Alltoall(sentElemsSize, 1, MPI_INT, recvElemsSize, 1, MPI_INT, mpiComm);

    //calculate offset for the data sent/received
    for (int cpu=0; cpu<mpiSize; cpu++)
	sentElemsOffset[cpu] = cpu==0 ? 0 : sentElemsOffset[cpu-1] + sentElemsSize[cpu-1];  

    for (int cpu=0; cpu<mpiSize; cpu++)
        recvElemsOffset[cpu] = cpu==0 ? 0 : recvElemsOffset[cpu-1] + recvElemsSize[cpu - 1];
    
    //receives data
    unsigned int totalCellsCount_T = (recvElemsOffset[mpiSize-1] + recvElemsSize[mpiSize-1])/sizeof(T);
    T * cells_T = new T[totalCellsCount_T];
    MPI_Alltoallv(cells, sentElemsSize, sentElemsOffset, MPI_BYTE, cells_T, recvElemsSize, recvElemsOffset, MPI_BYTE, mpiComm);

    delete [] cells; cells=NULL;
    delete [] sentElemsSize; sentElemsSize = NULL;
    delete [] recvElemsSize; recvElemsSize = NULL;
    delete [] sentElemsOffset; sentElemsOffset=NULL;
    delete [] recvElemsOffset; recvElemsOffset=NULL;

    //4th we will reconvert the multiple transposed matrices, into a single matrix
    cells = cells_T;
    qsortWrapper = qsortWrapper_T;
    myColumnsCount = myColumnsCount_T;
    totalCellsCount = totalCellsCount_T;

    //we set the pointers to the correct address of cells
    cellId = 0;
    for (unsigned int c = 0; c < myColumnsCount; c++)
    {
        qsortWrapper[c].cells = &(cells[cellId]);
        cellId += qsortWrapper[c].cellsCount;

	//make sure that all columns received were meant for this cpu
	assert(qsortWrapper[c].cpu == mpiRank);
    }

    //we sort by row, not by cpu (therefore converting into a single sparse matrix)    
    sortCellsByCpuRowColumn(myColumnsCount, totalCellsCount, qsortWrapper, cells);

    //set final data structures: convert qsort wrapper to sparse matrix arrays
    columnsIndex = new unsigned int[myColumnsCount];
    numberOfColumnsPerRow = new unsigned int[myRowsCount];
    numberOfCellsPerColumn = new unsigned int [myColumnsCount];

    for (unsigned int r=0; r<myRowsCount; r++)
	numberOfColumnsPerRow[r]=0;
    
    for (unsigned int c = 0; c < myColumnsCount; c++)
    {
	unsigned int & row = qsortWrapper[c].row;
	numberOfColumnsPerRow[row - myFirstRowIndex]++;
	
	columnsIndex[c]=qsortWrapper[c].col;
	numberOfCellsPerColumn[c]= qsortWrapper[c].cellsCount;
    }

    delete [] qsortWrapper;
  }
};

#endif	/* DISTRIBUTEDMEMORYSORTER_H */

