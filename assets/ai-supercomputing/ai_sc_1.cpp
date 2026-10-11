#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define N 100000

double y[N], fx[N];
double MAE = 0;

int main()
{
  std::fill_n(y, N, 1);   //set y to ones
  std::fill_n(fx, N, 0);  //set fx to zeros

  printf("Running single thread from index %d to %d...\n", 0, N); 
  for (int n=0; n<N; n++)
    for (int i=0; i<50000; i++)
      MAE += abs(y[n] - fx[n]);

  printf("%.1f\n", MAE);
  return 0;
}
