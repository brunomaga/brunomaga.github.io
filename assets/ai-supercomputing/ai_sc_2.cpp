#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <algorithm>

#define N 100000
#define THREADS 4 //<----- NEW

double y[N], fx[N];
double MAE = 0;

void * update_MAE(void *args)
{
  int N_per_thread = N/THREADS;
  int t = *(int*)args;
  int init = N_per_thread*t;
  int end = N_per_thread*(t+1);
  printf("Running thread %d, from index %d to %d...\n", t, init, end);

  for (int n=init; n<end; n++)
    for (int i=0; i<50000; i++)
    {
        MAE += abs(y[n] - fx[n]);
    }
}

int main()
{

  std::fill_n(y, N, 1);
  std::fill_n(fx, N, 0);

  //threads and thread-arguments declaration (NEW)
  pthread_t threads[THREADS];
  int args[THREADS];

  //Launch all threads (NEW)
  for (int t=0; t<THREADS; t++)
  {
    args[t] = t; //<--- Why do we need this?
    pthread_create(&threads[t], NULL, update_MAE, (void*) &args[t]);
  }

  //Wait for all threads (NEW) 
  for (int t=0; t<THREADS; t++) 
    pthread_join(threads[t], NULL); 

  printf("%.1f\n", MAE);
  return 0;
}
