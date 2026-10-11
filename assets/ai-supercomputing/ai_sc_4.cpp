#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <algorithm>

#define N 100000
#define THREADS 4

double y[N], fx[N];
double MAE = 0;
pthread_mutex_t MAE_mutex;

void * update_MAE(void *args)
{
  int N_per_thread = N/THREADS;
  int t = *(int*)args;
  int init = N_per_thread*t;
  int end = N_per_thread*(t+1);
  printf("Running thread %d, from index %d to %d...\n", t, init, end);

  double MAE_partial = 0; // <---- NEW
  for (int n=init; n<end; n++)
    for (int i=0; i<50000; i++)
    {
        MAE_partial += abs(y[n] - fx[n]);
    }
 
  //update contribution of this thread to MAE
  pthread_mutex_lock(&MAE_mutex);
  MAE += MAE_partial;
  pthread_mutex_unlock(&MAE_mutex);
}

int main()
{

  std::fill_n(y, N, 1);
  std::fill_n(fx, N, 0);

  //threads and thread-arguments declaration 
  pthread_t threads[THREADS];
  int args[THREADS];

  //Launch all threads
  for (int t=0; t<THREADS; t++)
  {
    args[t] = t;
    pthread_create(&threads[t], NULL, update_MAE, (void*) &args[t]);
  }

  //Wait for all threads
  for (int t=0; t<THREADS; t++) 
    pthread_join(threads[t], NULL); 

  printf("%.1f\n", MAE);
  return 0;
}
