#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "util.h"


// calculate the relative error
double* error(int N_p, double *u, double *v)
{
  static double a[2] = {0, 0};

  int i;
  for(i = 1; i <= N_p; i++) {
      a[0] += (u[i]-v[i])*(u[i]-v[i]);
      a[1] += u[i]*u[i];
  }

  return a;
}

// calculate the residual
double residual(int N, int N_p, double *f, double *u)
{
   double resi = 0, diff = 0;
   double hinv2 = (N+1)*(N+1);

   int i;
   for(i = 1; i <= N_p; i++) {
       diff = u[i]*2*hinv2-u[i-1]*hinv2-u[i+1]*hinv2-f[i];
       resi += diff*diff;
   }

   return resi;
}

// only check N_MAX iterations here, won't calculate the residual
int main(int argc, char **argv)
{
//  double eps = 1e-6; 
  long iter;
  double err, norm, resi, resi_p;
  double *err_p;
  double *f, *u_J, *u_temp;
  double *u_truth;

  int mpisize, mpirank, tag, left, right;
  MPI_Status status;
  tag = 99;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  if (argc != 3) {
      fprintf(stderr, "Function needs vector size as input arguments!\n");
      MPI_Abort(MPI_COMM_WORLD, 1); 
  } 
  long N = atol(argv[1]);
  long N_MAX = atol(argv[2]);
  long N_p = N/mpisize;

  if(N % mpisize != 0) {
      fprintf(stderr, "Vector size not divisible!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
  }

  f = (double *) malloc(sizeof(double)*(N_p + 2));
  u_J = (double *) malloc(sizeof(double)*(N_p + 2));
  u_temp = (double *) malloc(sizeof(double)*(N_p + 1));

  // initialize
  int i;
  for(i = 0; i <= N_p+1; i++) {
      f[i]=1; u_J[i]=0; 
  }
//  resi_p = residual(N, N/mpisize, f, u_J);
//  MPI_Allreduce(&resi_p, &resi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//  resi = sqrt(resi);
//  resi0 = resi;

  // setup time counter
  timestamp_type time1, time2;
  timestamp_type itert1, itert2, commt1, commt2;
  double itert_p=0, commt_p=0, elapsed_p=0;
  double itert, commt, elapsed;
  
  // Jacobi
  double hinv2 = (N+1)*(N+1);
  get_timestamp(&time1);
  iter = 0;
//  printf("Jacobi iteration of rank %d started.\n", mpirank);
  while( iter < N_MAX ) {
        get_timestamp(&itert1);
        for(i = 1; i <= N_p; i++) {
            u_temp[i] = (f[i]+hinv2*u_J[i-1]+hinv2*u_J[i+1]) / (2*hinv2);
            if(i > 1) {
                u_J[i-1] = u_temp[i-1];
            }
        }
        u_J[N_p] = u_temp[N_p];
        get_timestamp(&itert2);

        get_timestamp(&commt1);
        if(mpirank == 0) {
            right = mpirank+1;

            MPI_Send(u_J+N_p,   1, MPI_DOUBLE, right, tag, MPI_COMM_WORLD);
            MPI_Recv(u_J+N_p+1, 1, MPI_DOUBLE, right, tag, MPI_COMM_WORLD, &status);
        } else if(mpirank == mpisize-1) {
            left = mpirank-1;

            MPI_Send(u_J+1, 1, MPI_DOUBLE, left, tag, MPI_COMM_WORLD);
            MPI_Recv(u_J,   1, MPI_DOUBLE, left, tag, MPI_COMM_WORLD, &status);
        } else {
            right = mpirank+1;
            left = mpirank-1;

            MPI_Send(u_J+N_p,   1, MPI_DOUBLE, right, tag, MPI_COMM_WORLD);
            MPI_Send(u_J+1,           1, MPI_DOUBLE, left,  tag, MPI_COMM_WORLD);
            MPI_Recv(u_J,             1, MPI_DOUBLE, left,  tag, MPI_COMM_WORLD, &status);
            MPI_Recv(u_J+N_p+1, 1, MPI_DOUBLE, right, tag, MPI_COMM_WORLD, &status);
        }
        get_timestamp(&commt2);
	iter++;

        itert_p += timestamp_diff_in_seconds(itert1,itert2);
        commt_p += timestamp_diff_in_seconds(commt1,commt2);
  }
  get_timestamp(&time2);
  elapsed_p = timestamp_diff_in_seconds(time1,time2);
  MPI_Reduce(&elapsed_p, &elapsed, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  elapsed = elapsed/mpisize;
  MPI_Reduce(&itert_p, &itert, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  itert = itert/mpisize;
  MPI_Reduce(&commt_p, &commt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  commt = commt/mpisize;


  // get truth 
  double x;
  u_truth = (double *) malloc(sizeof(double)*(N_p+2));
  for(i = 0; i <= N_p+1; i++) {
      x = (double)(i+mpirank*N_p) / (N+1);
      u_truth[i] = .5 * x * (1-x);
  }
  // get error
  err_p = error(N_p, u_truth, u_J);
  MPI_Reduce(err_p, &err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(err_p+1, &norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  err = sqrt(err)/sqrt(norm);
  resi_p = residual(N, N_p, f, u_J);
  MPI_Reduce(&resi_p, &resi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  resi = sqrt(resi);



  // print results
  if(mpirank == 0) {
    printf("take %ld iterations, residual = %.10f, err = %.10f.\n", iter, resi, err);
    printf("total run_time = %fs; iteration time = %fs, communication time = %fs.\n", elapsed, itert, commt);
  }

  // pointwise value
/*
  int j;
  for(j=0; j < mpisize; j++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(j == mpirank) {

          for(i=0; i <= N/mpisize; i++) {
            if(i%(N/10) == 0) {
              printf("x=%f: u_truth=%.10f, u_J=%.10f\n", (double)(i+j*N/mpisize) / (N+1), u_truth[i],u_J[i]);
            }
          }

      }
  }
*/

  
  free(u_J);
  free(u_truth);
  free(u_temp);
  free(f);

  MPI_Finalize();
  return 0;
}
