
/* -------------------------------------------------------
 * Solve a distributed lasso problem, i.e.,
 *
 *   minimize \lambda * ||x||_1 + 0.5 * ||Ax - b||_2^2
 *
 * The implementation uses MPI for distributed communication
 * and the GNU Scientific Library (GSL) for math.
 * Compile: make
 * run code: mpiexec -n 1 ./IST data_directory
 *
 * Author:   Zhimin Peng
 * Date:     01/11/2013
 * Modified: 02/06/2013
 *--------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mmio.h"
#include <mpi.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

void shrink(gsl_matrix *m, gsl_vector *gamma, double sigma);
void objective(gsl_matrix *x, double lambda, gsl_matrix *z, double *obj);

int main(int argc, char **argv) {
  
  const int MAX_ITER      = 50; // number of iteration
  const double TOL        = 1e-4; // tolerence
//  const double lambda_tol = 1e-4; // tolerence for update lambda
  int rank; // process ID
  int size; // number of processes

  MPI_Init(&argc, &argv); // initialize MPI environment
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // determine current running process
  MPI_Comm_size(MPI_COMM_WORLD, &size); // total number of processes


  char* dataCenterDir = ".";
  char* big_dir; // directory of data
  if(argc==2)
    big_dir = argv[1];
  else
    big_dir = "data/tmp/1";

  /* Read in local data */
  FILE *f, *test;
  int m, n, row, col;
  double entry;
  double startTime, endTime, commStartTime, commEndTime, commTime;
  char s[100];

  /* -------------------------------------------------------
   * Subsystem n will look for files called An.dat and bn.dat
   * in the current directory; these are its local data and do 
   * not need to be visible to any other processes. Note that
   * m and n here refer to the dimensions of the 
   * local coefficient matrix.
   * -------------------------------------------------------*/

  /* Read A */
  sprintf(s, "%s/%s/A%d.dat",dataCenterDir,big_dir, rank + 1);
  printf("[%d] reading %s\n", rank, s);  
  f = fopen(s, "r");
  if (f == NULL) {
    printf("[%d] ERROR: %s does not exist, exiting.\n", rank, s);
   exit(EXIT_FAILURE);
  }
  mm_read_mtx_array_size(f, &m, &n);
  gsl_matrix *A = gsl_matrix_calloc(m, n);
  for (int i = 0; i < m*n; i++) {
    row = i % m;
    col = floor(i/m);
    fscanf(f, "%lf", &entry);
    gsl_matrix_set(A, row, col, entry);
  }
  fclose(f);

  /* Read b */
  sprintf(s, "%s/%s/b.dat", dataCenterDir, big_dir);
  f = fopen(s, "r");
  if (f == NULL) {
    printf("[%d] ERROR: %s does not exist, exiting.\n", rank, s);
    exit(EXIT_FAILURE);
  }
  mm_read_mtx_array_size(f, &m, &n);
  gsl_matrix *b = gsl_matrix_calloc(m, n);
  for (int i = 0; i < m*n; i++) {
    row = i % m;
    col = floor(i/m);
    fscanf(f, "%lf", &entry);
    gsl_matrix_set(b, row, col, entry);
  }
  fclose(f);
  
  /* Read Gamma */
  sprintf(s, "%s/%s/Gamma.dat", dataCenterDir, big_dir);
  f = fopen(s, "r");
  if (f == NULL) {
    printf("[%d] ERROR: %s does not exist, exiting.\n", rank, s);
    exit(EXIT_FAILURE);
  }
  mm_read_mtx_array_size(f, &m, &n);
  gsl_vector *G = gsl_vector_calloc(m);
  for (int i = 0; i < m; i++) {
    fscanf(f, "%lf", &entry);
    gsl_vector_set(G, i, entry);
  }
  fclose(f);

  /* Read xs */
  sprintf(s, "%s/%s/xs%d.dat", dataCenterDir, big_dir, rank + 1);
  printf("[%d] reading %s\n", rank, s);
  f = fopen(s, "r");
  if (f == NULL) {
    printf("[%d] ERROR: %s does not exist, exiting.\n", rank, s);
    exit(EXIT_FAILURE);
  }
  mm_read_mtx_array_size(f, &m, &n);
  gsl_vector *xs = gsl_matrix_calloc(m, n);
  for (int i = 0; i < m*n; i++) {
    row = i % m;
    col = floor(i/m);
    fscanf(f, "%lf", &entry);
    gsl_matrix_set(xs, row, col, entry);
  }
  fclose(f);
  // [m, n] = size(A);
  m = A->size1;
  n = A->size2;
  int bm = b->size1;
  int bn = b->size2;

  MPI_Barrier(MPI_COMM_WORLD);
 
  // These are all variables related to FISTA
  gsl_matrix *x      = gsl_matrix_calloc(n, bn);
  gsl_vector *X      = gsl_vector_calloc(n);
  gsl_matrix *u      = gsl_matrix_calloc(n, bn);
  gsl_matrix *xold   = gsl_matrix_calloc(n, bn);
  gsl_matrix *z      = gsl_matrix_calloc(m, bn);
  gsl_matrix *Z      = gsl_vector_calloc(m);
  gsl_vector *y      = gsl_vector_calloc(m);
  gsl_matrix *w      = gsl_matrix_calloc(n, bn);
  gsl_matrix *Ax     = gsl_matrix_calloc(m, bn);
  gsl_vector *ax     = gsl_vector_calloc(m);
  gsl_vector *xdiff  = gsl_vector_calloc(n);
  double send[1]; // an array used to aggregate 3 scalars at once
  double recv[1]; // used to receive the results of these aggregations
  double err;
//  double xs_local_nrm[1], xs_nrm[1]; // calculate the two norm of xs
//  gsl_vector_view vec = gsl_matrix_row (xs, 1);
//  xs_local_nrm[0] = gsl_blas_dnrm2(&vec.vector);
//  xs_local_nrm[0] = xs_local_nrm[0]* xs_local_nrm[0];
//  MPI_Allreduce(xs_local_nrm, xs_nrm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//  xs_nrm[0] = sqrt(xs_nrm[0]);

  // FISTA parameters
  double delta, t1=1, t2=1;
  double lambda = 0.01;

  // approximate ||A||_2 by power method
  double sum_x = 0, err0=0, err1;
  double tol= 1e-6,local_normx, normx, normy, local_err1;
  int cnt=0;
  // calcuate x = sum(abs(A), 1)';
   for(int j=0;j<n;j++){
    gsl_matrix_get_col(Z, A, j);
    sum_x = gsl_blas_dasum(Z);
    gsl_vector_set(X, j, sum_x);
  }
  local_err1 = gsl_blas_dnrm2(X);
  local_err1 = local_err1*local_err1;
  MPI_Allreduce(&local_err1, &err1,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // y = sum A * x
  err1 = sqrt(err1);
  gsl_vector_scale(X, 1.0/err1);
  while(fabs(err1 - err0) > tol * err1){
    err0 = err1;
    gsl_blas_dgemv(CblasNoTrans, 1, A, X, 0, ax); // Ax = A*x
    MPI_Allreduce(ax->data, y->data,  m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // y = sum A * x
    gsl_blas_dgemv(CblasTrans, 1, A, y, 0, X);
    local_normx = gsl_blas_dnrm2(X);
    local_normx = local_normx * local_normx;
    MPI_Allreduce(&local_normx, &normx,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // y = sum A * x
    normx = sqrt(normx);
    normy = gsl_blas_dnrm2(y);

    err1 = normx/normy;
    gsl_vector_scale(X, 1.0/normx);
    cnt = cnt+1;
    if(cnt > 100){
      break;
    }
  }

  /*----------------------
     initialize variables
    ----------------------*/
  gsl_vector_set_zero(X);
  gsl_vector_set_zero(y);
  gsl_vector_set_zero(Z);
//
  delta = 1.00/(err1*err1);

    sprintf(s, "Results/test%d.m", rank + 1);
    test = fopen(s, "w");

  int iter = 0;

  startTime = MPI_Wtime();

  /* Main FISTA solver loop */
  while (iter < MAX_ITER) {

    t1 = t2;
    gsl_matrix_memcpy(xold, x); // copy x to old x;
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, A, u, 0, z); // A_i x_i = A_i * x_i

    commTime = commEndTime - commStartTime; // calculate the communication time
    gsl_matrix_sub(z, b);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, A, z, 0, w); // w = A' (Ax - b)
    gsl_matrix_scale(w, delta);  // w = delta * w
    gsl_matrix_sub(u, w);  // u = x - delta * A'(Ax - b)
    gsl_matrix_memcpy(x, u);
    shrink(x, G, delta * lambda); // shrink(x, alpha*lambda)
 
    // FISTA 
    t2 = 0.5 + 0.5*sqrt(1+4*t1*t1);
    gsl_matrix_sub(xold, x);
    gsl_matrix_scale(xold, (1-t1)/t2);
    gsl_matrix_add(x, xold);

    /* termination check */
      double obj[x->size2];
      objective(x, lambda, z, &obj);
      for(int i = 0; i < x->size2; i++)
      {
       fprintf(test, "%e;\n", obj[i]);
      }
    
//    if(recv[0] < TOL){
//      break;
//    }

    iter++;
  }
  endTime = MPI_Wtime();

  /* Have the master write out the results to disk */
    fclose(test);
    sprintf(s, "Results/solution%d.dat",rank + 1);
    f = fopen(s, "w");
    gsl_matrix_fprintf(f, x, "%lf");
    fclose(f);
 
    printf("Elapsed time is: %lf \n", endTime - startTime);
  
  MPI_Finalize(); /* Shut down the MPI execution environment */
  
  /* Clear memory */
  gsl_matrix_free(A);
  gsl_matrix_free(b);
  gsl_matrix_free(x);
  gsl_vector_free(X);
  gsl_vector_free(y);
  gsl_matrix_free(z);
  gsl_vector_free(Z);
  gsl_matrix_free(w);
  gsl_vector_free(xdiff);
  gsl_matrix_free(Ax);
  gsl_vector_free(ax);

  return 0;
}

/* calculate objective function */
void objective(gsl_matrix *x, double lambda, gsl_matrix *z, double *obj) {
//  double obj[x->size2];
  double temp[z->size2];

  for (int j = 0; j < z->size2; j++)
    {
      gsl_vector_view column = gsl_matrix_column (z, j);
      temp[j] = gsl_blas_dnrm2 (&column.vector);
      temp[j] = temp[j]*temp[j]/2;
    }

  double foo[x->size2];
  for (int j = 0; j < x->size2; j++)
    {
      gsl_vector_view column = gsl_matrix_column (x, j);
      foo[j] = gsl_blas_dasum (&column.vector);
      foo[j] = foo[j]*foo[j]/2;
    }


  for (int i = 0; i < x->size2; i++)
  {
	  obj[i] = lambda*foo[i] + temp[i];
  }
}

/* shrinkage function */
void shrink(gsl_matrix *m, gsl_vector *G, double sigma) {
  double entry;
  double Gi;
  gsl_vector *gamma = gsl_vector_calloc(G->size);
  gsl_vector_memcpy(gamma, G);
  gsl_vector_scale(gamma, sigma);

  for (int i = 0; i < m->size1; i++) {
    Gi = gsl_vector_get(gamma, i);
    for (int j = 0; j < m->size2; j++) {
      entry = gsl_matrix_get(m, i, j);
      if (entry < Gi && entry > -Gi) {
	gsl_matrix_set(m, i, j, 0);
      }
    }
  }
  gsl_vector_free(gamma);
}
