
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

void shrink(gsl_vector *x, gsl_vector *gamma, double sigma);
double objective(gsl_vector *x, double lambda, gsl_vector *z);

int main(int argc, char **argv) {
  
  const int MAX_ITER      = 50; // number of iteration
//  const double TOL        = 1e-4; // tolerence
//  const double lambda_tol = 1e-4; // tolerence for update lambda
  int rank; // process ID
  int size; // number of processes

  MPI_Init(&argc, &argv); // initialize MPI environment
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // determine current running process
  MPI_Comm_size(MPI_COMM_WORLD, &size); // total number of processes


  char* dataCenterDir = "./";
  char* big_dir; // directory of data
  if(argc==2)
    big_dir = argv[1];
  else
    big_dir = "big1";

  /* Read in local data */
  FILE *f, *test;
  int m, n, row, col;
  double entry, startTime, endTime;
  char s[100];

  /* -------------------------------------------------------
   * Subsystem n will look for files called An.dat and bn.dat
   * in the current directory; these are its local data and do 
   * not need to be visible to any other processes. Note that
   * m and n here refer to the dimensions of the 
   * local coefficient matrix.
   * -------------------------------------------------------*/

  /* Read A */
  sprintf(s, "%s/%s/A.dat",dataCenterDir,big_dir);
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
    gsl_matrix_set(b, row, col, entry);  }
  fclose(f);
  
  /* Read Gamma */
  sprintf(s, "%s/%s/Gamma.dat", dataCenterDir, big_dir);
  printf("reading %s\n", s);
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

  // [m, n] = size(A);
  m = A->size1;
  n = A->size2;
  MPI_Barrier(MPI_COMM_WORLD);
 
  // These are all variables related to FISTA
  gsl_vector *x      = gsl_vector_calloc(n);
  gsl_vector *u      = gsl_vector_calloc(n);
  gsl_vector *xold   = gsl_vector_calloc(n);
  gsl_vector *w      = gsl_vector_calloc(n);
  gsl_vector *Ax     = gsl_vector_calloc(m);
  double err;

  // FISTA parameters
  double delta, t1=1, t2=1;
  double lambda = 0.01;

  // approximate ||A||_2 by power method
  double sum_x = 0, err0=0, err1;
  double tol= 1e-6, normx, normy;
  int cnt=0;
  // calcuate x = sum(abs(A), 1)';
  startTime = MPI_Wtime();
   for(int j=0;j<n;j++){
    gsl_matrix_get_col(Ax, A, j);
    sum_x = gsl_blas_dasum(Ax);
    gsl_vector_set(x, j, sum_x);
  }
  err1 = gsl_blas_dnrm2(x);
  gsl_vector_scale(x, 1.0/err1);
  while(fabs(err1 - err0) > tol * err1){
    err0 = err1;
    gsl_blas_dgemv(CblasNoTrans, 1, A, x, 0, Ax); // Ax = A*x
    gsl_blas_dgemv(CblasTrans, 1, A, Ax, 0, x);
    normx = gsl_blas_dnrm2(x);
    normy = gsl_blas_dnrm2(Ax);
  
    err1 = normx/normy;
    gsl_vector_scale(x, 1.0/normx);
    cnt = cnt+1;
    if(cnt > 100){
      break;
    }
  }
  endTime = MPI_Wtime();
  printf("spertral norm evaluation time: %e \n", endTime - startTime);
  
  delta = 1.00/(err1*err1);

  gsl_vector *bi = gsl_vector_calloc(b->size1);

  printf("B(MxN) = %dx%d\n", b->size1, b->size2);
  printf("Processing job %d\n", rank);

  startTime = MPI_Wtime();

  int END_COL, START_COL;
  START_COL = rank * b->size2/size;
  if(rank == size - 1)
    END_COL = START_COL + b->size2/size + b->size2%size;
  else
    END_COL = START_COL + b->size2/size;
  sprintf(s, "Results/solution%d.dat",rank + 1);
  f = fopen(s, "w");

  printf("Process %d works on b(%d:%d)\n", rank, START_COL, END_COL - 1);

  for(int col = START_COL; col < END_COL; ++col)
  {
    /*----------------------
     initialize variables
    ----------------------*/
    gsl_vector_set_zero(x);
    gsl_vector_set_zero(u);

    err = 0;

    int iter = 0;

    gsl_matrix_get_col(bi, b, col);

    /* Main FISTA solver loop */
    while (iter < MAX_ITER) {

    t1 = t2;
    gsl_vector_memcpy(xold, x); // copy x to old x;
    gsl_blas_dgemv(CblasNoTrans, 1, A, u, 0, Ax); // A_i x_i = A_i * x_i

    gsl_vector_sub(Ax, bi);
    gsl_blas_dgemv(CblasTrans, 1, A, Ax, 0, w); // w = A' (Ax - b)
    gsl_vector_scale(w, delta);  // w = delta * w
    gsl_vector_sub(u, w);  // u = x - delta * A'(Ax - b)
    gsl_vector_memcpy(x, u); 
    shrink(x, G, delta * lambda); // shrink(x, alpha*lambda)
 
    // FISTA 
    t2 = 0.5 + 0.5*sqrt(1+4*t1*t1);
    gsl_vector_sub(xold, x);
    gsl_vector_scale(xold, (1 - t1)/t2);
    gsl_vector_memcpy(u, x);
    gsl_vector_add(u, xold);

    /* termination check */
//    if (rank == 0) {
//      printf("%3d %e %10.4f %e \n", iter,
//	     recv[0],  objective(x, lambda, z), lambda);
//      fprintf(test, "%e %e %e %e;\n", recv[0], objective(x, lambda, z),
//	      endTime - startTime, commTime);
//    }
//
    iter++;
   }
  
  /* Have the master write out the results to disk */

  gsl_vector_fprintf(f, x, "%lf");


  }
  fclose(f);

  endTime = MPI_Wtime();

  printf("Elapsed time is: %lf \n", endTime - startTime);

  MPI_Finalize(); /* Shut down the MPI execution environment */
  
  /* Clear memory */
  gsl_matrix_free(A);
  gsl_vector_free(G);
  gsl_matrix_free(b);
  gsl_vector_free(bi);
  gsl_vector_free(x);
  gsl_vector_free(w);
  gsl_vector_free(Ax);
  return 0;
}

/* calculate objective function */
double objective(gsl_vector *x, double lambda, gsl_vector *z) {
  double obj = 0;
  double temp =0.0;
  temp = gsl_blas_dnrm2(z);
  temp = temp*temp/2;
  double foo;
  foo = gsl_blas_dasum(x);
  obj = lambda*foo + temp;
  return obj;
}

/* shrinkage function */
void shrink(gsl_vector *x, gsl_vector *G, double sigma) {
  double entry;
  double Gi;
  gsl_vector *gamma = gsl_vector_calloc(G->size);
  gsl_vector_memcpy(gamma, G);
  gsl_vector_scale(gamma, sigma);

  for (int i = 0; i < x->size; i++) {
    Gi = gsl_vector_get(gamma, i);
      entry = gsl_vector_get(x, i);
      if (entry < Gi && entry > -Gi) {
	gsl_vector_set(x, i, 0);
      }
  }
  gsl_vector_free(gamma);
}
