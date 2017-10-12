
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
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <omp.h>
#include <matio.h>

void shrink(gsl_vector *x, gsl_vector *G);
double objective(gsl_vector *x, double lambda, gsl_vector *z);
double err(gsl_matrix *A)
{
	  double m = A->size1;
	  double n = A->size2;
	  // These are all variables related to FISTA
	  gsl_vector *x      = gsl_vector_calloc(n);
	  gsl_vector *u      = gsl_vector_calloc(n);
	  gsl_vector *xold   = gsl_vector_calloc(n);
	  gsl_vector *w      = gsl_vector_calloc(n);
	  gsl_vector *Ax     = gsl_vector_calloc(m);
	  double err;

	  // approximate ||A||_2 by power method
	  double sum_x = 0, err0=0, err1;
	  double tol= 1e-6, normx, normy;
	  int cnt=0;
	  // calcuate x = sum(abs(A), 1)';
	//  startTime = MPI_Wtime();
	   for(int j=0;j<n;j++){
	    gsl_matrix_get_col(Ax, A, j);
	    sum_x = gsl_blas_dasum(Ax);
	    gsl_vector_set(x, j, sum_x);
	  }
	  err1 = gsl_blas_dnrm2(x);
	  gsl_vector_scale(x, 1.0/err1);
	  while(fabs(err1 - err0) > tol * err1){
	    gsl_blas_dgemv(CblasNoTrans, 1, A, x, 0, Ax); // Ax = A*x
	    gsl_blas_dgemv(CblasTrans, 1, A, Ax, 0, x);
	    normx = gsl_blas_dnrm2(x);
	    normy = gsl_blas_dnrm2(Ax);
	    gsl_vector_scale(x, 1.0/normx);

	    err0 = err1;
	    err1 = normx/normy;
	    cnt = cnt+1;
	    if(cnt > 100){
	      break;
	    }
	  }

	  gsl_vector_free(x);
	  gsl_vector_free(w);
	  gsl_vector_free(Ax);
	  gsl_vector_free(xold);
	  gsl_vector_free(u);
	  return err1;
}

int main(int argc, char **argv) {
  
  const int MAX_ITER      = 50; // number of iteration
//  const double TOL        = 1e-4; // tolerence
//  const double lambda_tol = 1e-4; // tolerence for update lambda

  char* dir; // directory of data
  unsigned int nthreads = 1;
  unsigned int use_matio = 0;
  if(argc==2)
    {
	  dir = argv[1];
	  printf("Using default 1 processing thread\n");
    }
  else if(argc == 3)
  	{
	  dir = argv[1];
	  nthreads = (unsigned)atoi(argv[2]);
    }
  else
    perror("Please provide data directory");

  /* Read in local data */
  FILE *f, *test;
  int m, n, row, col;
  double entry, startTime, endTime;
  char s[100];

  /* -------------------------------------------------------
   * Subsystem n will look for files called A.dat and b.dat
   * and Gamma.dat in the current directory; these are its
   * local data and do not need to be visible to any other
   * processes.
   * Note that m and n here refer to the dimensions of the
   * local coefficient matrix.
   * -------------------------------------------------------*/

  /* Read A */
  sprintf(s, "%s/A.dat",dir);
  printf("Reading %s\n", s);
  f = fopen(s, "r");
  if (f == NULL) {
   printf("ERROR: %s does not exist, exiting.\n", s);
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
  sprintf(s, "%s/b.dat", dir);
  printf("Reading %s\n", s);
  f = fopen(s, "r");
  if (f == NULL) {
    printf("ERROR: %s does not exist, exiting.\n", s);
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
  sprintf(s, "%s/Gamma.dat", dir);
  printf("Reading %s\n", s);
  f = fopen(s, "r");
  if (f == NULL) {
	printf("ERROR: %s does not exist, exiting.\n", s);
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

  double delta, t1=1, t2=1;
  double lambda = 0.01;

  double err1 = err(A);
  delta = 1.00/(err1*err1);


  // Scaling Gamma
  gsl_vector_scale(G, delta * lambda);

  // Preallocating solution matrix in memory
  gsl_matrix *X = gsl_matrix_calloc(n, b->size2);

  printf("Running pFISTA for LASSO\n");

  startTime = omp_get_wtime();
#pragma omp parallel for schedule(static) num_threads(nthreads)

  for(int col = 0; col < b->size2; ++col)
  {
    /*----------------------
     initialize local variables
    ----------------------*/
	gsl_vector *x      = gsl_vector_calloc(n);
    gsl_vector *u      = gsl_vector_calloc(n);
    gsl_vector *xold   = gsl_vector_calloc(n);
    gsl_vector *w      = gsl_vector_calloc(n);
    gsl_vector *Ax     = gsl_vector_calloc(m);
    gsl_vector *bi 	   = gsl_vector_calloc(b->size1);

    gsl_vector_set_zero(x);
    gsl_vector_set_zero(u);

    int err = 0;

    int iter = 0;

    gsl_matrix_get_col(bi, b, col);

    /* Main FISTA solver loop */
    while (iter < MAX_ITER)
    {
		t1 = t2;
		gsl_vector_memcpy(xold, x); // copy x to old x;
		gsl_blas_dgemv(CblasNoTrans, 1, A, u, 0, Ax); // A_i x_i = A_i * x_i

		gsl_vector_sub(Ax, bi);
		gsl_blas_dgemv(CblasTrans, 1, A, Ax, 0, w); // w = A' (Ax - b)
		gsl_vector_scale(w, delta);  // w = delta * w
		gsl_vector_sub(u, w);  // u = x - delta * A'(Ax - b)
		gsl_vector_memcpy(x, u);
		shrink(x, G); // shrink(x, alpha*lambda)

		// FISTA
		t2 = 0.5 + 0.5*sqrt(1+4*t1*t1);
		gsl_vector_sub(xold, x);
		gsl_vector_scale(xold, (1 - t1)/t2);
		gsl_vector_memcpy(u, x);
		gsl_vector_add(u, xold);

		iter++;
   }

    gsl_matrix_set_col (X, col, x);

    // Clear local memory allocations
    gsl_vector_free(x);
    gsl_vector_free(w);
    gsl_vector_free(Ax);
    gsl_vector_free(xold);
    gsl_vector_free(u);
  	gsl_vector_free(bi);
  }
  endTime = omp_get_wtime();
  printf("Elapsed time is: %lf \n", endTime - startTime);


  if(use_matio)
  {
	  sprintf(s, "Results/solution.mat");
	  printf("Writing solution matrix to: %s ", s);
	  mat_t *matfp = Mat_CreateVer(s, NULL, MAT_FT_MAT73); //or MAT_FT_MAT4 / MAT_FT_MAT73
	  size_t    dims[2] = {n, b->size2};
	  matvar_t *solution = Mat_VarCreate("X", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, X->data, MAT_F_DONT_COPY_DATA);
	  Mat_VarWrite(matfp, solution, MAT_COMPRESSION_ZLIB);
	  Mat_VarFree(solution);
  }
  else
  {
	  sprintf(s, "Results/solution.dat");
	  printf("Writing solution matrix to: %s ", s);
	  f = fopen(s, "w");
	  startTime = omp_get_wtime();
	  gsl_matrix_fprintf(f, X, "%lf");
	  endTime = omp_get_wtime();
	  printf("in %lf seconds\n", endTime - startTime);
	  fclose(f);
  }

  /* Clear memory */
  gsl_matrix_free(A);
  gsl_vector_free(G);
  gsl_matrix_free(b);
  gsl_matrix_free(X);

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
void shrink(gsl_vector *x, gsl_vector *G) {
  double entry;
  double Gi;

  for (int i = 0; i < x->size; i++) {
    Gi = gsl_vector_get(G, i);
    entry = gsl_vector_get(x, i);
    if (entry < - Gi) {
      gsl_vector_set(x, i, entry + Gi);
    }
    else if (entry > Gi)
    {
      gsl_vector_set(x, i, entry - Gi);
    }
  }
}
