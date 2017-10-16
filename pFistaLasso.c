
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
#include <omp.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <matio.h>

void shrink(gsl_vector *x, gsl_vector *gamma, double sigma);
double objective(gsl_vector *x, double lambda, gsl_vector *z);
double calcErr(gsl_matrix *A);

int main(int argc, char **argv) {
  
  const int MAX_ITER      = 50; // number of iteration
  unsigned int nthreads = 8;

//  const double TOL        = 1e-4; // tolerence
//  const double lambda_tol = 1e-4; // tolerence for update lambda

  char* dir; // directory of data
  if(argc==2)
    dir = argv[1];

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

  mat_t *matfp;
  matvar_t *matvar;
  gsl_matrix *A, *b;
  gsl_vector *G;

  printf("Reading A, b and Gamma from %s\n", dir);
  matfp = Mat_Open(dir, MAT_ACC_RDONLY);
    if (NULL == matfp) {
      fprintf(stderr, "Error opening MAT file \"%s\"!\n", dir);
      return EXIT_FAILURE;
    }
    matvar = Mat_VarRead(matfp, "A");
    if(NULL == matvar) {
      fprintf(stderr, "Error reading variable");
      return EXIT_FAILURE;
    } else {
      A = gsl_matrix_calloc(matvar->dims[0], matvar->dims[1]);
      m = A->size1;
      n = A->size2;
      for (int i = 0; i < m*n; i++) {
        row = i % m;
        col = floor(i/m);
        entry = ((double*)matvar->data)[i];
        gsl_matrix_set(A, row, col, entry);
      }
    }
    Mat_VarFree(matvar);

    matvar = Mat_VarRead(matfp, "b");
    if(NULL == matvar) {
      fprintf(stderr, "Error reading variable");
      return EXIT_FAILURE;
    } else {
      b = gsl_matrix_calloc(matvar->dims[0], matvar->dims[1]);
      m = b->size1;
      n = b->size2;
      for (int i = 0; i < m*n; i++) {
        row = i % m;
        col = floor(i/m);
        entry = ((double*)matvar->data)[i];
        gsl_matrix_set(b, row, col, entry);
      }
    }
    Mat_VarFree(matvar);

    matvar = Mat_VarRead(matfp, "G");
    if(NULL == matvar) {
      fprintf(stderr, "Error reading variable");
      return EXIT_FAILURE;
    } else {
      G = gsl_vector_calloc(matvar->dims[0]);
      m = G->size;
      n = 1;
      for (int i = 0; i < m*n; i++) {
        entry = ((double*)matvar->data)[i];
        gsl_vector_set(G, i, entry);
      }
    }

    Mat_VarFree(matvar);
    Mat_Close(matfp);
  // [m, n] = size(A);
  m = A->size1;
  n = A->size2;
//  MPI_Barrier(MPI_COMM_WORLD);
 


  // FISTA parameters
  double delta;
  double lambda = 10;

  double err1 = calcErr(A);

  delta = 1.00/(err1*err1);

  const gsl_matrix *X = gsl_matrix_calloc(b->size2, n);

  startTime = omp_get_wtime();

#pragma omp parallel for schedule(static) num_threads(nthreads) shared(X) shared(G) shared(delta) shared(lambda)
  for(int i = 0; i < b->size2; ++i)
  {
	  // These are all variables related to FISTA
	  gsl_vector *x      = gsl_vector_calloc(n);
	  gsl_vector *u      = gsl_vector_calloc(n);
	  gsl_vector *xold   = gsl_vector_calloc(n);
	  gsl_vector *w      = gsl_vector_calloc(n);
	  gsl_vector *Ax     = gsl_vector_calloc(m);
	  gsl_vector *bi 	 = gsl_vector_calloc(b->size1);

    /*----------------------
     initialize variables
    ----------------------*/
    gsl_vector_set_zero(x);
    gsl_vector_set_zero(u);
    gsl_vector_set_zero(xold);
    gsl_vector_set_zero(w);
    gsl_vector_set_zero(bi);
    gsl_vector_set_zero(Ax);


    double t1=1, t2=1;
    int iter = 0;

    gsl_matrix_get_col(bi, b, i);

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

    iter++;
   }
  
  //Fill solution Matrix X with column solution x
  gsl_matrix_set_col (X,i,x);

  gsl_vector_free(bi);
  gsl_vector_free(x);
  gsl_vector_free(w);
  gsl_vector_free(Ax);

  }


  endTime = omp_get_wtime();
  printf("Elapsed time is: %lf \n", endTime - startTime);

  sprintf(s, "Results/solution.mat");
  printf("Writing solution matrix to: %s ", s);
  startTime = omp_get_wtime();
  matfp = Mat_CreateVer(s, NULL, MAT_FT_MAT73); //or MAT_FT_MAT4 / MAT_FT_MAT73
  size_t    dims[2] = {X->size2, X->size1}; // The X will be transposed, this is because Matlab is column-major while C is row-major
  matvar_t *solution = Mat_VarCreate("X", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, X->data, MAT_F_DONT_COPY_DATA);
  Mat_VarWrite(matfp, solution, MAT_COMPRESSION_NONE);
  Mat_VarFree(solution);
  Mat_Close(matfp);
  endTime = omp_get_wtime();
  printf("in %lf seconds\n", endTime - startTime);

  
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
void shrink(gsl_vector *x, gsl_vector *G, double sigma) {
  double entry;
  double Gi;
  gsl_vector *gamma = gsl_vector_calloc(G->size);
  gsl_vector_memcpy(gamma, G);
  gsl_vector_scale(gamma, sigma);

  for (int i = 0; i < x->size; i++) {
    Gi = gsl_vector_get(gamma, i);
    entry = gsl_vector_get(x, i);
    if (entry < - Gi) {
      gsl_vector_set(x, i, entry + Gi);
    }
    else if (entry > Gi)
    {
      gsl_vector_set(x, i, entry - Gi);
    }
  }
  gsl_vector_free(gamma);
}

double calcErr(gsl_matrix *A)
{
	double m = A->size1;
	double n = A->size2;
	gsl_vector *x = gsl_vector_calloc(n);
	gsl_vector *Ax = gsl_vector_calloc(m);

	// approximate ||A||_2 by power method
	double sum_x = 0, err0 = 0, err1;
	double tol = 1e-6, normx, normy;
	int cnt = 0;
	for (int j = 0; j < n; j++) {
		gsl_matrix_get_col(Ax, A, j);
		sum_x = gsl_blas_dasum(Ax);
		gsl_vector_set(x, j, sum_x);
	}
	err1 = gsl_blas_dnrm2(x);
	gsl_vector_scale(x, 1.0 / err1);
	while (fabs(err1 - err0) > tol * err1) {
		err0 = err1;
		gsl_blas_dgemv(CblasNoTrans, 1, A, x, 0, Ax); // Ax = A*x
		gsl_blas_dgemv(CblasTrans, 1, A, Ax, 0, x);
		normx = gsl_blas_dnrm2(x);
		normy = gsl_blas_dnrm2(Ax);

		err1 = normx / normy;
		gsl_vector_scale(x, 1.0 / normx);
		cnt = cnt + 1;
		if (cnt > 100) {
			break;
		}
	}

	return err1;
}
