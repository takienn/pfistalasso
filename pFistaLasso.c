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
#include <gsl/gsl_vector_float.h>
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <matio.h>

void
shrink (gsl_vector_float *x, gsl_vector_float *gamma, float sigma);
float
objective (gsl_vector_float *x, float lambda, gsl_vector_float *z);
float
calcErr (gsl_matrix_float *A);

int
main (int argc, char **argv)
{

  const int MAX_ITER = 50; // number of iteration
  unsigned int nthreads = 8;

//  const float TOL        = 1e-4; // tolerence
//  const float lambda_tol = 1e-4; // tolerence for update lambda

  char* dir; // directory of data
  if (argc >= 2)
    dir = argv[1];
  if (argc >= 3)
    nthreads = (unsigned) atoi (argv[2]);

  printf ("Running with %d threads\n", nthreads);

  /* Read in local data */
  int m, n, row, col;
  float entry, startTime, endTime;
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
  gsl_matrix_float *A, *b;
  gsl_vector_float *G;

  printf ("Reading A, b and Gamma from %s\n", dir);
  matfp = Mat_Open (dir, MAT_ACC_RDONLY);
  if (NULL == matfp)
  {
    fprintf (stderr, "Error opening MAT file \"%s\"!\n", dir);
    return EXIT_FAILURE;
  }
  matvar = Mat_VarRead (matfp, "A");
  if (NULL == matvar)
  {
    fprintf (stderr, "Error reading variable A\n");
    return EXIT_FAILURE;
  }
  else if (matvar->class_type != MAT_C_SINGLE)
  {
    fprintf (stderr, "A has to be of type single\n");
    return EXIT_FAILURE;
  }
  else
  {
    A = gsl_matrix_float_calloc (matvar->dims[0], matvar->dims[1]);
    m = A->size1;
    n = A->size2;
    for (int i = 0; i < m * n; i++)
    {
      row = i % m;
      col = floor (i / m);
      entry = ((float*) matvar->data)[i];
      gsl_matrix_float_set (A, row, col, entry);
    }
  }
  Mat_VarFree (matvar);

  matvar = Mat_VarRead (matfp, "b");
  if (NULL == matvar)
  {
    fprintf (stderr, "Error reading variable b\n");
    return EXIT_FAILURE;
  }
  else if (matvar->class_type != MAT_C_SINGLE)
  {
    fprintf (stderr, "b has to be of type single\n");
    return EXIT_FAILURE;
  }
  else
  {
    b = gsl_matrix_float_calloc (matvar->dims[0], matvar->dims[1]);
    m = b->size1;
    n = b->size2;
    for (int i = 0; i < m * n; i++)
    {
      row = i % m;
      col = floor (i / m);
      entry = ((float*) matvar->data)[i];
      gsl_matrix_float_set (b, row, col, entry);
    }
  }
  Mat_VarFree (matvar);

  matvar = Mat_VarRead (matfp, "G");
  if (NULL == matvar)
  {
    fprintf (stderr, "Error reading variable Gamma\n");
    return EXIT_FAILURE;
  }
  else if (matvar->class_type != MAT_C_SINGLE)
  {
    fprintf (stderr, "Gamma has to be of type single\n");
    return EXIT_FAILURE;
  }
  else
  {
    G = gsl_vector_float_calloc (matvar->dims[0]);
    m = G->size;
    n = 1;
    for (int i = 0; i < m * n; i++)
    {
      entry = ((float*) matvar->data)[i];
      gsl_vector_float_set (G, i, entry);
    }
  }

  Mat_VarFree (matvar);
  Mat_Close (matfp);
  // [m, n] = size(A);
  m = A->size1;
  n = A->size2;

  // FISTA parameters
  float delta;
  float lambda = 10;

  float err1 = calcErr (A);

  delta = 1.00 / (err1 * err1);

  const gsl_matrix_float *X = gsl_matrix_float_calloc (n, b->size2);

  startTime = omp_get_wtime ();

#pragma omp parallel for schedule(static) num_threads(nthreads) shared(X) shared(G) shared(delta) shared(lambda)
  for (int i = 0; i < b->size2; ++i)
  {
    // These are all variables related to FISTA
    gsl_vector_float *x = gsl_vector_float_calloc (n);
    gsl_vector_float *u = gsl_vector_float_calloc (n);
    gsl_vector_float *xold = gsl_vector_float_calloc (n);
    gsl_vector_float *w = gsl_vector_float_calloc (n);
    gsl_vector_float *Ax = gsl_vector_float_calloc (m);
    gsl_vector_float *bi = gsl_vector_float_calloc (b->size1);

    /*----------------------
     initialize variables
     ----------------------*/
    gsl_vector_float_set_zero (x);
    gsl_vector_float_set_zero (u);
    gsl_vector_float_set_zero (xold);
    gsl_vector_float_set_zero (w);
    gsl_vector_float_set_zero (bi);
    gsl_vector_float_set_zero (Ax);

    float t1 = 1, t2 = 1;
    int iter = 0;

    gsl_matrix_float_get_col (bi, b, i);

    /* Main FISTA solver loop */
    while (iter < MAX_ITER)
    {

      t1 = t2;
      gsl_vector_float_memcpy (xold, x); // copy x to old x;
      gsl_blas_sgemv (CblasNoTrans, 1, A, u, 0, Ax); // A_i x_i = A_i * x_i

      gsl_vector_float_sub (Ax, bi);
      gsl_blas_sgemv (CblasTrans, 1, A, Ax, 0, w); // w = A' (Ax - b)
      gsl_vector_float_scale (w, delta);  // w = delta * w
      gsl_vector_float_sub (u, w);  // u = x - delta * A'(Ax - b)
      gsl_vector_float_memcpy (x, u);
      shrink (x, G, delta * lambda); // shrink(x, alpha*lambda)

      // FISTA
      t2 = 0.5 + 0.5 * sqrt (1 + 4 * t1 * t1);
      gsl_vector_float_sub (xold, x);
      gsl_vector_float_scale (xold, (1 - t1) / t2);
      gsl_vector_float_memcpy (u, x);
      gsl_vector_float_add (u, xold);

      iter++;
    }

    //Fill solution Matrix X with column solution x
    gsl_matrix_float_set_col (X, i, x);

    gsl_vector_float_free (x);
    gsl_vector_float_free (w);
    gsl_vector_float_free (Ax);
    gsl_vector_float_free (bi);
    gsl_vector_float_free (xold);
    gsl_vector_float_free (u);

  }

  endTime = omp_get_wtime ();
  printf ("Elapsed time is: %lf \n", endTime - startTime);

  sprintf (s, "Results/solution.mat");
  printf ("Writing solution matrix to: %s ", s);
  startTime = omp_get_wtime ();
  matfp = Mat_CreateVer (s, NULL, MAT_FT_MAT73); //or MAT_FT_MAT4 / MAT_FT_MAT73
  size_t dims[2] = { X->size2, X->size1 }; // The X will be transposed, this is because Matlab is column-major while C is row-major
  matvar_t *solution = Mat_VarCreate ("X", MAT_C_SINGLE, MAT_T_SINGLE, 2, dims,
				      X->data, MAT_F_DONT_COPY_DATA);
  Mat_VarWrite (matfp, solution, MAT_COMPRESSION_NONE);
  Mat_VarFree (solution);
  Mat_Close (matfp);
  endTime = omp_get_wtime ();
  printf ("in %lf seconds\n", endTime - startTime);

  /* Clear memory */
  gsl_matrix_float_free (A);
  gsl_vector_float_free (G);
  gsl_matrix_float_free (b);
  gsl_matrix_float_free (X);

  return 0;
}

/* calculate objective function */
float
objective (gsl_vector_float *x, float lambda, gsl_vector_float *z)
{
  float obj = 0;
  float temp = 0.0;
  temp = gsl_blas_snrm2 (z);
  temp = temp * temp / 2;
  float foo;
  foo = gsl_blas_sasum (x);
  obj = lambda * foo + temp;
  return obj;
}

/* shrinkage function */
void
shrink (gsl_vector_float *x, gsl_vector_float *G, float sigma)
{
  float entry;
  float Gi;
  gsl_vector_float *gamma = gsl_vector_float_calloc (G->size);
  gsl_vector_float_memcpy (gamma, G);
  gsl_vector_float_scale (gamma, sigma);

  for (int i = 0; i < x->size; i++)
  {
    Gi = gsl_vector_float_get (gamma, i);
    entry = gsl_vector_float_get (x, i);
    if (entry < -Gi)
    {
      gsl_vector_float_set (x, i, entry + Gi);
    }
    else if (entry > Gi)
    {
      gsl_vector_float_set (x, i, entry - Gi);
    }
    else
      gsl_vector_float_set (x, i, 0);
  }
  gsl_vector_float_free (gamma);
}

float
calcErr (gsl_matrix_float *A)
{
  int m = A->size1;
  int n = A->size2;
  gsl_vector_float *x = gsl_vector_float_calloc (n);
  gsl_vector_float *Ax = gsl_vector_float_calloc (m);

  // approximate ||A||_2 by power method
  float sum_x = 0, err0 = 0, err1;
  float tol = 1e-6, normx, normy;
  int cnt = 0;
  for (int j = 0; j < n; j++)
  {
    gsl_matrix_float_get_col (Ax, A, j);
    sum_x = gsl_blas_sasum (Ax);
    gsl_vector_float_set (x, j, sum_x);
  }
  err1 = gsl_blas_snrm2 (x);
  gsl_vector_float_scale (x, 1.0 / err1);
  while (fabs (err1 - err0) > tol * err1)
  {
    err0 = err1;
    gsl_blas_sgemv (CblasNoTrans, 1, A, x, 0, Ax); // Ax = A*x
    gsl_blas_sgemv (CblasTrans, 1, A, Ax, 0, x);
    normx = gsl_blas_snrm2 (x);
    normy = gsl_blas_snrm2 (Ax);

    err1 = normx / normy;
    gsl_vector_float_scale (x, 1.0 / normx);
    cnt = cnt + 1;
    if (cnt > 100)
    {
      break;
    }
  }

  return err1;
}
