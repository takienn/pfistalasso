#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <omp.h>
#include <mex.h>
#include <string.h>

#include "matrix.h"

void shrink(gsl_vector_float *x, gsl_vector_float *G, float sigma);
//double objective(gsl_vector *x, double lambda, gsl_vector *z);
float calcErr(gsl_matrix_float *A);
int pFistaLasso(gsl_matrix_float *A, gsl_matrix_float *b, gsl_vector_float *G, gsl_matrix_float *X, unsigned int nthreads);

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
	if(nlhs != 1) {
	    mexErrMsgTxt("One output required.");
	}

	if(nrhs != 4) {
	    mexErrMsgTxt("Four inputs required.");
	}

	if( !mxIsSingle(prhs[0]) ||
	     mxIsComplex(prhs[0])) {
		mexErrMsgTxt("Input A matrix must be type single.");
	}

	if( !mxIsSingle(prhs[1]) ||
	     mxIsComplex(prhs[1])) {
		mexErrMsgTxt("Input b matrix must be type single.");
	}

	if( !mxIsSingle(prhs[2]) ||
	     mxIsComplex(prhs[2])) {
		mexErrMsgTxt("Input G matrix must be type single.");
	}

	if( !mxIsDouble(prhs[3]) ||
	     mxIsComplex(prhs[3]) ||
	     mxGetNumberOfElements(prhs[3]) != 1 ) {
	    mexErrMsgTxt("nthreads must be a scalar.");
	}
	unsigned int nthreads = mxGetScalar(prhs[3]);

	float *Adata = (float*)mxGetData(prhs[0]);
	float *Bdata = (float*)mxGetData(prhs[1]);
	float *Gdata = (float*)mxGetData(prhs[2]);


	mwSize mA = mxGetM(prhs[0]);
	mwSize nA = mxGetN(prhs[0]);
	mwSize mB = mxGetM(prhs[1]);
	mwSize nB = mxGetN(prhs[1]);
	mwSize mG = mxGetM(prhs[2]);
//	mwSize nG = mxGetN(prhs[2]);



	int row, col;
	float entry;
    gsl_matrix_float *A = gsl_matrix_float_calloc(mA, nA);
    for (int i = 0; i < mA*nA; i++) {
      row = i % mA;
      col = floor(i/mA);
      entry = Adata[i];
      gsl_matrix_float_set(A, row, col, entry);
    }

    gsl_matrix_float *B = gsl_matrix_float_calloc(mB, nB);
    for (int i = 0; i < mB*nB; i++) {
      row = i % mB;
      col = floor(i/mB);
      entry = Bdata[i];
      gsl_matrix_float_set(B, row, col, entry);
    }

    gsl_vector_float *G = gsl_vector_float_calloc(mG);
    for (int i = 0; i < mG; i++) {
      gsl_vector_float_set(G, i, Gdata[i]);
    }

    mwSize dims[2];
    dims[0] = nB;
    dims[1] = nA;

	plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);

	mwSize nX = mxGetM(plhs[0]);
	mwSize mX = mxGetN(plhs[0]);

	gsl_matrix_float *X = gsl_matrix_float_calloc(mX, nX);

	pFistaLasso(A, B, G, X, nthreads);

	memcpy(mxGetData(plhs[0]), (void *)X->data, mX*nX*sizeof(float));

	gsl_matrix_float_free(X);
	gsl_matrix_float_free(A);
	gsl_matrix_float_free(B);
	gsl_vector_float_free(G);
}

int pFistaLasso(gsl_matrix_float *A, gsl_matrix_float *b, gsl_vector_float *G, gsl_matrix_float *X, unsigned int nthreads) {
  
  const int MAX_ITER      = 50; // number of iteration
//  const double TOL        = 1e-4; // tolerence
//  const double lambda_tol = 1e-4; // tolerence for update lambda

  float delta;
  float lambda = 0.01;

  float err1 = calcErr(A);
  delta = 1.00/(err1*err1);

  printf("Running pFISTA for LASSO\n");

#pragma omp parallel for schedule(static) num_threads(nthreads) shared(X) shared(delta) shared(G) shared(lambda)
  for(int i = 0; i < b->size2; ++i)
  {

	  int m, n;
	  m = A->size1;
	  n = A->size2;

	  // These are all variables related to FISTA
	  gsl_vector_float *x      = gsl_vector_float_calloc(n);
	  gsl_vector_float *u      = gsl_vector_float_calloc(n);
	  gsl_vector_float *xold   = gsl_vector_float_calloc(n);
	  gsl_vector_float *w      = gsl_vector_float_calloc(n);
	  gsl_vector_float *Ax     = gsl_vector_float_calloc(m);
	  gsl_vector_float *bi 	   = gsl_vector_float_calloc(b->size1);

    /*----------------------
     initialize variables
    ----------------------*/
    gsl_vector_float_set_zero(x);
    gsl_vector_float_set_zero(u);
    gsl_vector_float_set_zero(xold);
    gsl_vector_float_set_zero(w);
    gsl_vector_float_set_zero(bi);
    gsl_vector_float_set_zero(Ax);


    float t1=1, t2=1;
    int iter = 0;

    gsl_matrix_float_get_col(bi, b, i);

    /* Main FISTA solver loop */
    while (iter < MAX_ITER) {

		t1 = t2;
		gsl_vector_float_memcpy(xold, x); // copy x to old x;
		gsl_blas_sgemv(CblasNoTrans, 1, A, u, 0, Ax); // A_i x_i = A_i * x_i

		gsl_vector_float_sub(Ax, bi);
		gsl_blas_sgemv(CblasTrans, 1, A, Ax, 0, w); // w = A' (Ax - b)
		gsl_vector_float_scale(w, delta);  // w = delta * w
		gsl_vector_float_sub(u, w);  // u = x - delta * A'(Ax - b)
		gsl_vector_float_memcpy(x, u);
		shrink(x, G, delta * lambda); // shrink(x, alpha*lambda)

		// FISTA
		t2 = 0.5 + 0.5*sqrt(1+4*t1*t1);
		gsl_vector_float_sub(xold, x);
		gsl_vector_float_scale(xold, (1 - t1)/t2);
		gsl_vector_float_memcpy(u, x);
		gsl_vector_float_add(u, xold);

		iter++;
   }

  //Fill solution Matrix X with column solution x

  gsl_matrix_float_set_col(X, i, x);

  gsl_vector_float_free(x);
  gsl_vector_float_free(w);
  gsl_vector_float_free(Ax);
  gsl_vector_float_free(bi);
  gsl_vector_float_free(xold);
  gsl_vector_float_free(u);

  }
  printf("Finished pFISTA for LASSO\n");

  return 0;
}

/* calculate objective function */
//double objective(gsl_vector *x, double lambda, gsl_vector *z) {
//  double obj = 0;
//  double temp =0.0;
//  temp = gsl_blas_snrm2(z);
//  temp = temp*temp/2;
//  double foo;
//  foo = gsl_blas_sasum(x);
//  obj = lambda*foo + temp;
//  return obj;
//}

/* shrinkage function */
void shrink(gsl_vector_float *x, gsl_vector_float *G, float sigma) {
  float entry;
  float Gi;
  gsl_vector_float *gamma = gsl_vector_float_calloc(G->size);
  gsl_vector_float_memcpy(gamma, G);
  gsl_vector_float_scale(gamma, sigma);

  for (int i = 0; i < x->size; i++) {
    Gi = gsl_vector_float_get(gamma, i);
    entry = gsl_vector_float_get(x, i);
    if (entry < - Gi) {
      gsl_vector_float_set(x, i, entry + Gi);
    }
    else if (entry > Gi)
    {
      gsl_vector_float_set(x, i, entry - Gi);
    }
    else
    	gsl_vector_float_set(x, i, 0);
  }
  gsl_vector_float_free(gamma);
}

float calcErr(gsl_matrix_float *A)
{
	int m = A->size1;
	int n = A->size2;
	gsl_vector_float *x = gsl_vector_float_calloc(n);
	gsl_vector_float *Ax = gsl_vector_float_calloc(m);

	// approximate ||A||_2 by power method
	float sum_x = 0, err0 = 0, err1;
	float tol = 1e-6, normx, normy;
	int cnt = 0;
	for (int j = 0; j < n; j++) {
		gsl_matrix_float_get_col(Ax, A, j);
		sum_x = gsl_blas_sasum(Ax);
		gsl_vector_float_set(x, j, sum_x);
	}
	err1 = gsl_blas_snrm2(x);
	gsl_vector_float_scale(x, 1.0 / err1);
	while (fabs(err1 - err0) > tol * err1) {
		err0 = err1;
		gsl_blas_sgemv(CblasNoTrans, 1, A, x, 0, Ax); // Ax = A*x
		gsl_blas_sgemv(CblasTrans, 1, A, Ax, 0, x);
		normx = gsl_blas_snrm2(x);
		normy = gsl_blas_snrm2(Ax);

		err1 = normx / normy;
		gsl_vector_float_scale(x, 1.0 / normx);
		cnt = cnt + 1;
		if (cnt > 100) {
			break;
		}
	}

	return err1;
}
