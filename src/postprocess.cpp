#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

//[[Rcpp::export]]
arma::cube cube_tcrossprod(const arma::cube& x){
  arma::cube result = arma::zeros(x.n_rows, x.n_rows, x.n_slices);
  
  for(int i=0; i<x.n_slices; i++){
    result.slice(i) = x.slice(i) * x.slice(i).t();
  }
  return result;
}