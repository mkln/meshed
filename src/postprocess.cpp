#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

//[[Rcpp::export]]
arma::cube cube_tcrossprod(const arma::cube& x){
  arma::cube result = arma::zeros(x.n_rows, x.n_rows, x.n_slices);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<x.n_slices; i++){
    result.slice(i) = x.slice(i) * x.slice(i).t();
  }
  return result;
}

//[[Rcpp::export]]
arma::cube cube_correl_from_lambda(const arma::cube& lambda_mcmc){
  int q = lambda_mcmc.n_rows;
  int k = lambda_mcmc.n_cols;
  int m = lambda_mcmc.n_slices;
  
  arma::cube llt = arma::zeros(q, q, m);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<m; i++){
    llt.slice(i) = lambda_mcmc.slice(i) * arma::trans(lambda_mcmc.slice(i));
    arma::mat dllt = arma::diagmat(1.0/sqrt( llt.slice(i).diag() ));
    arma::mat cc = dllt * llt.slice(i) * dllt;
    llt.slice(i) = cc;
  }
  return llt;
}

//[[Rcpp::export]]
arma::mat summary_list_mean(const arma::field<arma::mat>& x, int n_threads=1){
  // all matrices in x must be the same size.
  int nrows = x(0).n_rows;
  int ncols = x(0).n_cols;
  
  arma::mat result = arma::zeros(nrows, ncols);
  
  // check how many list elements are nonempty
  int n = 0;
  for(unsigned int i=0; i<x.n_elem; i++){
    if(x(i).n_rows > 0){
      n ++;
    }
  }
  
#ifdef _OPENMP
  omp_set_num_threads(n_threads);
#endif
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int j=0; j<nrows*ncols; j++){
    arma::vec slices = arma::zeros(n);
    for(int i=0; i<n; i++){
        // we have stored something here
        slices(i) = x(i)(j);
    }
    result(j) = arma::mean(slices);
  }
  return result;
}

void prctile_stl(double* in, const int &len, const double &percent, std::vector<double> &range) {
  double r = (percent / 100.) * len;
  double lower = 0;
  double upper = 0;
  double* min_ptr = NULL;
  int k = 0;
  
  if(r >= len / 2.) {    
    int idx_lo = max(r - 1, (double) 0.);
    nth_element(in, in + idx_lo, in + len);            
    lower = in[idx_lo];
    if(idx_lo < len - 1) {
      min_ptr = min_element(&(in[idx_lo + 1]), in + len);
      upper = *min_ptr;
    }
    else
      upper = lower;
  } else {                  
    double* max_ptr;
    int idx_up = ceil(max(r - 1, (double) 0.));
    nth_element(in, in + idx_up, in + len);             
    upper = in[idx_up];
    if(idx_up > 0) {
      max_ptr = max_element(in, in + idx_up);
      lower = *max_ptr;
    }
    else
      lower = upper;
  }
  // Linear interpolation
  k = r + 0.5;        // Implicit floor
  r = r - k;
  range[1] = (0.5 - r) * lower + (0.5 + r) * upper;
  
  min_ptr = min_element(in, in + len);
  range[0] = *min_ptr;
}

double cqtile(arma::vec& v, double q){
  int n = v.n_elem;
  double* a = v.memptr();
  std::vector<double> result(2);
  prctile_stl(a, n, q*100.0, result);
  return result.at(1);
}

//[[Rcpp::export]]
arma::mat summary_list_q(const arma::field<arma::mat>& x, double q, int n_threads=1){
  // all matrices in x must be the same size.
  int nrows = x(0).n_rows;
  int ncols = x(0).n_cols;
  
  arma::mat result = arma::zeros(nrows, ncols);
  
  // check how many list elements are nonempty
  int n = 0;
  for(unsigned int i=0; i<x.n_elem; i++){
    if(x(i).n_rows > 0){
      n ++;
    }
  }
  
#ifdef _OPENMP
  omp_set_num_threads(n_threads);
#endif
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int j=0; j<nrows*ncols; j++){
    arma::vec slices = arma::zeros(n);
    for(int i=0; i<n; i++){
      slices(i) = x(i)(j);
    }
    result(j) = cqtile(slices, q);
  }
  return result;
}

