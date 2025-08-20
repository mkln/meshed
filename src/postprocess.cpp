#include <RcppArmadillo.h>
#include "covariance_lmc.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;



//[[Rcpp::export]]
arma::cube crosscov_matfun_h(double h, 
                             const arma::cube& lambda, 
                             const arma::cube& theta, 
                             bool correl=false,
                             int num_threads=1,
                             int dd=2, int matern_twonu_in=1){
  int q = lambda.n_rows;
  int k = lambda.n_cols;
  int m = lambda.n_slices;
  
  MaternParams matern;
  
  int bessel_ws_inc = 5;
  matern.bessel_ws = (double *) R_alloc(num_threads*bessel_ws_inc, sizeof(double));
  matern.twonu = matern_twonu_in;
  matern.using_ps = false;
  matern.estimating_nu = (dd == 2) & (theta.n_rows == 3);
  
  arma::mat x = arma::zeros(1, dd);
  arma::mat y = x;
  
  arma::cube sigma = arma::zeros(q, q, m);
//#ifdef _OPENMP
//#pragma omp parallel for num_threads(num_threads)
//#endif
  for(int i=0; i<m; i++){
    
    y(0) = h;
    arma::mat theta_iter = theta.slice(i);
    
    arma::mat covar = arma::eye(k, k);
    for(int j=0; j<k; j++){
      arma::vec thetaj = theta_iter.col(j);
      arma::mat cmat = Correlationc(x, y, thetaj, matern, false);
      covar(j,j) = cmat(0,0);
    }
    
    arma::mat lambdahere = lambda.slice(i); // * U.t();

    sigma.slice(i) = lambdahere * covar * lambdahere.t();
    
    if(correl){
      arma::vec covar0 = arma::zeros(k);
      for(int j=0; j<k; j++){
        arma::vec thetaj = theta_iter.col(j);
        arma::mat cmat = Correlationc(x, x, thetaj, matern, true);
        covar0(j) = cmat(0,0);
      }
      arma::mat omega = lambdahere * arma::diagmat(covar0) * lambdahere.t();
      
      arma::mat dsigma = arma::diagmat(1.0/sqrt(omega.diag()));
      arma::mat correlmat = dsigma * sigma.slice(i) * dsigma;
      sigma.slice(i) = correlmat;
    }
  }
  return sigma;
}


//[[Rcpp::export]]
arma::cube recover_W_cpp(const arma::cube& V, const arma::cube& L, const arma::uvec& mcmcix){

  arma::cube Lsub = L.slices(mcmcix-1);
  
  int n = V.n_rows;
  int q = L.n_rows;
  int mcmc = V.n_slices;
  
  arma::cube W = arma::zeros(n, q, mcmc);
  
  for(int m=0; m<mcmc; m++){
    W.slice(m) = V.slice(m) * arma::trans( Lsub.slice(m) );
  }
  
  return W;
}

//[[Rcpp::export]]
arma::cube recover_linear_predictor_cpp(const arma::mat& X, const arma::cube& B,
                         const arma::cube& V, const arma::cube& L, const arma::uvec& mcmcix){
  
  arma::cube Bsub = B.slices(mcmcix-1);
  arma::cube Lsub = L.slices(mcmcix-1);
  
  int n = X.n_rows;
  int q = L.n_rows;
  int mcmc = V.n_slices;
  
  arma::cube LP = arma::zeros(n, q, mcmc);
  
  for(int m=0; m<mcmc; m++){
    LP.slice(m) = X * Bsub.slice(m) + V.slice(m) * arma::trans( Lsub.slice(m) );
  }
  
  return LP;
}



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

