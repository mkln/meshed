#ifndef MSP_VECRND
#define MSP_VECRND

#include "RcppArmadillo.h"
#include "R.h"

using namespace std;


inline arma::mat mrstdnorm(int r, int c){
  arma::mat result = arma::zeros(r, c);
  for(int i=0; i<r; i++){
    for(int j=0; j<c; j++){
      result(i,j) = R::rnorm(0,1);
    }
  }
  return result;
}

inline arma::vec vrpois(const arma::vec& lambdas){
  arma::vec y = arma::zeros(lambdas.n_elem);
  for(unsigned int i=0; i<lambdas.n_elem; i++){
    y(i) = R::rpois(lambdas(i));
  }
  return y;
}

inline arma::vec vrunif(int n){
  arma::vec result = arma::zeros(n);
  for(int i=0; i<n; i++){
    result(i) = R::runif(0, 1);
  }
  return result;
}

inline arma::vec vrbern(const arma::vec& p){
  arma::vec result = arma::zeros(p.n_elem);
  
  for(unsigned int i=0; i<p.n_elem; i++){
    result(i) = R::rbinom(1, p(i));
  }
  return result;
}

inline arma::vec vrbeta(const arma::vec& a, const arma::vec& b){
  arma::vec result = arma::zeros(a.n_elem);
  for(unsigned int i=0; i<a.n_elem; i++){
    result(i) = R::rbeta(a(i), b(i));
  }
  return result;
}

inline arma::vec vrnb(const arma::vec& mu, double alpha){
  arma::vec result = arma::zeros(mu.n_elem);
  for(unsigned int i=0; i<mu.n_elem; i++){
    // alpha = 1/scale
    // rgamma defaults to using scale
    // mean = alpha * 1/alpha = 1
    // variance = alpha * 1/alpha^2 = 1/alpha = tausq
    double mixed = mu(i) * R::rgamma(alpha, 1.0/alpha);
    result(i) = R::rpois(mixed);
  }
  return result;
}


inline arma::mat mvn(int n, const arma::vec& mu, const arma::mat sigma){
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}


#endif
