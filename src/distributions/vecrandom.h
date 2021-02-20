#ifndef MSP_VECRND
#define MSP_VECRND

#include "RcppArmadillo.h"
#include "R.h"

using namespace std;


inline arma::vec vrpois(const arma::vec& lambdas){
  arma::vec y = arma::zeros(lambdas.n_elem);
  for(int i=0; i<lambdas.n_elem; i++){
    y(i) = R::rpois(lambdas(i));
  }
  return y;
}

inline arma::vec vrbern(const arma::vec& p){
  arma::vec result = arma::zeros(p.n_elem);
  
  for(int i=0; i<p.n_elem; i++){
    result(i) = R::rbinom(1, p(i));
  }
  return result;
}

inline arma::vec vrbeta(const arma::vec& a, const arma::vec& b){
  arma::vec result = arma::zeros(a.n_elem);
  for(int i=0; i<a.n_elem; i++){
    result(i) = R::rbeta(a(i), b(i));
  }
  return result;
}

#endif