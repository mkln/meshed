
#include "RcppArmadillo.h"

using namespace Rcpp;

arma::mat mvn(int n, const arma::vec& mu, const arma::mat sigma){
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}
