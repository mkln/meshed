#include <RcppArmadillo.h>

inline double check_symmetric(const arma::mat& temp){
  return arma::accu( abs( arma::trimatl(temp) - arma::trans(arma::trimatu(temp)) ) );
}

