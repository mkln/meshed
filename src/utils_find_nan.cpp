#include "utils_find_nan.h"

arma::field<arma::mat> find_not_nan(const arma::field<arma::mat>& infield, 
                                           const arma::field<arma::mat>& filtering){
  
  arma::field<arma::mat> filtered = arma::field<arma::mat>(infield.n_elem);
  
  for(unsigned int i=0; i<infield.n_elem; i++){
    filtered(i) = infield(i).rows(arma::find_finite(filtering(i).col(0)));
  }
  
  return filtered;
  
}

arma::field<arma::mat> find_nan(const arma::field<arma::mat>& infield, 
                                       const arma::field<arma::mat>& filtering){
  
  arma::field<arma::mat> filtered = arma::field<arma::mat>(infield.n_elem);
  
  for(unsigned int i=0; i<infield.n_elem; i++){
    filtered(i) = infield(i).rows(arma::find_nonfinite(filtering(i).col(0)));
  }
  return filtered;
}
