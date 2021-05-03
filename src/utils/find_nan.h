#include <RcppArmadillo.h>

using namespace std;

arma::field<arma::mat> find_not_nan(const arma::field<arma::mat>& infield, 
                                        const arma::field<arma::mat>& filtering);

arma::field<arma::mat> find_nan(const arma::field<arma::mat>& infield, 
                                         const arma::field<arma::mat>& filtering);
