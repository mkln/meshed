#include "RcppArmadillo.h"

arma::vec check_gibbs_groups(
           arma::vec block_groups,
     const arma::field<arma::vec>& parents,
     const arma::field<arma::vec>& children,
     const arma::vec& block_names,
     const arma::vec& blocks,
     int maxit=10);
