/*#include "pois_opt.h"

arma::mat pois_reg_ens_lbfgs(const arma::vec& init, 
                             const arma::vec& y, const arma::mat& X, const arma::vec& offset,
                             const arma::vec& mstar, const arma::mat& Vi, int maxit=100) {
  
  arma::vec Xty = X.t() * y;
  // Construct the first objective function.
  PoisReg poisf(y, X, offset, mstar, Vi, Xty);
  
  // Create the L_BFGS optimizer with default parameters.
  // The ens::L_BFGS type can be replaced with any ensmallen optimizer that can
  // handle differentiable functions.
  ens::L_BFGS lbfgs;
  
  lbfgs.MaxIterations() = maxit;
  
  // Create a starting point for our optimization randomly.
  // The model has p parameters, so the shape is p x 1.
  arma::mat beta = init;// arma::randn(X.n_cols, 1);
  
  // Run the optimization
  lbfgs.Optimize(poisf, beta);
  
  return beta;
}*/

