#include <RcppArmadillo.h>

using namespace std;

inline arma::mat ps_back(const arma::mat& theta, int d, int nutimes2, bool use_ps=true){
  if(!use_ps){
    return arma::eye(theta.n_cols, theta.n_cols);
  }
  
  arma::mat reparametrizer; 
  if(d == 3){
    // expand variance for spacetime gneiting covariance
    reparametrizer = arma::diagmat(pow(theta.row(3), .5)); 
  } else if (d == 2) {
    if(theta.n_rows > 2){
      // full matern
      arma::vec rdiag = arma::zeros(theta.n_cols);
      for(unsigned int j=0; j<rdiag.n_elem; j++){
        rdiag(j) = pow(theta(0, j), -theta(1, j));
      }
      reparametrizer = 
        arma::diagmat(rdiag) * 
        arma::diagmat(sqrt(theta.row(2))); 
    } else {
      // we use this from mcmc samples because those already have row0 as the transformed param
      // zhang 2004 corollary to theorem 2.
      reparametrizer = 
        arma::diagmat(pow(theta.row(0), -nutimes2/2.0)) * 
        arma::diagmat(sqrt(theta.row(1))); 
    }
  } else {
    // p-kernel
    reparametrizer = arma::diagmat(pow(theta.tail_rows(1), .5));
  }
  return reparametrizer;
}

inline arma::mat ps_forward(const arma::mat& theta, int d, int nutimes2, bool use_ps=true){
  if(!use_ps){
    return arma::eye(theta.n_cols, theta.n_cols);
  }
  
  arma::mat reparametrizer;
  if(d == 3){
    // expand variance for spacetime gneiting covariance
    reparametrizer = arma::diagmat(pow(
      theta.row(3), -0.5)); 
  } else if (d==2) {
    if(theta.n_rows > 2){
      // full matern: builds lambda*phi^nu
      arma::vec rdiag = arma::zeros(theta.n_cols);
      for(unsigned int j=0; j<rdiag.n_elem; j++){
        rdiag(j) = pow(theta(0, j), theta(1, j));
      }
      reparametrizer = 
        arma::diagmat(rdiag) * 
        arma::diagmat(pow(theta.row(2), -0.5)); ; 
    } else {
      // zhang 2004 corollary to theorem 2.
      
      reparametrizer = 
        arma::diagmat(pow(theta.row(0), + nutimes2/2.0)) * 
        arma::diagmat(pow(theta.row(1), -0.5)); 
    }
  } else {
    // p-kernel
    reparametrizer = arma::diagmat(pow(theta.tail_rows(1), -.5));
  }
  return reparametrizer;
}

inline arma::mat reparametrize_lambda_back(const arma::mat& Lambda_in, 
                                           const arma::mat& theta, int d, int nutimes2, bool use_ps=true){
  if(!use_ps){
    return Lambda_in;
  }
  
  arma::mat reparametrizer; 
  if(d == 3){
    // expand variance for spacetime gneiting covariance
    reparametrizer = arma::diagmat(pow(theta.row(3), .5)); 
  } else if (d == 2) {
    if(theta.n_rows > 2){
      // full matern
      arma::vec rdiag = arma::zeros(theta.n_cols);
      for(unsigned int j=0; j<rdiag.n_elem; j++){
        rdiag(j) = pow(theta(0, j), -theta(1, j));
      }
      reparametrizer = 
        arma::diagmat(rdiag) * 
        arma::diagmat(sqrt(theta.row(2))); 
    } else {
      // we use this from mcmc samples because those already have row0 as the transformed param
      // zhang 2004 corollary to theorem 2.
      reparametrizer = 
        arma::diagmat(pow(theta.row(0), -nutimes2/2.0)) * 
        arma::diagmat(sqrt(theta.row(1))); 
    }
  } else {
    // p-kernel
    reparametrizer = arma::diagmat(pow(theta.row(0), .5));
  }
  return Lambda_in * reparametrizer;
}


inline arma::mat reparametrize_lambda_forward(const arma::mat& Lambda_in, 
                                              const arma::mat& theta, int d, int nutimes2, bool use_ps=true){
  if(!use_ps){
    return Lambda_in;
  }
  
  arma::mat reparametrizer;
  if(d == 3){
    // expand variance for spacetime gneiting covariance
    reparametrizer = arma::diagmat(pow(
      theta.row(3), -0.5)); 
  } else if (d==2) {
    if(theta.n_rows > 2){
      // full matern: builds lambda*phi^nu
      arma::vec rdiag = arma::zeros(theta.n_cols);
      for(unsigned int j=0; j<rdiag.n_elem; j++){
        rdiag(j) = pow(theta(0, j), theta(1, j));
      }
      reparametrizer = 
        arma::diagmat(rdiag) * 
        arma::diagmat(pow(theta.row(2), -0.5)); ; 
    } else {
      // zhang 2004 corollary to theorem 2.
      
      reparametrizer = 
        arma::diagmat(pow(theta.row(0), + nutimes2/2.0)) * 
        arma::diagmat(pow(theta.row(1), -0.5)); 
    }
  } else {
    // p-kernel
    reparametrizer = arma::diagmat(pow(theta.row(0), -.5));
  }
  return Lambda_in * reparametrizer;
}
