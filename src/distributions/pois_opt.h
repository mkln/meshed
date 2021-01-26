/* 

#include <RcppArmadillo.h>
//#include <RcppEnsmallen.h>
#include "mesh_utils.h"

class PoisReg {
  public:
    // Construct the object with the given the design 
  // matrix and responses.
  PoisReg(const arma::vec& y, const arma::mat& X, const arma::vec& offset,
          const arma::vec& mstar, const arma::mat& Vi, const arma::vec& Xty) :
    X(X), y(y), mstar(mstar), offset(offset), Vi(Vi), Xty(Xty) { 
    //Xty = X.t() * y;
    //offset = offset_in;
  }
    
  // Return the objective function for model parameters beta.
  double Evaluate(const arma::mat& x)
  {
    double loglike = arma::conv_to<double>::from( 
      y.t() * X * x - arma::accu( exp(offset + X * x) ));
    
    double logprior = arma::conv_to<double>::from(
      x.t() * mstar - .5 * x.t() * Vi * x);
    return -loglike-logprior;//neg loglike
  }
  
  // Compute the gradient for model parameters beta
  void Gradient(const arma::mat& x, arma::mat& g)
  {
    arma::vec grad_loglike = X.t() * (y - exp(offset + X * x));
    arma::vec grad_logprior = mstar - Vi * x;
    g = - grad_loglike - grad_logprior; //neg grad
  }
  
  double EvaluateWithGradient(const arma::mat& x, arma::mat& g)
  {
    arma::vec exp_Off_plus_Xx = exp(offset + X * x);
    arma::vec Vix = Vi * x;
    
    arma::vec grad_loglike = Xty - X.t() * exp_Off_plus_Xx;
    arma::vec grad_logprior = mstar - Vix;
    
    double loglike = arma::conv_to<double>::from( 
      Xty.t() * x - arma::accu( exp_Off_plus_Xx ));
    
    double logprior = arma::conv_to<double>::from(
      x.t() * mstar - .5 * x.t() * Vix);
    
    g = - grad_loglike - grad_logprior; // neg grad
    return -loglike-logprior;           // neg loglike
  }
  
  //arma::vec offset;
    
  private:
    // The design matrix.
  const arma::mat& X;
  // The responses to each data point.
  const arma::vec& y;
  const arma::vec& offset;
  const arma::vec& mstar;
  const arma::mat& Vi;
  const arma::vec& Xty;
};



class PoisReg0m {
public:
  // Construct the object with the given the design 
  // matrix and responses.
  PoisReg0m(const arma::vec& y, const arma::mat& X, const arma::vec& offset,
          const arma::mat& Vi, const arma::vec& Xty) :
  X(X), y(y), offset(offset), Vi(Vi), Xty(Xty) { 
    //Xty = X.t() * y;
    //offset = offset_in;
  }
  
  // Return the objective function for model parameters beta.
  double Evaluate(const arma::mat& x)
  {
    double loglike = arma::conv_to<double>::from( 
      y.t() * X * x - arma::accu( exp(offset + X * x) ));
    
    double logprior = arma::conv_to<double>::from( - .5 * x.t() * Vi * x);
    return -loglike-logprior;//neg loglike
  }
  
  // Compute the gradient for model parameters beta
  void Gradient(const arma::mat& x, arma::mat& g)
  {
    arma::vec grad_loglike = X.t() * (y - exp(offset + X * x));
    arma::vec grad_logprior = - Vi * x;
    g = - grad_loglike - grad_logprior; //neg grad
  }
  
  double EvaluateWithGradient(const arma::mat& x, arma::mat& g)
  {
    arma::vec exp_Off_plus_Xx = exp(offset + X * x);
    arma::vec Vix = Vi * x;
    
    arma::vec grad_loglike = Xty - X.t() * exp_Off_plus_Xx;
    arma::vec grad_logprior = - Vix;
    
    double loglike = arma::conv_to<double>::from( 
      Xty.t() * x - arma::accu( exp_Off_plus_Xx ));
    
    double logprior = arma::conv_to<double>::from(- .5 * x.t() * Vix);
    
    g = - grad_loglike - grad_logprior; // neg grad
    return -loglike-logprior;           // neg loglike
  }
  
  //arma::vec offset;
  
private:
  // The design matrix.
  const arma::mat& X;
  // The responses to each data point.
  const arma::vec& y;
  const arma::vec& offset;
  const arma::mat& Vi;
  const arma::vec& Xty;
};


arma::mat pois_reg_ens_lbfgs(const arma::vec& init,
    const arma::vec& y, const arma::mat& X, const arma::vec& offset,
                             const arma::vec& mstar, const arma::mat& Vi, int maxit);
 

 
 inline arma::mat hess_lpoisson(const arma::vec& x, 
 const arma::vec& y, const arma::mat& X, const arma::vec& offset,
 const arma::vec& mstar, const arma::mat& Vi){
 //int n = y.n_elem;
 //arma::sp_mat C = arma::sp_mat(n, n);
 arma::vec diagC = exp(offset + X*x);
 arma::mat CX = diagmultiply(diagC, X);
 
 return X.t() * CX + Vi; // neg hess, cov
 }
 
 inline arma::vec grad_lpoisson(const arma::vec& x, 
 const arma::vec& y, const arma::mat& X, const arma::vec& offset,
 const arma::vec& mstar, const arma::mat& Vi){
 arma::vec grad_loglike = X.t() * (y - exp(offset + X * x));
 arma::vec grad_logprior = mstar - Vi * x;
 return grad_loglike + grad_logprior;
 }
 
 inline arma::mat invhess_lpoisson(const arma::vec& x, 
 const arma::vec& y, const arma::mat& X, const arma::vec& offset,
 const arma::vec& mstar, const arma::mat& Vi){
 //int n = y.n_elem;
 //arma::sp_mat C = arma::sp_mat(n, n);
 arma::vec diagC = exp(offset + X*x);
 arma::mat CX = diagmultiply(diagC, X);
 
 return arma::inv_sympd(X.t() * CX + Vi); // neg hess, cov
 }
*/
