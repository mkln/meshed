#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>

void fwdconditional_mvn(double& logtarget, arma::vec& gradient, 
                        const arma::mat& x, const arma::cube* Ri, const arma::mat& Kcxpar);
  
double fwdcond_dmvn(const arma::mat& x, 
                   const arma::cube* Ri,
                   const arma::mat& Kcxpar);

arma::vec grad_fwdcond_dmvn(const arma::mat& x, 
                                   const arma::cube* Ri,
                                   const arma::mat& Kcxpar);

double bwdcond_dmvn(const arma::mat& x, 
                           const arma::mat& w_child,
                           const arma::cube* Ri_of_child,
                           const arma::cube& Kcx_x,
                           const arma::mat& Kco_wo);

arma::vec grad_bwdcond_dmvn(const arma::mat& x, 
                                   const arma::mat& w_child,
                                   const arma::cube* Ri_of_child,
                                   const arma::cube& Kcx_x,
                                   const arma::mat& Kco_wo);

void bwdconditional_mvn(double& xtarget, arma::vec& gradient, 
                        const arma::mat& x, 
                        const arma::mat& w_child,
                        const arma::cube* Ri_of_child,
                        const arma::cube& Kcx_x,
                        const arma::mat& Kco_wo);

void neghess_fwdcond_dmvn(arma::mat& result, const arma::mat& x, 
                                   const arma::cube* Ri);

void neghess_bwdcond_dmvn(arma::mat& result, const arma::mat& x, 
                                   const arma::mat& w_child,
                                   const arma::cube* Ri_of_child,
                                   const arma::cube& Kcx_x);

void mvn_dens_grad_neghess(double& xtarget, arma::vec& gradient, arma::mat& neghess, const arma::mat& x, 
                           const arma::mat& w_child, const arma::cube* Ri_of_child,
                           const arma::cube& Kcx_x, const arma::mat& Kco_wo);