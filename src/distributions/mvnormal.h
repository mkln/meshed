#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>


inline double fwdcond_dmvn(const arma::mat& x, 
                           const arma::cube& Ri,
                           const arma::vec& parKxxpar, 
                           const arma::mat& Kcxpar){
  // conditional of x | parents
  
  double numer = 0;
  for(int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(parKxxpar(j) > -1){ // meaning w_parents.n_rows > 0
      xcentered -= Kcxpar.col(j);
    } 
    numer += arma::conv_to<double>::from( xcentered.t() * Ri.slice(j) * xcentered );
  }
  return -.5 * numer;//result;
}

inline arma::vec grad_fwdcond_dmvn(const arma::mat& x, 
                                   const arma::cube& Ri,
                                   const arma::vec& parKxxpar, 
                                   const arma::mat& Kcxpar){
  
  // gradient of conditional of x | parents
  arma::mat norm_grad = arma::zeros(arma::size(x));
  for(int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(parKxxpar(j) > -1){ // meaning w_parents.n_rows > 0
      xcentered -= Kcxpar.col(j);
    } 
    norm_grad.col(j) = -Ri.slice(j) * xcentered;
  }
  return arma::vectorise(norm_grad);//result;
}

inline double bwdcond_dmvn(const arma::mat& x, 
                           const arma::mat& w_child,
                           const arma::cube& Ri_of_child,
                           const arma::cube& Kcx_x,
                           const arma::cube& Kxxi_x,
                           const arma::mat& Kxo_wo,
                           const arma::mat& Kco_wo,
                           const arma::vec& woKoowo,
                           //double nu,
                           double num_par){
  // conditional of Y | x, others
  
  double numer = 0;
  for(int j=0; j<x.n_cols; j++){
    arma::vec xcentered = w_child.col(j) - Kcx_x.slice(j)*x.col(j);
    if(woKoowo(j) > -1){
      xcentered -= Kco_wo.col(j);
    } 
    numer += arma::conv_to<double>::from(xcentered.t() * Ri_of_child.slice(j) * xcentered);
  }
  
  return -0.5*numer;
}

inline arma::vec grad_bwdcond_dmvn(const arma::mat& x, 
                                   const arma::mat& w_child,
                                   const arma::cube& Ri_of_child,
                                   const arma::cube& Kcx_x,
                                   const arma::cube& Kxxi_x,
                                   const arma::mat& Kxo_wo,
                                   const arma::mat& Kco_wo,
                                   const arma::vec& woKoowo,
                                   //double nu,
                                   double num_par){
  // gradient of conditional of Y | x, others
  arma::mat result = arma::zeros(arma::size(x));
  for(int j=0; j<x.n_cols; j++){
    arma::mat wccenter = w_child.col(j) - Kcx_x.slice(j) * x.col(j);
    if(woKoowo(j) > -1){
      wccenter -= Kco_wo.col(j);
    }
    result.col(j) = Kcx_x.slice(j).t() * Ri_of_child.slice(j) * wccenter;
  }
  return arma::vectorise(result);
}

inline arma::mat neghess_fwdcond_dmvn(const arma::mat& x, 
                                   const arma::cube& Ri,
                                   const arma::vec& parKxxpar, 
                                   const arma::mat& Kcxpar){
  
  int k = Ri.n_slices;
  int nr = Ri.n_rows;
  int nc = Ri.n_cols;
  
  arma::mat result = arma::zeros(nr * k, nc * k);
  for(int j=0; j<k; j++){
    result.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) = Ri.slice(j);
  }
  return result;
}

inline arma::mat neghess_bwdcond_dmvn(const arma::mat& x, 
                                   const arma::mat& w_child,
                                   const arma::cube& Ri_of_child,
                                   const arma::cube& Kcx_x,
                                   const arma::cube& Kxxi_x,
                                   const arma::mat& Kxo_wo,
                                   const arma::mat& Kco_wo,
                                   const arma::vec& woKoowo,
                                   //double nu,
                                   double num_par){
  int k = Ri_of_child.n_slices;
  int nr = Ri_of_child.n_rows;
  int nc = Ri_of_child.n_cols;
  
  arma::mat result = arma::zeros(nr * k, nc * k);
  for(int j=0; j<k; j++){
    result.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) = Kcx_x.slice(j).t() * Ri_of_child.slice(j) * Kcx_x.slice(j);
  }
  
  return result;
}

