
#include "mvnormal.h"

double fwdcond_dmvn(const arma::mat& x, 
                           const arma::cube* Ri,
                           const arma::mat& Kcxpar){
  // conditional of x | parents
  
  double numer = 0;
  for(int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(Kcxpar.n_cols > 0){ // meaning w_parents.n_rows > 0
      xcentered -= Kcxpar.col(j);
    } 
    numer += arma::conv_to<double>::from( xcentered.t() * (*Ri).slice(j) * xcentered );
  }
  return -.5 * numer;//result;
}

arma::vec grad_fwdcond_dmvn(const arma::mat& x, 
                                   const arma::cube* Ri,
                                   const arma::mat& Kcxpar){
  
  // gradient of conditional of x | parents
  arma::mat norm_grad = arma::zeros(arma::size(x));
  for(int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(Kcxpar.n_cols > 0){ // meaning w_parents.n_rows > 0
      xcentered -= Kcxpar.col(j);
    } 
    norm_grad.col(j) = -(*Ri).slice(j) * xcentered;
  }
  return arma::vectorise(norm_grad);//result;
}

void fwdconditional_mvn(double& logtarget, arma::vec& gradient, 
                        const arma::mat& x, const arma::cube* Ri, const arma::mat& Kcxpar){
  arma::mat norm_grad = arma::zeros(arma::size(x));
  double numer = 0;
  for(int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(Kcxpar.n_cols > 0){ // meaning w_parents.n_rows > 0
      xcentered -= Kcxpar.col(j);
    } 
    arma::vec Rix = (*Ri).slice(j) * xcentered;
    numer += arma::conv_to<double>::from( xcentered.t() * Rix );
    norm_grad.col(j) = - Rix;
  }
  logtarget = -.5 * numer;//result;
  gradient = arma::vectorise(norm_grad);//result;
}



double bwdcond_dmvn(const arma::mat& x, 
                           const arma::mat& w_child,
                           const arma::cube* Ri_of_child,
                           const arma::cube& Kcx_x,
                           const arma::mat& Kco_wo){
  // conditional of Y | x, others
  
  double numer = 0;
  for(int j=0; j<x.n_cols; j++){
    arma::vec xcentered = w_child.col(j) - Kcx_x.slice(j)*x.col(j);
    if(Kco_wo.n_cols > 0){
      xcentered -= Kco_wo.col(j);
    } 
    numer += arma::conv_to<double>::from(xcentered.t() * (*Ri_of_child).slice(j) * xcentered);
  }
  
  return -0.5*numer;
}

arma::vec grad_bwdcond_dmvn(const arma::mat& x, 
                                   const arma::mat& w_child,
                                   const arma::cube* Ri_of_child,
                                   const arma::cube& Kcx_x,
                                   const arma::mat& Kco_wo){
  // gradient of conditional of Y | x, others
  arma::mat result = arma::zeros(arma::size(x));
  for(int j=0; j<x.n_cols; j++){
    arma::mat wccenter = w_child.col(j) - Kcx_x.slice(j) * x.col(j);
    if(Kco_wo.n_cols > 0){
      wccenter -= Kco_wo.col(j);
    } 
    result.col(j) = Kcx_x.slice(j).t() * (*Ri_of_child).slice(j) * wccenter;
  }
  return arma::vectorise(result);
}

void bwdconditional_mvn(double& xtarget, arma::vec& gradient, const arma::mat& x, 
      const arma::mat& w_child, const arma::cube* Ri_of_child,
      const arma::cube& Kcx_x, const arma::mat& Kco_wo){
  
  arma::mat result = arma::zeros(arma::size(x));
  double numer = 0;
  for(int j=0; j<x.n_cols; j++){
    arma::vec xcentered = w_child.col(j) - Kcx_x.slice(j)*x.col(j);
    if(Kco_wo.n_cols > 0){
      xcentered -= Kco_wo.col(j);
    } 
    arma::vec Rix = (*Ri_of_child).slice(j) * xcentered;
    numer += arma::conv_to<double>::from(xcentered.t() * Rix);
    result.col(j) = Kcx_x.slice(j).t() * Rix;
  }
  xtarget -= 0.5*numer;
  gradient += arma::vectorise(result);
}

void neghess_fwdcond_dmvn(arma::mat& result, const arma::mat& x, 
                                      const arma::cube* Ri){
  
  int k = (*Ri).n_slices;
  int nr = (*Ri).n_rows;
  int nc = (*Ri).n_cols;
  
  //Rcpp::Rcout << "neghess_fwdcond_dmvn " << arma::size((*Ri)) << "\n";
  //arma::mat result = arma::zeros(nr * k, nc * k);
  for(int j=0; j<k; j++){
    result.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) += (*Ri).slice(j);
  }
  //return result;
}

void neghess_bwdcond_dmvn(arma::mat& result,
    const arma::mat& x, const arma::mat& w_child,
    const arma::cube* Ri_of_child, const arma::cube& Kcx_x){
  
  int k = (*Ri_of_child).n_slices;
  int nr = Kcx_x.n_cols; //(*Ri_of_child).n_rows;
  int nc = Kcx_x.n_cols; //(*Ri_of_child).n_cols;
  
  //Rcpp::Rcout << "neghess_bwdcond_dmvn " << arma::size(Kcx_x) << " " << arma::size((*Ri_of_child)) << "\n";
  
  //arma::mat result = arma::zeros(nr * k, nc * k);
  for(int j=0; j<k; j++){
    result.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) += Kcx_x.slice(j).t() * (*Ri_of_child).slice(j) * Kcx_x.slice(j);
  }
  
  //return result;
}

void mvn_dens_grad_neghess(double& xtarget, arma::vec& gradient, arma::mat& neghess,
                           const arma::mat& x, const arma::mat& w_child, const arma::cube* Ri_of_child,
                           const arma::cube& Kcx_x, const arma::mat& Kco_wo){

  
  int k = (*Ri_of_child).n_slices;
  int nr = Kcx_x.n_cols; //(*Ri_of_child).n_rows;
  int nc = Kcx_x.n_cols; //(*Ri_of_child).n_cols;
  
  arma::mat result = arma::zeros(arma::size(x));
  double numer = 0;
  for(int j=0; j<k; j++){
    arma::vec xcentered = w_child.col(j) - Kcx_x.slice(j)*x.col(j);
    if(Kco_wo.n_cols > 0){
      xcentered -= Kco_wo.col(j);
    } 
    arma::mat KRichild = Kcx_x.slice(j).t() * (*Ri_of_child).slice(j);
    numer += arma::conv_to<double>::from(xcentered.t() * (*Ri_of_child).slice(j) * xcentered);
    result.col(j) = KRichild * xcentered;
    neghess.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) += KRichild * Kcx_x.slice(j);
  }
  xtarget -= 0.5*numer;
  gradient += arma::vectorise(result);
  
}
