#ifndef MSP_UTILS 
#define MSP_UTILS

#include "RcppArmadillo.h"


using namespace std;

const double hl2pi = -.5 * log(2 * M_PI);

// everything that changes during MCMC
struct MeshData {
  
  double nu;
  arma::vec theta; 
  
  arma::vec wcore; 
  arma::field<arma::mat> w_cond_mean_K;
  arma::field<arma::mat> w_cond_prec;
  arma::field<arma::mat> w_cond_cholprec;
  arma::field<arma::mat> w_cond_prec_parents;
  
  arma::vec logdetCi_comps;
  double logdetCi;
  
  arma::vec loglik_w_comps;
  double loglik_w;
  
  arma::uvec track_chol_fails;
  bool cholfail;
  
};

/*
struct MeshData {
  
  arma::vec theta; 
  
  arma::vec wcore; 
  arma::field<arma::mat> w_cond_mean_K;
  arma::field<arma::mat> w_cond_prec;
  arma::field<arma::mat> w_cond_cholprec;
  arma::field<arma::mat> w_cond_prec_parents;
  
  arma::vec logdetCi_comps;
  double logdetCi;
  
  arma::vec loglik_w_comps;
  double loglik_w;
  
  arma::uvec track_chol_fails;
  bool cholfail;
  
};
*/

inline arma::vec armarowsum(const arma::mat& x){
  return arma::sum(x, 1);
}

inline arma::vec armacolsum(const arma::mat& x){
  return arma::trans(arma::sum(x, 0));
}

inline arma::mat diagmultiply(const arma::vec& d, arma::mat X){
  // returns DX where D is diagonal
  for(unsigned int i=0; i<X.n_rows; i++){
    X.row(i) *= d(i);
  }
  return X;
}

inline arma::sp_mat Zify(const arma::mat& x) {
  //x: list of matrices 
  unsigned int n = x.n_rows;
  int rdimen = 0;
  int cdimen = 0;
  
  arma::ivec dimrow(n);
  arma::ivec dimcol(n);
  
  for(unsigned int i=0; i<n; i++) {
    dimrow(i) = 1;
    dimcol(i) = x.n_cols;
    rdimen += dimrow(i);
    cdimen += dimcol(i);
  }
  
  arma::mat X = arma::zeros(rdimen, cdimen);
  int idx=0;
  int cdx=0;
  
  for(unsigned int i=0; i<n; i++){
    arma::uvec store_rows = arma::regspace<arma::uvec>(idx, idx + dimrow(i) - 1);
    arma::uvec store_cols = arma::regspace<arma::uvec>(cdx, cdx + dimcol(i) - 1);
    
    X(store_rows, store_cols) = x.row(i);
    
    idx = idx + dimrow(i);
    cdx = cdx + dimcol(i);
  }
  
  return arma::conv_to<arma::sp_mat>::from(X);
}



inline void print_data(const MeshData& data){
  Rcpp::Rcout << "---- TMeshData ----" << endl; 
  Rcpp::Rcout << "phi: " << data.theta(0) << " " << data.theta(1) << endl;
  Rcpp::Rcout << "wcore sum: " << arma::accu(data.wcore) << endl;
  Rcpp::Rcout << "logdetCi: " << data.logdetCi << endl;
  Rcpp::Rcout << "loglik_w: " << data.loglik_w << endl;
  Rcpp::Rcout << "------------------" << endl;
}

inline arma::vec kdiagchol(const arma::mat& Koo, const arma::mat& Kos, const arma::mat& Kssi){
  arma::vec temp = arma::sum( ( Kos * Kssi ) % Kos, 1);
  return sqrt( 1.0/( Koo.diag() - temp ) );
}



#endif

