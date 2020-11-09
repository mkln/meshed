#ifndef MGP_UTILS 
#define MGP_UTILS

#include "RcppArmadillo.h"


using namespace std;


bool compute_block(bool predicting, int block_ct, bool rfc);


struct MeshDataLMC {
  arma::mat theta; 
  
  arma::field<arma::cube> w_cond_mean_K;
  arma::field<arma::cube> w_cond_prec;
  
  arma::vec logdetCi_comps;
  double logdetCi;
  
  arma::mat wcore; 
  arma::mat loglik_w_comps;
  
  arma::vec ll_y;
  
  double loglik_w; // will be pml
  double ll_y_all;
  
  arma::field<arma::cube> Hproject; // moves from w to observed coords
  arma::field<arma::cube> Rproject; // stores R for obs
  //arma::field<arma::field<arma::mat> > LambdaH; //***
  arma::mat Ddiag; //***
  //arma::field<arma::mat> KcxKxxi_obs; // ***
  
};


arma::vec drowcol_uv(const arma::field<arma::uvec>& diag_blocks);

arma::uvec field_v_concat_uv(arma::field<arma::uvec> const& fuv);
  

#endif

