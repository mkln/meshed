#ifndef MGP_UTILS 
#define MGP_UTILS

#include "RcppArmadillo.h"


using namespace std;


bool compute_block(bool predicting, int block_ct, bool rfc);


struct MeshDataLMC {
  arma::mat theta; 
  
  arma::field<arma::cube> Kxxi_cache;
  
  arma::field<arma::cube> w_cond_mean_K;
  arma::field<arma::cube> w_cond_prec;
  //arma::field<arma::cube> w_cond_precchol;
  arma::field<arma::cube> w_cond_prec_times_cmk;
  
  arma::vec logdetCi_comps;
  double logdetCi;
  
  arma::mat wcore; 
  arma::mat loglik_w_comps;
  arma::cube nr_wcore;
  arma::vec nr_ldetci;
  
  arma::vec ll_y;
  
  double loglik_w; // will be pml
  double ll_y_all;
  
  arma::field<arma::cube> Hproject; // moves from w to observed coords
  arma::field<arma::cube> Rproject; // stores R for obs
  //arma::field<arma::field<arma::mat> > LambdaH; //***
  arma::mat Ddiag; //***
  //arma::field<arma::mat> KcxKxxi_obs; // ***
  
  // w cache
  arma::field<arma::mat> Sigi_chol;
  arma::field<arma::mat> Smu_start;
  
  arma::field<arma::field<arma::cube> > AK_uP;
  //arma::field<arma::field<arma::mat> > LambdaH_Ditau; // for forced grids;
};


arma::vec drowcol_uv(const arma::field<arma::uvec>& diag_blocks);

arma::uvec field_v_concat_uv(arma::field<arma::uvec> const& fuv);

arma::mat rwishart(unsigned int df, const arma::mat& S);

arma::mat rinvwishart(unsigned int df, const arma::mat& S);

#endif

