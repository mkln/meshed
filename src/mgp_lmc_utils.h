#ifndef MGP_UTILS 
#define MGP_UTILS

#include "RcppArmadillo.h"


using namespace std;

bool compute_block(bool predicting, int block_ct, bool rfc);


struct MeshDataUni {
  arma::mat theta; 
  
  arma::field<arma::mat> Kxxi_cache;
  
  arma::field<arma::mat> w_cond_mean_K;
  arma::field<arma::mat> w_cond_prec;
  arma::field<arma::mat> w_cond_prec_times_cmk;
  
  arma::vec logdetCi_comps;
  double logdetCi;
  
  arma::mat wcore; 
  arma::mat loglik_w_comps;
  
  arma::vec ll_y;
  
  double loglik_w; // will be pml
  double ll_y_all;
  
  arma::field<arma::cube> Hproject; // moves from w to observed coords
  arma::field<arma::cube> Rproject; // stores R for obs
  arma::field<arma::cube> Riproject;
  
  arma::cube DplusSi;
  arma::cube DplusSi_c;
  arma::vec DplusSi_ldet;
  
  // w cache
  arma::field<arma::mat> Sigi_chol;
  arma::field<arma::mat> Smu_start;
  
  arma::field<arma::field<arma::mat> > AK_uP;
  //arma::field<arma::field<arma::mat> > LambdaH_Ditau; // for forced grids;
};

struct MeshDataLMC {
  arma::mat theta; 
  
  arma::field<arma::cube> Kxxi_cache;
  
  arma::field<arma::cube> H_cache;
  arma::field<arma::cube> Ri_cache;
  
  std::vector<arma::cube *> w_cond_prec_ptr;
  std::vector<arma::cube *> w_cond_mean_K_ptr;
    
  arma::vec logdetCi_comps;
  double logdetCi;
  
  arma::mat wcore; 
  arma::mat loglik_w_comps;
  
  arma::vec ll_y;
  
  double loglik_w; // will be pml
  double ll_y_all;
  
  arma::field<arma::cube> Hproject; // moves from w to observed coords
  arma::field<arma::cube> Rproject; // stores R for obs
  arma::field<arma::cube> Riproject;
  
  arma::cube DplusSi;
  arma::cube DplusSi_c;
  arma::vec DplusSi_ldet;
  
  // w cache
  arma::field<arma::mat> Sigi_chol;
  arma::field<arma::mat> Smu_start;
  
  arma::field<arma::field<arma::cube> > AK_uP;
  //arma::field<arma::field<arma::mat> > LambdaH_Ditau; // for forced grids;
};


arma::vec drowcol_uv(const arma::field<arma::uvec>& diag_blocks);

arma::uvec field_v_concat_uv(arma::field<arma::uvec> const& fuv);

void block_invcholesky_(arma::mat& X, const arma::uvec& upleft_cumblocksizes);

void add_smu_parents_(arma::mat& result, 
                             const arma::cube& condprec,
                             const arma::cube& cmk,
                             const arma::mat& wparents);


arma::cube AKuT_x_R(arma::cube& result, const arma::cube& x, const arma::cube& y);



void add_AK_AKu_multiply_(arma::mat& result,
                                 const arma::cube& x, const arma::cube& y);

arma::mat AK_vec_multiply(const arma::cube& x, const arma::mat& y);

void add_lambda_crossprod(arma::mat& result, const arma::mat& X, int j, int q, int k, int blocksize);

void add_LtLxD(arma::mat& result, const arma::mat& LjtLj, const arma::vec& Ddiagvec);


arma::mat build_block_diagonal(const arma::cube& x);

arma::cube cube_cols(const arma::cube& x, const arma::uvec& sel_cols);

arma::mat ortho(const arma::mat& x);

// ptr versions
arma::cube cube_cols_ptr(const arma::cube* x, const arma::uvec& sel_cols);
arma::cube AKuT_x_R_ptr(arma::cube& result, const arma::cube& x, const arma::cube* y);
void add_smu_parents_ptr_(arma::mat& result, 
                          const arma::cube* condprec,
                          const arma::cube* cmk,
                          const arma::mat& wparents);
arma::mat build_block_diagonal_ptr(const arma::cube* x);

#endif

