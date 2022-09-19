

#ifndef MESHEDSP 
#define MESHEDSP

// uncomment to disable openmp on compilation
//#undef _OPENMP

#include <RcppArmadillo.h>

//#include "distrib_truncmvnorm.h"
#include "distrib_vecrandom.h"
#include "mcmc_ramadapt.h"
#include "mcmc_hmc_sample.h"
#include "utils_caching.h"
#include "utils_lmc.h"
#include "utils_irls.h"
#include "utils_others.h"
#include "covariance_lmc.h"

class Meshed {
public:
  
  arma::uvec familyid;
  
  // meta
  unsigned int n; // number of locations, total
  unsigned int p; // number of covariates
  unsigned int q; // number of outcomes
  unsigned int k; // number of factors
  unsigned int dd; // dimension
  unsigned int n_blocks; // number of blocks
  
  // data
  arma::mat y;
  arma::mat X;
  arma::mat Z;
  
  arma::mat coords;
  
  arma::uvec reference_blocks; // named
  int n_ref_blocks;
  
  // indexing info
  arma::field<arma::uvec> indexing; 
  arma::field<arma::uvec> indexing_obs;
  arma::field<arma::uvec> parents_indexing; 
  //arma::field<arma::uvec> children_indexing;
  
  // NA data -- indicator vectors
  // at least one of q available
  arma::field<arma::uvec> na_1_blocks; 
  // at least one of q missing
  arma::field<arma::uvec> na_0_blocks; 
  // indices of avails
  arma::field<arma::uvec> na_ix_blocks;
  arma::umat na_mat;
  
  // variable data
  arma::field<arma::uvec> ix_by_q_a; // storing indices using only available data
  
  int n_loc_ne_blocks;
  
  // DAG
  arma::field<arma::sp_mat> Ib;
  arma::field<arma::uvec>   parents; // i = parent block names for i-labeled block (not ith block)
  arma::field<arma::uvec>   children; // i = children block names for i-labeled block (not ith block)
  arma::vec                 block_names; //  i = block name (+1) of block i. all dependence based on this name
  //arma::uvec                ref_block_names;
  arma::vec                 block_groups; // same group = sample in parallel given all others
  arma::vec                 block_ct_obs; // 0 if no available obs in this block, >0=count how many available
  int                       n_gibbs_groups;
  arma::field<arma::vec>    u_by_block_groups;
  int                       predict_group_exists;
  
  arma::uvec                u_predicts;
  arma::vec                 block_groups_labels;
  // for each block's children, which columns of parents of c is u? and which instead are of other parents
  arma::field<arma::field<arma::field<arma::uvec> > > u_is_which_col_f; 
  
  // utils
  arma::uvec oneuv;
  double hl2pi;
    
  // params
  arma::mat yhat;
  arma::mat offsets;
  arma::mat rand_norm_mat;
  arma::vec rand_unif;
  arma::vec rand_unif2;
  // regression
  arma::mat Lambda;
  arma::umat Lambda_mask; // 1 where we want lambda to be nonzero
  arma::mat LambdaHw; // part of the regression mean explained by the latent process
  arma::mat wU; // nonreference locations
  
  arma::mat XB; // by outcome
  arma::mat linear_predictor;
  
  arma::field<arma::mat> XtX;
  arma::mat Vi; 
  arma::mat Vim;
  arma::vec bprim;
  
  arma::mat w;
  arma::mat Bcoeff; // sampled
  
  // covariance info
  int nThreads;
  MaternParams matern;
  
  // setup
  bool predicting;
  bool cached;
  bool forced_grid;
  
  bool verbose;
  bool debug;
  
  // predictions
  arma::field<arma::cube> Hpred;
  arma::field<arma::mat> Rcholpred;
  
  // init / indexing
  void init_indexing();
  void na_study();
  void make_gibbs_groups();
  void init_gibbs_index();
  void init_matern(int num_threads, int matern_twonu_in, bool use_ps);
  void init_for_mcmc();
  void init_cache();
  void init_meshdata(const arma::mat&);
  
  // caching for theta updates
  MeshDataLMC param_data; 
  MeshDataLMC alter_data;
  
  // Theta
  bool refresh_cache(MeshDataLMC& data);
  void update_block_covpars(int u, MeshDataLMC& data);
  void update_block_wlogdens(int, MeshDataLMC& data);
  bool get_loglik_comps_w(MeshDataLMC& data);
  
  // - caching
  arma::uvec coords_caching; 
  arma::uvec coords_caching_ix;
  //arma::uvec parents_caching;
  //arma::uvec parents_caching_ix;
  arma::uvec kr_caching;
  arma::uvec kr_caching_ix;
  arma::uvec cx_and_kr_caching; // merge of coords and kr
  
  arma::uvec findkr;
  arma::uvec findcc;
  
  unsigned int starting_kr;
  

  double logpost;
  
  // changing the values, no sampling;
  //void theta_update(MeshDataLMC&, const arma::mat&);
  void beta_update(const arma::vec&);
  void tausq_update(double);
  
  // RAMA for theta
  void metrop_theta();
  bool theta_adapt_active;
  int theta_mcmc_counter;
  RAMAdapt theta_adapt;
  arma::mat theta_unif_bounds;
  arma::mat theta_metrop_sd;
  void accept_make_change();
  
  // --------------------------------------------------------------- from Gaussian
  

  // tausq 
  arma::vec tausq_ab;
  arma::vec tausq_inv; // tausq for the l=q variables
  
  // MCMC
  // ModifiedPP-like updates for tausq -- used if not forced_grid
  int tausq_mcmc_counter;
  RAMAdapt tausq_adapt;
  arma::mat tausq_unif_bounds;
  
  // tausq for Beta regression
  arma::vec brtausq_mcmc_counter;
  std::vector<RAMAdapt> opt_tausq_adapt;
  
  int lambda_mcmc_counter;
  int n_lambda_pars;
  arma::uvec lambda_sampling;
  arma::mat lambda_unif_bounds; // 1x2: lower and upper for off-diagonal
  RAMAdapt lambda_adapt;
  
  void init_betareg();
  void init_gaussian();
  void update_lly(int, MeshDataLMC&, const arma::mat& LamHw, bool map=false);
  void calc_DplusSi(int, MeshDataLMC& data, const arma::mat& Lam, const arma::vec& tsqi);
  void update_block_w_cache(int, MeshDataLMC& data);
  void refresh_w_cache(MeshDataLMC& data);
  
  // W
  int which_hmc;
  bool w_do_hmc;
  bool w_hmc_nuts;
  bool w_hmc_rm;
  bool w_hmc_srm;
  void deal_with_w(MeshDataLMC& data, bool sample=true);
  void gaussian_w(MeshDataLMC& data, bool sample);
  void gaussian_nonreference_w(int, MeshDataLMC& data, const arma::mat&, bool sample);
  void nongaussian_w(MeshDataLMC& data, bool sample);
  void w_prior_sample(MeshDataLMC& data);
  std::vector<NodeDataW> w_node;
  arma::vec hmc_eps;
  std::vector<AdaptE> hmc_eps_adapt;
  arma::uvec hmc_eps_started_adapting;
  
  
  
  bool calc_ywlogdens(MeshDataLMC& data);
  
  // Beta
  void deal_with_beta(bool sample=true);
  void hmc_sample_beta(bool sample=true);
  //void tester_beta(bool sample=true);
  std::vector<NodeDataB> beta_node; // std::vector
  std::vector<AdaptE> beta_hmc_adapt; // std::vector
  arma::uvec beta_hmc_started;
  
  void deal_with_BetaLambdaTau(MeshDataLMC& data, bool sample, 
                               bool sample_beta, bool sample_lambda, bool sample_tau);
  arma::vec sample_BetaLambda_row(bool sample, int j, const arma::mat& rnorm_precalc);
  void sample_hmc_BetaLambdaTau(bool sample, 
                                bool sample_beta, bool sample_lambda, bool sample_tau);
  
  // Lambda
  void deal_with_Lambda(MeshDataLMC& data);
  void sample_nc_Lambda_std(); // noncentered
  void sample_nc_Lambda_fgrid(MeshDataLMC& data);
  arma::vec sample_Lambda_row(int j);
  void sample_hmc_Lambda();
  std::vector<NodeDataB> lambda_node; // std::vector
  std::vector<AdaptE> lambda_hmc_adapt; // std::vector
  arma::uvec lambda_hmc_started;
  
  
  // Tausq
  void deal_with_tausq(MeshDataLMC& data, bool ref_pardata=false);
  void gibbs_sample_tausq_std(bool ref_pardata);
  void gibbs_sample_tausq_fgrid(MeshDataLMC& data, bool ref_pardata);
  
  void logpost_refresh_after_gibbs(MeshDataLMC& data, bool sample=true); 
  
  // Predictions for W and Y
  void predict(bool sample=true);
  void predicty(bool sample=true);
  
  // --------------------------------------------------------------- from SP
  
  // need to adjust these
  int npars;
  //int nugget_in;
  
  // --------------------------------------------------------------- timers
  std::chrono::steady_clock::time_point start_overall;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point end_overall;
  
  
  // --------------------------------------------------------------- constructors
  
  Meshed(){};
  Meshed(
    const arma::mat& y_in, 
    const arma::uvec& familyid,
    
    const arma::mat& X_in, 
    
    const arma::mat& coords_in, 
    
    int k_in,
    
    const arma::field<arma::uvec>& parents_in,
    const arma::field<arma::uvec>& children_in,
    
    const arma::vec& block_names_in,
    const arma::vec& block_groups_in,
    
    const arma::field<arma::uvec>& indexing_in,
    const arma::field<arma::uvec>& indexing_obs_in,
    
    int matern_twonu_in,
    
    const arma::mat& w_in,
    const arma::mat& beta_in,
    const arma::mat& lambda_in,
    const arma::umat& lambda_mask_in,
    const arma::mat& theta_in,
    const arma::vec& tausq_inv_in,
    
    const arma::mat& beta_Vi_in,
    const arma::vec& tausq_ab_in,
    
    int which_hmc_in,
    bool adapting_theta,
    const arma::mat& metrop_theta_sd,
    const arma::mat& metrop_theta_bounds,
    
    bool use_cache,
    bool use_forced_grid,
    bool use_ps,
    
    bool verbose_in,
    bool debugging,
    int num_threads);
  
  // for prior sampling
  Meshed(
    const arma::mat& coords_in, 
    
    const arma::field<arma::uvec>& parents_in,
    const arma::field<arma::uvec>& children_in,
    
    const arma::vec& block_names_in,
    const arma::vec& block_groups_in,
    
    const arma::field<arma::uvec>& indexing_in,
    const arma::field<arma::uvec>& indexing_obs_in,
    
    int matern_twonu_in,
    
    const arma::mat& theta_in,
    
    bool use_cache,
    
    bool verbose_in,
    bool debugging,
    int num_threads);
};

#endif
