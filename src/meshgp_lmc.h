#define ARMA_DONT_PRINT_ERRORS

#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdexcept>

#include "R.h"
#include "mgp_lmc_utils.h"
#include "mh_adapt.h"
#include "caching_pairwise_compare.h"
#include "covariance_lmc.h"

#include "truncmvnorm.h"

using namespace std;

const double hl2pi = -.5 * log(2.0 * M_PI);

class LMCMeshGP {
public:
  // meta
  //int n; // number of observations
  int p; // number of covariates
  int q; // number of outcomes
  int k; // number of factors
  int dd; // dimension
  int n_blocks; // number of blocks
  
  // data
  arma::mat y;
  arma::mat X;
  
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
  
  // regression
  arma::field<arma::mat> XtX;
  arma::mat Vi; 
  arma::mat Vim;
  arma::vec bprim;
  
  // tausq priors
  arma::vec tausq_ab;
  
  // dependence
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
  
  arma::uvec oneuv;
  
  // params
  bool matern_estim;
  arma::mat w;
  bool recover_generator;
  arma::mat wgen;
  arma::mat Bcoeff; // sampled
  arma::mat rand_norm_mat;
  
  arma::mat Lambda;
  
  arma::umat Lambda_mask; // 1 where we want lambda to be nonzero
  arma::mat LambdaHw; // part of the regression mean explained by the latent process
  arma::mat wU; // sampling all nonreference locations
  
  arma::mat XB; // by outcome
  arma::vec tausq_inv; // tausq for the l=q variables
  
  // ModifiedPP-like updates for tausq -- used if not forced_grid
  int tausq_mcmc_counter;
  int lambda_mcmc_counter;
  
  RAMAdapt tausq_adapt;
  arma::mat tausq_unif_bounds;
  
  int n_lambda_pars;
  arma::uvec lambda_sampling;
  arma::mat lambda_unif_bounds; // 1x2: lower and upper for off-diagonal
  RAMAdapt lambda_adapt;
  
  // ----------------
  
  // params with mh step
  MeshDataLMC param_data; 
  MeshDataLMC alter_data;
  
  // Matern
  int nThreads;
  MaternParams matern;
  
  // setup
  bool predicting;
  bool cached;
  bool forced_grid;
  
  bool verbose;
  bool debug;
  
  void message(string s);
  
  // init / indexing
  void init_indexing();
  void init_meshdata(const arma::mat&);
  void init_finalize();
  void fill_zeros_Kcache();
  void na_study();
  void make_gibbs_groups();
  
  // init / caching obj
  void init_cache();
  
  // caching
  arma::uvec coords_caching; 
  arma::uvec coords_caching_ix;
  //arma::uvec parents_caching;
  //arma::uvec parents_caching_ix;
  arma::uvec kr_caching;
  arma::uvec kr_caching_ix;
  arma::uvec cx_and_kr_caching; // merge of coords and kr
  
  arma::uvec findkr;
  arma::uvec findcc;
  
  int starting_kr;
  
  // caching some matrices // ***
  //arma::field<arma::cube> H_cache;
  //arma::field<arma::cube> Ri_cache;
  arma::field<arma::cube> CC_cache;
  
  //arma::field<arma::cube> Kxxi_cache; //*** we store the components not the Kroned matrix
  arma::vec Ri_chol_logdet;
  
  arma::field<arma::cube> Hpred;
  arma::field<arma::mat> Rcholpred;
  
  // MCMC
  void update_block_covpars(int, MeshDataLMC& data);
  void update_block_wlogdens(int, MeshDataLMC& data);
  void update_lly(int, MeshDataLMC&, const arma::mat& LamHw);
  
  bool refresh_cache(MeshDataLMC& data);
  bool calc_ywlogdens(MeshDataLMC& data);
  // 
  bool get_loglik_comps_w(MeshDataLMC& data);
  
  // update_block_wpars used for sampling and proposing w
  // calculates all conditional means and variances
  void calc_DplusSi(int, MeshDataLMC& data, const arma::mat& Lam, const arma::vec& tsqi);
  void update_block_w_cache(int, MeshDataLMC& data);
  void sample_nonreference_w(int, MeshDataLMC& data, const arma::mat& );
  void refresh_w_cache(MeshDataLMC& data);
  void gibbs_sample_w(MeshDataLMC& data, bool needs_update);
  
  //
  void gibbs_sample_beta();
  
  void deal_with_Lambda(MeshDataLMC& data);
  void sample_nc_Lambda_std(); // noncentered
  void sample_nc_Lambda_fgrid(MeshDataLMC& data);
  
  void deal_with_tausq(MeshDataLMC& data, double, double, bool);
  void gibbs_sample_tausq_std(double, double);
  void gibbs_sample_tausq_fgrid(MeshDataLMC& data, double, double, bool);
  
  void logpost_refresh_after_gibbs(MeshDataLMC& data); //***
  
  void predict();
  
  double logpost;
  
  // changing the values, no sampling;
  void theta_update(MeshDataLMC&, const arma::mat&);
  void beta_update(const arma::vec&);
  void tausq_update(double);
  
  // avoid expensive copies
  void accept_make_change();
  
  std::chrono::steady_clock::time_point start_overall;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point end_overall;
  
  // empty
  LMCMeshGP();
  
  // build everything
  LMCMeshGP(
    const arma::mat& y_in, 
    const arma::mat& X_in, 
    
    const arma::mat& coords_in, 
    
    int k_in,
    
    const arma::field<arma::uvec>& parents_in,
    const arma::field<arma::uvec>& children_in,
    
    const arma::vec& block_names_in,
    const arma::vec& block_group_in,
    
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
    
    bool use_cache,
    bool use_forced_grid,
    
    bool verbose_in,
    bool debugging,
    int num_threads);
  
};

void LMCMeshGP::message(string s){
  if(verbose & debug){
    Rcpp::Rcout << s << "\n";
  }
}

LMCMeshGP::LMCMeshGP(){
  
}

LMCMeshGP::LMCMeshGP(
  const arma::mat& y_in, 
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
  
  bool use_cache=true,
  bool use_forced_grid=false,
  
  bool verbose_in=false,
  bool debugging=false,
  int num_threads = 1){
  
  oneuv = arma::ones<arma::uvec>(1);//utils
  
  
  verbose = verbose_in;
  debug = debugging;
  
  message("LMCMeshGP::LMCMeshGP initialization.\n");
  
  forced_grid = use_forced_grid;
  
  start_overall = std::chrono::steady_clock::now();
  
  cached = true;
  
  if(forced_grid){
    message("MGP on a latent grid");
  } else {
    message("MGP on data grid, caching activated");
  }
  
  message("[LMCMeshGP::LMCMeshGP] assign values.");
  // data
  y                   = y_in;
  X                   = X_in;
  
  na_mat = arma::zeros<arma::umat>(arma::size(y));
  na_mat.elem(arma::find_finite(y)).fill(1);
  
  p  = X.n_cols;
  
  // spatial coordinates and dimension
  coords              = coords_in;
  dd = coords.n_cols;
  
  q = y.n_cols;
  k = k_in;
  
  
  Lambda = lambda_in; 
  Lambda_mask = lambda_mask_in;
  
  if(forced_grid){
    tausq_mcmc_counter = 0;
    lambda_mcmc_counter = 0;
    
    tausq_adapt = RAMAdapt(q, arma::eye(q,q)*.05, .25);
    tausq_unif_bounds = arma::join_horiz(1e-6 * arma::ones(q), 1e3 * arma::ones(q));
    
    // lambda prepare
    n_lambda_pars = arma::accu(Lambda_mask);
    lambda_adapt = RAMAdapt(n_lambda_pars, arma::eye(n_lambda_pars, n_lambda_pars)*.05, .25);
    
    lambda_sampling = arma::find(Lambda_mask == 1);
    lambda_unif_bounds = arma::zeros(n_lambda_pars, 2);
    for(int i=0; i<n_lambda_pars; i++){
      arma::uvec rc = arma::ind2sub( arma::size(Lambda), lambda_sampling(i) );
      if(rc(0) == rc(1)){
        lambda_unif_bounds(i, 0) = 0;
        lambda_unif_bounds(i, 1) = 1e6;
      } else {
        lambda_unif_bounds(i, 0) = -1e6;
        lambda_unif_bounds(i, 1) = 1e6;
      }
    }
  }
  
  // NAs at blocks of outcome variables 
  ix_by_q_a = arma::field<arma::uvec>(q);
  for(int j=0; j<q; j++){
    ix_by_q_a(j) = arma::find_finite(y.col(j));
    Rcpp::Rcout << "Y(" << j+1 << ") : " << ix_by_q_a(j).n_elem << " observed locations.\n";
  }
  
  // DAG
  parents             = parents_in;
  children            = children_in;
  block_names         = block_names_in;
  block_groups        = block_groups_in;
  block_groups_labels = arma::unique(block_groups);
  n_gibbs_groups      = block_groups_labels.n_elem;
  n_blocks            = block_names.n_elem;
  
  // domain partitioning
  indexing    = indexing_in;
  indexing_obs = indexing_obs_in;
  
  // initial values
  w = w_in; 
  
  matern_estim = (dd == 2) & (theta_in.n_rows == 3);
  recover_generator = matern_estim;
  if(recover_generator){
    wgen = arma::zeros(arma::size(w)); // *** remove
  }
  
  Rcpp::Rcout << "Lambda size: " << arma::size(Lambda) << "\n";
  
  tausq_inv = tausq_inv_in;
  XB = arma::zeros(coords.n_rows, q);
  Bcoeff = beta_in; 
  for(int j=0; j<q; j++){
    XB.col(j) = X * Bcoeff.col(j);
  }
  
  Rcpp::Rcout << "Beta size: " << arma::size(Bcoeff) << "\n"; 
  
  // prior params
  XtX = arma::field<arma::mat>(q);
  for(int j=0; j<q; j++){
    XtX(j) = X.rows(ix_by_q_a(j)).t() * X.rows(ix_by_q_a(j));
  }
  Vi    = beta_Vi_in;
  bprim = arma::zeros(p);
  Vim   = Vi * bprim;
  
  tausq_ab = tausq_ab_in;
  
  // init
  u_is_which_col_f    = arma::field<arma::field<arma::field<arma::uvec> > > (n_blocks);
  
  predicting = true;
  
  // now elaborate
  message("LMCMeshGP::LMCMeshGP : init_indexing()");
  init_indexing();
  
  message("LMCMeshGP::LMCMeshGP : na_study()");
  na_study();
  // now we know where NAs are, we can erase them
  y.elem(arma::find_nonfinite(y)).fill(0);
  
  message("LMCMeshGP::LMCMeshGP : init_finalize()");
  init_finalize();
  
  message("LMCMeshGP::LMCMeshGP : make_gibbs_groups()");
  // quick check for groups
  make_gibbs_groups();
  
  //caching;
  if(cached){
    message("LMCMeshGP::LMCMeshGP : init_cache()");
    init_cache();
    fill_zeros_Kcache();
  }
  
  init_meshdata(theta_in);
  
  nThreads = num_threads;
  
  int bessel_ws_inc = 3;
  matern.bessel_ws = (double *) R_alloc(nThreads*bessel_ws_inc, sizeof(double));
  matern.twonu = matern_twonu_in;
  
  // predict_initialize
  message("predict initialize");
  if(predict_group_exists == 1){
    Hpred = arma::field<arma::cube>(u_predicts.n_elem);
    Rcholpred = arma::field<arma::mat>(u_predicts.n_elem);
    
    for(int i=0; i<u_predicts.n_elem; i++){
      int u = u_predicts(i);
      if(block_ct_obs(u) > 0){
        Hpred(i) = arma::zeros(k,indexing(u).n_elem,indexing_obs(u).n_elem);
      } else {
        Hpred(i) = arma::zeros(k,parents_indexing(u).n_elem,indexing_obs(u).n_elem);
      }
      Rcholpred(i) = arma::zeros(k,indexing_obs(u).n_elem);
    }
  }
  
  message("LambdaHw initialize");
  LambdaHw = arma::zeros(coords.n_rows, q); 
  wU = w;
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i) - 1;
    LambdaHw.rows(indexing_obs(u)) = w.rows(indexing_obs(u)) * 
      Lambda.t();
  }
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "LMCMeshGP::LMCMeshGP initializing took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
}


void LMCMeshGP::make_gibbs_groups(){
  message("[make_gibbs_groups] start");
  
  // checks -- errors not allowed. use check_groups.cpp to fix errors.
  for(int g=0; g<n_gibbs_groups; g++){
    for(int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      if(block_groups(u) == block_groups_labels(g)){
        if(indexing(u).n_elem > 0){ //**
          
          for(int pp=0; pp<parents(u).n_elem; pp++){
            if(block_groups(parents(u)(pp)) == block_groups_labels(g)){
              Rcpp::Rcout << u << " <--- " << parents(u)(pp) 
                          << ": same group (" << block_groups(u) 
                          << ")." << "\n";
              throw 1;
            }
          }
          for(int cc=0; cc<children(u).n_elem; cc++){
            if(block_groups(children(u)(cc)) == block_groups_labels(g)){
              Rcpp::Rcout << u << " ---> " << children(u)(cc) 
                          << ": same group (" << block_groups(u) 
                          << ")." << "\n";
              throw 1;
            }
          }
        }
      }
    }
  }
  
  int gx=0;
  arma::field<arma::vec> u_by_block_groups_temp(n_gibbs_groups);
  u_by_block_groups = arma::field<arma::vec>(n_gibbs_groups);
  /// create list of groups for gibbs
  
  for(int g=0; g<n_gibbs_groups; g++){
    u_by_block_groups_temp(g) = arma::zeros(0);
    
    for(int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      
      if(block_groups(u) == block_groups_labels(g)){
        if(block_ct_obs(u) > 0){ //**
          arma::vec uhere = arma::zeros(1) + u;
          u_by_block_groups_temp(g) = arma::join_vert(u_by_block_groups_temp(g), uhere);
        } 
      }
    }
    if(u_by_block_groups_temp(g).n_elem > 0){
      u_by_block_groups(gx) = u_by_block_groups_temp(g);
      gx ++;
    }
  }
  
  int pblocks = 0;
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i) - 1;
    if(forced_grid){
      // forced grid, then predict blocks are all those that have some missing
      if(block_ct_obs(u) < na_1_blocks(u).n_elem){
        pblocks ++;
      }
    } else {
      // original grid, then predict blocks are the empty ones
      if(block_ct_obs(u) == 0){
        pblocks ++;
      }
    }
  }
  
  if(pblocks > 0){
    u_predicts = arma::zeros<arma::uvec>(pblocks);
    predict_group_exists = 1;
  } else {
    predict_group_exists = 0;
  }
  
  if(predict_group_exists == 1){
    int p=0; 
    for(int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      if(forced_grid){
        // forced grid, then predict blocks are all those that have some missing
        if(block_ct_obs(u) < na_1_blocks(u).n_elem){
          u_predicts(p) = u;
          p ++;
        }
      } else {
        // original grid, then predict blocks are the empty ones
        if(block_ct_obs(u) == 0){
          u_predicts(p) = u;
          
          p ++;
        }
      }
    }
  } else {
    Rcpp::Rcout << "No prediction group " << endl;
  }
  
  message("[make_gibbs_groups] done.");
}

void LMCMeshGP::na_study(){
  // prepare stuff for NA management
  message("[na_study] start");
  na_1_blocks = arma::field<arma::uvec> (n_blocks);
  na_0_blocks = arma::field<arma::uvec> (n_blocks);
  na_ix_blocks = arma::field<arma::uvec> (n_blocks);
  n_loc_ne_blocks = 0;
  block_ct_obs = arma::zeros(n_blocks);
  
  message("[na_study] step 1.");
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks;i++){
    arma::mat yvec = y.rows(indexing_obs(i));
    na_1_blocks(i) = arma::zeros<arma::uvec>(yvec.n_rows);
    na_0_blocks(i) = arma::zeros<arma::uvec>(yvec.n_rows);
    // consider NA if all margins are missing
    // otherwise it's available
    for(int ix=0; ix<yvec.n_rows; ix++){
      arma::uvec yfinite_row = arma::find_finite(yvec.row(ix));
      if(yfinite_row.n_elem > 0){
        // at least one is available
        na_1_blocks(i)(ix) = 1;
      }
      if(yfinite_row.n_elem < q){
        // at least one is missing
        na_0_blocks(i)(ix) = 1;
      }
    }
    na_ix_blocks(i) = arma::find(na_1_blocks(i) == 1); 
  }
  
  message("[na_study] step 2.");
  n_ref_blocks = 0;
  for(int i=0; i<n_blocks; i++){
    block_ct_obs(i) = arma::accu(na_1_blocks(i));
    if(block_ct_obs(i) > 0){
      n_loc_ne_blocks += indexing(i).n_elem;
      n_ref_blocks += 1;
    } 
  }
  
  message("[na_study] step 3.");
  int j=0;
  reference_blocks = arma::zeros<arma::uvec>(n_ref_blocks);
  //ref_block_names = arma::zeros<arma::uvec>(n_ref_blocks);
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i) - 1;
    if(block_ct_obs(u) > 0){
      reference_blocks(j) = i;
      //ref_block_names(j) = u;
      j ++;
    } 
  }
  message("[na_study] done.");
}

void LMCMeshGP::fill_zeros_Kcache(){
  message("[fill_zeros_Kcache]");
  
  CC_cache = arma::field<arma::cube>(coords_caching.n_elem);
  //H_cache = arma::field<arma::cube> (kr_caching.n_elem);
  //Ri_cache = arma::field<arma::cube> (kr_caching.n_elem);
  Ri_chol_logdet = arma::zeros(kr_caching.n_elem);
  
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i); 
    if(block_ct_obs(u) > 0){
      CC_cache(i) = arma::cube(indexing(u).n_elem, indexing(u).n_elem, k);
    }
  }
  
  /*for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    H_cache(i) = 
      arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem, k);
    Ri_cache(i) = 
      arma::zeros(indexing(u).n_elem, indexing(u).n_elem, k);
  }*/
  message("[fill_zeros_Kcache] done.");
}

void LMCMeshGP::init_cache(){
  // coords_caching stores the layer names of those layers that are representative
  // coords_caching_ix stores info on which layers are the same in terms of rel. distance
  
  message("[init_cache]");
  //coords_caching_ix = caching_pairwise_compare_uc(coords_blocks, block_names, block_ct_obs); // uses block_names(i)-1 !
  coords_caching_ix = caching_pairwise_compare_uci(coords, indexing, block_names, block_ct_obs); // uses block_names(i)-1 !
  coords_caching = arma::unique(coords_caching_ix);
  
  //parents_caching_ix = caching_pairwise_compare_uc(parents_coords, block_names, block_ct_obs);
  //parents_caching_ix = caching_pairwise_compare_uci(coords, parents_indexing, block_names, block_ct_obs);
  //parents_caching = arma::unique(parents_caching_ix);
  
  arma::field<arma::mat> kr_pairing(n_blocks);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_blocks; i++){
    int u = block_names(i)-1;
    arma::mat cmat = coords.rows(indexing(u));
    if(parents_indexing(u).n_elem > 0){
      arma::mat pmat = coords.rows(parents_indexing(u));
      arma::mat kr_mat_c = arma::join_vert(cmat, pmat);
      kr_pairing(u) = kr_mat_c;
    } else {
      kr_pairing(u) = cmat;
    }
  }
  
  kr_caching_ix = caching_pairwise_compare_uc(kr_pairing, block_names, block_ct_obs);
  kr_caching = arma::unique(kr_caching_ix);
  
  starting_kr = 0;
  if(forced_grid){
    cx_and_kr_caching = arma::join_vert(coords_caching,
                                        kr_caching);
    starting_kr = coords_caching.n_elem;
  } else {
    cx_and_kr_caching = kr_caching;
  }
  
  // 
  findkr = arma::zeros<arma::uvec>(n_blocks);
  findcc = arma::zeros<arma::uvec>(n_blocks);
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i) - 1;
    int kr_cached_ix = kr_caching_ix(u);
    arma::uvec cpx = arma::find(kr_caching == kr_cached_ix, 1, "first");
    findkr(u) = cpx(0);
    
    //if(forced_grid){
      int u_cached_ix = coords_caching_ix(u);
      arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first" );
      findcc(u) = cx(0);
    //}
  }
  

  
  
  
  if(verbose & debug || true){
    Rcpp::Rcout << "Caching c: " << coords_caching.n_elem 
                << " k: " << kr_caching.n_elem << "\n";
  }
  message("[init_cache]");
}

void LMCMeshGP::init_meshdata(const arma::mat& theta_in){
  message("[init_meshdata]");
  
  // block params
  //param_data.w_cond_mean_K = arma::field<arma::cube> (n_blocks);
  //param_data.w_cond_prec   = arma::field<arma::cube> (n_blocks);
  
  param_data.Rproject = arma::field<arma::cube>(n_blocks);
  param_data.Riproject = arma::field<arma::cube>(n_blocks);
  param_data.Hproject = arma::field<arma::cube>(n_blocks);
  
  param_data.Smu_start = arma::field<arma::mat>(n_blocks);
  param_data.Sigi_chol = arma::field<arma::mat>(n_blocks);
  param_data.AK_uP = arma::field<arma::field<arma::cube> >(n_blocks);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    int u=block_names(i) - 1;
    //param_data.w_cond_mean_K(i) = arma::zeros(indexing(i).n_elem, parents_indexing(i).n_elem, k);
    //param_data.w_cond_prec(i) = arma::zeros(indexing(i).n_elem, indexing(i).n_elem, k);
   
   if(forced_grid){
     param_data.Hproject(i) = arma::zeros(k, indexing(i).n_elem, indexing_obs(i).n_elem);
     param_data.Rproject(i) = arma::zeros(k, k, indexing_obs(i).n_elem);
     param_data.Riproject(i) = arma::zeros(k, k, indexing_obs(i).n_elem);
   }
    
    param_data.Smu_start(i) = arma::zeros(k*indexing(i).n_elem, 1);
    param_data.Sigi_chol(i) = arma::zeros(k*indexing(i).n_elem, k*indexing(i).n_elem);
    param_data.AK_uP(i) = arma::field<arma::cube>(children(i).n_elem);
    for(int c=0; c<children(i).n_elem; c++){
      int child = children(i)(c);
      param_data.AK_uP(i)(c) = arma::zeros(indexing(i).n_elem, indexing(child).n_elem, k);
    }
  }
  
  param_data.w_cond_prec_ptr.reserve(n_blocks);
  param_data.w_cond_mean_K_ptr.reserve(n_blocks);
  for(int i=0; i<n_blocks; i++){
    arma::cube jibberish = arma::zeros(1,1,1);
    param_data.w_cond_prec_ptr.push_back(&jibberish);
    param_data.w_cond_mean_K_ptr.push_back(&jibberish);
  }
  
  param_data.Kxxi_cache = arma::field<arma::cube>(coords_caching.n_elem);
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i);
    param_data.Kxxi_cache(i) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem, k);
  }
  
  // loglik w for updating theta
  param_data.logdetCi_comps = arma::zeros(n_blocks);
  param_data.logdetCi       = 0;
  
  // ***
  param_data.wcore = arma::zeros(n_blocks, 1);
  param_data.loglik_w_comps = arma::zeros(n_blocks, 1);
  param_data.loglik_w       = 0; 
  param_data.theta          = theta_in;//##
  
  // noncentral parameters
  param_data.ll_y = arma::zeros(coords.n_rows, 1);
  param_data.ll_y_all       = 0; 
  param_data.DplusSi = arma::zeros(q, q, y.n_rows);
  param_data.DplusSi_c = arma::zeros(q, q, y.n_rows);
  param_data.DplusSi_ldet = arma::zeros(y.n_rows);
  
  param_data.H_cache = arma::field<arma::cube> (kr_caching.n_elem);
  param_data.Ri_cache = arma::field<arma::cube> (kr_caching.n_elem);
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    param_data.Ri_cache(i) = 
      arma::zeros(indexing(u).n_elem, indexing(u).n_elem, k);
    if(parents(u).n_elem > 0){
      param_data.H_cache(i) = 
        arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem, k);
    }
  }
  
  
  
  
  alter_data = param_data; 
  
  message("[init_meshdata] done.");
}

void LMCMeshGP::init_indexing(){
  
  parents_indexing = arma::field<arma::uvec> (n_blocks);

  message("[init_indexing] parent_indexing");
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(parents(u).n_elem > 0){
      arma::field<arma::uvec> pixs(parents(u).n_elem);
      for(int pi=0; pi<parents(u).n_elem; pi++){
        pixs(pi) = indexing(parents(u)(pi));
      }
      parents_indexing(u) = field_v_concat_uv(pixs);
    }
  }
  message("[init_indexing] done.");
}

void LMCMeshGP::init_finalize(){
  message("[init_finalize] dim_by_parent, parents_coords, children_coords");
  
  arma::field<arma::uvec> dim_by_parent(n_blocks);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){ // all blocks
    int u = block_names(i)-1; // block name
    
    if(indexing_obs(u).n_elem > 0){ 
      // number of coords of the jth parent of the child
      dim_by_parent(u) = arma::zeros<arma::uvec>(parents(u).n_elem + 1);
      for(int j=0; j<parents(u).n_elem; j++){
        dim_by_parent(u)(j+1) = indexing(parents(u)(j)).n_elem;
      }
      dim_by_parent(u) = arma::cumsum(dim_by_parent(u));
    }
  }
  message("[init_finalize] u_is_which_col_f");
  
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(indexing(u).n_elem > 0){
      // children-parent relationship variables
      u_is_which_col_f(u) = arma::field<arma::field<arma::uvec> > (children(u).n_elem);
      
      for(int c=0; c<children(u).n_elem; c++){
        int child = children(u)(c);
        // which parent of child is u which we are sampling
        arma::uvec u_is_which = arma::find(parents(child) == u, 1, "first"); 
        
        // which columns correspond to it
        int firstcol = dim_by_parent(child)(u_is_which(0));
        int lastcol = dim_by_parent(child)(u_is_which(0)+1);
        
        int dimen = parents_indexing(child).n_elem;
        
        // this is for w=mat and fields
        arma::vec colix = arma::zeros(dimen);
        for(int s=0; s<1; s++){
          int shift = s * dimen;
          colix.subvec(shift + firstcol, shift + lastcol-1).fill(1);
        }
        
        u_is_which_col_f(u)(c) = arma::field<arma::uvec> (2);
        u_is_which_col_f(u)(c)(0) = arma::find(colix == 1); // u parent of c is in these columns for c
        u_is_which_col_f(u)(c)(1) = arma::find(colix != 1); // u parent of c is NOT in these columns for c
      }
    }
  }
  
  message("[init_finalize] done.");
}

bool LMCMeshGP::refresh_cache(MeshDataLMC& data){
  start_overall = std::chrono::steady_clock::now();
  message("[refresh_cache] start.");
  
  Ri_chol_logdet = arma::zeros(kr_caching.n_elem);
  
  int errtype = -1;
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i); 
    if(block_ct_obs(u) > 0){
      for(int j=0; j<k; j++){
        CC_cache(i).slice(j) = Correlationf(coords, indexing(u), indexing(u), //coords.rows(indexing(u)), coords.rows(indexing(u)), 
                 data.theta.col(j), matern, true);
      }
    }
  }
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int it=0; it<cx_and_kr_caching.n_elem; it++){
    int i = 0;
    if(it < starting_kr){
      // this means we are caching coords
      i = it;
      int u = coords_caching(i); // block name of ith representative
      try {
        CviaKron_invsympd_(data.Kxxi_cache(i),
                           coords, indexing(u), k, data.theta, matern);
      } catch (...) {
        errtype = 1;
      }
    } else {
      // this means we are caching kr
      i = it - starting_kr;
      int u = kr_caching(i);
      try {
        if(block_ct_obs(u) > 0){
          //int u_cached_ix = coords_caching_ix(u);
          //arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first");
          
          int ccfound = findcc(u);
          //arma::cube Cxx = CC_cache(ccfound);
          
          Ri_chol_logdet(i) = CviaKron_HRi_(data.H_cache(i), data.Ri_cache(i), CC_cache(ccfound),
                         coords, indexing(u), parents_indexing(u), k, data.theta, matern);
        }
      } catch (...) {
        errtype = 2;
      }
    }
  }
  
  if(false & (verbose & debug)){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[refresh_cache] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
  if(errtype > 0){
    if(verbose & debug){
      Rcpp::Rcout << "Cholesky failed at some point. Here's the value of theta that caused this" << "\n";
      Rcpp::Rcout << "theta: " << data.theta.t() << "\n";
      Rcpp::Rcout << " -- auto rejected and proceeding." << "\n";
    }
    return false;
  }
  return true;
}

void LMCMeshGP::update_block_covpars(int u, MeshDataLMC& data){
  //message("[update_block_covpars] start.");
  // given block u as input, this function updates H and R
  // which will be used later to compute logp(w | theta)
  int krfound = findkr(u);
  
  //data.w_cond_prec(u) = data.Ri_cache(krfound);
  data.w_cond_prec_ptr.at(u) = &data.Ri_cache(krfound);
  
  data.logdetCi_comps(u) = Ri_chol_logdet(krfound);
  
  if( parents(u).n_elem > 0 ){
    //data.w_cond_mean_K(u) = H_cache(krfound);
    data.w_cond_mean_K_ptr.at(u) = &data.H_cache(krfound);
  } 
  
  if(forced_grid){
    int ccfound = findcc(u);
    CviaKron_HRj_bdiag_(data.Hproject(u), data.Rproject(u), data.Riproject(u),
                        data.Kxxi_cache(ccfound),
                        coords, indexing_obs(u), 
                        na_1_blocks(u), indexing(u), 
                        k, data.theta, matern);
    
  }
  //message("[update_block_covpars] done.");
}

void LMCMeshGP::update_block_wlogdens(int u, MeshDataLMC& data){
  //message("[update_block_wlogdens].");
  arma::mat wx = w.rows(indexing(u));
  arma::mat wcoresum = arma::zeros(1, k);
  if( parents(u).n_elem > 0 ){
    arma::mat wpar = w.rows(parents_indexing(u));
    for(int j=0; j<k; j++){
      wx.col(j) = wx.col(j) - 
        (*data.w_cond_mean_K_ptr.at(u)).slice(j) *
        //data.w_cond_mean_K(u).slice(j) * 
        wpar.col(j);
    }
  }
  
  for(int j=0; j<k; j++){
    wcoresum(j) = 
      arma::conv_to<double>::from(arma::trans(wx.col(j)) * 
      //data.w_cond_prec(u).slice(j) * 
      (*data.w_cond_prec_ptr.at(u)).slice(j) *
      wx.col(j));
  }
  
  data.wcore.row(u) = arma::accu(wcoresum);
  data.loglik_w_comps.row(u) = (indexing(u).n_elem+.0) * hl2pi -.5 * arma::accu(wcoresum); //
  //arma::accu(data.wcore.slice(u).diag());

  //message("[update_block_wlogdens] done.");
}

void LMCMeshGP::calc_DplusSi(int u, 
          MeshDataLMC & data, const arma::mat& Lam, const arma::vec& tsqi){
  //message("[calc_DplusSi] start.");
  int indxsize = indexing(u).n_elem;
  
  if((k==1) & (q==1)){
    for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
      if(na_1_blocks(u)(ix) == 1){
        arma::mat Dtau = Lam(0, 0) * Lam(0, 0) * data.Rproject(u).slice(ix) + 1/tsqi(0);
        // fill 
        data.DplusSi_ldet(indexing_obs(u)(ix)) = - log(Dtau(0,0));
        data.DplusSi.slice(indexing_obs(u)(ix)) = 1.0/Dtau; // 1.0/ (L * L);
        data.DplusSi_c.slice(indexing_obs(u)(ix)) = pow(Dtau, -0.5);
      }
    }
  } else {
    for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
      if(na_1_blocks(u)(ix) == 1){
        arma::mat Dtau = Lam * data.Rproject(u).slice(ix) * Lam.t();
        arma::vec II = arma::ones(q);
        for(int j=0; j<q; j++){
          if(na_mat(indexing_obs(u)(ix), j) == 1){
            // this outcome margin observed at this location
            Dtau(j, j) += 1/tsqi(j);
          } else {
            II(j) = 0;
          }
        }
        arma::uvec obs = arma::find(II == 1);
        // Dtau = D + S
        arma::mat L = arma::chol(Dtau.submat(obs, obs), "lower");
        // L Lt = D + S, therefore Lti Li = (D + S)^-1
        arma::mat Li = arma::inv(arma::trimatl(L));
        
        arma::mat Ditau = arma::zeros(q, q);
        arma::mat Ditau_obs = Li.t() * Li;
        Ditau.submat(obs, obs) = Ditau_obs;
        
        arma::mat Lifull = arma::zeros(arma::size(Ditau));
        Lifull.submat(obs, obs) = Li;
        
        // fill 
        data.DplusSi_ldet(indexing_obs(u)(ix)) = 2.0 * arma::accu(log(Li.diag()));
        data.DplusSi.slice(indexing_obs(u)(ix)) = Ditau;
        data.DplusSi_c.slice(indexing_obs(u)(ix)) = Lifull;
      }
    }
  }
  
  
}

void LMCMeshGP::update_lly(int u, MeshDataLMC& data, const arma::mat& LamHw){
  //message("[update_lly] start.");
  start = std::chrono::steady_clock::now();
  data.ll_y.rows(indexing_obs(u)).fill(0.0);
  
  for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
    if(na_1_blocks(u)(ix) == 1){
      // at least one outcome available
      arma::vec ymean = arma::trans(y.row(indexing_obs(u)(ix)) - 
        XB.row(indexing_obs(u)(ix)) - LamHw.row(indexing_obs(u)(ix)));
      data.ll_y.row(indexing_obs(u)(ix)) += 
        + 0.5 * data.DplusSi_ldet(indexing_obs(u)(ix)) - 0.5 * ymean.t() * 
        data.DplusSi.slice(indexing_obs(u)(ix)) * ymean;
    }
  }
  end = std::chrono::steady_clock::now();
  //message("[update_lly] done.");
}

void LMCMeshGP::logpost_refresh_after_gibbs(MeshDataLMC& data){
  message("[logpost_refresh_after_gibbs]");
  if(verbose & debug){
    start_overall = std::chrono::steady_clock::now();
  }
  
#ifdef _OPENMP
  #pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    //update_block_covpars(u, data);
    update_block_wlogdens(u, data);
    
    if(forced_grid & true){
      calc_DplusSi(u, data, Lambda, tausq_inv);
      update_lly(u, data, LambdaHw);
    }
  }
  
  data.loglik_w = arma::accu(data.logdetCi_comps) + 
    arma::accu(data.loglik_w_comps) + 
    arma::accu(data.ll_y);
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[logpost_refresh_after_gibbs] " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count() 
                << "us.\n"
                << "of which " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                << "us to do [update_lly].\n";
  }
}

bool LMCMeshGP::calc_ywlogdens(MeshDataLMC& data){
  start_overall = std::chrono::steady_clock::now();
  
  //message("[calc_ywlogdens] start.");
  // called for a proposal of theta
  // updates involve the covariances
  // and Sigma for adjusting the error terms
  
  //start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    update_block_covpars(u, data);
    update_block_wlogdens(u, data);
    
    if(forced_grid){
      calc_DplusSi(u, data, Lambda, tausq_inv);
      update_lly(u, data, LambdaHw);
    }
  }
  
  //Rcpp::Rcout << "loglik_w_comps " << arma::accu(data.loglik_w_comps) << endl;
  
  data.loglik_w = arma::accu(data.logdetCi_comps) + 
    arma::accu(data.loglik_w_comps) + arma::accu(data.ll_y);
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[calc_ywlogdens] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n"
                << "of which " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us. for [update_lly]\n";
  }
  
  return true;
}

bool LMCMeshGP::get_loglik_comps_w(MeshDataLMC& data){
  bool acceptable = refresh_cache(data);
  if(acceptable){
    acceptable = calc_ywlogdens(data);
    return acceptable;
  } else {
    return acceptable;
  }
}

void LMCMeshGP::gibbs_sample_beta(){
  message("[gibbs_sample_beta]");
  start = std::chrono::steady_clock::now();
  
  Rcpp::RNGScope scope;
  arma::mat bmat = arma::randn(p, q);
  
  // for forced grid we use non-reference samples 
  arma::mat LHW = wU * Lambda.t();
  
  for(int j=0; j<q; j++){
    arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * XtX(j) + Vi), "lower");
    arma::mat Sigma_chol_Bcoeff = arma::inv(arma::trimatl(Si_chol));
    
    arma::mat Xprecy_j = Vim + tausq_inv(j) * X.rows(ix_by_q_a(j)).t() * 
      (y.submat(ix_by_q_a(j), oneuv*j) - LambdaHw.submat(ix_by_q_a(j), oneuv*j)); //***
    
    Bcoeff.col(j) = Sigma_chol_Bcoeff.t() * (Sigma_chol_Bcoeff * Xprecy_j + bmat.col(j));
    XB.col(j) = X * Bcoeff.col(j);
  }
  
  if(verbose & debug){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_beta] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

void LMCMeshGP::deal_with_Lambda(MeshDataLMC& data){
  bool randomize_update = (R::runif(0,1) > .5) || (y.n_rows < 50000);
  if(forced_grid & randomize_update){
    sample_nc_Lambda_fgrid(data);
  } else {
    sample_nc_Lambda_std();
  }
}

void LMCMeshGP::deal_with_tausq(MeshDataLMC& data, double aprior=2.001, double bprior=1, bool ref_pardata=false){
  bool randomize_update = (R::runif(0,1) > .5) || (y.n_rows < 50000);
  // ref_pardata: set to true if this is called without calling deal_with_Lambda first
  if(forced_grid & randomize_update){
    gibbs_sample_tausq_fgrid(data, aprior, bprior, ref_pardata);
  } else {
    gibbs_sample_tausq_std(aprior, bprior);
  }
}

void LMCMeshGP::sample_nc_Lambda_fgrid(MeshDataLMC& data){
  message("[gibbs_sample_Lambda_fgrid] start (sampling via Robust adaptive Metropolis)");
  start = std::chrono::steady_clock::now();
  
  lambda_adapt.count_proposal();
  Rcpp::RNGScope scope;
  
  arma::vec U_update = arma::randn(n_lambda_pars);
  arma::vec lambda_vec_current = Lambda.elem(lambda_sampling);
  
  arma::vec lambda_vec_proposal = par_huvtransf_back(par_huvtransf_fwd(lambda_vec_current, lambda_unif_bounds) + 
    lambda_adapt.paramsd * U_update, lambda_unif_bounds);
  arma::mat Lambda_proposal = Lambda;
  Lambda_proposal.elem(lambda_sampling) = lambda_vec_proposal;
  
  arma::mat LambdaHw_proposal = w * Lambda_proposal.t();
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    // we are doing this after sampling w so need to update the current likelihood too
    update_lly(u, param_data, LambdaHw);
    //calc_DplusSi(u, param_data, tausq_inv);
    
    // then for the proposal
    calc_DplusSi(u, alter_data, Lambda_proposal, tausq_inv);
    update_lly(u, alter_data, LambdaHw_proposal);
  }

  arma::vec Lambda_prop_d = Lambda_proposal.diag();
  arma::vec Lambda_d = Lambda.diag();
  arma::mat L_prior_prec = 1 * arma::eye(Lambda_d.n_elem, Lambda_d.n_elem);
  double log_prior_ratio = arma::conv_to<double>::from(
    -0.5*Lambda_prop_d.t() * L_prior_prec * Lambda_prop_d
    +0.5*Lambda_d.t() * L_prior_prec * Lambda_d);
    
    double new_logpost = arma::accu(alter_data.ll_y);
    double start_logpost = arma::accu(param_data.ll_y);
    
    double jacobian  = calc_jacobian(lambda_vec_proposal, lambda_vec_current, lambda_unif_bounds);
    double logaccept = new_logpost - start_logpost + log_prior_ratio + jacobian;
    
    //double u = R::runif(0,1);//arma::randu();
    bool accepted = do_I_accept(logaccept);
    
    if(accepted){
      lambda_adapt.count_accepted();
      // accept the move
      Lambda = Lambda_proposal;
      LambdaHw = LambdaHw_proposal;
      
      param_data.DplusSi = alter_data.DplusSi;
      param_data.DplusSi_c = alter_data.DplusSi_c;
      param_data.DplusSi_ldet = alter_data.DplusSi_ldet;
      param_data.ll_y = alter_data.ll_y;
      
      logpost = new_logpost;
    } else {
      logpost = start_logpost;
    }
    
    lambda_adapt.update_ratios();
    lambda_adapt.adapt(U_update, exp(logaccept), lambda_mcmc_counter); 
    lambda_mcmc_counter ++;
    if(verbose & debug){
      end = std::chrono::steady_clock::now();
      Rcpp::Rcout << "[gibbs_sample_Lambda_fgrid] " << lambda_adapt.accept_ratio << " average acceptance rate, "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                  << "us.\n";
    }
}

void LMCMeshGP::sample_nc_Lambda_std(){
  message("[gibbs_sample_Lambda] starting");
  start = std::chrono::steady_clock::now();
  
  //arma::mat wmean = LambdaHw * Lambdati;
  
  // new with botev's 2017 method to sample from truncated normal
  for(int j=0; j<q; j++){
    // build W
    int maxwhere = std::min(j, k-1);
    arma::uvec subcols = arma::find(Lambda_mask.row(j) == 1);
    
    // filter: choose value of spatial processes at locations of Yj that are available
    arma::mat WWj = wU.submat(ix_by_q_a(j), subcols); // acts as X //*********
    //wmean.submat(ix_by_q_a(j), subcols); // acts as X
    
    arma::mat Wcrossprod = WWj.t() * WWj; 
    
    arma::mat Lprior_inv = arma::eye(WWj.n_cols, WWj.n_cols); 
    
    arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * Wcrossprod + Lprior_inv
      ), "lower");
    
    arma::mat Sigma_chol_L = arma::inv(arma::trimatl(Si_chol));
    
    arma::mat Simean_L = tausq_inv(j) * WWj.t() * 
      (y.submat(ix_by_q_a(j), oneuv*j) - XB.submat(ix_by_q_a(j), oneuv*j));
    
    arma::mat Lambdarow_Sig = Sigma_chol_L.t() * Sigma_chol_L;
    
    arma::vec Lprior_mean = arma::zeros(subcols.n_elem);
    //if(j < 0){
    //  Lprior_mean(j) = arma::stddev(arma::vectorise( y.submat(ix_by_q_a(j), oneuv*j) ));
    //}
    arma::mat Lambdarow_mu = Lprior_inv * Lprior_mean + 
      Lambdarow_Sig * Simean_L;
    
    // truncation limits: start with (-inf, inf) then replace lower lim at appropriate loc
    arma::vec upper_lim = arma::zeros(subcols.n_elem);
    upper_lim.fill(arma::datum::inf);
    arma::vec lower_lim = arma::zeros(subcols.n_elem);
    lower_lim.fill(-arma::datum::inf);
    if(j < q){
      lower_lim(j) = 0;
    }
    arma::rowvec sampled = arma::trans(mvtruncnormal(Lambdarow_mu, lower_lim, upper_lim, Lambdarow_Sig, 1));
    
    //Rcpp::Rcout << "Lambda:  " << sampled;
    Lambda.submat(oneuv*j, subcols) = sampled;
  } 
  
  LambdaHw = w * Lambda.t();
  
  // refreshing density happens in the 'logpost_refresh_after_gibbs' function
  
  if(verbose & debug){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_Lambda] " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

void LMCMeshGP::gibbs_sample_tausq_std(double aprior, double bprior){
  message("[gibbs_sample_tausq_std] start");
  start = std::chrono::steady_clock::now();
  // note that at the available locations w already includes Lambda 
  
  arma::mat LHW = wU * Lambda.t();
  
  logpost = 0;
  for(int j=0; j<q; j++){
    arma::mat yrr = y.submat(ix_by_q_a(j), oneuv*j) - 
      XB.submat(ix_by_q_a(j), oneuv*j) - 
      LHW.submat(ix_by_q_a(j), oneuv*j); //***
    
    double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
    
    double aparam = aprior + ix_by_q_a(j).n_elem/2.0;
    double bparam = 1.0/( bprior + .5 * bcore );
    
    Rcpp::RNGScope scope;
    tausq_inv(j) = R::rgamma(aparam, bparam);
    logpost += 0.5 * (ix_by_q_a(j).n_elem + .0) * log(tausq_inv(j)) - 0.5*tausq_inv(j)*bcore;
    
    if(verbose & debug){
      Rcpp::Rcout << "[gibbs_sample_tausq] " << j << " | "
                  << aparam << " : " << bparam << " " << bcore << " --> " << 1.0/tausq_inv(j)
                  << "\n";
    }
  }
  if(verbose & debug){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_tausq] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                << "us.\n";
  }
  
}

void LMCMeshGP::gibbs_sample_tausq_fgrid(MeshDataLMC& data, double aprior, double bprior, bool ref_pardata){
  message("[gibbs_sample_tausq_fgrid] start (sampling via Robust adaptive Metropolis)");
  start = std::chrono::steady_clock::now();
  
  tausq_adapt.count_proposal();
  Rcpp::RNGScope scope;
  arma::vec new_tausq = 1.0/tausq_inv;
  arma::vec U_update = arma::randn(q);
  new_tausq = par_huvtransf_back(par_huvtransf_fwd(1.0/tausq_inv, tausq_unif_bounds) + 
    tausq_adapt.paramsd * U_update, tausq_unif_bounds);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    //for(int i=0; i<n_blocks; i++){
    //int u = block_names(i)-1;
    calc_DplusSi(u, alter_data, Lambda, 1.0/new_tausq);
    update_lly(u, alter_data, LambdaHw);
    
    //calc_DplusSi(u, param_data, tausq_inv);
    if(ref_pardata){
      update_lly(u, param_data, LambdaHw);
    }
  }
  
  double new_logpost = arma::accu(alter_data.ll_y);
  double start_logpost = arma::accu(param_data.ll_y);
  
  double prior_logratio = calc_prior_logratio(new_tausq, 1.0/tausq_inv, aprior, bprior);
  double jacobian  = calc_jacobian(new_tausq, 1.0/tausq_inv, tausq_unif_bounds);
  double logaccept = new_logpost - start_logpost + 
    prior_logratio +
    jacobian;
  
  bool accepted = do_I_accept(logaccept);
  if(accepted){
    tausq_adapt.count_accepted();
    // make the move
    tausq_inv = 1.0/new_tausq;
    param_data.DplusSi = alter_data.DplusSi;
    param_data.DplusSi_c = alter_data.DplusSi_c;
    param_data.DplusSi_ldet = alter_data.DplusSi_ldet;
    
    logpost = new_logpost;
  } else {
    logpost = start_logpost;
  }
  
  tausq_adapt.update_ratios();
  tausq_adapt.adapt(U_update, exp(logaccept), tausq_mcmc_counter); 
  tausq_mcmc_counter ++;
  if(verbose & debug){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_tausq_fgrid] " << tausq_adapt.accept_ratio << " average acceptance rate, "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                << "us.\n";
  }
}

void LMCMeshGP::update_block_w_cache(int u, MeshDataLMC& data){
  // 
  arma::mat Sigi_tot = build_block_diagonal_ptr(data.w_cond_prec_ptr.at(u));
  arma::mat Smu_tot = arma::zeros(k*indexing(u).n_elem, 1); // replace with fill(0)
  for(int c=0; c<children(u).n_elem; c++){
    int child = children(u)(c);
    arma::cube AK_u = cube_cols_ptr(data.w_cond_mean_K_ptr.at(child), u_is_which_col_f(u)(c)(0));
    AKuT_x_R_ptr(data.AK_uP(u)(c), AK_u, data.w_cond_prec_ptr.at(child)); 
    add_AK_AKu_multiply_(Sigi_tot, data.AK_uP(u)(c), AK_u);
  }

  if(forced_grid){
    int indxsize = indexing(u).n_elem;
    arma::mat yXB = arma::trans(y.rows(indexing_obs(u)) - XB.rows(indexing_obs(u)));
    for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
      if(na_1_blocks(u)(ix) == 1){
        arma::mat LambdaH = arma::zeros(q, k*indxsize);
        for(int j=0; j<q; j++){
          if(na_mat(indexing_obs(u)(ix), j) == 1){
            arma::mat Hloc = data.Hproject(u).slice(ix);
            for(int jx=0; jx<k; jx++){
              arma::mat Hsub = Hloc.row(jx); //data.Hproject(u).subcube(jx,0,ix,jx,indxsize-1, ix);
              // this outcome margin observed at this location
              
              LambdaH.submat(j, jx*indxsize, j, (jx+1)*indxsize-1) += Lambda(j, jx) * Hsub;
            }
          }
        }
        arma::mat LambdaH_DplusSi = LambdaH.t() * data.DplusSi.slice(indexing_obs(u)(ix));
        Smu_tot += LambdaH_DplusSi * yXB.col(ix);
        Sigi_tot += LambdaH_DplusSi * LambdaH;
      }
    }
  } else {
    arma::mat u_tau_inv = arma::zeros(indexing_obs(u).n_elem, q);
    arma::mat ytilde = arma::zeros(indexing_obs(u).n_elem, q);
    
    for(int j=0; j<q; j++){
      for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
        if(na_mat(indexing_obs(u)(ix), j) == 1){
          u_tau_inv(ix, j) = pow(tausq_inv(j), .5);
          ytilde(ix, j) = (y(indexing_obs(u)(ix), j) - XB(indexing_obs(u)(ix), j))*u_tau_inv(ix, j);
        }
      }
      // dont erase:
      //Sigi_tot += arma::kron( arma::trans(Lambda.row(j)) * Lambda.row(j), arma::diagmat(u_tau_inv%u_tau_inv));
      arma::mat LjtLj = arma::trans(Lambda.row(j)) * Lambda.row(j);
      arma::vec u_tausq_inv = u_tau_inv.col(j) % u_tau_inv.col(j);
      add_LtLxD(Sigi_tot, LjtLj, u_tausq_inv);
      
      Smu_tot += arma::vectorise(arma::diagmat(u_tau_inv.col(j)) * ytilde.col(j) * Lambda.row(j));
    }
  }
  data.Smu_start(u) = Smu_tot;
  data.Sigi_chol(u) = Sigi_tot;
  
  if((k>q) & (q>1)){
    arma::uvec blockdims = arma::cumsum( indexing(u).n_elem * arma::ones<arma::uvec>(k) );
    blockdims = arma::join_vert(oneuv * 0, blockdims);
    arma::uvec blockdims_q = blockdims.subvec(0, q-1);
    // WARNING: if k>q then we have only updated the lower block-triangular part of Sigi_tot for cholesky!
    // WARNING: we make use ONLY of the lower-blocktriangular part of Sigi_tot here.
    block_invcholesky_(data.Sigi_chol(u), blockdims_q);
  } else {
    data.Sigi_chol(u) = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
  }
}

void LMCMeshGP::sample_nonreference_w(int u, MeshDataLMC& data, const arma::mat& rand_norm_mat){
  //message("[sample_nonreference_w] start.");
  // for updating lambda and tau which will only look at observed locations
  // centered updates instead use the partially marginalized thing

  for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
    if(na_1_blocks(u)(ix) == 1){
      arma::mat Stemp = data.Riproject(u).slice(ix);
      arma::mat tsqi = tausq_inv;
      for(int j=0; j<q; j++){
        if(na_mat(indexing_obs(u)(ix), j) == 0){
          tsqi(j) = 0;
        }
      }
      
      arma::mat Smu_par = Stemp.diag() % arma::vectorise(w.row(indexing_obs(u)(ix)));
      arma::mat Smu_y = Lambda.t() * (tsqi % arma::trans(y.row(indexing_obs(u)(ix)) - XB.row(indexing_obs(u)(ix))));
      arma::mat Smu_tot = Smu_par + Smu_y;
      
      arma::mat Sigi_tot = Stemp + Lambda.t() * arma::diagmat(tsqi) * Lambda;
      arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
      
      arma::vec rnvec = arma::vectorise(rand_norm_mat.row(indexing_obs(u)(ix)));
      arma::vec wmean = Sigi_chol.t() * Sigi_chol * Smu_tot;
      
      wU.row(indexing_obs(u)(ix)) = arma::trans(wmean + Sigi_chol.t() * rnvec);
    }
  }
  //message("[sample_nonreference_w] done.");
}

void LMCMeshGP::gibbs_sample_w(MeshDataLMC& data, bool needs_update=true){
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_w] " << "\n";
  }
  //Rcpp::Rcout << "Lambda from:  " << Lambda_orig(0, 0) << " to  " << Lambda(0, 0) << endl;
  
  Rcpp::RNGScope scope;
  rand_norm_mat = arma::randn(coords.n_rows, k);
  start_overall = std::chrono::steady_clock::now();
  
  arma::field<arma::mat> LCi_cache(coords_caching.n_elem);
  if(recover_generator & matern_estim){
    for(int i=0; i<coords_caching.n_elem; i++){
      int u = coords_caching(i); // block name of ith representative
      arma::mat cx = coords.rows(indexing(u));
      arma::mat CCu = CmaternInv(cx, pow(Lambda(0,0), 2.0),
                                 data.theta(0,0), data.theta(1, 0), 1.0/tausq_inv(0));
      LCi_cache(i) = arma::inv(arma::trimatl(arma::chol(CCu, "lower")));
    }
  }
  
  for(int g=0; g<n_gibbs_groups; g++){
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      
      if((block_ct_obs(u) > 0)){
        
        update_block_w_cache(u, data);
        
        // recompute conditional mean
        arma::mat Smu_tot = data.Smu_start(u); //
        
        if(parents(u).n_elem>0){
          add_smu_parents_ptr_(Smu_tot, data.w_cond_prec_ptr.at(u), data.w_cond_mean_K_ptr.at(u),
                           w.rows( parents_indexing(u) ));
        } 
        
        for(int c=0; c<children(u).n_elem; c++){
          int child = children(u)(c);
          //---------------------
          arma::cube AK_u = cube_cols_ptr(data.w_cond_mean_K_ptr.at(child), u_is_which_col_f(u)(c)(0));
          
          arma::mat w_child = w.rows(indexing(child));
          arma::mat w_parchild = w.rows(parents_indexing(child));
          //---------------------
          if(parents(child).n_elem > 1){
            arma::cube AK_others = cube_cols_ptr(data.w_cond_mean_K_ptr.at(child), u_is_which_col_f(u)(c)(1));
            
            arma::mat w_parchild_others = w_parchild.rows(u_is_which_col_f(u)(c)(1));
            Smu_tot += 
              arma::vectorise(AK_vec_multiply(data.AK_uP(u)(c), 
                                              w_child - AK_vec_multiply(AK_others, w_parchild_others)));
          } else {
            Smu_tot += 
              arma::vectorise(AK_vec_multiply(data.AK_uP(u)(c), w_child));
          }
        }
        
        // sample
        arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
        arma::vec wmean = data.Sigi_chol(u).t() * data.Sigi_chol(u) * Smu_tot;
        arma::vec wtemp = wmean + data.Sigi_chol(u).t() * rnvec;
        
        w.rows(indexing(u)) = 
          arma::mat(wtemp.memptr(), wtemp.n_elem/k, k); 
        
        // non-ref effect on y at all locations. here we have already sampled
        wU.rows(indexing(u)) = w.rows(indexing(u));
        
        if(forced_grid){
          for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
            if(na_1_blocks(u)(ix) == 1){
              arma::mat wtemp = arma::sum(arma::trans(data.Hproject(u).slice(ix) % arma::trans(w.rows(indexing(u)))), 0);
              
              w.row(indexing_obs(u)(ix)) = wtemp;
              // do not sample here
              //LambdaHw.row(indexing_obs(u)(ix)) = w.row(indexing_obs(u)(ix)) * Lambda.t();
            }
          }
          sample_nonreference_w(u, data, rand_norm_mat);
        } 
        
        // *** kaust
        if(recover_generator & matern_estim){
          int ccfound = findcc(u);
          wgen.rows(indexing(u)) = LCi_cache(ccfound) * w.rows(indexing(u));
        }
      } 
    }
  }
  LambdaHw = w * Lambda.t();
  
  if(false || verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
}

void LMCMeshGP::refresh_w_cache(MeshDataLMC& data){
  if(verbose & debug){
    Rcpp::Rcout << "[refresh_w_cache] \n";
  }
  start_overall = std::chrono::steady_clock::now();
  for(int i=0; i<n_blocks; i++){
    int u=block_names(i)-1;
    update_block_w_cache(u, data);
  }
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[refresh_w_cache] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
}

void LMCMeshGP::predict(){
  start_overall = std::chrono::steady_clock::now();
  if(predict_group_exists == 1){
    message("[predict] start ");
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_predicts.n_elem; i++){ //*** subset to blocks with NA
      int u = u_predicts(i);// u_predicts(i);
      // only predictions at this block. 
      arma::uvec predict_parent_indexing, cx;
      arma::cube Kxxi_parents;
      
      if((block_ct_obs(u) > 0) & forced_grid){
        // this is a reference set with some observed locations
        predict_parent_indexing = indexing(u); // uses knots which by construction include all k processes
        int ccfound = findcc(u);
        CviaKron_HRj_chol_bdiag_wcache(Hpred(i), Rcholpred(i), param_data.Kxxi_cache(ccfound), na_1_blocks(u),
                                       coords, indexing_obs(u), predict_parent_indexing, k, param_data.theta, matern);
      } else {
        // no observed locations, use line of sight
        predict_parent_indexing = parents_indexing(u);
        CviaKron_HRj_chol_bdiag(Hpred(i), Rcholpred(i), Kxxi_parents,
                                na_1_blocks(u),
                                coords, indexing_obs(u), predict_parent_indexing, k, param_data.theta, matern);
      }

      for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
        if(na_1_blocks(u)(ix) == 0){
          arma::mat wpars = w.rows(predict_parent_indexing);
          
          arma::rowvec wtemp = arma::sum(arma::trans(Hpred(i).slice(ix)) % wpars, 0) + 
            arma::trans(Rcholpred(i).col(ix)) % rand_norm_mat.row(indexing_obs(u)(ix));

          w.row(indexing_obs(u)(ix)) = wtemp;
          wU.row(indexing_obs(u)(ix)) = wtemp;
          
          LambdaHw.row(indexing_obs(u)(ix)) = w.row(indexing_obs(u)(ix)) * Lambda.t();
        }
      }
      
      if(false){
        // this makes the prediction at grid points underlying non-observed areas
        // ONLY useful to make a full raster map at the end
        if((block_ct_obs(u) == 0) & forced_grid){
          arma::cube Hpredx = arma::zeros(k, predict_parent_indexing.n_elem, indexing(u).n_elem);
          arma::mat Rcholpredx = arma::zeros(k, indexing(u).n_elem);
          arma::uvec all_na = arma::zeros<arma::uvec>(indexing(u).n_elem);
          CviaKron_HRj_chol_bdiag_wcache(Hpredx, Rcholpredx, Kxxi_parents, all_na, 
                                         coords, indexing(u), predict_parent_indexing, k, param_data.theta, matern);
          for(int ix=0; ix<indexing(u).n_elem; ix++){
            arma::mat wpars = w.rows(predict_parent_indexing);
            arma::rowvec wtemp = arma::sum(arma::trans(Hpredx.slice(ix)) % wpars, 0) + 
              arma::trans(Rcholpredx.col(ix)) % rand_norm_mat.row(indexing(u)(ix));
            w.row(indexing(u)(ix)) = wtemp;
            LambdaHw.row(indexing(u)(ix)) = w.row(indexing(u)(ix)) * Lambda.t();
          }
        }
      }
    }
    
    if(verbose & debug){
      end_overall = std::chrono::steady_clock::now();
      Rcpp::Rcout << "[predict] "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                  << "us. ";
    }
  }
}

void LMCMeshGP::theta_update(MeshDataLMC& data, const arma::mat& new_theta){
  message("[theta_update] Updating theta");
  data.theta = new_theta;
}

void LMCMeshGP::tausq_update(double new_tausq){
  tausq_inv = 1.0/new_tausq;
}

void LMCMeshGP::beta_update(const arma::vec& new_beta){ 
  Bcoeff = new_beta;
}

void LMCMeshGP::accept_make_change(){
  std::swap(param_data, alter_data);
}

