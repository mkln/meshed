#define ARMA_DONT_PRINT_ERRORS

#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdexcept>

#include "R.h"
#include "mgp_utils.h"
#include "mh_adapt.h"
#include "caching_pairwise_compare.h"
#include "covariance_lmc.h"

#include "truncmvnorm.h"

using namespace std;

const double hl2pi = -.5 * log(2.0 * M_PI);


class UniMeshGP {
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
  arma::field<arma::uvec> children_indexing;
  
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
  arma::uvec                ref_block_names;
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
  MeshDataUni param_data; 
  MeshDataUni alter_data;
  
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
  int starting_kr;
  
  // caching some matrices // ***
  arma::field<arma::mat> H_cache;
  arma::field<arma::mat> Ri_cache;
  
  arma::vec Ri_chol_logdet;
  
  arma::field<arma::cube> Hpred;
  arma::field<arma::mat> Rcholpred;
  
  // MCMC
  void update_block_covpars(int, MeshDataUni& data);
  void update_block_wlogdens(int, MeshDataUni& data);
  void update_lly(int, MeshDataUni&, const arma::mat& LamHw);
  
  bool refresh_cache(MeshDataUni& data);
  bool calc_ywlogdens(MeshDataUni& data);
  // 
  bool get_loglik_comps_w(MeshDataUni& data);
  
  // update_block_wpars used for sampling and proposing w
  // calculates all conditional means and variances
  void calc_DplusSi(int, MeshDataUni& data, const arma::mat& Lam, const arma::vec& tsqi);
  void update_block_w_cache(int, MeshDataUni& data, arma::vec& );
  void sample_nonreference_w(int, MeshDataUni& data, const arma::mat& );
  void refresh_w_cache(MeshDataUni& data);
  void gibbs_sample_w(MeshDataUni& data, bool needs_update);
  
  //
  void gibbs_sample_beta();
  
  void deal_with_Lambda(MeshDataUni& data);
  void sample_nc_Lambda_std(); // noncentered
  void sample_nc_Lambda_fgrid(MeshDataUni& data);
  void refresh_after_lambda();
  
  void deal_with_tausq(MeshDataUni& data, double, double, bool);
  void gibbs_sample_tausq_std(double, double);
  void gibbs_sample_tausq_fgrid(MeshDataUni& data, double, double, bool);
  
  void logpost_refresh_after_gibbs(MeshDataUni& data); //***
  
  void predict();
  
  double logpost;
  
  // changing the values, no sampling;
  void theta_update(MeshDataUni&, const arma::mat&);
  void beta_update(const arma::vec&);
  void tausq_update(double);
  
  // avoid expensive copies
  void accept_make_change();
  
  std::chrono::steady_clock::time_point start_overall;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point end_overall;
  
  // empty
  UniMeshGP();
  
  // build everything
  UniMeshGP(
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

void UniMeshGP::message(string s){
  if(verbose & debug){
    Rcpp::Rcout << s << "\n";
  }
}

UniMeshGP::UniMeshGP(){
  
}

UniMeshGP::UniMeshGP(
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
  
  message("UniMeshGP::UniMeshGP initialization.\n");
  
  forced_grid = use_forced_grid;
  
  start_overall = std::chrono::steady_clock::now();
  
  cached = true;
  
  if(forced_grid){
    message("MGP on a latent grid");
  } else {
    message("MGP on data grid, caching activated");
  }
  
  message("[UniMeshGP::UniMeshGP] assign values.");
  // data
  y                   = y_in;
  X                   = X_in;
  
  na_mat = arma::zeros<arma::umat>(arma::size(y));
  na_mat.elem(arma::find_finite(y)).fill(1);
  
  //n  = na_ix_all.n_elem;
  p  = X.n_cols;
  
  // spatial coordinates and dimension
  coords              = coords_in;
  dd = coords.n_cols;
  
  // outcome variables
  //qvblock_c           = mv_id_in-1;
  //arma::uvec mv_id_uniques = arma::unique(mv_id_in);
  q = y.n_cols;//mv_id_uniques.n_elem;
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
    Rcpp::Rcout << "Y : " << ix_by_q_a(j).n_elem << " observed locations.\n";
  }
  
  
  // DAG
  parents             = parents_in;
  children            = children_in;
  block_names         = block_names_in;
  block_groups        = block_groups_in;
  block_groups_labels = arma::unique(block_groups);
  n_gibbs_groups      = block_groups_labels.n_elem;
  n_blocks            = block_names.n_elem;
  
  //Z_available = Z.rows(na_ix_all);
  //Zw = arma::zeros(coords.n_rows);
  
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
  
  tausq_inv        = //arma::ones(q) * 
    tausq_inv_in;
  XB = arma::zeros(coords.n_rows, q);
  Bcoeff           = beta_in; //arma::zeros(p, q);
  for(int j=0; j<q; j++){
    XB.col(j) = X * Bcoeff.col(j);// beta_in;
    //Bcoeff.col(j) = beta_in;
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
  message("UniMeshGP::UniMeshGP : init_indexing()");
  init_indexing();
  
  message("UniMeshGP::UniMeshGP : na_study()");
  na_study();
  // now we know where NAs are, we can erase them
  y.elem(arma::find_nonfinite(y)).fill(0);
  
  message("UniMeshGP::UniMeshGP : init_finalize()");
  init_finalize();
  
  message("UniMeshGP::UniMeshGP : make_gibbs_groups()");
  // quick check for groups
  make_gibbs_groups();
  
  //caching;
  if(cached){
    message("UniMeshGP::UniMeshGP : init_cache()");
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
    Rcpp::Rcout << "UniMeshGP::UniMeshGP initializing took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
}


void UniMeshGP::make_gibbs_groups(){
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

void UniMeshGP::na_study(){
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
  ref_block_names = arma::zeros<arma::uvec>(n_ref_blocks);
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i) - 1;
    if(block_ct_obs(u) > 0){
      reference_blocks(j) = i;
      ref_block_names(j) = u;
      j ++;
    } 
  }
  message("[na_study] done.");
}

void UniMeshGP::fill_zeros_Kcache(){
  message("[fill_zeros_Kcache]");
  // ***
  H_cache = arma::field<arma::mat> (kr_caching.n_elem);
  Ri_cache = arma::field<arma::mat> (kr_caching.n_elem);
  //Richol_cache = arma::field<arma::cube> (kr_caching.n_elem);
  
  Ri_chol_logdet = arma::zeros(kr_caching.n_elem);
  
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    H_cache(i) = 
      arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem);
    Ri_cache(i) = 
      arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
    //Richol_cache(i) = 
    //  arma::zeros(indexing(u).n_elem, indexing(u).n_elem, k);
  }
  
  message("[fill_zeros_Kcache] done.");
}

void UniMeshGP::init_cache(){
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
    if(parents_indexing(u).n_elem > 0){//parents_coords(u).n_rows > 0){
      arma::mat cmat = coords.rows(indexing(u));
      arma::mat pmat = coords.rows(parents_indexing(u));
      arma::mat kr_mat_c = arma::join_vert(cmat, pmat);
      
      kr_pairing(u) = kr_mat_c;//arma::join_horiz(kr_mat_c, kr_mat_mvid);
    } else {
      kr_pairing(u) = arma::zeros(arma::size(parents_indexing(u)));
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
  
  if(verbose & debug || true){
    Rcpp::Rcout << "Caching c: " << coords_caching.n_elem 
                << " k: " << kr_caching.n_elem << "\n";
  }
  message("[init_cache]");
}

void UniMeshGP::init_meshdata(const arma::mat& theta_in){
  message("[init_meshdata]");
  
  // block params
  param_data.w_cond_mean_K = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_prec   = arma::field<arma::mat> (n_blocks);
  //param_data.w_cond_precchol   = arma::field<arma::cube> (n_blocks);
  param_data.w_cond_prec_times_cmk = arma::field<arma::mat> (n_blocks);
  
  param_data.Rproject = arma::field<arma::cube>(n_blocks);
  param_data.Riproject = arma::field<arma::cube>(n_blocks);
  param_data.Hproject = arma::field<arma::cube>(n_blocks);
  //param_data.Ddiag = arma::zeros(arma::size(y)); 
  
  param_data.Smu_start = arma::field<arma::mat>(n_blocks);
  param_data.Sigi_chol = arma::field<arma::mat>(n_blocks);
  param_data.AK_uP = arma::field<arma::field<arma::mat> >(n_blocks);
  //param_data.LambdaH_Ditau = arma::field<arma::field<arma::mat> >(n_blocks);
  
  for(int i=0; i<n_blocks; i++){
    int u=block_names(i) - 1;
    param_data.w_cond_mean_K(i) = arma::zeros(indexing(i).n_elem, parents_indexing(i).n_elem);
    param_data.w_cond_prec(i) = arma::zeros(indexing(i).n_elem, indexing(i).n_elem);
    //param_data.w_cond_precchol(i) = arma::zeros(indexing(i).n_elem, indexing(i).n_elem, k);
    param_data.w_cond_prec_times_cmk(i) = arma::zeros(indexing(i).n_elem, parents_indexing(i).n_elem);
    
    param_data.Hproject(i) = arma::zeros(k, //k*
                        indexing(i).n_elem, indexing_obs(i).n_elem);
    param_data.Rproject(i) = arma::zeros(k, k, indexing_obs(i).n_elem);
    param_data.Riproject(i) = arma::zeros(k, k, indexing_obs(i).n_elem);
    
    param_data.Smu_start(i) = arma::zeros(k*indexing(i).n_elem, 1);
    param_data.Sigi_chol(i) = arma::zeros(k*indexing(i).n_elem, k*indexing(i).n_elem);
    param_data.AK_uP(i) = arma::field<arma::mat>(children(i).n_elem);
    for(int c=0; c<children(i).n_elem; c++){
      int child = children(i)(c);
      param_data.AK_uP(i)(c) = arma::zeros(indexing(i).n_elem, indexing(child).n_elem);
    }
    //param_data.LambdaH_Ditau(i) = arma::field<arma::mat> (q);
  }
  
  param_data.Kxxi_cache = arma::field<arma::mat>(coords_caching.n_elem);
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i);
    param_data.Kxxi_cache(i) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
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
  
  alter_data = param_data; 
  
  message("[init_meshdata] done.");
}

void UniMeshGP::init_indexing(){
  
  parents_indexing = arma::field<arma::uvec> (n_blocks);
  children_indexing = arma::field<arma::uvec> (n_blocks);
  
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

void UniMeshGP::init_finalize(){
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
      //Rcpp::Rcout << "u: " << u << " has children: " << children(u).t() << "\n";
      
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

bool UniMeshGP::refresh_cache(MeshDataUni& data){
  start_overall = std::chrono::steady_clock::now();
  message("[refresh_cache] start.");
  
  Ri_chol_logdet = arma::zeros(kr_caching.n_elem);
  arma::vec timings = arma::zeros(2);
  int errtype = -1;
  
  arma::field<arma::mat> CC_cache(coords_caching.n_elem);
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i); 
    if(block_ct_obs(u) > 0){
      CC_cache(i) = Correlationf(coords.rows(indexing(u)), coords.rows(indexing(u)), 
               data.theta, matern, true);
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
        data.Kxxi_cache(i) = arma::inv_sympd( Correlationf(coords.rows(indexing(u)), coords.rows(indexing(u)), 
                                              data.theta, matern, true) );
      } catch (...) {
        errtype = 1;
      }
    } else {
      // this means we are caching kr
      i = it - starting_kr;
      int u = kr_caching(i);
      try {
        if(block_ct_obs(u) > 0){
          //arma::mat Cxx = Correlationf(coords.rows(indexing(u)), coords.rows(indexing(u)), 
          //                             data.theta, matern, true);
          int u_cached_ix = coords_caching_ix(u);
          arma::uvec cx = arma::find( coords_caching == u_cached_ix );
          arma::mat Cxx = CC_cache(cx(0));
          
          arma::mat Cxy = Correlationf(coords.rows(indexing(u)), coords.rows(parents_indexing(u)), 
                                       data.theta, matern, false);
          arma::mat Cyy_i = arma::inv_sympd(
            Correlationf(coords.rows(parents_indexing(u)), coords.rows(parents_indexing(u)), 
                         data.theta, matern, true) );
          
          arma::mat Hloc = Cxy * Cyy_i;
          arma::mat Rloc_ichol = arma::inv(arma::trimatl(arma::chol( arma::symmatu(
            Cxx - Hloc * Cxy.t()) , "lower")));
          Ri_chol_logdet(i) = arma::accu(log(Rloc_ichol.diag()));
          
          if(parents_indexing(u).n_elem > 0){
            H_cache(i) = Hloc;
          }
          Ri_cache(i) = Rloc_ichol.t() * Rloc_ichol;//Ri.submat(firstrow, firstrow, lastrow, lastrow) = Rloc_ichol.t() * Rloc_ichol; // symmetric
          
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

void UniMeshGP::update_block_covpars(int u, MeshDataUni& data){
  //message("[update_block_covpars] start.");
  // given block u as input, this function updates H and R
  // which will be used later to compute logp(w | theta)
  if( parents(u).n_elem > 0 ){
    int kr_cached_ix = kr_caching_ix(u);
    arma::uvec cpx = arma::find(kr_caching == kr_cached_ix, 1, "first");
    data.w_cond_mean_K(u) = H_cache(cpx(0));
    data.w_cond_prec(u) = Ri_cache(cpx(0));
    
    //data.w_cond_prec_times_cmk(u) = data.w_cond_prec(u) * data.w_cond_mean_K(u);
    
    data.logdetCi_comps(u) = Ri_chol_logdet(cpx(0));
  } else {
    arma::mat Kcc = Correlationf(coords.rows(indexing(u)), coords.rows(indexing(u)), data.theta, matern, true);
    arma::mat CC_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Kcc ), "lower")));

    data.w_cond_prec(u) = CC_chol.t()*CC_chol;
    data.logdetCi_comps(u) = arma::accu(log(CC_chol.diag()));
  }
  
  if(forced_grid){
    int u_cached_ix = coords_caching_ix(u);
    arma::uvec cx = arma::find( coords_caching == u_cached_ix);//, 1, "first" );
    arma::mat Cyy_i = data.Kxxi_cache(cx(0));
    //arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), matern, true) );
    
    for(int ix=0; ix<indexing_obs(u).n_rows; ix++){
      if(na_1_blocks(u)(ix) == 1){
        arma::mat Cxx = Correlationf(coords.row(indexing_obs(u)(ix)), coords.row(indexing_obs(u)(ix)), 
                                     data.theta, matern, true);
        arma::mat Cxy = Correlationf(coords.row(indexing_obs(u)(ix)), coords.rows(indexing(u)), 
                                     data.theta, matern, false);
        arma::mat Hloc = Cxy * Cyy_i;
        arma::mat R = Cxx - Hloc * Cxy.t();
        
        data.Hproject(u).subcube(0, 0, ix, 0, 0, ix) = Hloc;
        data.Rproject(u)(0, 0, ix) = R(0,0) < 0 ? 0.0 : R(0,0);
        data.Riproject(u)(0, 0, ix) = R(0,0) < 1e-14 ? 0.0 : 1.0/R(0,0); //1e-10
      }
    }
  }
  
  //message("[update_block_covpars] done.");
}

void UniMeshGP::update_block_wlogdens(int u, MeshDataUni& data){
  //message("[update_block_wlogdens].");
  arma::mat wx = w.rows(indexing(u));
  arma::mat wcoresum = arma::zeros(1, 1);
  if( parents(u).n_elem > 0 ){
    arma::mat wpar = w.rows(parents_indexing(u));
      wx = wx - data.w_cond_mean_K(u) * wpar;
  }
  
    wcoresum(0) = 
      arma::conv_to<double>::from(wx.t() * data.w_cond_prec(u) * wx);
  
  
  data.wcore.row(u) = arma::accu(wcoresum);
  data.loglik_w_comps.row(u) = (indexing(u).n_elem+.0) * hl2pi -.5 * arma::accu(wcoresum); //
  //arma::accu(data.wcore.slice(u).diag());

  //message("[update_block_wlogdens] done.");
}

void UniMeshGP::calc_DplusSi(int u, 
          MeshDataUni & data, const arma::mat& Lam, const arma::vec& tsqi){
  //message("[calc_DplusSi] start.");
  int indxsize = indexing(u).n_elem;

  for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
    if(na_1_blocks(u)(ix) == 1){
      arma::mat Dtau = Lam(0, 0) * Lam(0, 0) * data.Rproject(u).slice(ix) + 1/tsqi(0);
      // fill 
      data.DplusSi_ldet(indexing_obs(u)(ix)) = - log(Dtau(0,0));
      data.DplusSi.slice(indexing_obs(u)(ix)) = 1.0/Dtau; // 1.0/ (L * L);
      data.DplusSi_c.slice(indexing_obs(u)(ix)) = pow(Dtau, -0.5);
    }
  }
}

void UniMeshGP::update_lly(int u, MeshDataUni& data, const arma::mat& LamHw){
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

void UniMeshGP::logpost_refresh_after_gibbs(MeshDataUni& data){
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

bool UniMeshGP::calc_ywlogdens(MeshDataUni& data){
  start_overall = std::chrono::steady_clock::now();
  //message("[calc_ywlogdens] start.");
  // called for a proposal of theta
  // updates involve the covariances
  // and Sigma for adjusting the error terms
  arma::vec timing = arma::zeros(2);
  
  //start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    
    start = std::chrono::steady_clock::now();
    update_block_covpars(u, data);
    end = std::chrono::steady_clock::now();
    //timing(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::steady_clock::now();
    update_block_wlogdens(u, data);
    end = std::chrono::steady_clock::now();
    //timing(1) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    if(forced_grid){
      calc_DplusSi(u, data, Lambda, tausq_inv);
      update_lly(u, data, LambdaHw);
    }
  }
  
  //Rcpp::Rcout << "Timing: " << timing.t();
  
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

bool UniMeshGP::get_loglik_comps_w(MeshDataUni& data){
  start = std::chrono::steady_clock::now();
  bool acceptable = refresh_cache(data);
  end = std::chrono::steady_clock::now();
  
  //Rcpp::Rcout << 
  //  "cache timing: " << 
  //  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << endl;
  
  if(acceptable){
    acceptable = calc_ywlogdens(data);
    return acceptable;
  } else {
    return acceptable;
  }
}

void UniMeshGP::gibbs_sample_beta(){
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

void UniMeshGP::deal_with_Lambda(MeshDataUni& data){
  bool randomize_update = (R::runif(0,1) > .5) || (y.n_rows < 50000);
  if(forced_grid & randomize_update){
    sample_nc_Lambda_fgrid(data);
  } else {
    sample_nc_Lambda_std();
  }
}

void UniMeshGP::deal_with_tausq(MeshDataUni& data, double aprior=2.001, double bprior=1, bool ref_pardata=false){
  bool randomize_update = (R::runif(0,1) > .5) || (y.n_rows < 50000);
  // ref_pardata: set to true if this is called without calling deal_with_Lambda first
  if(forced_grid & randomize_update){
    gibbs_sample_tausq_fgrid(data, aprior, bprior, ref_pardata);
  } else {
    gibbs_sample_tausq_std(aprior, bprior);
  }
}

void UniMeshGP::sample_nc_Lambda_fgrid(MeshDataUni& data){
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

void UniMeshGP::sample_nc_Lambda_std(){
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
    
    arma::mat Lprior_inv = tausq_inv(j) * 
      arma::eye(WWj.n_cols, WWj.n_cols) * .001; 
    
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

void UniMeshGP::refresh_after_lambda(){
  // refresh LambdaHw and Ddiag
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    
    if(forced_grid){
      //calc_DplusSi(u, param_data, tausq_inv);
      //update_lly(u, param_data, LambdaHw);
    }
    
    //arma::mat wtemp = LambdaHw.rows(indexing_obs(u)) * Lambdati;
    LambdaHw.rows(indexing_obs(u)) = w.rows(indexing_obs(u)) * 
      Lambda.t();
    LambdaHw.rows(indexing(u)) = w.rows(indexing(u)) * 
      Lambda.t();
  }
}

void UniMeshGP::gibbs_sample_tausq_std(double aprior, double bprior){
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

void UniMeshGP::gibbs_sample_tausq_fgrid(MeshDataUni& data, double aprior, double bprior, bool ref_pardata){
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

void UniMeshGP::update_block_w_cache(int u, MeshDataUni& data, arma::vec& timing){
  // 
  arma::mat Sigi_tot = data.w_cond_prec(u);
  arma::mat Smu_tot = arma::zeros(k*indexing(u).n_elem, 1); // replace with fill(0)
  for(int c=0; c<children(u).n_elem; c++){
    int child = children(u)(c);
    arma::mat AK_u = data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(0));
    data.AK_uP(u)(c) = AK_u.t() * data.w_cond_prec(child);
    Sigi_tot += data.AK_uP(u)(c) * AK_u;// childdims);
  }

  //start_overall = std::chrono::steady_clock::now();
  if(forced_grid){
    int indxsize = indexing(u).n_elem;

      for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
        if(na_1_blocks(u)(ix) == 1){
          arma::mat LambdaH = data.Hproject(u).slice(ix) * Lambda(0,0);
          //start = std::chrono::steady_clock::now();
          arma::mat LambdaH_DplusSi = LambdaH.t() * data.DplusSi(0, 0, indexing_obs(u)(ix));
          Smu_tot += LambdaH_DplusSi * (y(indexing_obs(u)(ix)) - XB(indexing_obs(u)(ix)));
          Sigi_tot += LambdaH_DplusSi * LambdaH;
          //end = std::chrono::steady_clock::now();
          //timing(1) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
      }
    
  } else {
    
      for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
        if(na_1_blocks(u)(ix) == 1){
          Sigi_tot(ix, ix) += Lambda(0,0) * Lambda(0,0) * tausq_inv(0);
          Smu_tot(ix) += Lambda(0,0) * tausq_inv(0) * (y(indexing_obs(u)(ix)) - XB(indexing_obs(u)(ix)));
        }
      }
    
  }
  //end_overall = std::chrono::steady_clock::now();
  //timing(2) += std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count();
  
  //start = std::chrono::steady_clock::now();
  data.Smu_start(u) = Smu_tot;
  
  data.Sigi_chol(u) = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
  
  //Rcpp::Rcout << "size of Sigi_tot " << arma::size(Sigi_tot) << endl;
  //end = std::chrono::steady_clock::now();
  //timing(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void UniMeshGP::sample_nonreference_w(int u, MeshDataUni& data, const arma::mat& rand_norm_mat){
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

void UniMeshGP::gibbs_sample_w(MeshDataUni& data, bool needs_update=true){
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_w] " << "\n";
  }
  //Rcpp::Rcout << "Lambda from:  " << Lambda_orig(0, 0) << " to  " << Lambda(0, 0) << endl;
  
  Rcpp::RNGScope scope;
  rand_norm_mat = arma::randn(coords.n_rows, k);
  start_overall = std::chrono::steady_clock::now();
  arma::vec timing = arma::zeros(5);
  
  //start = std::chrono::steady_clock::now();
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
  end = std::chrono::steady_clock::now();
  //timing(0) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
  /*
  start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    
    update_block_w_cache(u, data, timing);
  }
  end = std::chrono::steady_clock::now();
  timing(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  */
  
  for(int g=0; g<n_gibbs_groups; g++){
    //for(int g=n_gibbs_groups-1; g>=0; g--){
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      
      if((block_ct_obs(u) > 0)){
        
      
        update_block_w_cache(u, data, timing);
        
        // recompute conditional mean
        arma::mat Smu_tot = data.Smu_start(u); //
        
        if(parents(u).n_elem>0){
          Smu_tot += data.w_cond_prec(u) * data.w_cond_mean_K(u) * //data.w_cond_prec_times_cmk(u) * 
            w.rows(parents_indexing(u));
        } 
        for(int c=0; c<children(u).n_elem; c++){
          int child = children(u)(c);
          //---------------------
          
          arma::mat AK_u = data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(0));
          
          arma::mat w_child = w.rows(indexing(child));
          arma::mat w_parchild = w.rows(parents_indexing(child));
          //---------------------
          if(parents(child).n_elem > 1){
            start = std::chrono::steady_clock::now();
            arma::mat AK_others = data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(1));
            
            arma::mat w_parchild_others = w_parchild.rows(u_is_which_col_f(u)(c)(1));
            Smu_tot += data.AK_uP(u)(c) * (w_child - AK_others*w_parchild_others);
            end = std::chrono::steady_clock::now();
            
          } else {
            Smu_tot += data.AK_uP(u)(c) * w_child;
          }
        }
        end = std::chrono::steady_clock::now();
        //timing(1) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::steady_clock::now();
        // sample
        
        arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
        arma::vec wmean = data.Sigi_chol(u).t() * data.Sigi_chol(u) * Smu_tot;
        
        arma::vec wtemp = wmean + data.Sigi_chol(u).t() * rnvec;
        
        w.rows(indexing(u)) = //arma::trans(arma::mat(wtemp.memptr(), k, wtemp.n_elem/k)); 
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
        
        end = std::chrono::steady_clock::now();
        //timing(2) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        //start = std::chrono::steady_clock::now();
        // *** kaust
        if(recover_generator & matern_estim){
          int u_cached_ix = coords_caching_ix(u);
          arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first" );
          wgen.rows(indexing(u)) = LCi_cache(cx(0)) * w.rows(indexing(u));
        }
        //end = std::chrono::steady_clock::now();
        //timing(4) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
      } 
    }
  }

  //start = std::chrono::steady_clock::now();
  LambdaHw = w * Lambda.t();
  //end = std::chrono::steady_clock::now();
  //timing(3) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  
  if(false || (verbose & debug)){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "timing: " << timing.t();
    Rcpp::Rcout << "[gibbs_sample_w] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
}

void UniMeshGP::refresh_w_cache(MeshDataUni& data){
  if(verbose & debug){
    Rcpp::Rcout << "[refresh_w_cache] \n";
  }
  start_overall = std::chrono::steady_clock::now();
  arma::vec timing = arma::zeros(2);
  for(int i=0; i<n_blocks; i++){
    int u=block_names(i)-1;
    update_block_w_cache(u, data, timing);
  }
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[refresh_w_cache] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
    
  }
}

void UniMeshGP::predict(){
  //arma::vec timings = arma::zeros(5);
  if(predict_group_exists == 1){
    start_overall = std::chrono::steady_clock::now();
    arma::vec timer = arma::zeros(3);
    
    message("[predict] start ");
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_predicts.n_elem; i++){ //*** subset to blocks with NA
      int u = u_predicts(i);// u_predicts(i);
      // only predictions at this block. 
      arma::uvec predict_parent_indexing, cx;
      
      arma::mat Kxxi_parents;
      start = std::chrono::steady_clock::now();
      if((block_ct_obs(u) > 0) & forced_grid){
        // this is a reference set with some observed locations
        int u_cached_ix = coords_caching_ix(u);
        predict_parent_indexing = indexing(u); // uses knots which by construction include all k processes
        arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first" );
        //arma::mat Iselect = arma::eye(k, k);
        
        arma::mat coordsy = coords.rows(predict_parent_indexing);
        for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
          if(na_1_blocks(u)(ix) == 0){ // otherwise it's not missing
            arma::mat Cxx = Correlationf(coords.row(indexing_obs(u)(ix)), 
                                         coords.row(indexing_obs(u)(ix)), param_data.theta, matern, true);
            arma::mat Cxy = Correlationf(coords.row(indexing_obs(u)(ix)), coordsy, param_data.theta, matern, false);
            arma::mat Hloc = Cxy * param_data.Kxxi_cache(cx(0));
              
            Hpred(i).slice(ix).row(0) = Hloc;//+=arma::kron(arma::diagmat(Iselect.col(j)), Hloc);
            double Rcholtemp = arma::conv_to<double>::from(
              Cxx - Hloc * Cxy.t() );
            Rcholtemp = Rcholtemp < 0 ? 0.0 : Rcholtemp;
            Rcholpred(i)(0,ix) = pow(Rcholtemp, .5); // 0 could be numerically negative
          }
        }
      
      } else {
        // no observed locations, use line of sight
        predict_parent_indexing = parents_indexing(u);
        arma::mat coordsy = coords.rows(predict_parent_indexing);
        Kxxi_parents = arma::inv_sympd( Correlationf(coordsy, coordsy, param_data.theta, matern, true) );
        for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
          if(na_1_blocks(u)(ix) == 0){ // otherwise it's not missing
            arma::mat Cxx = Correlationf(coords.row(indexing_obs(u)(ix)), 
                                         coords.row(indexing_obs(u)(ix)), param_data.theta, matern, true);
            arma::mat Cxy = Correlationf(coords.row(indexing_obs(u)(ix)), coordsy, param_data.theta, matern, false);
            arma::mat Hloc = Cxy * Kxxi_parents;
            
            Hpred(i).slice(ix).row(0) = Hloc;//+=arma::kron(arma::diagmat(Iselect.col(j)), Hloc);
            double Rcholtemp = arma::conv_to<double>::from(
              Cxx - Hloc * Cxy.t() );
            Rcholtemp = Rcholtemp < 0 ? 0.0 : Rcholtemp;
            Rcholpred(i)(0, ix) = pow(Rcholtemp, .5); // 0 could be numerically negative
          }
        }
      }
      end = std::chrono::steady_clock::now();
      timer(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      //Rcpp::Rcout << "step 2" << "\n";
      
      start = std::chrono::steady_clock::now();
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
      end = std::chrono::steady_clock::now();
      timer(1) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    
    if(verbose & debug){
      end_overall = std::chrono::steady_clock::now();
      Rcpp::Rcout << "[predict] "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                  << "us. ";
      Rcpp::Rcout << timer.t() << "\n";
    }
  }
}

void UniMeshGP::theta_update(MeshDataUni& data, const arma::mat& new_theta){
  message("[theta_update] Updating theta");
  data.theta = new_theta;
}

void UniMeshGP::tausq_update(double new_tausq){
  tausq_inv = 1.0/new_tausq;
}

void UniMeshGP::beta_update(const arma::vec& new_beta){ 
  Bcoeff = new_beta;
}

void UniMeshGP::accept_make_change(){
  std::swap(param_data, alter_data);
}

