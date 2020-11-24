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

inline void block_invcholesky_(arma::mat& X, const arma::uvec& upleft_cumblocksizes){
  // inplace
  // this function computes inv(chol(X)) when 
  // X is a block matrix in which the upper left block itself is block diagonal
  // the cumulative block sizes of the blockdiagonal block of X are specified in upleft_cumblocksizes
  // the size of upleft_cumblocksizes is 1+number of blocks, and its first element is 0
  // --
  // the function proceeds by computing the upper left inverse cholesky
  // ends with the schur matrix for the lower right block of the result
  
  int upperleft_nblocks = upleft_cumblocksizes.n_elem - 1;
  int n_upleft = upleft_cumblocksizes(upleft_cumblocksizes.n_elem-1);
  
  X.submat(0, n_upleft, n_upleft-1, X.n_cols-1).fill(0); // upper-right block-corner erased
  
  arma::mat B = X.submat(n_upleft, 0, X.n_rows-1, n_upleft-1);
  arma::mat D = X.submat(n_upleft, n_upleft, X.n_rows-1, X.n_rows-1);
  arma::mat BLAinvt = arma::zeros(B.n_rows, n_upleft);
  arma::mat BLAinvtLAinv = arma::zeros(B.n_rows, n_upleft);
  
  for(int i=0; i<upperleft_nblocks; i++){
    arma::mat LAinv = arma::inv(arma::trimatl(arma::chol(
      arma::symmatu(X.submat(upleft_cumblocksizes(i), upleft_cumblocksizes(i), 
                             upleft_cumblocksizes(i+1) - 1, upleft_cumblocksizes(i+1) - 1)), "lower")));
    
    X.submat(upleft_cumblocksizes(i), upleft_cumblocksizes(i), 
                  upleft_cumblocksizes(i+1) - 1, upleft_cumblocksizes(i+1) - 1) = LAinv;
    BLAinvt.cols(upleft_cumblocksizes(i), upleft_cumblocksizes(i+1) - 1) = 
      B.cols(upleft_cumblocksizes(i), upleft_cumblocksizes(i+1) - 1) * LAinv.t();
    BLAinvtLAinv.cols(upleft_cumblocksizes(i), upleft_cumblocksizes(i+1) - 1) =
      BLAinvt.cols(upleft_cumblocksizes(i), upleft_cumblocksizes(i+1) - 1) * LAinv;
  }
  
  arma::mat invcholSchur = arma::inv(arma::trimatl(arma::chol(
    arma::symmatu(D - BLAinvt * BLAinvt.t()), "lower")));
  X.submat(n_upleft, 0, X.n_rows-1, n_upleft-1) = - invcholSchur * BLAinvtLAinv;
  X.submat(n_upleft, n_upleft, X.n_rows-1, X.n_rows-1) = invcholSchur;
}

inline void add_smu_parents_(arma::mat& result, 
                             const arma::cube& condprec_cmk, 
                             const arma::mat& wparents,
                             const arma::uvec& blockdims){
  int n_blocks = condprec_cmk.n_slices;
  for(int i=0; i<n_blocks; i++){
    result.rows(blockdims(i), blockdims(i+1)-1) +=
      condprec_cmk.slice(i) * wparents.col(i);
  }
}

inline arma::cube AKuT_x_R(const arma::cube& x, const arma::cube& y){ 
  arma::cube result = arma::zeros(x.n_cols, y.n_cols, x.n_slices);
  for(int i=0; i<x.n_slices; i++){
    result.slice(i) = arma::trans(x.slice(i)) * y.slice(i); 
  }
  return result;
}

inline void add_AK_AKu_multiply_(arma::mat& result,
                            const arma::cube& x, const arma::cube& y, 
                                 const arma::uvec& outerdims){
  int n_blocks = outerdims.n_elem-1;
  for(int i=0; i<n_blocks; i++){
    result.submat(outerdims(i), outerdims(i), 
                  outerdims(i+1)-1, outerdims(i+1)-1) +=
                    x.slice(i) * y.slice(i);
  }
}

inline arma::mat AK_vec_multiply(const arma::cube& x, const arma::mat& y){ 
  arma::mat result = arma::zeros(x.n_rows, y.n_cols);
  int n_blocks = x.n_slices;
  for(int i=0; i<n_blocks; i++){
    result.col(i) = x.slice(i) * y.col(i);
  }
  return result;
}

inline void add_lambda_crossprod(arma::mat& result, const arma::mat& X, int j, int q, int k, int blocksize){
  // X has blocksize rows and k*blocksize columns
  // we want to output X.t() * X
  //arma::mat result = arma::zeros(X.n_cols, X.n_cols);
  
  // WARNING: this does NOT update result to the full crossproduct,
  // but ONLY the lower-blocktriangular part!
  int kstar = k-q;
  arma::uvec lambda_nnz = arma::zeros<arma::uvec>(1+kstar);
  lambda_nnz(0) = j;
  for(int i=0; i<kstar; i++){
    lambda_nnz(i+1) = q + i;
  }
  for(int h=0; h<lambda_nnz.n_elem; h++){
    int indh = lambda_nnz(h);
    for(int i=h; i<lambda_nnz.n_elem; i++){
      int indi = lambda_nnz(i);
      result.submat(indi*blocksize, indh*blocksize, (indi+1)*blocksize-1, (indh+1)*blocksize-1) +=
        X.cols(indi*blocksize, (indi+1)*blocksize-1).t() * X.cols(indh*blocksize, (indh+1)*blocksize-1);
    }
  }
}

inline void add_LtLxD(arma::mat& result, const arma::mat& LjtLj, const arma::vec& Ddiagvec){
  // computes LjtLj %x% diag(Ddiagvec)
  //arma::mat result = arma::zeros(LjtLj.n_rows * Ddiagvec.n_elem, LjtLj.n_cols * Ddiagvec.n_elem);
  int blockx = Ddiagvec.n_elem;
  int blocky = Ddiagvec.n_elem;
  
  for(int i=0; i<LjtLj.n_rows; i++){
    int startrow = i*blockx;
    for(int j=0; j<LjtLj.n_cols; j++){
      int startcol = j*blocky;
      if(LjtLj(i,j) != 0){
        for(int h=0; h<Ddiagvec.n_elem; h++){
          if(Ddiagvec(h) != 0){
            result(startrow + h, startcol+h) += LjtLj(i, j) * Ddiagvec(h);
          }
        }
      }
    }
  }
}

inline arma::mat build_block_diagonal(const arma::cube& x, const arma::uvec& blockdims){
  arma::mat result = arma::zeros(x.n_rows * x.n_slices, x.n_cols * x.n_slices);
  for(int i=0; i<x.n_slices; i++){
    result.submat(blockdims(i), blockdims(i), blockdims(i+1)-1, blockdims(i+1)-1) = x.slice(i);
  }
  return result;
}

inline arma::cube cube_cols(const arma::cube& x, const arma::uvec& sel_cols){
  arma::cube result = arma::zeros(x.n_rows, sel_cols.n_elem, x.n_slices);
  for(int i=0; i<x.n_slices; i++){
    result.slice(i) = x.slice(i).cols(sel_cols);
  }
  return result;
}

inline arma::mat ortho(const arma::mat& x){
  return arma::eye(arma::size(x)) - x * arma::inv_sympd(arma::symmatu(x.t() * x)) * x.t();
}

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
  arma::field<arma::uvec> children_indexing;
  
  // NA data
  arma::field<arma::uvec> na_1_blocks; // indicator vector by block
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
  arma::mat w;
  arma::mat Bcoeff; // sampled
  arma::mat rand_norm_mat;
  int parts;
  
  arma::mat Lambda;
  arma::umat Lambda_mask; // 1 where we want lambda to be nonzero
  arma::mat LambdaHw; // part of the regression mean explained by the latent process
  
  arma::mat XB; // by outcome
  arma::vec tausq_inv; // tausq for the l=q variables
  
  // ModifiedPP-like updates for tausq -- used if not forced_grid
  RAMAdapt tausq_adapt;
  int mcmc_counter;
  arma::mat tausq_unif_bounds;
  
  // params with mh step
  MeshDataLMC param_data; 
  MeshDataLMC alter_data;
  
  
  // Matern
  int nThreads;
  double * bessel_ws;
  
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
  arma::field<arma::cube> H_cache;
  arma::field<arma::cube> Ri_cache;
  //arma::field<arma::cube> Kxxi_cache; //*** we store the components not the Kroned matrix
  arma::vec Ri_chol_logdet;
  
  arma::field<arma::cube> Hpred;
  arma::field<arma::mat> Rcholpred;
  
  
  // MCMC
  void update_block_covpars(int, MeshDataLMC& data);
  void update_block_logdens(int, MeshDataLMC& data);
  void update_lly(int, MeshDataLMC&);
  
  bool refresh_cache(MeshDataLMC& data);
  bool calc_ywlogdens(MeshDataLMC& data);
  // 
  bool get_loglik_comps_w(MeshDataLMC& data);
  
  // update_block_wpars used for sampling and proposing w
  // calculates all conditional means and variances
  void update_block_w_cache(int, MeshDataLMC& data);
  void refresh_w_cache(MeshDataLMC& data);
  void gibbs_sample_w(bool needs_update);
  
  //
  void gibbs_sample_beta();
  
  void deal_with_Lambda();
  void gibbs_sample_Lambda();
  
  void deal_with_tausq();
  void gibbs_sample_tausq();
  
  void logpost_refresh_after_gibbs(); //***
  
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
  
  parts = 1; 
  
  if(forced_grid){
    tausq_adapt = RAMAdapt(q, arma::eye(q,q)*.1, q==1? .45 : .25);
    mcmc_counter = 0;
    tausq_unif_bounds = arma::join_horiz(1e-5 * arma::ones(q), 1 * arma::ones(q));
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

  //Z_available = Z.rows(na_ix_all);
  //Zw = arma::zeros(coords.n_rows);
  
  // domain partitioning
  indexing    = indexing_in;
  indexing_obs = indexing_obs_in;

  
  
  // initial values
  w = w_in; //arma::zeros(w_in.n_rows, k); 
  LambdaHw = arma::zeros(coords.n_rows, q); 
  
  Lambda = lambda_in; 
  Lambda_mask = lambda_mask_in;
  
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

  int bessel_ws_inc = 5;
  bessel_ws = (double *) R_alloc(nThreads*bessel_ws_inc, sizeof(double));
  
  // predict_initialize
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
  
  if(pblocks > 1){
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
  }
  
  message("[make_gibbs_groups] done.");
}

void LMCMeshGP::na_study(){
  // prepare stuff for NA management
  message("[na_study] start");
  na_1_blocks = arma::field<arma::uvec> (n_blocks);
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
    // consider NA if all margins are missing
    // otherwise it's available
    for(int ix=0; ix<yvec.n_rows; ix++){
      arma::uvec yfinite_row = arma::find_finite(yvec.row(ix));
      if(yfinite_row.n_elem > 0){
        na_1_blocks(i)(ix) = 1;
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

void LMCMeshGP::fill_zeros_Kcache(){
  message("[fill_zeros_Kcache]");
  // ***
  H_cache = arma::field<arma::cube> (kr_caching.n_elem);
  Ri_cache = arma::field<arma::cube> (kr_caching.n_elem);
  
  Ri_chol_logdet = arma::zeros(kr_caching.n_elem);
  
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    H_cache(i) = 
      arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem, k);
    Ri_cache(i) = 
      arma::zeros(indexing(u).n_elem, indexing(u).n_elem, k);
  }
  
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
  
  if(verbose & debug){
    Rcpp::Rcout << "Caching stats c: " << coords_caching.n_elem 
                << " k: " << kr_caching.n_elem << "\n";
  }
  message("[init_cache]");
}

void LMCMeshGP::init_meshdata(const arma::mat& theta_in){
  message("[init_meshdata]");
  
  // block params
  param_data.w_cond_mean_K = arma::field<arma::cube> (n_blocks);
  param_data.w_cond_prec   = arma::field<arma::cube> (n_blocks);
  param_data.w_cond_prec_times_cmk = arma::field<arma::cube> (n_blocks);
  
  param_data.Rproject = arma::field<arma::cube>(n_blocks);
  param_data.Hproject = arma::field<arma::cube>(n_blocks);
  param_data.Ddiag = arma::zeros(arma::size(y)); 
  
  param_data.Smu_start = arma::field<arma::mat>(n_blocks);
  param_data.Sigi_chol = arma::field<arma::mat>(n_blocks);
  param_data.AK_uP = arma::field<arma::field<arma::cube> >(n_blocks);
  //param_data.LambdaH_Ditau = arma::field<arma::field<arma::mat> >(n_blocks);
  
  for(int i=0; i<n_blocks; i++){
    int u=block_names(i) - 1;
    param_data.w_cond_mean_K(i) = arma::zeros(indexing(i).n_elem, parents_indexing(i).n_elem, k);
    param_data.w_cond_prec(i) = arma::zeros(indexing(i).n_elem, indexing(i).n_elem, k);
    param_data.w_cond_prec_times_cmk(i) = arma::zeros(indexing(i).n_elem, parents_indexing(i).n_elem, k);
    
    param_data.Hproject(i) = arma::zeros(k, //k*
      indexing(i).n_elem, indexing_obs(i).n_elem);
    param_data.Rproject(i) = arma::zeros(k, k, indexing_obs(i).n_elem);

    param_data.Smu_start(i) = arma::zeros(k*indexing(i).n_elem, 1);
    param_data.Sigi_chol(i) = arma::zeros(k*indexing(i).n_elem, k*indexing(i).n_elem);
    param_data.AK_uP(i) = arma::field<arma::cube>(children(i).n_elem);
    //param_data.LambdaH_Ditau(i) = arma::field<arma::mat> (q);
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
  param_data.wcore         = arma::zeros(n_blocks, 1);
  param_data.loglik_w_comps = arma::zeros(n_blocks, 1);
  param_data.ll_y = arma::zeros(coords.n_rows, 1);
  param_data.loglik_w       = 0; 
  param_data.ll_y_all       = 0; 
  
  param_data.theta          = theta_in;//##
    
  alter_data                = param_data; 
  message("[init_meshdata] done.");
}

void LMCMeshGP::init_indexing(){
  
  parents_indexing = arma::field<arma::uvec> (n_blocks);
  children_indexing = arma::field<arma::uvec> (n_blocks);
  
  message("[init_indexing] parent_indexing");
#ifdef _OPENMP
//#pragma omp parallel for 
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

bool LMCMeshGP::refresh_cache(MeshDataLMC& data){
  start_overall = std::chrono::steady_clock::now();
  message("[refresh_cache] start.");
  
  Ri_chol_logdet = arma::zeros(kr_caching.n_elem);
  
  arma::vec timings = arma::zeros(2);
  int errtype = -1;
  
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
                           coords, indexing(u), k, data.theta, bessel_ws);
      } catch (...) {
        errtype = 1;
      }
    } else {
      // this means we are caching kr
      i = it - starting_kr;
      int u = kr_caching(i);
      try {
        if(block_ct_obs(u) > 0){
          Ri_chol_logdet(i) = CviaKron_HRi_(H_cache(i), Ri_cache(i),
                         coords, indexing(u), parents_indexing(u), k, data.theta, bessel_ws);
        }
      } catch (...) {
        errtype = 2;
      }
    }
  }
  
  if(errtype > 0){
    if(verbose & debug){
      Rcpp::Rcout << "Cholesky failed at some point. Here's the value of theta that caused this" << "\n";
      Rcpp::Rcout << "theta: " << data.theta.t() << "\n";
      Rcpp::Rcout << " -- auto rejected and proceeding." << "\n";
    }
    return false;
  }
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[refresh_cache] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
  return true;
}

/*
bool LMCMeshGP::refresh_cache(MeshDataLMC& data){
  start_overall = std::chrono::steady_clock::now();
  message("[refresh_cache] start.");
  
  Ri_chol_logdet = arma::zeros(kr_caching.n_elem);
  
  arma::vec timings = arma::zeros(2);
  int errtype = -1;
  
  start = std::chrono::steady_clock::now();
  if(forced_grid){
    // for forced grids we compute the cached inverse of each blocks
#ifdef _OPENMP
#pragma omp parallel for if(coords_caching.n_elem > 3)
#endif
    for(int i=0; i<coords_caching.n_elem; i++){
      int u = coords_caching(i); // block name of ith representative
      arma::mat cx = coords.rows(indexing(u));
      try {
        for(int j=0; j<k; j++){
          data.Kxxi_cache(i).slice(j) = arma::inv_sympd(
            Correlationf(cx, cx, data.theta.col(j), true));
        }
      } catch(...) {
        errtype = 1;
      }
    }
  }

  
  if(errtype > 0){
    return false;
  }
  end = std::chrono::steady_clock::now();
  timings(0) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  
  start = std::chrono::steady_clock::now();
#ifdef _OPENMP
//***#pragma omp parallel for 
#endif
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    try {
      if(block_ct_obs(u) > 0){
        Ri_chol_logdet(i) = CviaKron_HRi_(H_cache(i), Ri_cache(i),
                       coords, indexing(u), parents_indexing(u), k, data.theta);
      }
    } catch (...) {
      errtype = 2;
    }
  }
  end = std::chrono::steady_clock::now();
  timings(1) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  
  if(errtype > 0){
    if(verbose & debug){
      Rcpp::Rcout << "Cholesky failed at some point. Here's the value of theta that caused this" << "\n";
      Rcpp::Rcout << "theta: " << data.theta.t() << "\n";
      Rcpp::Rcout << " -- auto rejected and proceeding." << "\n";
    }
    return false;
  }
  
  Rcpp::Rcout << "timings: " << endl << timings.t() << endl;
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[refresh_cache] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
  return true;
}
*/
void LMCMeshGP::update_block_covpars(int u, MeshDataLMC& data){
  //message("[update_block_covpars] start.");
  // given block u as input, this function updates H and R
  // which will be used later to compute logp(w | theta)
  
  //int u_cached_ix = coords_caching_ix(u);
  //Rcpp::Rcout << "u: " << u << "\n";
  
  //arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first" );
  //arma::mat wx = w.rows(indexing(u));
  
  if( parents(u).n_elem > 0 ){
    //arma::vec vecwpar = arma::vectorise(w.rows(parents_indexing(u)));
    //arma::mat wpar = w.rows(parents_indexing(u));
    
    int kr_cached_ix = kr_caching_ix(u);
    arma::uvec cpx = arma::find(kr_caching == kr_cached_ix, 1, "first" );
    data.w_cond_mean_K(u) = H_cache(cpx(0));
    data.w_cond_prec(u) = Ri_cache(cpx(0));
    for(int j=0; j<k; j++){
      data.w_cond_prec_times_cmk(u).slice(j) = data.w_cond_prec(u).slice(j) * data.w_cond_mean_K(u).slice(j);
    }
    
    //for(int j=0; j<k; j++){
    //  wx.col(j) = wx.col(j) - data.w_cond_mean_K(u).slice(j) * wpar.col(j);
    //}
    
    data.logdetCi_comps(u) = Ri_chol_logdet(cpx(0));
  } else {
    data.logdetCi_comps(u) = CviaKron_invsympd_wdet_(data.w_cond_prec(u), coords, indexing(u), k, data.theta, bessel_ws);
  }
  //message("[update_block_covpars] done.");
}

void LMCMeshGP::update_block_logdens(int u, MeshDataLMC& data){
  //message("[update_block_logdens] start.");
  
  arma::mat wx = w.rows(indexing(u));
  arma::mat wcoresum = arma::zeros(1, data.wcore.n_cols);
  
  if( parents(u).n_elem > 0 ){
    //arma::vec vecwpar = arma::vectorise(w.rows(parents_indexing(u)));
    arma::mat wpar = w.rows(parents_indexing(u));
    for(int j=0; j<k; j++){
      wx.col(j) = wx.col(j) - data.w_cond_mean_K(u).slice(j) * wpar.col(j);
    }
  } 
  
  for(int j=0; j<k; j++){
    wcoresum += arma::trans(wx.col(j)) * data.w_cond_prec(u).slice(j) * wx.col(j);
  }
  
  data.wcore.row(u) = wcoresum;
  data.loglik_w_comps.row(u) = (indexing(u).n_elem+.0) * hl2pi -.5 * data.wcore.row(u);
  
  if(forced_grid){
    int u_cached_ix = coords_caching_ix(u);
    arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first" );
    CviaKron_HRj_bdiag_(data.Hproject(u), data.Rproject(u), 
                        data.Kxxi_cache(cx(0)),
                        coords, indexing_obs(u), 
                        na_1_blocks(u), indexing(u), 
                        k, data.theta, bessel_ws);
    
    update_lly(u, data);
  }  
  //message("[update_block_logdens] done.");
}

bool LMCMeshGP::calc_ywlogdens(MeshDataLMC& data){
  // this is for standard mcmc not pseudomarginal
  // in pseudomarginal we do this along with sampling in 1 loop
  start_overall = std::chrono::steady_clock::now();
  message("[calc_ywlogdens] start.");

  //start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    update_block_covpars(u, data);
    update_block_logdens(u, data);
  }
  
  data.loglik_w = arma::accu(data.logdetCi_comps) + 
    arma::accu(data.loglik_w_comps) +
    arma::accu(data.ll_y);
  
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
  // for MCMC without pseudomarginalization
  bool acceptable = refresh_cache(data);
  if(acceptable){
    acceptable = calc_ywlogdens(data);
  } else {
    return acceptable;
  }
}

void LMCMeshGP::gibbs_sample_beta(){
  message("[gibbs_sample_beta]");
  start = std::chrono::steady_clock::now();
  
  Rcpp::RNGScope scope;
  arma::mat bmat = arma::randn(p, q);
  
  //arma::vec LambdaHw_available = LambdaHw.rows(na_ix_all);
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

void LMCMeshGP::deal_with_Lambda(){
  gibbs_sample_Lambda();
}

void LMCMeshGP::gibbs_sample_Lambda(){
  message("[gibbs_sample_Lambda] starting");
  start = std::chrono::steady_clock::now();
  
  // old with accept/reject if diagonal element is negative
  if(false){
    for(int j=0; j<q; j++){
      // build W
      int maxwhere = std::min(j, k-1);
      arma::uvec subcols = arma::find(Lambda_mask.row(j) == 1);
      
      // filter: choose value of spatial processes at locations of Yj that are available
      arma::mat WWj = w.submat(ix_by_q_a(j), subcols); // acts as X
      arma::mat Wcrossprod = WWj.t() * WWj;
      arma::mat Lprior_inv = tausq_inv(j) * 
          arma::eye(WWj.n_cols, WWj.n_cols) * .0001; 
      
      arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * Wcrossprod// + Lprior_inv
                                                     ), "lower");
      arma::mat Sigma_chol_L = arma::inv(arma::trimatl(Si_chol));
      
      arma::mat Simean_L = tausq_inv(j) * WWj.t() * 
        (y.submat(ix_by_q_a(j), oneuv*j) - XB.submat(ix_by_q_a(j), oneuv*j));
      
      bool looping = true;
      int ctr = 0;
      arma::vec Lambdarowj_new;
      while(looping & (ctr < 10)){
        ctr ++;
        Rcpp::RNGScope scope;
        arma::vec rLambdasample = arma::randn(k);
        
        Lambdarowj_new = Sigma_chol_L.t() * 
          (Sigma_chol_L * Simean_L + rLambdasample.elem(subcols));//, oneuv*j));
        
        if((j<k) & (j==maxwhere)){
          if(Lambdarowj_new(j) > 0){
            looping = false;
          }
        }
      }
      if(ctr < 10){
        // update lambda 
        Lambda.submat(oneuv*j, subcols) = 
          Lambdarowj_new.t();    
      }
    } 
  } else {
  
    // new with botev's 2017 method to sample from truncated normal
    for(int j=0; j<q; j++){
      // build W
      int maxwhere = std::min(j, k-1);
      arma::uvec subcols = arma::find(Lambda_mask.row(j) == 1);
      
      // filter: choose value of spatial processes at locations of Yj that are available
      arma::mat WWj = w.submat(ix_by_q_a(j), subcols); // acts as X
      arma::mat Wcrossprod = WWj.t() * WWj;
      arma::mat Lprior_inv = tausq_inv(j) * 
        arma::eye(WWj.n_cols, WWj.n_cols) * .0001; 
      
      arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * Wcrossprod // + Lprior_inv
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

      Lambda.submat(oneuv*j, subcols) = sampled;
    } 
  
  
  }
  // refreshing density happens in the 'logpost_refresh_after_gibbs' function
  
  if(verbose & debug){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_Lambda] " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}


void LMCMeshGP::deal_with_tausq(){
  gibbs_sample_tausq();
}

void LMCMeshGP::gibbs_sample_tausq(){
  if(!forced_grid){
    message("[gibbs_sample_tausq] start");
    start = std::chrono::steady_clock::now();
    // note that at the available locations w already includes Lambda 
    logpost = 0;
    for(int j=0; j<q; j++){
      
      arma::mat yrr = y.submat(ix_by_q_a(j), oneuv*j) - 
        XB.submat(ix_by_q_a(j), oneuv*j) - 
        LambdaHw.submat(ix_by_q_a(j), oneuv*j); //***
      
      double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
      
      double aparam = 2.00001 + ix_by_q_a(j).n_elem/2.0;
      double bparam = 1.0/( 1 + .5 * bcore );
      
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
  } else {
    message("[gibbs_sample_tausq] start (sampling via Robust adaptive Metropolis)");
    start = std::chrono::steady_clock::now();
    
    tausq_adapt.count_proposal();
    Rcpp::RNGScope scope;
    arma::vec new_tausq = 1.0/tausq_inv;
    arma::vec U_update = arma::randn(q);
    new_tausq = par_huvtransf_back(par_huvtransf_fwd(1.0/tausq_inv, tausq_unif_bounds) + 
      tausq_adapt.paramsd * U_update, tausq_unif_bounds);
    
    double start_logpost = 0;
    for(int j=0; j<q; j++){
      arma::vec regprecis = 1.0/(param_data.Ddiag.submat(ix_by_q_a(j), oneuv*j) + 1.0/tausq_inv(j));
      arma::mat yrr = (y.submat(ix_by_q_a(j), oneuv*j) - 
        XB.submat(ix_by_q_a(j), oneuv*j) - 
        LambdaHw.submat(ix_by_q_a(j), oneuv*j)) % pow(regprecis, .5); // ***
      double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
      start_logpost += 0.5 * arma::accu(log(regprecis)) - 0.5*bcore;
    }
    
    double new_logpost = 0;
    // note that at the available locations w already includes Lambda 
    for(int j=0; j<q; j++){
      arma::vec regprecis = 1.0/(param_data.Ddiag.submat(ix_by_q_a(j), oneuv*j) + new_tausq(j));
      arma::mat yrr = (y.submat(ix_by_q_a(j), oneuv*j) - 
        XB.submat(ix_by_q_a(j), oneuv*j) - 
        LambdaHw.submat(ix_by_q_a(j), oneuv*j)) % pow(regprecis, .5); // ***
      double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
      new_logpost += 0.5 * arma::accu(log(regprecis)) - 0.5*bcore;
    }
    
    double prior_logratio = calc_prior_logratio(new_tausq, 1.0/tausq_inv);
    double jacobian  = calc_jacobian(new_tausq, 1.0/tausq_inv, tausq_unif_bounds);
    double logaccept = new_logpost - start_logpost + 
      prior_logratio +
      jacobian;
    
    bool accepted = do_I_accept(logaccept);
    if(accepted){
      tausq_adapt.count_accepted();
      tausq_inv = 1.0/new_tausq;
      logpost = new_logpost;
    } else {
      logpost = start_logpost;
    }
    
    tausq_adapt.update_ratios();
    tausq_adapt.adapt(U_update, exp(logaccept), mcmc_counter); 
    mcmc_counter ++;
    if(verbose & debug){
      end = std::chrono::steady_clock::now();
      Rcpp::Rcout << "[gibbs_sample_tausq] " << tausq_adapt.accept_ratio << " average acceptance rate, "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                  << "us.\n";
    }
  }
}


void LMCMeshGP::update_lly(int u, MeshDataLMC& data){
  message("[update_lly] start.");
  start = std::chrono::steady_clock::now();
  // if the grid is forced it likely is not overlapping with the data
  // therefore the spatial ranges end up in the likelihood
  for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
    data.ll_y.row(indexing_obs(u)(ix)) = 0;
    //arma::vec wxu = arma::vectorise(w.rows(indexing(u)));
    
    for(int j=0; j<q; j++){
      if(na_mat(indexing_obs(u)(ix), j) == 1){
        arma::rowvec Lambda_var = Lambda.row(j);
        
        double Rix = arma::conv_to<double>::from(
          Lambda_var * data.Rproject(u).slice(ix) * Lambda_var.t());
        //data.Ddiag(u)(ix, j) = std::max(Rix, 0.0); // Lambda of this variable to collapse Rix
        data.Ddiag(indexing_obs(u)(ix), j) = std::max(Rix, 0.0); // Lambda of this variable to collapse Rix
        
        // if the observation is numerically overlapping with the knot coordinate
        // then the MH ratio at this location will always be zero
        // therefore skip if coordinate is numerically overlapping with a knot
        if(data.Ddiag(indexing_obs(u)(ix), j) > 1e-6){
          double ysigmasq = data.Ddiag(indexing_obs(u)(ix), j) + 
            1.0/tausq_inv(j); // this used to be tausq_inv_long but we probably don't need it
          
          // here we dont use the non-reference conditional means that was calculated before
          // because we have proposed values for theta
          double KXw = arma::conv_to<double>::from(Lambda_var * 
                          arma::sum(data.Hproject(u).slice(ix) % arma::trans(w.rows(indexing(u))), 1));
          
          double ytilde = y(indexing_obs(u)(ix), j) - XB(indexing_obs(u)(ix), j) - KXw;
          data.ll_y.row(indexing_obs(u)(ix)) += hl2pi -.5 * log(ysigmasq) - 1.0/(2*ysigmasq)*pow(ytilde, 2);
        }
      }
    }
  }
  
  end = std::chrono::steady_clock::now();
  message("[update_lly] done.");
}

void LMCMeshGP::logpost_refresh_after_gibbs(){
  message("[logpost_refresh_after_gibbs]");
  start_overall = std::chrono::steady_clock::now();
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    
    arma::mat wx = w.rows(indexing(u));
    
    if(parents(u).n_elem > 0){
      arma::mat wpar = w.rows(parents_indexing(u));
      for(int j=0; j<k; j++){
        wx.col(j) = wx.col(j) - param_data.w_cond_mean_K(u).slice(j) * wpar.col(j);
      }
    }
    
    arma::mat wcoresum = arma::zeros(1, param_data.wcore.n_cols);
    for(int j=0; j<k; j++){
      wcoresum += arma::trans(wx.col(j)) * param_data.w_cond_prec(u).slice(j) * wx.col(j);
    }
    param_data.wcore.row(u) = wcoresum;
    param_data.loglik_w_comps.row(u) = (indexing(u).n_elem+.0) * hl2pi -.5 * param_data.wcore.row(u);

    if(forced_grid){
      // if the grid is forced it likely is not overlapping with the data
      update_lly(u, param_data);
    }
  }
  
  param_data.loglik_w = arma::accu(param_data.logdetCi_comps) + 
    arma::accu(param_data.loglik_w_comps) + 
    arma::accu(param_data.ll_y);
  
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[logpost_refresh_after_gibbs] " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count() 
                << "us.\n"
                << "of which " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                << "us to do [update_lly].\n";
  }
}

void LMCMeshGP::update_block_w_cache(int u, MeshDataLMC& data){
  // 
  arma::uvec blockdims = arma::cumsum( indexing(u).n_elem * arma::ones<arma::uvec>(k) );
  blockdims = arma::join_vert(oneuv * 0, blockdims);
  
  arma::mat Sigi_tot = build_block_diagonal(data.w_cond_prec(u), blockdims);
  arma::mat Smu_tot = arma::zeros(k*indexing(u).n_elem, 1); // replace with fill(0)
  
  for(int c=0; c<children(u).n_elem; c++){
    int child = children(u)(c);
    //---------------------
    //Rcpp::Rcout << "3 " << "\n";
    arma::cube AK_u = cube_cols(data.w_cond_mean_K(child), u_is_which_col_f(u)(c)(0));
    
    //Rcpp::Rcout << "4 " << "\n";
    start = std::chrono::steady_clock::now();
    data.AK_uP(u)(c) = AKuT_x_R(AK_u, data.w_cond_prec(child));
    end = std::chrono::steady_clock::now();
    
    //Rcpp::Rcout << "7 " << "\n";
    start = std::chrono::steady_clock::now();
    add_AK_AKu_multiply_(Sigi_tot, data.AK_uP(u)(c), AK_u, blockdims);// childdims);
    end = std::chrono::steady_clock::now();
  }
  
  // each row of u_tausq_inv stores the diagonal of the nugget variance 
  // for the (q,1) vector Y(s) 
  
  arma::mat u_tau_inv = arma::zeros(indexing_obs(u).n_elem, q);
  arma::mat ytilde = arma::zeros(indexing_obs(u).n_elem, q);
  for(int j=0; j<q; j++){
    for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
      if(na_mat(indexing_obs(u)(ix), j) == 1){
        u_tau_inv(ix, j) = pow(1.0/tausq_inv(j) + data.Ddiag(indexing_obs(u)(ix), j), -.5);
        ytilde(ix, j) = (y(indexing_obs(u)(ix), j) - XB(indexing_obs(u)(ix), j))*u_tau_inv(ix, j);
      }
    }
  }
  
  if(forced_grid){
    for(int j=0; j<q; j++){
      start = std::chrono::steady_clock::now();
      int indxsize = indexing(u).n_elem;
      //data.LambdaH_Ditau(u)(j) = 
      arma::mat LambdaH_Ditau = arma::zeros(indexing_obs(u).n_elem, k*indxsize);
      for(int ix=0; ix<indexing_obs(u).n_rows; ix++){
        if(na_mat(indexing_obs(u)(ix), j) == 1){
          // if k > q fill LambdaH_Ditau only at nonzero places
          for(int jx=0; jx<k; jx++){
            arma::mat Hsub = data.Hproject(u).subcube(jx,0,ix,jx,indxsize-1, ix);
            //data.LambdaH_Ditau(u)(j)
            LambdaH_Ditau.submat(ix, jx*indxsize, ix, (jx+1)*indxsize - 1) =
              u_tau_inv(ix, j) * Lambda(j, jx) * Hsub;
          }
        }
      }
      end = std::chrono::steady_clock::now();
      Smu_tot += //param_data.LambdaH_Ditau(u)(j).t() * 
        LambdaH_Ditau.t() * ytilde.col(j);//
      if(k > q){
        add_lambda_crossprod(Sigi_tot, LambdaH_Ditau, //data.LambdaH_Ditau(u)(j), 
                             j, q, k, indexing(u).n_elem);
      } else {
        Sigi_tot += LambdaH_Ditau.t() * LambdaH_Ditau; //data.LambdaH_Ditau(u)(j).t() * data.LambdaH_Ditau(u)(j); 
      }
    }
  } else {
    for(int j=0; j<q; j++){
      start = std::chrono::steady_clock::now();
      // dont erase:
      //Sigi_tot += arma::kron( arma::trans(Lambda.row(j)) * Lambda.row(j), arma::diagmat(u_tau_inv%u_tau_inv));
      arma::mat LjtLj = arma::trans(Lambda.row(j)) * Lambda.row(j);
      arma::vec u_tausq_inv = u_tau_inv.col(j) % u_tau_inv.col(j);
      add_LtLxD(Sigi_tot, LjtLj, u_tausq_inv);
      end = std::chrono::steady_clock::now();
      Smu_tot += arma::vectorise(arma::diagmat(u_tau_inv.col(j)) * ytilde.col(j) * Lambda.row(j));
    }
  }
  
  start = std::chrono::steady_clock::now();
  data.Smu_start(u) = Smu_tot;
  data.Sigi_chol(u) = Sigi_tot;
  if((k>q) & true){
    arma::uvec blockdims_q = blockdims.subvec(0, q-1);
    // WARNING: if k>q then we have only updated the lower block-triangular part of Sigi_tot for cholesky!
    // WARNING: we make use ONLY of the lower-blocktriangular part of Sigi_tot here.
    block_invcholesky_(data.Sigi_chol(u), blockdims_q);
  } else {
    data.Sigi_chol(u) = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
  }
}

void LMCMeshGP::gibbs_sample_w(bool needs_update=true){
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_w] " << "\n";
  }
  
  Rcpp::RNGScope scope;
  rand_norm_mat = arma::randn(coords.n_rows, k);
  start_overall = std::chrono::steady_clock::now();
  
  double timing=0;
  
  for(int g=0; g<n_gibbs_groups; g++){
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      
      if((block_ct_obs(u) > 0)){
        
        start = std::chrono::steady_clock::now();
        if(needs_update){
          // recompute tchol of conditional variance
          update_block_w_cache(u, param_data);
        }
        end = std::chrono::steady_clock::now();
        timing += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // recompute conditional mean
        arma::uvec blockdims = arma::cumsum( indexing(u).n_elem * arma::ones<arma::uvec>(k) );
        blockdims = arma::join_vert(oneuv * 0, blockdims);
        
        arma::mat Smu_tot = param_data.Smu_start(u); //
        
        if(parents(u).n_elem>0){
          add_smu_parents_(Smu_tot, param_data.w_cond_prec_times_cmk(u), 
                           w.rows( parents_indexing(u) ), blockdims);
        } 
        for(int c=0; c<children(u).n_elem; c++){
          int child = children(u)(c);
          //---------------------
          
          arma::cube AK_u = cube_cols(param_data.w_cond_mean_K(child), u_is_which_col_f(u)(c)(0));
          
          arma::mat w_child = w.rows(indexing(child));
          arma::mat w_parchild = w.rows(parents_indexing(child));
          //---------------------
          if(parents(child).n_elem > 1){
            start = std::chrono::steady_clock::now();
            arma::cube AK_others = cube_cols(param_data.w_cond_mean_K(child), u_is_which_col_f(u)(c)(1));
            
            arma::mat w_parchild_others = w_parchild.rows(u_is_which_col_f(u)(c)(1));
            Smu_tot += 
              arma::vectorise(AK_vec_multiply(param_data.AK_uP(u)(c), 
                                              w_child - AK_vec_multiply(AK_others, w_parchild_others)));
            end = std::chrono::steady_clock::now();
            
          } else {
            Smu_tot += 
              arma::vectorise(AK_vec_multiply(param_data.AK_uP(u)(c), w_child));
          }
        }
        
        // sample
        arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
        arma::vec wmean = param_data.Sigi_chol(u).t() * param_data.Sigi_chol(u) * Smu_tot;
        arma::vec wtemp = wmean + param_data.Sigi_chol(u).t() * rnvec; 
        
        w.rows(indexing(u)) = //arma::trans(arma::mat(wtemp.memptr(), k, wtemp.n_elem/k)); 
            arma::mat(wtemp.memptr(), wtemp.n_elem/k, k); 
        
        if(forced_grid){
          for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
            if(na_1_blocks(u)(ix) == 1){
              w.row(indexing_obs(u)(ix)) = //arma::trans(param_data.Hproject(u).slice(ix) * wtemp);
                arma::sum(arma::trans(param_data.Hproject(u).slice(ix) % arma::trans(w.rows(indexing(u)))), 0);
              w.row(indexing_obs(u)(ix)) = w.row(indexing_obs(u)(ix));
              // do not sample here
              //LambdaHw.row(indexing_obs(u)(ix)) = w.row(indexing_obs(u)(ix)) * Lambda.t();
            }
          }
        }
        LambdaHw.rows(indexing_obs(u)) = w.rows(indexing_obs(u)) * Lambda.t();
        
        end = std::chrono::steady_clock::now();
        //timings(6) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      } 
    }
  }
  //Zw = armarowsum(Z % w);
  
  //Rcpp::Rcout << "timing for refresh: " << timing << endl;
  if(verbose & debug){
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
      
      arma::cube Kxxi_parents;
      start = std::chrono::steady_clock::now();
      if((block_ct_obs(u) > 0) & forced_grid){
        // this is a reference set with some observed locations
        int u_cached_ix = coords_caching_ix(u);
        predict_parent_indexing = indexing(u); // uses knots which by construction include all k processes
        arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first" );
        
        CviaKron_HRj_chol_bdiag_wcache(Hpred(i), Rcholpred(i), param_data.Kxxi_cache(cx(0)), na_1_blocks(u),
                                coords, indexing_obs(u), predict_parent_indexing, k, param_data.theta, bessel_ws);
      } else {
        // no observed locations, use line of sight
        predict_parent_indexing = parents_indexing(u);
        CviaKron_HRj_chol_bdiag(Hpred(i), Rcholpred(i), Kxxi_parents,
                                na_1_blocks(u),
                                coords, indexing_obs(u), predict_parent_indexing, k, param_data.theta, bessel_ws);
      }
      end = std::chrono::steady_clock::now();
      timer(0) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      //Rcpp::Rcout << "step 2" << "\n";
      
      start = std::chrono::steady_clock::now();
      for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
        if(na_1_blocks(u)(ix) == 0){
          //Rcpp::Rcout << indexing_obs(u)(ix) << "\n";
          //Rcpp::Rcout << "substep 1" << "\n";
          // if == 1 this has already been sampled.
          arma::mat wpars = w.rows(predict_parent_indexing);
         
          arma::rowvec wtemp = arma::sum(arma::trans(Hpred(i).slice(ix)) % wpars, 0) + 
            arma::trans(Rcholpred(i).col(ix)) % rand_norm_mat.row(indexing_obs(u)(ix));
          //Rcpp::Rcout << "substep 3" << "\n";
          w.row(indexing_obs(u)(ix)) = wtemp;//arma::mat(wtemp.memptr(), wtemp.n_elem/k, k); 
          //Rcpp::Rcout << "substep 4" << "\n";
          LambdaHw.row(indexing_obs(u)(ix)) = w.row(indexing_obs(u)(ix)) * Lambda.t();
        }
      }
      end = std::chrono::steady_clock::now();
      timer(1) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      
      if(false){
        // this makes the prediction at grid points underlying non-observed areas
        // ONLY useful to make a full raster map at the end
        start = std::chrono::steady_clock::now();
        if((block_ct_obs(u) == 0) & forced_grid){
          arma::cube Hpredx = arma::zeros(k, predict_parent_indexing.n_elem, indexing(u).n_elem);
          arma::mat Rcholpredx = arma::zeros(k, indexing(u).n_elem);
          arma::uvec all_na = arma::zeros<arma::uvec>(indexing(u).n_elem);
          CviaKron_HRj_chol_bdiag_wcache(Hpredx, Rcholpredx, Kxxi_parents, all_na, 
                                         coords, indexing(u), predict_parent_indexing, k, param_data.theta, bessel_ws);
          for(int ix=0; ix<indexing(u).n_elem; ix++){
            arma::mat wpars = w.rows(predict_parent_indexing);
            arma::rowvec wtemp = arma::sum(arma::trans(Hpredx.slice(ix)) % wpars, 0) + 
              arma::trans(Rcholpredx.col(ix)) % rand_norm_mat.row(indexing(u)(ix));
            w.row(indexing(u)(ix)) = wtemp;
            LambdaHw.row(indexing(u)(ix)) = w.row(indexing(u)(ix)) * Lambda.t();
          }
        }
        end = std::chrono::steady_clock::now();
        timer(2) += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      }
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






