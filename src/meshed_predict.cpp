
#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>
#include "covariance_lmc.h"
#include "utils_field_v_concatm.h"
#include "utils_parametrize.h"

using namespace std;

//[[Rcpp::export]]
Rcpp::List spmeshed_predict(
    const arma::mat& predx,
    const arma::mat& predcoords,
    const arma::uvec& predblock,
    const arma::mat& coords,
    const arma::field<arma::uvec>& parents,
    const arma::uvec& block_names,
    const arma::field<arma::uvec>& indexing,
    const arma::field<arma::mat>& v_sampled,
    const arma::cube& theta_sampled,
    const arma::cube& lambda_sampled,
    const arma::cube& beta_sampled,
    const arma::mat& tausq_sampled,
    int twonu,
    bool use_ps,
    bool verbose=false,
    int num_threads=4){
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  int q = lambda_sampled.n_rows;
  int k = v_sampled(0).n_cols;
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  int nsamples = v_sampled.n_elem;
  
  MaternParams matern;
  matern.using_ps = use_ps,
  matern.estimating_nu = false;
  matern.twonu = twonu;
  int bessel_ws_inc = 5;
  matern.bessel_ws = (double *) R_alloc(num_threads*bessel_ws_inc, sizeof(double));
  
  int d = coords.n_cols;
  
  arma::uvec ublock = arma::unique(predblock);
  arma::field<arma::mat> coords_out_list(ublock.n_elem);
  arma::field<arma::cube> preds_out_list(ublock.n_elem);
  
  for(unsigned int i=0; i<ublock.n_elem; i++){
    if(verbose){
      Rcpp::Rcout << "Block " << i+1 << " of " << ublock.n_elem << endl;
    }
    
    int iblock = ublock(i);
    arma::uvec block_ix = arma::find(predblock == iblock);
    arma::mat block_coords = predcoords.rows(block_ix);
    arma::mat block_x = predx.rows(block_ix);
    
    coords_out_list(i) = block_coords;
    preds_out_list(i) = arma::zeros(block_coords.n_rows, q, nsamples);
    
    int u = iblock-1;
    arma::uvec block_parents = parents(u);

    if(block_parents.n_elem <= d){
      // reference block so add itself to parents for predictions
      block_parents = arma::join_vert(block_parents, u * arma::ones<arma::uvec>(1) );
    }
    
    arma::field<arma::uvec> pixs(block_parents.n_elem);
    for(unsigned int pi=0; pi<block_parents.n_elem; pi++){
      pixs(pi) = indexing(block_parents(pi));
    }
    arma::uvec parents_indexing = field_v_concat_uv(pixs);
    arma::mat parents_coords = coords.rows(parents_indexing);
    
    
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int m=0; m<nsamples; m++){
      arma::mat theta = theta_sampled.slice(m);
      arma::cube Hpred = arma::zeros(k, parents_coords.n_rows, block_coords.n_rows);
      arma::mat Rcholpred = arma::zeros(k, block_coords.n_rows);
      
      for(int j=0; j<k; j++){
        arma::mat Cyy = Correlationc(parents_coords, parents_coords, theta.col(j), matern, true);
        arma::mat Cyyi = arma::inv_sympd(Cyy);
        
        for(unsigned int ix=0; ix<block_coords.n_rows; ix++){
          arma::mat Cxx = Correlationc(block_coords.rows(oneuv*ix), block_coords.rows(oneuv*ix), 
                                       theta.col(j), matern, true);
          arma::mat Cxy = Correlationc(block_coords.rows(oneuv*ix), parents_coords,  
                                       theta.col(j), matern, false);
          arma::mat Hloc = Cxy * Cyyi;
          
          Hpred.slice(ix).row(j) = Hloc;
          double Rcholtemp = arma::conv_to<double>::from(
            Cxx - Hloc * Cxy.t() );
          Rcholtemp = Rcholtemp < 0 ? 0.0 : Rcholtemp;
          Rcholpred(j,ix) = pow(Rcholtemp, .5); 
        }
      }
      
      
      arma::mat wpars = v_sampled(m).rows(parents_indexing);
      
      arma::mat Lambda = reparametrize_lambda_forward(lambda_sampled.slice(m), theta, d, matern.twonu, matern.using_ps);
      
      for(unsigned int ix=0; ix<block_coords.n_rows; ix++){
        arma::rowvec wtemp = arma::sum(arma::trans(Hpred.slice(ix)) % wpars, 0);
        arma::vec normrnd = arma::randn(k);
        wtemp += arma::trans(Rcholpred.col(ix) % normrnd);
        arma::mat LW = wtemp * Lambda.t();
        
        for(int j=0; j<q; j++){
          double yerr = arma::conv_to<double>::from(arma::randn(1)) * pow(tausq_sampled(j,m), .5);
          
          preds_out_list(i)(ix, j, m) = 
            arma::conv_to<double>::from(
              block_x.row(ix) * beta_sampled.slice(m).col(j) + 
            LW(j) + yerr);
        }
      }
    }
  }
  
  arma::mat coords_out = field_v_concatm(coords_out_list);
  arma::cube preds_out = field_v_concatc(preds_out_list);
  
  return Rcpp::List::create(
    Rcpp::Named("preds_out") = preds_out,
    Rcpp::Named("coords_out") = coords_out
  );
}
