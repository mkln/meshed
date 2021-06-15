

#include "covariance_lmc.h"
#include "utils_field_v_concatm.h"
#include "utils_parametrize.h"

//#undef _OPENMP 

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
  
  unsigned int q = lambda_sampled.n_rows;
  unsigned int k = v_sampled(0).n_cols;
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  unsigned int nsamples = v_sampled.n_elem;
  
  MaternParams matern;
  matern.using_ps = use_ps,
  matern.estimating_nu = false;
  matern.twonu = twonu;
  int bessel_ws_inc = 5;
  matern.bessel_ws = (double *) R_alloc(num_threads*bessel_ws_inc, sizeof(double));
  
  unsigned int d = coords.n_cols;
  
  arma::uvec ublock = arma::unique(predblock);
  arma::field<arma::mat> coords_out_list(ublock.n_elem);
  arma::field<arma::cube> preds_out_list(ublock.n_elem);
  
  for(unsigned int i=0; i<ublock.n_elem; i++){
    if(verbose){
      Rcpp::Rcout << "Block " << i+1 << " of " << ublock.n_elem << endl;
    }
    
    unsigned int iblock = ublock(i);
    arma::uvec block_ix = arma::find(predblock == iblock);
    arma::mat block_coords = predcoords.rows(block_ix);
    arma::mat block_x = predx.rows(block_ix);
    
    coords_out_list(i) = block_coords;
    preds_out_list(i) = arma::zeros(block_coords.n_rows, q, nsamples);
    
    unsigned int u = iblock-1;
    arma::uvec block_parents = parents(u);

    if(block_parents.n_elem <= d){
      unsigned int n_parents_upd = 1+parents(u).n_elem;
      arma::uvec block_parents_upd = arma::zeros<arma::uvec>(n_parents_upd);
      block_parents_upd(0) = u;
      for(unsigned int k=1; k<n_parents_upd; k++){
        block_parents_upd(k) = parents(u)(k-1);
      }
      block_parents = block_parents_upd;
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
    for(unsigned int m=0; m<nsamples; m++){
      arma::mat theta = theta_sampled.slice(m);
      
      arma::cube Hpred = arma::zeros(k, parents_coords.n_rows, block_coords.n_rows);
      arma::mat Rcholpred = arma::zeros(k, block_coords.n_rows);
      
      try {
        for(unsigned int j=0; j<k; j++){
          arma::mat Cyy = Correlationc(parents_coords, parents_coords, theta.col(j), matern, true);
          arma::mat Cyyc = arma::chol(Cyy, "lower");
          
          arma::mat Cyy_ichol = arma::inv(arma::trimatl(Cyyc));
          arma::mat Cyyi = Cyy_ichol.t() * Cyy_ichol;
          
          for(unsigned int ix=0; ix<block_coords.n_rows; ix++){
            arma::mat block_coords_ix = block_coords.rows(oneuv*ix);
            arma::vec thetaj = theta.col(j);
            arma::mat Cxx = Correlationc(block_coords_ix, block_coords_ix, 
                                         thetaj, matern, true);
            arma::mat Cxy = Correlationc(block_coords_ix, parents_coords,  
                                         thetaj, matern, false);
            arma::mat Hloc = Cxy * Cyyi;
            
            Hpred.slice(ix).row(j) = Hloc;
            double Rcholtemp = arma::conv_to<double>::from(
              Cxx - Hloc * Cxy.t() );
            Rcholtemp = Rcholtemp < 0 ? 0.0 : Rcholtemp;
            Rcholpred(j,ix) = pow(Rcholtemp, .5); 
          }
        }
      } catch(...) {
        // thank you solaris
        
        
      }
      
      
      
      arma::mat wpars = v_sampled(m).rows(parents_indexing);
      
      arma::mat Lambda = reparametrize_lambda_forward(lambda_sampled.slice(m), theta, d, matern.twonu, matern.using_ps);
      
      for(unsigned int ix=0; ix<block_coords.n_rows; ix++){
        arma::rowvec wtemp = arma::sum(arma::trans(Hpred.slice(ix)) % wpars, 0);
        arma::vec normrnd = arma::randn(k);
        wtemp += arma::trans(Rcholpred.col(ix) % normrnd);
        arma::mat LW = wtemp * Lambda.t();
        
        for(unsigned int j=0; j<q; j++){
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
