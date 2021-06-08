#include "meshed.h"

using namespace std;

//[[Rcpp::export]]
arma::mat rmeshedgp_internal(const arma::mat& coords, 
                             
                             const arma::field<arma::uvec>& parents,
                             const arma::field<arma::uvec>& children,
                             
                             const arma::vec& layer_names,
                             const arma::vec& layer_gibbs_group,
                             
                             const arma::field<arma::uvec>& indexing,
                             const arma::field<arma::uvec>& indexing_obs,
                             
                             int matern_twonu,
                             
                             const arma::mat& theta,
                             int num_threads = 1,
                             
                             bool use_cache=true,
                             
                             bool verbose=false,
                             bool debug=false){
  
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  
  Meshed msp(coords, parents, children, layer_names, layer_gibbs_group,
             
             indexing, indexing_obs,
             
             matern_twonu, theta, 
             use_cache,
             verbose, debug, num_threads);
  
  msp.w_prior_sample(msp.param_data);
  
  return msp.w;
  
}
