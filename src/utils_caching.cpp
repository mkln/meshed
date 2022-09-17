#include "utils_caching.h"

using namespace std;

arma::vec caching_pairwise_compare_u(const arma::field<arma::mat>& blocks,
                                            const arma::vec& names,
                                            const arma::vec& block_ct_obs){
  
  // result(x) = y
  // means
  // blocks(x) r= blocks(y)
  
  arma::vec result = arma::zeros(blocks.n_elem)-1;
  arma::field<arma::mat> sorted(blocks.n_elem);
  
  // remodel blocks so they are all relative to the first row (which is assumed sorted V1-V2-V3 from R)
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(unsigned int i=0; i<blocks.n_elem; i++){
    sorted(i) = blocks(i);
    //Rcpp::Rcout << "i: " << i << endl;
    if(blocks(i).n_rows > 1){
      for(unsigned int j=0; j<blocks(i).n_rows; j++){
        sorted(i).row(j) = blocks(i).row(j) - blocks(i).row(0);
      }
    }
  }
  
  // now find if the blocks are relatively the same
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(unsigned int j=0; j<blocks.n_elem; j++){
    int u_target = names(j)-1;
    if(block_ct_obs(u_target) > 0){
      bool foundsame = false;
      //Rcpp::Rcout << "u_target: " << u_target << endl;
      for(unsigned int k=0; k<j; k++){ //blocks.n_elem; k++){
        int u_prop = names(k)-1;
        if(sorted(u_target).n_rows == sorted(u_prop).n_rows){
          // these are knots so designed, no risk of making mistakes based on tolerance here
          // unless there are knots closer than 1e-4 apart which should be considered different!
          bool same = arma::approx_equal(sorted(u_target), sorted(u_prop), "absdiff", 1e-4);
          if(same){
            result(u_target) = u_prop;
            foundsame = true;
            break;
          }
        }
      }
      if(!foundsame){
        result(u_target) = u_target;
      }
    } else {
      result(u_target) = u_target;
    }
    
  }
  return result;
}


arma::uvec caching_pairwise_compare_uc(const arma::field<arma::mat>& blocks,
                                              const arma::vec& names,
                                              const arma::vec& ct_obs, bool cached){
  
  // result(x) = y
  // means
  // blocks(x) r= blocks(y)
  
  // if ct_obs == 0 then don't search for cache and assign own
  
  arma::uvec result = arma::zeros<arma::uvec>(blocks.n_elem)-1;
  arma::field<arma::mat> sorted(blocks.n_elem);
  
  // remodel blocks so they are all relative to the first row (which is assumed sorted V1-V2-V3 from R)
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(unsigned int i=0; i<blocks.n_elem; i++){
    sorted(i) = blocks(i);
    //Rcpp::Rcout << "i: " << i << endl;
    if(blocks(i).n_rows > 1){
      for(unsigned int j=0; j<blocks(i).n_rows; j++){
        sorted(i).row(j) = blocks(i).row(j) - blocks(i).row(0);
      }
    }
  }
  
  // now find if the blocks are relatively the same
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(unsigned int j=0; j<blocks.n_elem; j++){
    int u_target = names(j)-1;
    if((!cached) || (ct_obs(u_target) == 0)){
      result(u_target) = u_target;
    } else {
      bool foundsame = false;
      //Rcpp::Rcout << "u_target: " << u_target << endl;
      for(unsigned int k=0; k<j; k++){ //blocks.n_elem; k++){
        int u_prop = names(k)-1;
        // predicting blocks match anything, others only match themselves
        
        // both predicting or both observed
        // or prop is observed
        if( ( (ct_obs(u_prop) == 0) == (ct_obs(u_target) == 0) ) +
            (ct_obs(u_prop) > 0)
        ){ 
          if(sorted(u_target).n_rows == sorted(u_prop).n_rows){
            // these are knots so designed, no risk of making mistakes based on tolerance here
            // unless there are knots closer than 1e-4 apart which should be considered different!
            bool same = arma::approx_equal(sorted(u_target), sorted(u_prop), "absdiff", 1e-4);
            if(same){
              result(u_target) = u_prop;
              foundsame = true;
              break;
            }
          }
        } 
      }
      if(!foundsame){
        result(u_target) = u_target;
      }
    }
  }
  return result;
}


arma::uvec caching_pairwise_compare_uci(const arma::mat& coords,
                                               const arma::field<arma::uvec>& indexing,
                                               const arma::vec& names,
                                               const arma::vec& ct_obs, bool cached){
  
  // result(x) = y
  // means
  // blocks(x) r= blocks(y)
  
  // if ct_obs == 0 then don't search for cache and assign own
  
  arma::uvec result = arma::zeros<arma::uvec>(indexing.n_elem)-1;
  arma::field<arma::mat> sorted(indexing.n_elem);
  
  // remodel blocks so they are all relative to the first row (which is assumed sorted V1-V2-V3 from R)
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(unsigned int i=0; i<indexing.n_elem; i++){
    sorted(i) = coords.rows(indexing(i));
    //Rcpp::Rcout << "i: " << i << endl;
    if(indexing(i).n_elem > 1){
      arma::mat cmat = coords.rows(indexing(i));
      for(unsigned int j=0; j<indexing(i).n_elem; j++){
        sorted(i).row(j) = cmat.row(j) - cmat.row(0);
      }
    }
  }
  
  // now find if the blocks are relatively the same
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(unsigned int j=0; j<indexing.n_elem; j++){
    int u_target = names(j)-1;
    if(cached){ //ct_obs(u_target) == 0){
      //  result(u_target) = u_target;
      //} else {
      bool foundsame = false;
      //Rcpp::Rcout << "u_target: " << u_target << endl;
      for(unsigned int k=0; k<j; k++){ //blocks.n_elem; k++){
        int u_prop = names(k)-1;
        // predicting blocks match anything, others only match themselves
        
        // both predicting or both observed
        // or prop is observed
        if( ( (ct_obs(u_prop) == 0) == (ct_obs(u_target) == 0) ) +
            (ct_obs(u_prop) > 0)
        ){ 
          if(sorted(u_target).n_rows == sorted(u_prop).n_rows){
            bool same = arma::approx_equal(sorted(u_target), sorted(u_prop), "absdiff", 1e-4);
            if(same){
              result(u_target) = u_prop;
              foundsame = true;
              break;
            }
          }
        } 
      }
      if(!foundsame){
        result(u_target) = u_target;
      }
    } else {
      result(u_target) = u_target;
    }
  }
  return result;
}

