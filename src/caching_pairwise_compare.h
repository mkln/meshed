#include <RcppArmadillo.h>

arma::vec caching_pairwise_compare_u(const arma::field<arma::mat>& blocks,
                                     const arma::vec& names,
                                     const arma::vec& ct_obs);

arma::vec caching_pairwise_compare_uc(const arma::field<arma::mat>& blocks,
                                     const arma::vec& names,
                                     const arma::vec& ct_obs);

arma::vec caching_pairwise_compare_uci(const arma::mat& coords,
                                       const arma::field<arma::uvec>& insexing,
                                       const arma::vec& names,
                                       const arma::vec& ct_obs);

inline arma::vec caching_pairwise_compare_ux(const arma::field<arma::mat>& blocks){
  
  // result(x) = y
  // means
  // blocks(x) r= blocks(y)
  
  arma::vec result = arma::zeros(blocks.n_elem)-1;
  arma::field<arma::mat> sorted(blocks.n_elem);
  
  // remodel blocks so they are all relative to the first row (which is assumed sorted V1-V2-V3 from R)
  //***#pragma omp parallel for
  for(int i=0; i<blocks.n_elem; i++){
    sorted(i) = blocks(i);
    //Rcpp::Rcout << "i: " << i << endl;
    if(blocks(i).n_rows > 1){
      for(int j=0; j<blocks(i).n_rows; j++){
        sorted(i).row(j) = blocks(i).row(j) - blocks(i).row(0);
      }
    }
  }
  
  // now find if the blocks are relatively the same
  //***#pragma omp parallel for
  for(int j=0; j<blocks.n_elem; j++){
    int u_target = j;
    bool foundsame = false;
    //Rcpp::Rcout << "u_target: " << u_target << endl;
    for(int k=0; k<j; k++){ //blocks.n_elem; k++){
      int u_prop = k;
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
  }
  return result;
}