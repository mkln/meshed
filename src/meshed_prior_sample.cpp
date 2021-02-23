#include "meshed/meshed.h"

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
  
  Meshed msp(coords, parents, children, layer_names, layer_gibbs_group,
             
             indexing, indexing_obs,
             
             matern_twonu, theta, 
             use_cache,
             verbose, debug, num_threads);
  
  msp.w_prior_sample(msp.param_data);
  
  return msp.w;
  
  //---
  
  /*
   //Rcpp::Rcout << "triplets " << endl;
   typedef Eigen::Triplet<double> T;
   std::vector<T> tripletList_Ci;
   tripletList_Ci.reserve(Cilocs.n_rows);
   for(int i=0; i<Cilocs.n_rows; i++){
   tripletList_Ci.push_back(T(Cilocs(i, 0), Cilocs(i, 1), Civals(i)));
   }
   
   Rcpp::Rcout << "solving after cholesky " << endl;
   
   //Rcpp::Rcout << "gen sparse mat " << endl;
   int n = grid.n_rows;
   Eigen::SparseMatrix<double> eCi(n,n);
   eCi.setFromTriplets(tripletList_Ci.begin(), tripletList_Ci.end());
   end = std::chrono::steady_clock::now();
   //timings(0) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
   
   Eigen::VectorXd eHDy_long = armavec_to_vectorxd(HDy_long);
   
   start = std::chrono::steady_clock::now();
   Eigen::CholmodDecomposition<Eigen::SparseMatrix<double> > solver;
   solver.analyzePattern(eCi);   // for this step the numerical values of A are not used
   end = std::chrono::steady_clock::now();
   
   Rcpp::Rcout << "analyzing pattern: " <<
   std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << endl;
   
   start = std::chrono::steady_clock::now();
   solver.factorize(eCi);
   Eigen::VectorXd x1 = solver.solve(eHDy_long);
   end = std::chrono::steady_clock::now();
   
   Rcpp::Rcout << "solving: " <<
   std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << endl;
   
   
   // Eigen::VectorXd Ltx = eCi.triangularView<Eigen::Lower>().solve(eHDy_long);
   // Eigen::VectorXd emu = eCi.transpose().triangularView<Eigen::Upper>().solve(Ltx);
   // 
   // arma::vec mu = arma::vec(emu.data(), emu.rows(), false, false);
   // 
   // Eigen::VectorXd cholCimu = cholCi.transpose() * emu;
   // double muCmu = (cholCimu.transpose() * cholCimu)(0, 0);
   // 
   
   
   
   // --
   
   arma::vec Cmu = spsymmat_vec_mult(armaCi, mu);
   
   double astar = 2 + available_ix.n_rows/2.0;
   double bstar = 1 + 0.5 * arma::conv_to<double>::from( 
   y.t() * (Dtaui_long % y) - mu.t() * Cmu);
   
   arma::vec sigmainfo = arma::zeros(3);
   sigmainfo(0) = astar;
   sigmainfo(1) = bstar;
   sigmainfo(2) = bstar/(astar-1);
   
   Rcpp::Rcout << sigmainfo.t() << endl;
   
   Rcpp::Rcout << "do predictions" << endl;
   
   // predict
   arma::vec yhat = arma::zeros(y.n_elem);
   for(int i = 0; i<n_blocks; i++){
   int u = block_names(i)-1;
   int datablock = data_block_link(u);
   if(datablock>0){
   arma::uvec indx = indexing_data(datablock-1);
   yhat.rows(indx) = data.Hproject(u) * mu.rows(indexing(u));
   //Rcpp::Rcout << coords.rows(indx) << endl << grid.rows(indexing(u)) << endl;
   //Rcpp::Rcout << " -- -- " << endl;
   }
   }
   
   // predict at remaining locations
   for(int i=0; i<u_predicts.n_elem; i++){ //*** subset to blocks with NA
   int u = u_predicts(i);// u_predicts(i);
   int datablock = data_block_link(u);
   if(datablock>0){
   // no observed locations, use line of sight
   arma::uvec indx = indexing_data(datablock-1);
   arma::uvec predict_parent_indexing = parents_indexing(u);
   
   arma::mat coordx = coords.rows(indx);
   arma::mat gridpar = grid.rows(predict_parent_indexing);      
   
   arma::mat Kxxi = arma::inv_sympd( 
   Correlationx(gridpar, gridpar, data.theta, matern, true) );
   
   arma::mat Cxx = Correlationx(coordx, coordx, data.theta, matern, true);
   arma::mat Cxy = Correlationx(coordx, gridpar, data.theta, matern, false);
   arma::mat Hloc = Cxy * Kxxi;
   
   yhat.rows(indx) = Hloc * mu.rows(predict_parent_indexing);
   } 
   }
   
   
   if(verbose & debug){
   end_overall = std::chrono::steady_clock::now();
   double timer_all = std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count();
   //Rcpp::Rcout << timings.t() << endl;
   }
   
   return Rcpp::List::create(
   Rcpp::Named("Ci_bsp") = Ci_bsp,
   Rcpp::Named("Ci") = armaCi,
   Rcpp::Named("HDy") = HDy_long,
   Rcpp::Named("yhat") = yhat,
   Rcpp::Named("mu") = mu
   );
   */
}
