

#include "utils_lmc.h"
#include "utils_interrupt_handler.h"
#include "utils_parametrize.h"
#include "meshed.h"

//[[Rcpp::export]]
Rcpp::List meshed_casc(
    const arma::mat& y, 
    const arma::uvec& family,
    const int k,
    const arma::mat& X, 
    
    const arma::mat& coords, 
    
    const arma::field<arma::uvec>& parents,
    const arma::field<arma::uvec>& children,
    
    const arma::vec& layer_names,
    const arma::vec& layer_gibbs_group,
    
    const arma::field<arma::uvec>& indexing,
    const arma::field<arma::uvec>& indexing_obs,
    
    const arma::mat& beta_Vi,
    
    int matern_twonu,
    
    const arma::mat& start_w,
    const arma::mat& lambda_values,
    const arma::umat& lambda_mask,
    const arma::field<arma::mat>& theta_values,
    const arma::mat& beta,
    const arma::mat& tausq_values,
    
    int maxit = 1000,
    int num_threads = 1,
    
    bool adapting=false,
    
    bool use_cache=true,
    bool forced_grid=true,
    
    bool verbose=false,
    bool debug=false,
    int print_every=false,
    
    bool casc_beta=true,
    bool casc_w=true){
  
  if(verbose & debug){
    Rcpp::Rcout << "Initializing.\n";
  }
  
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  // timers
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point start_all = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_all = std::chrono::steady_clock::now();
  // ------
  
  bool printall = print_every == 1;
  bool verbose_mcmc = printall;
  
  int n = coords.n_rows;
  int d = coords.n_cols;
  int q  = y.n_cols;

  int n_alts = theta_values.n_elem;
  
  arma::mat wllsave = arma::zeros(maxit, n_alts);
  
  arma::cube beta_map = arma::zeros(X.n_cols, q, n_alts);
  arma::cube v_map = arma::zeros(n, k, n_alts);
  arma::cube w_map = arma::zeros(n, q, n_alts);
  arma::cube lambda_map = arma::zeros(q, k, n_alts);
  arma::cube lp_map = arma::zeros(n, q, n_alts);
  arma::cube yhat_map = arma::zeros(n, q, n_alts);
  
  
  arma::vec unused_tausq_ab;
  arma::mat unused_mcmcsd, unused_unifbounds;
  bool unused_adapting=false;
  int unused_hmc = 0;
  bool use_ps = false;
  bool acceptable=true;
  
  arma::mat start_lambda = lambda_values;
  arma::mat start_theta = theta_values(0);
  arma::vec start_tausqi = 1.0/tausq_values.col(0);
  
  bool verbose_msp = verbose & debug;
  Meshed msp(y, family,
             X, coords, k,
             parents, children, layer_names, layer_gibbs_group,
             
             indexing, indexing_obs,
             
             matern_twonu,
             start_w, beta, start_lambda, lambda_mask, start_theta, start_tausqi, 
             
             beta_Vi, 
             
             unused_tausq_ab, 
             unused_hmc,
             unused_adapting,
             unused_mcmcsd, unused_unifbounds,
             
             use_cache, forced_grid, use_ps,
             verbose_msp, debug, num_threads);
  

  //bool acceptable = false;
  
  if(verbose & debug){
    Rcpp::Rcout << "Starting... " << endl; 
  }
  
  // do all alternatives for theta
  for(int i=0; i<n_alts; i++){
    
    start_all = std::chrono::steady_clock::now();
    bool keep_running = true;
    int m=0;
    
    // set theta and tausq
    arma::mat theta_here = theta_values(i);
    msp.param_data.theta = theta_here;
    
    msp.tausq_inv = 1.0/tausq_values.col(i);
    
    if(maxit > 0){
      acceptable = msp.get_loglik_comps_w( msp.param_data );
    }
    double current_loglik = msp.param_data.loglik_w;
    
    for(m=0; (m<maxit) & keep_running; m++){
      msp.predicting = false;
      
      if(casc_w){
        start = std::chrono::steady_clock::now();
        msp.deal_with_w(msp.param_data, false);
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[w] "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
        }
      }
        
      if(casc_beta){
        start = std::chrono::steady_clock::now();
        //msp.deal_with_beta(false);
        msp.deal_with_BetaLambdaTau(msp.param_data, false, false, false, false); // false = no sample
        //
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[beta] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n"; 
        }
      }
      
      if(true){
        start = std::chrono::steady_clock::now();
        msp.logpost_refresh_after_gibbs(msp.param_data, false);
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[logpost_refresh_after_gibbs] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
        }
      }
      
      
      if(false){
        start = std::chrono::steady_clock::now();
        msp.deal_with_tausq(msp.param_data);
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[tausq] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
        }
      }
      
      
      wllsave(m, i) = msp.param_data.loglik_w;
      
      if(abs((current_loglik - wllsave(m, i))/wllsave(m, i)) < 1e-5){
        keep_running = false;
      } else {
        current_loglik = wllsave(m, i);
      }
      
    }
    
    if(true){
      msp.predicting = true;
      start = std::chrono::steady_clock::now();
      msp.predict(false); 
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & verbose){
        Rcpp::Rcout << "[predict] "
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
      }
    }
    
    beta_map.slice(i) = msp.Bcoeff;
    v_map.slice(i) = msp.w;
    w_map.slice(i) = msp.LambdaHw;
    lambda_map.slice(i) = msp.Lambda;
    lp_map.slice(i) = msp.linear_predictor;
    Rcpp::RNGScope scope;
    msp.predicty(false);
    yhat_map.slice(i) = msp.yhat;
    
    end_all = std::chrono::steady_clock::now();
    double comp_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    
    if(verbose){
      Rprintf("%d of %d, %d iters in %.1fs. \n", i+1, n_alts, m, comp_time/1000.0);
    }
    
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }
    
  
  
  return Rcpp::List::create(
    Rcpp::Named("yhat_map") = yhat_map,
    Rcpp::Named("beta_map") = beta_map,
    Rcpp::Named("lambda_map") = lambda_map,
    Rcpp::Named("w_map") = w_map,
    Rcpp::Named("v_map") = v_map,
    Rcpp::Named("lp_map") = lp_map,
    Rcpp::Named("w_logdens") = wllsave
  );
}

