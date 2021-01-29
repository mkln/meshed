#define ARMA_DONT_PRINT_ERRORS

#include "utils/mesh_lmc_utils.h"
#include "utils/interrupt_handler.h"
#include "mcmc/parametrize.h"

#include "meshed/meshed.h"

//[[Rcpp::export]]
Rcpp::List meshed_mcmc(
    const arma::mat& y, 
    const arma::uvec& family,
    
    const arma::mat& X, 
    
    const arma::mat& coords, 
    
    int k,
    
    const arma::field<arma::uvec>& parents,
    const arma::field<arma::uvec>& children,
    
    const arma::vec& layer_names,
    const arma::vec& layer_gibbs_group,
    
    
    const arma::field<arma::uvec>& indexing,
    const arma::field<arma::uvec>& indexing_obs,
    
    const arma::mat& set_unif_bounds_in,
    const arma::mat& beta_Vi,
    
    const arma::vec& sigmasq_ab,
    const arma::vec& tausq_ab,
    
    int matern_twonu,
    
    const arma::mat& start_w,
    const arma::mat& lambda,
    const arma::umat& lambda_mask,
    const arma::mat& theta,
    const arma::mat& beta,
    const arma::vec& tausq,
    
    const arma::mat& mcmcsd,
    
    int mcmc_keep = 100,
    int mcmc_burn = 100,
    int mcmc_thin = 1,
    
    int mcmc_startfrom = 0,
    
    int num_threads = 1,
    
    bool adapting=false,
    
    bool use_cache=true,
    bool forced_grid=true,
    bool use_ps=true,
    
    bool verbose=false,
    bool debug=false,
    int print_every=false,
    
    bool sample_beta=true,
    bool sample_tausq=true,
    bool sample_lambda=true,
    bool sample_theta=true,
    bool sample_w=true){
  
  Rcpp::Rcout << "Initializing.\n";
  
  int twonu = 1; // choose between 1, 3, 5. 1=exp covar
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  // timers
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point start_all = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_all = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point start_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point tick_mcmc = std::chrono::steady_clock::now();
  // ------
  
  bool printall = print_every == 1;
  bool verbose_mcmc = printall;
  
  double tempr = 1;
  
  int n = coords.n_rows;
  int d = coords.n_cols;
  int q  = y.n_cols;
  
  //arma::mat set_unif_bounds = set_unif_bounds_in;
  
  if(verbose & debug){
    Rcpp::Rcout << "Limits to MCMC search for theta:\n";
    Rcpp::Rcout << set_unif_bounds_in << endl;
  }
  // adaptive params
  int mcmc = mcmc_thin*mcmc_keep + mcmc_burn;
  
  //arma::mat metropolis_sd = mcmcsd;
  
  arma::mat start_lambda = reparametrize_lambda_forward(lambda, theta, d, matern_twonu, use_ps);
  
  arma::mat start_theta = theta;
  Rcpp::Rcout << "start theta \n" << theta;
  
  Meshed msp(y, family,
            X, coords, k,
                parents, children, layer_names, layer_gibbs_group,
                
                indexing, indexing_obs,
                
                matern_twonu,
                start_w, beta, start_lambda, lambda_mask, start_theta, 1.0/tausq, 
                beta_Vi, tausq_ab,
                
                adapting,
                mcmcsd,
                set_unif_bounds_in,
                
                use_cache, forced_grid, use_ps,
                verbose, debug, num_threads);
  
  
  arma::vec param = arma::vectorise(msp.param_data.theta);
  
  arma::cube b_mcmc = arma::zeros(X.n_cols, q, mcmc_thin*mcmc_keep);
  arma::mat tausq_mcmc = arma::zeros(q, mcmc_thin*mcmc_keep);
  arma::cube theta_mcmc = arma::zeros(param.n_elem/k, k, mcmc_thin*mcmc_keep);
  
  arma::field<arma::vec> eps_mcmc(mcmc_thin * mcmc_keep);
  arma::field<arma::vec> ratios_mcmc(mcmc_thin * mcmc_keep);
  
  //arma::cube lambdastar_mcmc = arma::zeros(q, k, mcmc_thin*mcmc_keep);
  arma::cube lambda_mcmc = arma::zeros(q, k, mcmc_thin*mcmc_keep);
  
  arma::vec logaccept_mcmc = arma::zeros(mcmc);
  
  arma::vec llsave = arma::zeros(mcmc_thin*mcmc_keep);
  arma::vec wllsave = arma::zeros(mcmc_thin*mcmc_keep);
  
  // field avoids limit in size of objects -- ideally this should be a cube
  arma::field<arma::mat> w_mcmc(mcmc_keep);
  arma::field<arma::mat> lw_mcmc(mcmc_keep);
  arma::field<arma::mat> yhat_mcmc(mcmc_keep);
  
  for(int i=0; i<mcmc_keep; i++){
    w_mcmc(i) = arma::zeros(msp.w.n_rows, k);
    lw_mcmc(i) = arma::zeros(msp.y.n_rows, q);
    yhat_mcmc(i) = arma::zeros(msp.y.n_rows, q);
  }
  
  bool acceptable = false;
  
  if(mcmc > 0){
    acceptable = msp.get_loglik_comps_w( msp.param_data );
    acceptable = msp.get_loglik_comps_w( msp.alter_data );
  }
  

  double current_loglik = tempr*msp.param_data.loglik_w;
  if(verbose & debug){
    Rcpp::Rcout << "Starting from logdens: " << current_loglik << endl; 
  }
  
  double logaccept;

  //RAMAdapt adaptivemc(param.n_elem, metropolis_sd, .25);
  
  bool interrupted = false;
  Rcpp::Rcout << "Running MCMC for " << mcmc << " iterations.\n\n";
  
  start_all = std::chrono::steady_clock::now();
  int m=0; int mx=0; int num_chol_fails=0;
  int mcmc_saved = 0; int w_saved = 0;
  //try {
    for(m=0; m<mcmc & !interrupted; m++){
      
      msp.predicting = false;
      mx = m-mcmc_burn;
      if(mx >= 0){
        if(mx % mcmc_thin == 0){
          msp.predicting = true;
        }
      }
      
      if(printall){
        tick_mcmc = std::chrono::steady_clock::now();
      }
      
      if(sample_theta){
        start = std::chrono::steady_clock::now();
        msp.metrop_theta();
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[theta] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
        }
      }
      
      if(sample_w){
        start = std::chrono::steady_clock::now();
        msp.deal_with_w(msp.param_data);
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[w] "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
        }
        if(msp.predicting){
          start = std::chrono::steady_clock::now();
          msp.predict(); 
          end = std::chrono::steady_clock::now();
          if(verbose_mcmc & verbose){
            Rcpp::Rcout << "[predict] "
                        << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
          }
        }
      }
    
      if(sample_lambda){
        start = std::chrono::steady_clock::now();
        msp.deal_with_Lambda(msp.param_data);
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[Lambda] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n"; 
        }
      }
      
      if(sample_tausq){
        start = std::chrono::steady_clock::now();
        msp.deal_with_tausq(msp.param_data);
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[tausq] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
        }
      }
      
      if(sample_beta){
        start = std::chrono::steady_clock::now();
        msp.deal_with_beta();
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[beta] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n"; 
        }
      }
      
      if(sample_tausq || sample_beta || sample_w || sample_lambda){
        start = std::chrono::steady_clock::now();
        msp.logpost_refresh_after_gibbs(msp.param_data);
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[logpost_refresh_after_gibbs] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
        }
      }
      
      //save
      logaccept_mcmc(m) = logaccept > 0 ? 0 : logaccept;
      
      arma::mat lambda_transf_back = reparametrize_lambda_back(msp.Lambda, msp.param_data.theta, d, msp.matern.twonu, use_ps);
      
      if(mx >= 0){
        tausq_mcmc.col(w_saved) = 1.0 / msp.tausq_inv;
        b_mcmc.slice(w_saved) = msp.Bcoeff;
        
        theta_mcmc.slice(w_saved) = msp.param_data.theta;
        eps_mcmc(w_saved) = msp.hmc_eps;
        
        // lambda here reconstructs based on 1/phi Matern reparametrization
        //lambdastar_mcmc.slice(w_saved) = msp.Lambda;
        lambda_mcmc.slice(w_saved) = lambda_transf_back;
          
        llsave(w_saved) = msp.logpost;
        wllsave(w_saved) = msp.param_data.loglik_w;
        w_saved++;
        
        if(mx % mcmc_thin == 0){
          w_mcmc(mcmc_saved) = msp.w;
          lw_mcmc(mcmc_saved) = msp.LambdaHw;
          Rcpp::RNGScope scope;
        
          msp.predicty();
          yhat_mcmc(mcmc_saved) = msp.yhat;
          mcmc_saved++;
        }
      }
      
      if((m>0) & (mcmc > 100)){
        if(!(m % print_every)){
          interrupted = checkInterrupt();
          if(interrupted){
            Rcpp::stop("Interrupted by the user.");
          }
          end_mcmc = std::chrono::steady_clock::now();
          
          int time_tick = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - tick_mcmc).count();
          int time_mcmc = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - start_mcmc).count();
          msp.theta_adapt.print_summary(time_tick, time_mcmc, m, mcmc);
          
          tick_mcmc = std::chrono::steady_clock::now();
          Rprintf("  p(w|theta) = %.2f    p(y|...) = %.2f  \n  theta = ", msp.param_data.loglik_w, msp.logpost);
          for(int pp=0; pp<msp.param_data.theta.n_elem; pp++){
            Rprintf("%.3f ", msp.param_data.theta(pp));
          }
          Rprintf("\n  tsq = ");
          for(int pp=0; pp<q; pp++){
            if(msp.familyid(pp) == 0){
              Rprintf("(%d) %.6f ", pp+1, 1.0/msp.tausq_inv(pp));
            }
          }
          if(use_ps){
            arma::vec lvec = arma::vectorise(msp.Lambda);
            Rprintf("\n  lambdastar = ");
            for(int pp=0; pp<lvec.n_elem; pp++){
              Rprintf("%.3f ", lvec(pp));
            }
            lvec = arma::vectorise(lambda_transf_back);
            Rprintf("\n  lambda = ");
            for(int pp=0; pp<lvec.n_elem; pp++){
              Rprintf("%.3f ", lvec(pp));
            }
          }
          
          Rprintf("\n\n");
        } 
      } else {
        tick_mcmc = std::chrono::steady_clock::now();
      }
    }
    
    end_all = std::chrono::steady_clock::now();
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC done [" << mcmc_time/1000.0 <<  "s]\n";

    
    // 
    // Rcpp::Named("p_logdetCi_comps") = msp.param_data.logdetCi_comps,
    //   Rcpp::Named("a_logdetCi_comps") = msp.alter_data.logdetCi_comps,
    //   Rcpp::Named("p_wcore") = msp.param_data.wcore,
    //   Rcpp::Named("a_wcore") = msp.alter_data.wcore
    //   
    return Rcpp::List::create(
      Rcpp::Named("eps") = eps_mcmc,
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("lw_mcmc") = lw_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("lambda_mcmc") = lambda_mcmc,
      Rcpp::Named("paramsd") = msp.theta_adapt.paramsd,
      Rcpp::Named("mcmc") = mcmc,
      Rcpp::Named("logpost") = llsave,
      Rcpp::Named("logaccept") = logaccept_mcmc,
      Rcpp::Named("w_logdens") = wllsave,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0,
      Rcpp::Named("proposal_failures") = num_chol_fails
    );
  
  /*} catch (...) {
    end_all = std::chrono::steady_clock::now();
    
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC has been interrupted. Returning partial saved results if any.\n";
    
    return Rcpp::List::create(
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("lw_mcmc") = lw_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("lambda_mcmc") = lambda_mcmc,
      Rcpp::Named("paramsd") = adaptivemc.paramsd,
      Rcpp::Named("mcmc") = mcmc,
      Rcpp::Named("logpost") = llsave,
      Rcpp::Named("w_logdens") = wllsave,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0,
      Rcpp::Named("proposal_failures") = num_chol_fails
    );
  }*/
}

