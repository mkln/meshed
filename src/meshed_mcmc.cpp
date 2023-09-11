

#include "utils_lmc.h"
#include "utils_interrupt_handler.h"
#include "utils_parametrize.h"
#include "meshed.h"

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
    
    int which_hmc=0,
    bool adapting=false,
    
    bool use_cache=true,
    bool forced_grid=true,
    bool use_ps=true,
    
    bool verbose=false,
    bool debug=false,
    int print_every=false,
    bool low_mem=false,
    
    bool sample_beta=true,
    bool sample_tausq=true,
    bool sample_lambda=true,
    bool sample_theta=true,
    bool sample_w=true){
  
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
  std::chrono::steady_clock::time_point start_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point tick_mcmc = std::chrono::steady_clock::now();
  // ------
  
  bool printall = print_every == 1;
  bool verbose_mcmc = printall;
  
  double tempr = 1;
  
  //unsigned int n = coords.n_rows;
  unsigned int d = coords.n_cols;
  unsigned int q  = y.n_cols;
  
  //arma::mat set_unif_bounds = set_unif_bounds_in;
  
  if(verbose & debug){
    Rcpp::Rcout << "Limits to MCMC search for theta:\n";
    Rcpp::Rcout << set_unif_bounds_in << endl;
  }
  // adaptive params
  int mcmc = mcmc_thin*mcmc_keep + mcmc_burn;

  arma::mat start_lambda = lambda * 
    ps_forward(theta, d, matern_twonu, use_ps);
  
  arma::mat start_theta = theta;
  if(verbose & debug){
    Rcpp::Rcout << "start theta \n" << theta;
  }
  
  Meshed msp(y, family,
            X, coords, k,
                parents, children, layer_names, layer_gibbs_group,
                
                indexing, indexing_obs,
                
                matern_twonu,
                start_w, beta, start_lambda, lambda_mask, start_theta, 1.0/tausq, 
                beta_Vi, tausq_ab,
                
                which_hmc,
                adapting,
                mcmcsd,
                set_unif_bounds_in,
                
                use_cache, forced_grid, use_ps,
                verbose, debug, num_threads);
  
  Rcpp::List caching_info;
  caching_info["coords"] = msp.coords_caching.n_elem; 
  caching_info["hrmats"] = msp.kr_caching.n_elem;
  
  arma::vec param = arma::vectorise(msp.param_data.theta);
  
  arma::cube b_mcmc = arma::zeros(X.n_cols, q, mcmc_thin*mcmc_keep);
  arma::mat tausq_mcmc = arma::zeros(q, mcmc_thin*mcmc_keep);
  arma::cube theta_mcmc = arma::zeros(param.n_elem/k, k, mcmc_thin*mcmc_keep);
  
  arma::cube lambda_mcmc = arma::zeros(q, k, mcmc_thin*mcmc_keep);
  arma::cube lambdastar_mcmc = arma::zeros(1,1,1);
  if(use_ps){
    lambdastar_mcmc = arma::zeros(q, k, mcmc_thin*mcmc_keep);
  }
  
  arma::vec logaccept_mcmc = arma::zeros(mcmc);
  
  arma::uvec mcmc_ix = arma::zeros<arma::uvec>(mcmc_keep);
  arma::vec llsave = arma::zeros(mcmc_thin*mcmc_keep);
  arma::vec wllsave = arma::zeros(mcmc_thin*mcmc_keep);
  
  Rcpp::List v_mcmc;
  Rcpp::List w_mcmc;
  Rcpp::List lp_mcmc;
  Rcpp::List yhat_mcmc;
  
  for(int i=0; i<mcmc_keep; i++){
    std::string iname = std::to_string(i);
    v_mcmc[iname] = Rcpp::wrap(arma::zeros(msp.w.n_rows, k));
    yhat_mcmc[iname] = Rcpp::wrap(arma::zeros(msp.y.n_rows, q)); 
    if(!low_mem){
      w_mcmc[iname] = Rcpp::wrap(arma::zeros(msp.w.n_rows, q));
      lp_mcmc[iname] = Rcpp::wrap(arma::zeros(msp.y.n_rows, q));
    }
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
  
  double logaccept = 0;
  
  bool interrupted = false;
  
  if(verbose){
    Rcpp::Rcout << "Running MCMC for " << mcmc << " iterations.\n\n";
  }
  
  
  start_all = std::chrono::steady_clock::now();
  int m=0; int mx=0; int num_chol_fails=0;
  int mcmc_saved = 0; int w_saved = 0;
  
  try {
    
    for(m=0; (m<mcmc) & (!interrupted); m++){
      
      msp.predicting = false;
      mx = m-mcmc_burn;
      if(mx >= 0){
        if((mx % mcmc_thin) == 0){
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
    
      // do block beta&lambda update if 
      // we can do Gibbs for both in all-Gaussian case, or
      // we have at least one non-Gaussian outcome (i.e. no MPP-like adjustment if using grid)
      if(arma::any(family > 0) + ((!forced_grid) & arma::all(family==0))){
      //if(false){
        if(sample_lambda+sample_beta+sample_tausq){
          start = std::chrono::steady_clock::now();
          msp.deal_with_BetaLambdaTau(msp.param_data, true, sample_beta, sample_lambda, sample_tausq); // true = sample
          end = std::chrono::steady_clock::now();
          if(verbose_mcmc & verbose){
            Rcpp::Rcout << "[BetaLambdaTau] " 
                        << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n"; 
          }
        }
      } else {
        /*if(sample_lambda+sample_beta){
          start = std::chrono::steady_clock::now();
          msp.deal_with_BetaLambdaTau(msp.param_data, true, sample_beta, sample_lambda, false); // true = sample
          end = std::chrono::steady_clock::now();
          if(verbose_mcmc & verbose){
            Rcpp::Rcout << "[BetaLambdaTau] " 
                        << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n"; 
          }
        }*/
        if(sample_lambda){
          start = std::chrono::steady_clock::now();
          msp.deal_with_Lambda(msp.param_data);
          end = std::chrono::steady_clock::now();
          if(verbose_mcmc & verbose){
            Rcpp::Rcout << "[Lambda] " 
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
        
        if(sample_tausq){
          start = std::chrono::steady_clock::now();
          msp.deal_with_tausq(msp.param_data);
          end = std::chrono::steady_clock::now();
          if(verbose_mcmc & verbose){
            Rcpp::Rcout << "[tausq] " 
                        << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
          }
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
      
      arma::mat lambda_transf_back = msp.Lambda * 
        ps_back(msp.param_data.theta, d, msp.matern.twonu, use_ps);
      
      if(mx >= 0){
        tausq_mcmc.col(w_saved) = 1.0 / msp.tausq_inv;
        b_mcmc.slice(w_saved) = msp.Bcoeff;
        
        theta_mcmc.slice(w_saved) = msp.param_data.theta;

        // lambda here reconstructs based on 1/phi Matern reparametrization
        if(use_ps){
          lambdastar_mcmc.slice(w_saved) = msp.Lambda;
        }
        lambda_mcmc.slice(w_saved) = lambda_transf_back;
          
        llsave(w_saved) = msp.logpost;
        wllsave(w_saved) = msp.param_data.loglik_w;
        w_saved++;
        
        if(mx % mcmc_thin == 0){
          std::string iname = std::to_string(mcmc_saved);
          
          v_mcmc[iname] = Rcpp::wrap(msp.w * ps_forward(msp.param_data.theta, 
                                                     d, msp.matern.twonu, use_ps));
          
          Rcpp::RNGScope scope;
          msp.predicty();
          yhat_mcmc[iname] = Rcpp::wrap(msp.yhat);
          
          if(!low_mem){
            w_mcmc[iname] = Rcpp::wrap(msp.LambdaHw);
            lp_mcmc[iname] = Rcpp::wrap(msp.linear_predictor);
          }
          
          mcmc_ix(mcmc_saved) = w_saved;
          
          mcmc_saved++;
        }
      }
      
      interrupted = checkInterrupt();
      if(interrupted){
        Rcpp::stop("Interrupted by the user.");
      }
      
      if((m>0) & (mcmc > 100)){
        
        bool print_condition = (print_every>0);
        if(print_condition){
          print_condition = print_condition & (!(m % print_every));
        };
        
        if(print_condition){
          end_mcmc = std::chrono::steady_clock::now();
          
          int time_tick = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - tick_mcmc).count();
          int time_mcmc = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - start_mcmc).count();
          msp.theta_adapt.print_summary(time_tick, time_mcmc, m, mcmc);
          
          tick_mcmc = std::chrono::steady_clock::now();
          if(verbose & debug){
            Rprintf("  p(w|theta) = %.2f    p(y|...) = %.2f  \n ", msp.param_data.loglik_w, msp.logpost);
          }
          unsigned int printlimit = 10;
          
          msp.theta_adapt.print_acceptance();
          Rprintf("  theta = ");
          unsigned int n_theta = msp.param_data.theta.n_elem;
          unsigned int n_print_theta = min(printlimit, n_theta);
          for(unsigned int pp=0; pp<n_print_theta; pp++){
            Rprintf("%.3f ", msp.param_data.theta(pp));
          }
          
          
          if(arma::any(msp.familyid == 0)){
            Rprintf("\n  tausq = ");
            unsigned int n_print_tsq = min(printlimit, q);
            for(unsigned int pp=0; pp<n_print_tsq; pp++){
              if(msp.familyid(pp) == 0){
                Rprintf("(%d) %.6f ", pp+1, 1.0/msp.tausq_inv(pp));
              }
            }
          }
          if(arma::any(msp.familyid == 3)){
            Rprintf("\n  tau (betareg) = ");
            unsigned int n_print_tsq = min(printlimit, q);
            for(unsigned int pp=0; pp<n_print_tsq; pp++){
              if(msp.familyid(pp) == 3){
                Rprintf("(%d) %.4f ", pp+1, 1.0/msp.tausq_inv(pp));
              }
            }
          }
          if(arma::any(msp.familyid == 4)){
            Rprintf("\n  tau (negbinom) = ");
            unsigned int n_print_tsq = min(printlimit, q);
            for(unsigned int pp=0; pp<n_print_tsq; pp++){
              if(msp.familyid(pp) == 4){
                Rprintf("(%d) %.4f ", pp+1, 1.0/msp.tausq_inv(pp));
              }
            }
          }
          if(use_ps || q > 1){
            arma::vec lvec = arma::vectorise(msp.Lambda);
            unsigned int n_lambda = lvec.n_elem;
            unsigned int n_print_lambda = min(printlimit, n_lambda);
            if(debug){
              Rprintf("\n  lambdastar = ");
              for(unsigned int pp=0; pp<n_print_lambda; pp++){
                Rprintf("%.3f ", lvec(pp));
              } 
            }
            lvec = arma::vectorise(lambda_transf_back);
            Rprintf("\n  lambda = ");
            for(unsigned int pp=0; pp<n_print_lambda; pp++){
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
    if(print_every>0){
      Rcpp::Rcout << "MCMC done [" << mcmc_time/1000.0 <<  "s]\n";
    }
    
    return Rcpp::List::create(
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("v_mcmc") = v_mcmc,
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("lp_mcmc") = lp_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("lambda_mcmc") = lambda_mcmc,
      Rcpp::Named("lambdastar_mcmc") = lambdastar_mcmc,
      Rcpp::Named("paramsd") = msp.theta_adapt.paramsd,
      Rcpp::Named("mcmc") = mcmc,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0,
      Rcpp::Named("proposal_failures") = num_chol_fails,
      Rcpp::Named("caching_info") = caching_info,
      Rcpp::Named("mcmc_ix") = mcmc_ix,
      Rcpp::Named("success") = true
    );
  
  } catch(const std::exception& e) {
    Rcpp::Rcout << "Caught exception \"" << e.what() << "\"\n";

    end_all = std::chrono::steady_clock::now();
    
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::warning("MCMC has been interrupted. Returning partial saved results if any.\n");
    
    return Rcpp::List::create(
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("v_mcmc") = v_mcmc,
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("lp_mcmc") = lp_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("lambda_mcmc") = lambda_mcmc,
      Rcpp::Named("lambdastar_mcmc") = lambdastar_mcmc,
      Rcpp::Named("paramsd") = msp.theta_adapt.paramsd,
      Rcpp::Named("mcmc") = mcmc,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0,
      Rcpp::Named("proposal_failures") = num_chol_fails,
      Rcpp::Named("caching_info") = caching_info,
      Rcpp::Named("mcmc_ix") = mcmc_ix,
      Rcpp::Named("success") = false
    );
  }
}

