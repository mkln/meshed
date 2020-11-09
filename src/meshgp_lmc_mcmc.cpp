#include "meshgp_lmc.h"
#include "interrupt_handler.h"
#include "mgp_utils.h"


//[[Rcpp::export]]
Rcpp::List lmc_mgp_mcmc(
    const arma::mat& y, 
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
    
    const arma::vec& tausq_ab,
    
    const arma::vec& start_w,
    const double& sigmasq,
    const arma::mat& theta,
    const arma::vec& beta,
    const double& tausq,
    
    const arma::mat& mcmcsd,
    
    int mcmc_keep = 100,
    int mcmc_burn = 100,
    int mcmc_thin = 1,
    
    int num_threads = 1,
    
    bool adapting=false,
    
    bool forced_grid=true,
    
    bool verbose=false,
    bool debug=false,
    int print_every=false,
    
    bool sample_beta=true,
    bool sample_tausq=true,
    bool sample_lambda=true,
    bool sample_theta=true,
    bool sample_w=true){
  
  Rcpp::Rcout << "Initializing.\n";
  
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
  
  double tempr = 1.0;
  
  int n = coords.n_rows;
  int d = coords.n_cols;
  
  int q  = y.n_cols;
  
  arma::mat set_unif_bounds = set_unif_bounds_in;
  
  if(verbose & debug){
    Rcpp::Rcout << "Limits to MCMC search for theta:\n";
    Rcpp::Rcout << set_unif_bounds << endl;
  }
  
  arma::mat metropolis_sd = mcmcsd;
  
  LMCMeshGP mesh(y, X, coords, k,
                
                parents, children, layer_names, layer_gibbs_group,
                
                indexing, indexing_obs,
                
                start_w, beta, sigmasq, theta, 1.0/tausq, 
                beta_Vi, tausq_ab,
                
                true, forced_grid, 
                verbose, debug);

  
  arma::vec param = arma::vectorise(mesh.param_data.theta);
  
  arma::cube b_mcmc = arma::zeros(X.n_cols, q, mcmc_thin*mcmc_keep);
  arma::mat tausq_mcmc = arma::zeros(q, mcmc_thin*mcmc_keep);
  arma::mat theta_mcmc = arma::zeros(param.n_elem, mcmc_thin*mcmc_keep);
  arma::cube lambda_mcmc = arma::zeros(q, k, mcmc_thin*mcmc_keep);
  
  arma::vec llsave = arma::zeros(mcmc_thin*mcmc_keep);
  arma::vec wllsave = arma::zeros(mcmc_thin*mcmc_keep);
  
  // field avoids limit in size of objects -- ideally this should be a cube
  arma::field<arma::mat> w_mcmc(mcmc_keep);
  arma::field<arma::mat> yhat_mcmc(mcmc_keep);
  
  for(int i=0; i<mcmc_keep; i++){
    w_mcmc(i) = arma::zeros(mesh.w.n_rows, k);
    yhat_mcmc(i) = arma::zeros(mesh.y.n_rows, q);
  }
  
  bool acceptable = false;
  acceptable = mesh.get_loglik_comps_w( mesh.param_data );
  acceptable = mesh.get_loglik_comps_w( mesh.alter_data );

  double current_loglik = tempr*mesh.param_data.loglik_w;
  if(verbose & debug){
    Rcpp::Rcout << "Starting from logdens: " << current_loglik << endl; 
  }
  
  double logaccept;
  
  // adaptive params
  int mcmc = mcmc_thin*mcmc_keep + mcmc_burn;
  
  RAMAdapt adaptivemc(param.n_elem, metropolis_sd, param.n_elem == 1 ? .45 : .25);
  
  
  
  bool interrupted = false;
  
  Rcpp::Rcout << "Running MCMC for " << mcmc << " iterations.\n\n";
  
  bool needs_update = true;
  
  //Rcpp::List recovered;
  
  //arma::vec predict_theta = arma::vectorise( mesh.param_data.theta );
  
  start_all = std::chrono::steady_clock::now();
  int m=0; int mx=0; int num_chol_fails=0;
  int mcmc_saved = 0; int w_saved = 0;
  try { 
    for(m=0; m<mcmc & !interrupted; m++){
      
      mesh.predicting = false;
      mx = m-mcmc_burn;
      if(mx >= 0){
        if(mx % mcmc_thin == 0){
          mesh.predicting = true;
        }
      }
      
      if(printall){
        tick_mcmc = std::chrono::steady_clock::now();
      }
      
      // --------- METROPOLIS STEP ---------
      start = std::chrono::steady_clock::now();
      if(sample_theta){
        adaptivemc.count_proposal();
        
        // theta
        Rcpp::RNGScope scope;
        arma::vec new_param = param;
        arma::vec U_update = arma::randn(param.n_elem);
        new_param = par_huvtransf_back(par_huvtransf_fwd(param, set_unif_bounds) + 
          adaptivemc.paramsd * U_update, set_unif_bounds);

        bool out_unif_bounds = unif_bounds(new_param, set_unif_bounds);
        
        arma::mat theta_proposal = arma::trans(arma::mat(new_param.memptr(), k, new_param.n_elem/k));

        mesh.theta_update(mesh.alter_data, theta_proposal);
        acceptable = mesh.get_loglik_comps_w( mesh.alter_data );
        
        bool accepted = !out_unif_bounds;
        double new_loglik = 0;
        double prior_logratio = 0;
        double jacobian = 0;
        
        if(acceptable){
          new_loglik = tempr*mesh.alter_data.loglik_w;
          current_loglik = tempr*mesh.param_data.loglik_w;
          
          prior_logratio = calc_prior_logratio(new_param, param);
          jacobian  = calc_jacobian(new_param, param, set_unif_bounds);
          logaccept = new_loglik - current_loglik + 
            prior_logratio +
            jacobian;

          if(std::isnan(logaccept)){
            Rcpp::Rcout << "Error: NaN logdensity in MCMC. Something went wrong\n"; 
            Rcpp::Rcout << new_param.t();
            Rcpp::Rcout << new_loglik << " " << current_loglik << " " << jacobian << endl;
            Rcpp::stop("Got NaN logdensity -- something went wrong.");
          }
          
          accepted = do_I_accept(logaccept);
          
        } else {
          accepted = false;
          num_chol_fails ++;
          if(verbose & debug){
            Rcpp::Rcout << "[warning] numerical failure at MH proposal -- auto rejected\n";
          }
        }
      
        if(accepted){
          adaptivemc.count_accepted();
          
          current_loglik = new_loglik;
          mesh.accept_make_change();
          param = new_param;
          needs_update = true;
      
        } else {
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] rejected (log accept. " << logaccept << ")\n";;
          }
          needs_update = false;
        }
        
        adaptivemc.update_ratios();
        
        if(adapting){
          adaptivemc.adapt(U_update, acceptable*exp(logaccept), m); 
        }
        
      }
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & verbose){
        Rcpp::Rcout << "[theta] " 
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
      }
      
      // --------- GIBBS STEPS ---------
      if(sample_w){
        start = std::chrono::steady_clock::now();
        mesh.gibbs_sample_w();
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[w] "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
        }
      }
      
      if(mesh.predicting){
        start = std::chrono::steady_clock::now();
        mesh.predict(); 
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[predict] "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
        }
      }
      
      if(sample_lambda){
        start = std::chrono::steady_clock::now();
        mesh.gibbs_sample_Lambda();
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[Lambda] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n"; 
        }
      }
      
      if(sample_beta){
        start = std::chrono::steady_clock::now();
        mesh.gibbs_sample_beta();
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[beta] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n"; 
        }
      }
      
      if(sample_tausq){
        start = std::chrono::steady_clock::now();
        mesh.gibbs_sample_tausq();
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[tausq] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
        }
      }
      
      if(sample_tausq || sample_beta || sample_w || sample_lambda){
        start = std::chrono::steady_clock::now();
        mesh.logpost_refresh_after_gibbs();
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[logpost_refresh_after_gibbs] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
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
          adaptivemc.print_summary(time_tick, time_mcmc, m, mcmc);
          
          tick_mcmc = std::chrono::steady_clock::now();
          
          printf("  p(w|theta) = %.2f    p(y|...) = %.2f  \n  phi = ", mesh.param_data.loglik_w, mesh.logpost);
          for(int pp=0; pp<mesh.param_data.theta.n_elem; pp++){
            printf("%.3f ", mesh.param_data.theta(pp));
          }
          printf("\n  tsq = ");
          for(int pp=0; pp<q; pp++){
            printf("%.3f ", 1.0/mesh.tausq_inv(pp));
          }
          printf("\n\n");
        
        } 
      } else {
        tick_mcmc = std::chrono::steady_clock::now();
      }
      //save
      if(mx >= 0){
        tausq_mcmc.col(w_saved) = 1.0 / mesh.tausq_inv;
        b_mcmc.slice(w_saved) = mesh.Bcoeff;
        theta_mcmc.col(w_saved) = arma::vectorise( mesh.param_data.theta );
        lambda_mcmc.slice(w_saved) = mesh.Lambda;
        llsave(w_saved) = mesh.logpost;
        wllsave(w_saved) = mesh.param_data.loglik_w;
        w_saved++;
        
        if(mx % mcmc_thin == 0){
          w_mcmc(mcmc_saved) = mesh.w;
          Rcpp::RNGScope scope;
          yhat_mcmc(mcmc_saved) = mesh.XB + mesh.LambdaHw + 
            arma::kron(arma::trans(pow(1.0/mesh.tausq_inv, .5)), arma::ones(n,1)) % arma::randn(n, q);
          mcmc_saved++;
        }
      }
    }
    
    end_all = std::chrono::steady_clock::now();
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC done [" << mcmc_time/1000.0 <<  "s]\n";
    
    return Rcpp::List::create(
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("lambda_mcmc") = lambda_mcmc,
      Rcpp::Named("paramsd") = adaptivemc.paramsd,
      Rcpp::Named("logpost") = llsave,
      Rcpp::Named("w_loglik") = wllsave,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0
    );
    
  } catch (...) {
    end_all = std::chrono::steady_clock::now();
    
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC has been interrupted. Returning partial saved results if any.\n";
    
    return Rcpp::List::create(
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("lambda_mcmc") = lambda_mcmc,
      Rcpp::Named("parents_indexing") = mesh.parents_indexing,
      Rcpp::Named("paramsd") = adaptivemc.paramsd,
      Rcpp::Named("whoswho") = mesh.u_is_which_col_f,
      Rcpp::Named("logpost") = llsave,
      Rcpp::Named("w_loglik") = wllsave,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0
    );
  }
}

