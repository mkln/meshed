#define ARMA_DONT_PRINT_ERRORS

#include "meshgp_lmc.h"
#include "interrupt_handler.h"
#include "mgp_utils.h"

arma::mat reparametrize_lambda_back(const arma::mat& Lambda_in, const arma::mat& theta, int d, int nutimes2){
  arma::mat reparametrizer; 
  if(d == 3){
    // exponential reparametrization of gneiting's ? 
    reparametrizer = arma::diagmat(pow(theta.row(1), - 0.5)); 
  } else {
    if(theta.n_rows > 2){
      // full matern
      arma::vec rdiag = arma::zeros(theta.n_cols);
      for(int j=0; j<rdiag.n_elem; j++){
        rdiag(j) = pow(theta(0, j), -theta(1, j));
      }
      reparametrizer = 
        arma::diagmat(rdiag) * 
        arma::diagmat(sqrt(theta.row(2))); 
    } else {
      // we use this from mcmc samples because those already have row0 as the transformed param
      // zhang 2004 corollary to theorem 2.
      reparametrizer = 
        arma::diagmat(pow(theta.row(0), -nutimes2/2.0)) * 
        arma::diagmat(sqrt(theta.row(1))); 
    }
  }
  return Lambda_in * reparametrizer;
}

arma::mat reparametrize_lambda_forward(const arma::mat& Lambda_in, const arma::mat& theta, int d, int nutimes2){
  arma::mat reparametrizer;
  if(d == 3){
    // exponential
    reparametrizer = arma::diagmat(pow(
      theta.row(1), + 0.5)); 
  } else {
    if(theta.n_rows > 2){
      // full matern: builds lambda*phi^nu
      arma::vec rdiag = arma::zeros(theta.n_cols);
      for(int j=0; j<rdiag.n_elem; j++){
        rdiag(j) = pow(theta(0, j), theta(1, j));
      }
      reparametrizer = 
        arma::diagmat(rdiag) * 
        arma::diagmat(pow(theta.row(2), -0.5)); ; 
    } else {
      // zhang 2004 corollary to theorem 2.
      
      reparametrizer = 
        arma::diagmat(pow(theta.row(0), + nutimes2/2.0)) * 
        arma::diagmat(pow(theta.row(1), -0.5)); 
    }
  }
  return Lambda_in * reparametrizer;
}

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
  
  arma::mat set_unif_bounds = set_unif_bounds_in;
  
  if(verbose & debug){
    Rcpp::Rcout << "Limits to MCMC search for theta:\n";
    Rcpp::Rcout << set_unif_bounds << endl;
  }
  // adaptive params
  int mcmc = mcmc_thin*mcmc_keep + mcmc_burn;
  
  arma::mat metropolis_sd = mcmcsd;
  
  arma::mat start_lambda = reparametrize_lambda_forward(lambda, theta, d, matern_twonu);
  
  arma::mat start_theta = theta;
  Rcpp::Rcout << "start theta \n" << theta;
  
  LMCMeshGP mesh(y, X, coords, k,
                
                parents, children, layer_names, layer_gibbs_group,
                
                indexing, indexing_obs,
                
                matern_twonu,
                start_w, beta, start_lambda, lambda_mask, start_theta, 1.0/tausq, 
                beta_Vi, tausq_ab,
                
                true, forced_grid, 
                verbose, debug, num_threads);

  
  arma::vec param = arma::vectorise(mesh.param_data.theta);
  
  arma::cube b_mcmc = arma::zeros(X.n_cols, q, mcmc_thin*mcmc_keep);
  arma::mat tausq_mcmc = arma::zeros(q, mcmc_thin*mcmc_keep);
  arma::cube theta_mcmc = arma::zeros(param.n_elem/k, k, mcmc_thin*mcmc_keep);
  
  arma::cube lambdastar_mcmc = arma::zeros(q, k, mcmc_thin*mcmc_keep);
  arma::cube lambda_mcmc = arma::zeros(q, k, mcmc_thin*mcmc_keep);
  
  arma::vec logaccept_mcmc = arma::zeros(mcmc);
  
  arma::vec llsave = arma::zeros(mcmc_thin*mcmc_keep);
  arma::vec wllsave = arma::zeros(mcmc_thin*mcmc_keep);
  
  // field avoids limit in size of objects -- ideally this should be a cube
  arma::field<arma::mat> w_mcmc(mcmc_keep);
  arma::field<arma::mat> lw_mcmc(mcmc_keep);
  arma::field<arma::mat> wgen_mcmc(mcmc_keep); // remove me
  arma::field<arma::mat> yhat_mcmc(mcmc_keep);
  
  for(int i=0; i<mcmc_keep; i++){
    w_mcmc(i) = arma::zeros(mesh.w.n_rows, k);
    lw_mcmc(i) = arma::zeros(mesh.y.n_rows, q);
    wgen_mcmc(i) = arma::zeros(mesh.w.n_rows, k); // remove me
    yhat_mcmc(i) = arma::zeros(mesh.y.n_rows, q);
  }
  
  bool acceptable = false;
  
  if(mcmc > 0){
    acceptable = mesh.get_loglik_comps_w( mesh.param_data );
    acceptable = mesh.get_loglik_comps_w( mesh.alter_data );
  }
  

  double current_loglik = tempr*mesh.param_data.loglik_w;
  if(verbose & debug){
    Rcpp::Rcout << "Starting from logdens: " << current_loglik << endl; 
  }
  
  double logaccept;

  RAMAdapt adaptivemc(param.n_elem, metropolis_sd, .25);
  
  bool interrupted = false;
  Rcpp::Rcout << "Running MCMC for " << mcmc << " iterations.\n\n";
  
  start_all = std::chrono::steady_clock::now();
  int m=0; int mx=0; int num_chol_fails=0;
  int mcmc_saved = 0; int w_saved = 0;
  //try {
    for(m=0; m<mcmc & !interrupted; m++){
      //Rcpp::Rcout << "m: " << m << endl;
      
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
        
        arma::vec new_param = param;
        Rcpp::RNGScope scope;
        arma::vec U_update = arma::randn(param.n_elem);
        
        // theta
        new_param = par_huvtransf_back(par_huvtransf_fwd(param, set_unif_bounds) + 
          adaptivemc.paramsd * U_update, set_unif_bounds);
        
        bool out_unif_bounds = unif_bounds(new_param, set_unif_bounds);
        arma::mat theta_proposal = //arma::trans(
          arma::mat(new_param.memptr(), new_param.n_elem/k, k);
        
        mesh.theta_update(mesh.alter_data, theta_proposal);
        
        acceptable = mesh.get_loglik_comps_w( mesh.alter_data );
        
        
        bool accepted = false;
        double new_loglik = 0;
        double prior_logratio = 0;
        double jacobian = 0;
        
        if(acceptable){
          new_loglik = tempr*mesh.alter_data.loglik_w;
          current_loglik = tempr*mesh.param_data.loglik_w;
          
          prior_logratio = calc_prior_logratio(
              mesh.alter_data.theta.tail_rows(1).t(), mesh.param_data.theta.tail_rows(1).t(), 1e-4, 1e-4); // sigmasq
          
          jacobian  = calc_jacobian(new_param, param, set_unif_bounds);
          logaccept = new_loglik - current_loglik + 
            prior_logratio +
            jacobian;
          
          
          if(false){
            Rcpp::Rcout << "current " << current_loglik << " parm: " << param.t();
            Rcpp::Rcout << "new " << new_loglik << " proposal: " << new_param.t();
            Rcpp::Rcout << "log accept: " << logaccept << " j:" << jacobian << " prior: " << prior_logratio << endl;

            Rcpp::Rcout << "-- LogDens components:\n"
                        << arma::accu(mesh.alter_data.logdetCi_comps) << " vs " << arma::accu(mesh.param_data.logdetCi_comps) << endl
                        << arma::accu(mesh.alter_data.loglik_w_comps) << " vs " << arma::accu(mesh.param_data.loglik_w_comps) << endl
                        << arma::accu(mesh.alter_data.ll_y) << " vs " << arma::accu(mesh.param_data.ll_y) << endl;
            
            
          }
          
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
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] accepted (log accept. " << logaccept << ")\n";;
          }
        } else {
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] rejected (log accept. " << logaccept << ")\n";;
          }
        }
        
        adaptivemc.update_ratios();
        
        if(adapting){
          adaptivemc.adapt(U_update, acceptable*exp(logaccept), m + mcmc_startfrom); 
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
        mesh.gibbs_sample_w(mesh.param_data, true);
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[w] "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
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
      }
    
      if(sample_lambda){
        start = std::chrono::steady_clock::now();
        mesh.deal_with_Lambda(mesh.param_data);
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[Lambda] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n"; 
        }
      }
      
      if(sample_tausq){
        start = std::chrono::steady_clock::now();
        mesh.deal_with_tausq(mesh.param_data, 2.01, 1);
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[tausq] " 
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
      
      
      if(sample_tausq || sample_beta || sample_w || sample_lambda){
        start = std::chrono::steady_clock::now();
        mesh.logpost_refresh_after_gibbs(mesh.param_data);
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
          Rprintf("  p(w|theta) = %.2f    p(y|...) = %.2f  \n  theta = ", mesh.param_data.loglik_w, mesh.logpost);
          for(int pp=0; pp<mesh.param_data.theta.n_elem; pp++){
            Rprintf("%.3f ", mesh.param_data.theta(pp));
          }
          Rprintf("\n  tsq = ");
          for(int pp=0; pp<q; pp++){
            Rprintf("%.3f ", 1.0/mesh.tausq_inv(pp));
          }
          Rprintf("\n\n");
        } 
      } else {
        tick_mcmc = std::chrono::steady_clock::now();
      }
      
      //save
      logaccept_mcmc(m) = logaccept > 0 ? 0 : logaccept;
      
      if(mx >= 0){
        tausq_mcmc.col(w_saved) = 1.0 / mesh.tausq_inv;
        b_mcmc.slice(w_saved) = mesh.Bcoeff;
        
        theta_mcmc.slice(w_saved) = mesh.param_data.theta;
        // lambda here reconstructs based on 1/phi Matern reparametrization
        lambdastar_mcmc.slice(w_saved) = mesh.Lambda;
        lambda_mcmc.slice(w_saved) = reparametrize_lambda_back(mesh.Lambda, mesh.param_data.theta, d, mesh.matern.twonu);
        
        llsave(w_saved) = mesh.logpost;
        wllsave(w_saved) = mesh.param_data.loglik_w;
        w_saved++;
        
        if(mx % mcmc_thin == 0){
          w_mcmc(mcmc_saved) = mesh.w;
          lw_mcmc(mcmc_saved) = mesh.LambdaHw;
          wgen_mcmc(mcmc_saved) = mesh.wgen; // remove me
          Rcpp::RNGScope scope;
          
          yhat_mcmc(mcmc_saved) = mesh.XB + 
            mesh.wU * mesh.Lambda.t() + 
            arma::kron(arma::trans(pow(1.0/mesh.tausq_inv, .5)), arma::ones(n,1)) % arma::randn(n, q);
          mcmc_saved++;
        }
      }
    }
    
    end_all = std::chrono::steady_clock::now();
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC done [" << mcmc_time/1000.0 <<  "s]\n";
    
    return Rcpp::List::create(
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("lw_mcmc") = lw_mcmc,
      Rcpp::Named("wgen_mcmc") = wgen_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("lambda_mcmc") = lambda_mcmc,
      Rcpp::Named("lambdastar_mcmc") = lambdastar_mcmc,
      Rcpp::Named("paramsd") = adaptivemc.paramsd,
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

