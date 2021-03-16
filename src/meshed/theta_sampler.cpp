#include "meshed.h"

using namespace std;

void Meshed::metrop_theta(){
  message("[metrop_theta] start");
  
  theta_adapt.count_proposal();
  
  arma::vec param = arma::vectorise(param_data.theta);
  arma::vec new_param = arma::vectorise(param_data.theta);
  
  Rcpp::RNGScope scope;
  arma::vec U_update = mrstdnorm(new_param.n_elem, 1);
  
  
  // theta
  new_param = par_huvtransf_back(par_huvtransf_fwd(param, theta_unif_bounds) + 
    theta_adapt.paramsd * U_update, theta_unif_bounds);
  
  
  bool out_unif_bounds = unif_bounds(new_param, theta_unif_bounds);
  
  arma::mat theta_proposal = 
    arma::mat(new_param.memptr(), new_param.n_elem/k, k);
  
  alter_data.theta = theta_proposal;
  
 
  bool acceptable = get_loglik_comps_w( alter_data );
  
  bool accepted = false;
  double logaccept = 0;
  double current_loglik = 0;
  double new_loglik = 0;
  double prior_logratio = 0;
  double jacobian = 0;
  
  if(acceptable){
    new_loglik = alter_data.loglik_w;
    
    double before_loglik = param_data.loglik_w;
    
    bool from = get_loglik_comps_w( param_data );
    current_loglik = param_data.loglik_w;
    
    prior_logratio = calc_prior_logratio(
        alter_data.theta.tail_rows(1).t(), param_data.theta.tail_rows(1).t(), 2, 1); // sigmasq
    
    jacobian  = calc_jacobian(new_param, param, theta_unif_bounds);
    logaccept = new_loglik - current_loglik + 
      prior_logratio +
      jacobian;
  
    accepted = do_I_accept(logaccept);
  } else {
    accepted = false;
    //num_chol_fails ++;
    if(verbose & debug){
      Rcpp::Rcout << "[warning] numerical failure at MH proposal -- auto rejected\n";
    }
  }
  
  if(accepted){
    theta_adapt.count_accepted();
    
    accept_make_change();
    param_data.theta = theta_proposal;
    
    if(debug & verbose){
      Rcpp::Rcout << "log accept. " << logaccept << " : " << new_loglik << " " << current_loglik << 
        " " << prior_logratio << " " << jacobian << "s\n";
    }
  } else {
    if(debug & verbose){
      Rcpp::Rcout << "[theta] rejected (log accept. " << logaccept << " : " << new_loglik << " " << current_loglik << 
        " " << prior_logratio << " " << jacobian << ")\n";
    }
  }
  if(std::isnan(logaccept)){
    Rcpp::stop("NaN in logdensity");
  }
  
  theta_adapt.update_ratios();
  
  if(theta_adapt_active){
    theta_adapt.adapt(U_update, acceptable*exp(logaccept), theta_mcmc_counter); 
  }
  theta_mcmc_counter++;
  
  message("[metrop_theta] end");
}