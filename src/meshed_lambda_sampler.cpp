#include "meshed.h"

using namespace std;

void Meshed::deal_with_Lambda(MeshDataLMC& data){
  if(arma::any(familyid > 0)){
    sample_hmc_Lambda();
  } else {
    if(forced_grid){
      sample_nc_Lambda_fgrid(data);
    } else {
      sample_nc_Lambda_std();
    }
  }
}

void Meshed::sample_nc_Lambda_fgrid(MeshDataLMC& data){
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_Lambda_fgrid] start (sampling via Robust adaptive Metropolis)\n";
  }
  
  start = std::chrono::steady_clock::now();
  
  lambda_adapt.count_proposal();
  Rcpp::RNGScope scope;
  
  arma::vec U_update = arma::randn(n_lambda_pars);
  arma::vec lambda_vec_current = Lambda.elem(lambda_sampling);
  
  arma::vec lambda_vec_proposal = par_huvtransf_back(
    par_huvtransf_fwd(lambda_vec_current, lambda_unif_bounds) + 
      lambda_adapt.paramsd * U_update, lambda_unif_bounds);
  arma::mat Lambda_proposal = Lambda;
  Lambda_proposal.elem(lambda_sampling) = lambda_vec_proposal;
  
  arma::mat LambdaHw_proposal = w * Lambda_proposal.t();
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    // we are doing this after sampling w so need to update the current likelihood too
    update_lly(u, param_data, LambdaHw);
    //calc_DplusSi(u, param_data, tausq_inv);
    
    // then for the proposal
    calc_DplusSi(u, alter_data, Lambda_proposal, tausq_inv);
    update_lly(u, alter_data, LambdaHw_proposal);
  }
  
  arma::vec Lambda_prop_d = Lambda_proposal.diag();
  arma::vec Lambda_d = Lambda.diag();
  arma::mat L_prior_prec = 1e-6 * 
    arma::eye(Lambda_d.n_elem, Lambda_d.n_elem);
  //double log_prior_ratio = arma::conv_to<double>::from(
  //  -0.5*Lambda_prop_d.t() * L_prior_prec * Lambda_prop_d
  //  +0.5*Lambda_d.t() * L_prior_prec * Lambda_d);
    
    double new_logpost = arma::accu(alter_data.ll_y);
    double start_logpost = arma::accu(param_data.ll_y);
    
    double jacobian  = calc_jacobian(lambda_vec_proposal, lambda_vec_current, lambda_unif_bounds);
    double logaccept = 
      new_logpost - 
      start_logpost + 
      //log_prior_ratio + 
      jacobian;
    // 
    // Rcpp::Rcout << "bounds " << endl;
    // Rcpp::Rcout << lambda_unif_bounds << endl;
    // Rcpp::Rcout << "proposal vs current " << endl;
    // Rcpp::Rcout << Lambda_proposal << endl; 
    // Rcpp::Rcout << Lambda << endl;
    // Rcpp::Rcout << arma::join_horiz(lambda_vec_proposal, lambda_vec_current) << endl;
    // Rcpp::Rcout << "logpost " << new_logpost << " vs " << start_logpost << endl;
    // Rcpp::Rcout << "logprior " << log_prior_ratio << " jac " << jacobian << endl;
    // Rcpp::Rcout << lambda_adapt.paramsd << endl;
    // Rcpp::Rcout << "tran " << endl
    //             << par_huvtransf_back(
    // par_huvtransf_fwd(lambda_vec_current, lambda_unif_bounds) + 
    // lambda_adapt.paramsd * U_update, lambda_unif_bounds) << endl;
    // 
    
    //double u = R::runif(0,1);//arma::randu();
    bool accepted = do_I_accept(logaccept);
    
    if(accepted){
      lambda_adapt.count_accepted();
      // accept the move
      Lambda = Lambda_proposal;
      LambdaHw = LambdaHw_proposal;
      
      param_data.DplusSi = alter_data.DplusSi;
      param_data.DplusSi_c = alter_data.DplusSi_c;
      param_data.DplusSi_ldet = alter_data.DplusSi_ldet;
      param_data.ll_y = alter_data.ll_y;
      
      logpost = new_logpost;
    } else {
      logpost = start_logpost;
    }
    
    lambda_adapt.update_ratios();
    lambda_adapt.adapt(U_update, exp(logaccept), lambda_mcmc_counter); 
    lambda_mcmc_counter ++;
    
    if(verbose & debug){
      end = std::chrono::steady_clock::now();
      Rcpp::Rcout << "[gibbs_sample_Lambda_fgrid] " << lambda_adapt.accept_ratio << " average acceptance rate, "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                  << "us.\n";
    }
}

arma::vec Meshed::sample_Lambda_row(int j){
  // build W
  
  arma::uvec subcols = arma::find(Lambda_mask.row(j) == 1);
  // filter: choose value of spatial processes at locations of Yj that are available
  arma::mat WWj = wU.submat(ix_by_q_a(j), subcols); // acts as X //*********
  //wmean.submat(ix_by_q_a(j), subcols); // acts as X
  
  arma::mat Wcrossprod = WWj.t() * WWj; 
  
  
  arma::mat Lprior_inv = 1e-6 * 
    arma::eye(WWj.n_cols, WWj.n_cols); 
  
  arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * Wcrossprod + Lprior_inv
  ), "lower");
  
  
  arma::mat Sigma_chol_L = arma::inv(arma::trimatl(Si_chol));
  
  arma::mat Simean_L = tausq_inv(j) * WWj.t() * 
    (y.submat(ix_by_q_a(j), oneuv*j) - XB.submat(ix_by_q_a(j), oneuv*j));
  
  arma::mat Lambdarow_Sig = Sigma_chol_L.t() * Sigma_chol_L;

  arma::vec Lprior_mean = arma::zeros(subcols.n_elem);
  //if(j < 0){
  //  Lprior_mean(j) = arma::stddev(arma::vectorise( y.submat(ix_by_q_a(j), oneuv*j) ));
  //}
  
  
  arma::mat Lambdarow_mu = Lprior_inv * Lprior_mean + 
    Lambdarow_Sig * Simean_L;
  
  arma::vec sampled = Lambdarow_mu + Sigma_chol_L.t() * arma::randn(subcols.n_elem);
  return sampled;
}

void Meshed::sample_nc_Lambda_std(){
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_Lambda] starting\n";
  }
  
  start = std::chrono::steady_clock::now();
  
  //arma::mat wmean = LambdaHw * Lambdati;
  
  // new with botev's 2017 method to sample from truncated normal
  for(unsigned int j=0; j<q; j++){
    arma::vec sampled = sample_Lambda_row(j);
    
    arma::uvec subcols = arma::find(Lambda_mask.row(j) == 1);
    Lambda.submat(oneuv*j, subcols) = sampled.t();
  } 
  
  LambdaHw = w * Lambda.t();
  
  // refreshing density happens in the 'logpost_refresh_after_gibbs' function
  
  if(verbose & debug){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_Lambda] " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

void Meshed::sample_hmc_Lambda(){
  if(verbose & debug){
    Rcpp::Rcout << "[sample_hmc_Lambda] starting\n";
  }
  
  start = std::chrono::steady_clock::now();
  
  arma::vec lambda_runif = vrunif(q);
  arma::vec lambda_runif2 = vrunif(q);
  for(unsigned int j=0; j<q; j++){
    arma::uvec subcols = arma::find(Lambda_mask.row(j) == 1);
    if(familyid(j) == 0){
      arma::vec sampled = sample_Lambda_row(j);
      Lambda.submat(oneuv*j, subcols) = sampled.t();
    } else {
      
      //Rcpp::Rcout << "j: " << j << endl;
      arma::vec offsets_obs = offsets(ix_by_q_a(j), oneuv * j);
      arma::vec xb_obs = XB(ix_by_q_a(j), oneuv * j);
      arma::vec offsets_for_lambda = offsets_obs + xb_obs;
      
      // build W
      // filter: choose value of spatial processes at locations of Yj that are available
      arma::mat WWj = wU.submat(ix_by_q_a(j), subcols); // acts as X //*********
      //wmean.submat(ix_by_q_a(j), subcols); // acts as X
      
      arma::mat Wcrossprod = WWj.t() * WWj; 
      
      arma::mat Vi = 1 * arma::eye(WWj.n_cols, WWj.n_cols); 
      arma::vec Vim = arma::zeros(WWj.n_cols);
      
      lambda_node.at(j).update_mv(offsets_for_lambda, 1.0/tausq_inv(j), Vim, Vi);
      lambda_node.at(j).X = WWj;
      lambda_node.at(j).XtX = Wcrossprod;
      
      arma::vec curLrow = arma::trans(Lambda.submat(oneuv*j, subcols));
      arma::mat rnorm_row = mrstdnorm(curLrow.n_elem, 1);
      
      // nongaussian
      //Rcpp::Rcout << "step " << endl;
      lambda_hmc_adapt.at(j).step();
      if((lambda_hmc_started(j) == 0) && (lambda_hmc_adapt.at(j).i == 10)){
        // wait a few iterations before starting adaptation
        //Rcpp::Rcout << "reasonable stepsize " << endl;
        
        
        double lambda_eps = find_reasonable_stepsize(curLrow, lambda_node.at(j), rnorm_row);
        //Rcpp::Rcout << "adapting scheme starting " << endl;
        AdaptE new_adapting_scheme;
        new_adapting_scheme.init(lambda_eps, k, w_hmc_srm, w_hmc_nuts, 1e4);
        lambda_hmc_adapt.at(j) = new_adapting_scheme;
        lambda_hmc_started(j) = 1;
        //Rcpp::Rcout << "done initiating adapting scheme" << endl;
      }
      
      
      arma::vec sampled = manifmala_cpp(curLrow, lambda_node.at(j), lambda_hmc_adapt.at(j), 
                         rnorm_row, lambda_runif(j), lambda_runif2(j), true, debug); 
      
      Lambda.submat(oneuv*j, subcols) = sampled.t();
      
      //Rcpp::Rcout << sampled.t();
      
    }
    
  } 
  
  LambdaHw = w * Lambda.t();
  
  // refreshing density happens in the 'logpost_refresh_after_gibbs' function
  if(verbose & debug){
    Rcpp::Rcout << "[sample_hmc_Lambda] done\n";
  }
  
}
