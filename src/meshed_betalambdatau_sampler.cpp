#include "meshed.h"

using namespace std;

void apply2sd(arma::mat& x){
  for(unsigned int j=0; j<x.n_cols; j++){
    x.col(j) = x.col(j) / arma::stddev(x.col(j));  
  }
}

void Meshed::deal_with_BetaLambdaTau(MeshDataLMC& data, bool sample,
                                     bool sample_beta, bool sample_lambda, bool sample_tau){
  sample_hmc_BetaLambdaTau(sample, sample_beta, sample_lambda, sample_tau);
}


arma::vec Meshed::sample_BetaLambda_row(bool sample, int j, const arma::mat& rnorm_precalc){
  // build W
  arma::uvec subcols = arma::find(Lambda_mask.row(j) == 1);
  // filter: choose value of spatial processes at locations of Yj that are available
  
  arma::mat WWj = wU.submat(ix_by_q_a(j), subcols); // acts as X //*********
  //wmean.submat(ix_by_q_a(j), subcols); // acts as X
  if(!sample) { apply2sd(WWj); } // ***
  
  arma::mat XW = arma::join_horiz(X.rows(ix_by_q_a(j)), WWj);
  arma::mat Wcrossprod = XW.t() * XW; 
  
  arma::mat Lprior_inv = 1e-6 * arma::eye( XW.n_cols, XW.n_cols );
  Lprior_inv.submat(0, 0, p-1, p-1) = Vi; // prior precision for beta
  arma::vec Lprior_mean = arma::zeros(XW.n_cols);
  
  arma::vec sampled;
  if(sample){
    arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * Wcrossprod + Lprior_inv
      ), "lower");
    
    arma::mat Sigma_chol_L = arma::inv(arma::trimatl(Si_chol));
    arma::mat Simean_L = tausq_inv(j) * XW.t() * y.submat(ix_by_q_a(j), oneuv*j);  
       // - XB.submat(ix_by_q_a(j), oneuv*j));
    
    arma::mat Lambdarow_Sig = Sigma_chol_L.t() * Sigma_chol_L;
    arma::mat Lambdarow_mu = Lprior_inv * Lprior_mean + 
      Lambdarow_Sig * Simean_L;
  
    sampled = Lambdarow_mu + Sigma_chol_L.t() * 
        arma::trans(rnorm_precalc.submat(j, 0, j, XW.n_cols-1));
  } else {
    arma::mat Si_chol = arma::chol(
      arma::symmatu(Wcrossprod + Lprior_inv), "lower");
    
    arma::mat Sigma_chol_L = arma::inv(arma::trimatl(Si_chol));
    arma::mat Simean_L = XW.t() * y.submat(ix_by_q_a(j), oneuv*j);  
    // - XB.submat(ix_by_q_a(j), oneuv*j));
    
    arma::mat Lambdarow_Sig = Sigma_chol_L.t() * Sigma_chol_L;
    arma::mat Lambdarow_mu = Lprior_inv * Lprior_mean + 
      Lambdarow_Sig * Simean_L;
    sampled = Lambdarow_mu;
  }
  
  return sampled;
}

void Meshed::sample_hmc_BetaLambdaTau(bool sample, bool sample_beta, bool sample_lambda, bool sample_tau){
  if(verbose & debug){
    Rcpp::Rcout << "[sample_hmc_BetaLambdaTau] starting\n";
  }
  start = std::chrono::steady_clock::now();
  
  Rcpp::RNGScope scope;
  arma::mat rnorm_precalc = mrstdnorm(q, k+p);
  arma::vec lambda_runif = vrunif(q);
  arma::vec lambda_runif2 = vrunif(q);
  
  arma::vec tau_rnorm_precalc = mrstdnorm(q, 1);
  arma::vec tau_runif_precalc = vrunif(q);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int j=0; j<q; j++){
    
    ///
    /// ** Beta & Lambda update **
    ///
    arma::uvec subcols = arma::find(Lambda_mask.row(j) == 1);
    if(familyid(j) == 0){
      arma::vec sampled = sample_BetaLambda_row(sample, j, rnorm_precalc);
      
      if(sample_beta){
        Bcoeff.col(j) = sampled.head(p);
      }
      if(sample_lambda){
        Lambda.submat(oneuv*j, subcols) = arma::trans(sampled.tail(subcols.n_elem));
      }
    } else {
      arma::vec offsets_obs = offsets(ix_by_q_a(j), oneuv * j);
      
      // build W
      // filter: choose value of spatial processes at locations of Yj that are available
      arma::mat WWj = wU.submat(ix_by_q_a(j), subcols); // acts as X //*********
      //wmean.submat(ix_by_q_a(j), subcols); // acts as X
      if(!sample) { apply2sd(WWj); } // ***
      
      arma::mat XW = arma::join_horiz(X.rows(ix_by_q_a(j)), WWj);
      //arma::mat Wcrossprod = XW.t() * XW; 
      
      arma::mat BL_Vi = arma::eye( XW.n_cols, XW.n_cols );
      BL_Vi.submat(0, 0, p-1, p-1) = Vi; // prior precision for beta
      arma::vec BL_Vim = arma::zeros(XW.n_cols);
      
      lambda_node.at(j).update_mv(offsets_obs, 1.0/tausq_inv(j), BL_Vim, BL_Vi);
      lambda_node.at(j).X = XW;
      
      arma::vec curLrow = arma::join_vert(
        Bcoeff.col(j),
        arma::trans(Lambda.submat(oneuv*j, subcols)));
      arma::mat rnorm_row = arma::trans(rnorm_precalc.row(j).head(curLrow.n_elem));
      
      arma::vec sampled;
      if(sample){
        // nongaussian
        //Rcpp::Rcout << "step " << endl;
        lambda_hmc_adapt.at(j).step();
        if((lambda_hmc_started(j) == 0) & (lambda_hmc_adapt.at(j).i == 10)){
          // wait a few iterations before starting adaptation
          //Rcpp::Rcout << "reasonable stepsize " << endl;
          
          double lambda_eps = find_reasonable_stepsize(curLrow, lambda_node.at(j), rnorm_row);
          
          int n_params = curLrow.n_elem;
          AdaptE new_adapting_scheme(lambda_eps, n_params, w_hmc_srm, w_hmc_nuts, 1e4);
          lambda_hmc_adapt.at(j) = new_adapting_scheme;
          lambda_hmc_started(j) = 1;
          //Rcpp::Rcout << "done initiating adapting scheme" << endl;
        }
        
        if(which_hmc == 1){
          // mala
          sampled = mala_cpp(curLrow, lambda_node.at(j), lambda_hmc_adapt.at(j), 
                             rnorm_row, lambda_runif(j), true, debug);
        }
        if(which_hmc == 2){
          // nuts
          sampled = nuts_cpp(curLrow, lambda_node.at(j), lambda_hmc_adapt.at(j)); 
        }
        
        if((which_hmc == 3) || (which_hmc == 4)){
          // some form of manifold mala
          sampled = manifmala_cpp(curLrow, lambda_node.at(j), lambda_hmc_adapt.at(j), 
                                  rnorm_row, lambda_runif(j), lambda_runif2(j), 
                                  true, debug);
        }
        
        //sampled = manifmala_cpp(curLrow, lambda_node.at(j), lambda_hmc_adapt.at(j), 
        //                    rnorm_row, lambda_runif(j), lambda_runif2(j), 
        //                    true, true, debug); 
      } else {
        //sampled = irls_step(curLrow, lambda_node.at(j).y, lambda_node.at(j).X, lambda_node.at(j).offset, BL_Vi, familyid(j), tausq_inv(j));
        sampled = newton_step(curLrow, lambda_node.at(j), lambda_hmc_adapt.at(j), 1, false);
      }
      if(sample_beta){
        Bcoeff.col(j) = sampled.head(p);
      } 
      if(sample_lambda){
        Lambda.submat(oneuv*j, subcols) = arma::trans(sampled.tail(subcols.n_elem));
      }
    }
    
    XB.col(j) = X * Bcoeff.col(j);
    LambdaHw.col(j) = w * arma::trans(Lambda.row(j));

    if(sample & sample_tau){
      ///
      /// ** tausq update for beta/negbinom outcomes **
      ///
      double aprior = tausq_ab(0);
      double bprior = tausq_ab(1);
      if((familyid(j) == 3) || (familyid(j) == 4)){
        
        opt_tausq_adapt.at(j).count_proposal();
        
        arma::vec one = arma::ones(1);
        arma::vec U_update = one * tau_rnorm_precalc(j);
        
        arma::vec new_tsqiv = 
          par_huvtransf_back(par_huvtransf_fwd(one*tausq_inv(j), tausq_unif_bounds.rows(oneuv * j)) + 
          opt_tausq_adapt.at(j).paramsd * U_update, tausq_unif_bounds.rows(oneuv * j));
        
        double new_tsqi = new_tsqiv(0);
        //Rcpp::Rcout << arma::size(offsets) << " " << arma::size(XB) << " " << arma::size(LHW) << " " << arma::size(y) << endl;
        
        arma::vec start_logpost_vec = arma::zeros(ix_by_q_a(j).n_elem);
        arma::vec new_logpost_vec = arma::zeros(ix_by_q_a(j).n_elem);
        
        if(familyid(j) == 3){
          for(unsigned int ix=0; ix<ix_by_q_a(j).n_elem; ix++){
            int i = ix_by_q_a(j)(ix);
            
            double sigmoid = 1.0/(1.0 + exp(-offsets(i, j) - XB(i, j) - LambdaHw(i, j)));
            
            start_logpost_vec(ix) = betareg_logdens(y(i, j), sigmoid, tausq_inv(j));
            new_logpost_vec(ix) = betareg_logdens(y(i, j), sigmoid, new_tsqi);
          }
        } else if(familyid(j) == 4){
          for(unsigned int ix=0; ix<ix_by_q_a(j).n_elem; ix++){
            int i = ix_by_q_a(j)(ix);
            
            double logmu = offsets(i, j) + XB(i, j) + LambdaHw(i, j);
            double mu = exp(logmu);
            double oldalpha = tausq_inv(j);
            double newalpha = new_tsqi;
            start_logpost_vec(ix) = negbin_logdens(y(i, j), mu, logmu, oldalpha);
            new_logpost_vec(ix) = negbin_logdens(y(i, j), mu, logmu, newalpha);
          }
        }
        double start_logpost = arma::accu(start_logpost_vec);
        double new_logpost = arma::accu(new_logpost_vec);
        
        double prior_logratio = 0;
        
        if(aprior != 0){
          // for(int i=0; i<q; i++){
          //   prior_logratio += aprior * 
          //     (- log(new_tausq(i)) - log(tausq_inv(i)));
          // }
          
          prior_logratio = calc_prior_logratio(one /new_tsqi, one /tausq_inv(j), aprior, bprior);
        }
        
        if(std::isnan(prior_logratio)){
          Rcpp::Rcout << "NaN value from prior on tausq: a=" << aprior << " b=" << bprior << endl;
          Rcpp::stop("Terminated.");
        }
        
        double jacobian  = calc_jacobian(one*new_tsqi, one*tausq_inv(j), tausq_unif_bounds.rows(oneuv * j));
        double logaccept = new_logpost - start_logpost + 
          prior_logratio +
          jacobian;
  
        double u = tau_runif_precalc(j);
        bool accepted = exp(logaccept) > u;
        
        if(accepted){
          opt_tausq_adapt.at(j).count_accepted();
          // make the move
          tausq_inv(j) = new_tsqi;
        } 
        
        opt_tausq_adapt.at(j).update_ratios();
        opt_tausq_adapt.at(j).adapt(U_update, exp(logaccept), brtausq_mcmc_counter(j)); 
        brtausq_mcmc_counter(j) ++;
      }
    }
  }
  
  ///
  /// ** tausq update for gaussian outcomes when others are non-gaussian **
  ///
  if(sample & sample_tau & arma::any(familyid == 0)){
    double aprior = tausq_ab(0);
    double bprior = tausq_ab(1);
    for(unsigned int j=0; j<q; j++){
      if(familyid(j) == 0){
        // gibbs update
        arma::mat yrr = 
          y.submat(ix_by_q_a(j), oneuv*j) - 
          XB.submat(ix_by_q_a(j), oneuv*j) - 
          LambdaHw.submat(ix_by_q_a(j), oneuv*j); //***
        
        double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
        
        double aparam = aprior + ix_by_q_a(j).n_elem/2.0;
        double bparam = 1.0/( bprior + .5 * bcore );
        
        Rcpp::RNGScope scope;
        tausq_inv(j) = R::rgamma(aparam, bparam);
        logpost += 0.5 * (ix_by_q_a(j).n_elem + .0) * log(tausq_inv(j)) - 0.5*tausq_inv(j)*bcore;
        
        if(verbose & debug){
          Rcpp::Rcout << "[gibbs_sample_tausq] " << j << " | "
                      << aparam << " : " << bparam << " " << bcore << " --> " << 1.0/tausq_inv(j)
                      << "\n";
        }
      }
    }
  }

  // refreshing density happens in the 'logpost_refresh_after_gibbs' function
  if(verbose & debug){
    Rcpp::Rcout << "[sample_hmc_Lambda] done\n";
  }
  
}
