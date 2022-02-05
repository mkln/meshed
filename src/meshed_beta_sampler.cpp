#include "meshed.h"

using namespace std;

void Meshed::deal_with_beta(bool sample){
  hmc_sample_beta(sample);
}

void Meshed::hmc_sample_beta(bool sample){
  if(verbose & debug){
    Rcpp::Rcout << "[hmc_sample_beta]\n";
  }
  
  // choose between NUTS or Gibbs
  start = std::chrono::steady_clock::now();
  
  arma::mat bmat = mrstdnorm(p, q);
  arma::vec bunifv = vrunif(q);
  arma::vec bunifv2 = vrunif(q);
  
  arma::mat LHW = wU * Lambda.t();
  
  for(unsigned int j=0; j<q; j++){
    //Rcpp::Rcout << "j: " << j << endl;
    arma::vec offsets_obs = offsets(ix_by_q_a(j), oneuv * j);
    arma::vec lw_obs = LHW(ix_by_q_a(j), oneuv * j);
    arma::vec offsets_for_beta = offsets_obs + lw_obs;
    
    beta_node.at(j).update_mv(offsets_for_beta, 1.0/tausq_inv(j), Vim, Vi);
    
    if(familyid(j) == 0){ // disable for testing only
      // gaussian
      arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * beta_node.at(j).XtX + Vi), "lower");
      arma::mat Sigma_chol_Bcoeff = arma::inv(arma::trimatl(Si_chol));
      //arma::vec w_available = w.rows(na_ix_all);
      arma::mat Xprecy_j = Vim + tausq_inv(j) * beta_node.at(j).Xres;//beta_node.at(j).X.t() * 
      //(beta_node.at(j).y - offsets_for_beta.rows(ix_by_q_a(j)));
      
      Bcoeff.col(j) = Sigma_chol_Bcoeff.t() * Sigma_chol_Bcoeff * Xprecy_j;
      if(sample){
        Bcoeff.col(j) += Sigma_chol_Bcoeff.t() * bmat.col(j);
      }
      
    } else {
      // nongaussian
      beta_hmc_adapt.at(j).step();
      if(sample & (beta_hmc_started(j) == 0)){
        // wait a few iterations before starting adaptation
        //Rcpp::Rcout << "reasonable stepsize " << endl;
        double beta_eps = find_reasonable_stepsize(Bcoeff.col(j), beta_node.at(j), bmat.cols(oneuv * j));
        //Rcpp::Rcout << "adapting scheme starting " << endl;
        AdaptE new_adapting_scheme;
        new_adapting_scheme.init(beta_eps, p, w_hmc_srm, w_hmc_nuts, 1e4);
        beta_hmc_adapt.at(j) = new_adapting_scheme;
        beta_hmc_started(j) = 1;
        //Rcpp::Rcout << "done initiating adapting scheme" << endl;
      }
      
      if(sample){
        Bcoeff.col(j) = manifmala_cpp(Bcoeff.col(j), beta_node.at(j), 
                   beta_hmc_adapt.at(j), bmat.cols(oneuv * j), bunifv(j), bunifv2(j),
                   true, false); 
      } else {
        //Bcoeff.col(j) = newton_step(Bcoeff.col(j), beta_node.at(j), beta_hmc_adapt.at(j), 1, true);
        Bcoeff.col(j) = irls_step(Bcoeff.col(j), beta_node.at(j).y, beta_node.at(j).X, beta_node.at(j).offset, Vi, familyid(j));
      }
      
    }
    XB.col(j) = X * Bcoeff.col(j);
  }
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[hmc_sample_beta] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}


