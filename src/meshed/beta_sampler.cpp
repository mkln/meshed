#include "meshed.h"

using namespace std;

void Meshed::deal_with_beta(){
  if(false & (gibbs_or_hmc == false)){
    gibbs_sample_beta();
  } else {
    hmc_sample_beta();
  }
}

void Meshed::gibbs_sample_beta(){
  message("[gibbs_sample_beta]");
  start = std::chrono::steady_clock::now();
  
  Rcpp::RNGScope scope;
  arma::mat bmat = arma::randn(p, q);
  
  // for forced grid we use non-reference samples 
  arma::mat LHW = wU * Lambda.t();
  
  for(int j=0; j<q; j++){
    arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * XtX(j) + Vi), "lower");
    arma::mat Sigma_chol_Bcoeff = arma::inv(arma::trimatl(Si_chol));
    
    arma::mat Xprecy_j = Vim + tausq_inv(j) * X.rows(ix_by_q_a(j)).t() * 
      (y.submat(ix_by_q_a(j), oneuv*j) - LHW.submat(ix_by_q_a(j), oneuv*j)); //***
    
    Bcoeff.col(j) = Sigma_chol_Bcoeff.t() * (Sigma_chol_Bcoeff * Xprecy_j + bmat.col(j));
    XB.col(j) = X * Bcoeff.col(j);
  }
  
  if(verbose & debug){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_beta] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}



void Meshed::hmc_sample_beta(){
  message("[hmc_sample_beta]");
  // choose between NUTS or Gibbs
  start = std::chrono::steady_clock::now();
  
  arma::mat bmat = arma::randn(p, q);
  arma::mat LHW = wU * Lambda.t();
  
  for(int j=0; j<q; j++){
    //Rcpp::Rcout << "j: " << j << endl;
    arma::vec offsets_obs = offsets(ix_by_q_a(j), oneuv * j);
    arma::vec lw_obs = LHW(ix_by_q_a(j), oneuv * j);
    arma::vec offsets_for_beta = offsets_obs + lw_obs;
    
    available_data.at(j).update_mv(offsets_for_beta, 1.0/tausq_inv(j), Vim, Vi);
    
    if(familyid(j) == -1){ // disable for testing only
      // gaussian
      arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * available_data.at(j).XtX + Vi), "lower");
      arma::mat Sigma_chol_Bcoeff = arma::inv(arma::trimatl(Si_chol));
      //arma::vec w_available = w.rows(na_ix_all);
      arma::mat Xprecy_j = Vim + tausq_inv(j) * available_data.at(j).Xres;//available_data.at(j).X.t() * 
      //(available_data.at(j).y - offsets_for_beta.rows(ix_by_q_a(j)));
      Bcoeff.col(j) = Sigma_chol_Bcoeff.t() * (Sigma_chol_Bcoeff * Xprecy_j + bmat.col(j));
    } else {
      // nongaussian
      //Rcpp::Rcout << "step " << endl;
      beta_nuts.at(j).step();
      if(beta_nuts_started(j) == 0){
        // wait a few iterations before starting adaptation
        //Rcpp::Rcout << "reasonable stepsize " << endl;
        double beta_eps = find_reasonable_stepsize(Bcoeff.col(j), available_data.at(j));
        //Rcpp::Rcout << "adapting scheme starting " << endl;
        AdaptE new_adapting_scheme(beta_eps, 1e6);
        beta_nuts.at(j) = new_adapting_scheme;
        beta_nuts_started(j) = 1;
        //Rcpp::Rcout << "done initiating adapting scheme" << endl;
      }
      
      Bcoeff.col(j) = sample_one_mala_cpp(Bcoeff.col(j), available_data.at(j), beta_nuts.at(j)); 

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
