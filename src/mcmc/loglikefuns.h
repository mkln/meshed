
// GENERIC and not being used anymore

#include <RcppArmadillo.h>
#include "../distributions/studentt.h"
#include "../distributions/mvnormal.h"

#include "distparams.h"
using namespace std;


// log posterior 
inline double loglike_cpp(const arma::vec& x, const DistParams& postparams){
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  start = std::chrono::steady_clock::now();
  
  if(postparams.latent == "gaussian"){
    // likelihood part
    double loglike = 0;
    if(postparams.family == "gaussian"){
      // simple example: normal-normal model
      // argument x here is w in y = Xb + Zw + e where e ~ N(0, tausq)
      // calculate unnormalized log posterior
      if(postparams.which == "w"){
        loglike = 1.0/postparams.tausq * arma::conv_to<double>::from(
          postparams.Zres.t() * x - .5 * x.t() * postparams.ZtZ * x);
        //Rcpp::Rcout << "gaussian loglike: " << loglike << "\n";
        //Rcpp::Rcout << x.t() * postparams.ZtZ * x << "\n";
        //Rcpp::Rcout << postparams.Zres.t() * x << "\n";
      } else {
        //beta
        loglike = 1.0/postparams.tausq * arma::conv_to<double>::from(
          postparams.Zres.t() * x - .5 * x.t() * postparams.XtX * x);
      }
      
    }
    if(postparams.family == "poisson"){
      // poisson
      if(postparams.which == "w"){
        loglike = arma::conv_to<double>::from( postparams.yZ * x - //postparams.y.t() * postparams.Z * x - 
          postparams.na_1_blocks.t() * exp(postparams.offset + postparams.Z * x) );
      } else {
        //beta
        loglike = arma::conv_to<double>::from( postparams.yX * x - //postparams.y.t() * postparams.X * x - 
          postparams.na_1_blocks.t() * exp(postparams.offset + postparams.X * x) );
      }
    }
    if(postparams.family == "binomial"){
      //arma::vec sigmoid;
      arma::vec XB;
      if(postparams.which == "w"){
        // x=w
        //sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.Z * x ));
        XB = postparams.offset + postparams.Z * x;
      } else {
        // x=beta
        //sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.X * x ));
        XB = postparams.offset + postparams.X * x;
      }
      
      // y and y1 are both zero when missing data
      //loglike = arma::conv_to<double>::from( 
      //  postparams.y.t() * log(sigmoid) + postparams.y1.t() * log(1-sigmoid)
      //);
      
      loglike = arma::conv_to<double>::from(-postparams.ones.t() * log1p(exp(-XB)) - postparams.y1.t() * XB);
      //Rcpp::Rcout << "v " << loglike << "\n";
      
      //Rcpp::Rcout << "loglike at: " << x.t() << "\n";
      //Rcpp::Rcout << "loglike: " << loglike << "\n";
      if(std::isnan(loglike)){
        loglike = -arma::datum::inf;
        //Rcpp::Rcout << "loglike new: " << loglike << "\n";
      }
    }
    
    if(postparams.which != "w"){
      // unless it's the latent spatial process, this is gaussian
      double logprior = arma::conv_to<double>::from(
        x.t() * postparams.mstar - .5 * x.t() * postparams.Vw_i * x);
      
      //end = std::chrono::steady_clock::now();
      //loglike_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
      //Rcpp::Rcout << "intermediate: " << loglike + logprior << "\n";
      return ( loglike + logprior );
    } else {
      std::chrono::steady_clock::time_point start;
      std::chrono::steady_clock::time_point end;
      double timing = 0;
      
      //Rcpp::Rcout << "> loglike > U: " << postparams.u << "\n";
      
      //Rcpp::Rcout << "entering sauce \n";
      //Rcpp::Rcout << "-- Kxxi \n";
      // prior part : t process prior. here: full conditionals
      //arma::mat Kxxi = (*postparams.param_data).w_cond_prec_parents(postparams.u);
      //Rcpp::Rcout << "-- Kcx \n";
      //arma::mat Kcx = (*postparams.param_data).w_cond_mean_K(postparams.u);
      //Rcpp::Rcout << "-- Ri \n";
      //arma::mat Ri = (*postparams.param_data).w_cond_prec(postparams.u);
      //Rcpp::Rcout << "-- p_ix \n";
      //arma::uvec p_ix = (*postparams.parents_indexing)(postparams.u);
      //Rcpp::Rcout << "-- w_parents \n";
      //arma::vec w_parents = arma::vectorise( (*postparams.w_full).rows(p_ix) );
      
      //Rcpp::Rcout << "----> fwdcond_dmvt \n";
      //start = std::chrono::steady_clock::now();
      
      double logprior = fwdcond_dmvn(x, postparams.Ri, postparams.parKxxpar, 
                                     postparams.Kcxpar, 
                                     postparams.parents_dim);
      
      //end = std::chrono::steady_clock::now();
      //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
      //Rcpp::Rcout << "loglike1 " << llcount << " "
      //            << timing
      //            << "us.\n";
      
      //start = std::chrono::steady_clock::now();
      
      //Rcpp::Rcout << "-- children_of_this \n";
      //arma::uvec children_of_this = (*postparams.children)(postparams.u);
      
      for(int c=0; c<postparams.num_children; c++ ){
        //int child = children_of_this(c);
        
        //Rcpp::Rcout << "> loglike > child #" << c << "\n";
        //Rcpp::Rcout << "-- child [" << child << "]\n";
        //Rcpp::Rcout << "-- c_ix \n";
        //arma::uvec c_ix = (*postparams.indexing)(child);
        //Rcpp::Rcout << "-- " << arma::size(c_ix) << " \n";
        //Rcpp::Rcout << "-- pofc_ix \n";
        //arma::uvec pofc_ix = (*postparams.parents_indexing)(child);
        //Rcpp::Rcout << "-- pofc_ix_x \n";
        //arma::uvec pofc_ix_x = (*postparams.u_is_which_col_f_u)(c)(0);
        //Rcpp::Rcout << "-- pofc_ix_other \n";
        //arma::uvec pofc_ix_other = (*postparams.u_is_which_col_f_u)(c)(1);
        //Rcpp::Rcout << "-- w_childs_parents \n";
        //arma::vec w_childs_parents = arma::vectorise( (*postparams.w_full).rows(pofc_ix) );
        //Rcpp::Rcout << "-- w_child \n";
        //arma::vec w_child = arma::vectorise( (*postparams.w_full).rows(c_ix) );
        //Rcpp::Rcout << "-- w_otherparents \n";
        //arma::vec w_otherparents = w_childs_parents.rows(pofc_ix_other);
        //Rcpp::Rcout << "-- Kxxi_of_child \n";
        //arma::mat Kxxi_of_child = (*postparams.param_data).w_cond_prec_parents(child);
        //Rcpp::Rcout << "-- Kcx_of_child \n";
        //arma::mat Kcx_of_child = (*postparams.param_data).w_cond_mean_K(child);
        //Rcpp::Rcout << "-- Ri_of_child \n";
        //arma::mat Ri_of_child = (*postparams.param_data).w_cond_prec(child);
        
        //arma::mat Kcx_x = Kcx_of_child.cols(ix_x);
        //arma::mat Kcx_other = Kcx_of_child.cols(ix_otherparents);
        //arma::mat Kxxi_x = Kxxi_of_child(ix_x, ix_x);
        //arma::mat Kxxi_other = Kxxi_of_child(ix_otherparents, ix_otherparents);
        //arma::mat Kxxi_xo = Kxxi_of_child(ix_x, ix_otherparents);
        
        //Rcpp::Rcout << "----> bwdcond_dmvt \n";
        //int num_parents = (*postparams.parents_indexing)(child).n_elem;
        logprior += bwdcond_dmvn(x, postparams.w_child(c), postparams.Ri_of_child(c), 
                                 postparams.Kcx_x(c), postparams.Kxxi_x(c),
                                 postparams.Kxo_wo(c), postparams.Kco_wo(c), postparams.woKoowo(c), 
                                 postparams.dim_of_pars_of_children(c));
        
      }
      
      //end = std::chrono::steady_clock::now();
      //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
      //Rcpp::Rcout << "loglike2 " << llcount << " "
      //            << timing
      //            << "us.\n";
      //llcount ++;
      //Rcpp::Rcout << "loglike: " << loglike << " logprior: " << logprior << "\n";
      return ( loglike + logprior );
    }
  }
  
  if(postparams.latent == "student"){
    // likelihood part
    double loglike = 0;
    if(postparams.family == "gaussian"){
      // simple example: normal-normal model
      // argument x here is w in y = Xb + Zw + e where e ~ N(0, tausq)
      // calculate unnormalized log posterior
      if(postparams.which == "w"){
        loglike = 1.0/postparams.tausq * arma::conv_to<double>::from(
          postparams.Zres.t() * x - .5 * x.t() * postparams.ZtZ * x);
        //Rcpp::Rcout << "gaussian loglike: " << loglike << "\n";
        //Rcpp::Rcout << x.t() * postparams.ZtZ * x << "\n";
        //Rcpp::Rcout << postparams.Zres.t() * x << "\n";
      } else {
        //beta
        loglike = 1.0/postparams.tausq * arma::conv_to<double>::from(
          postparams.Zres.t() * x - .5 * x.t() * postparams.XtX * x);
      }
      
    }
    if(postparams.family == "poisson"){
      // poisson
      if(postparams.which == "w"){
        loglike = arma::conv_to<double>::from( postparams.y.t() * postparams.Z * x - 
          postparams.na_1_blocks.t() * exp(postparams.offset + postparams.Z * x) );
      } else {
        //beta
        loglike = arma::conv_to<double>::from( postparams.y.t() * postparams.X * x - 
          postparams.na_1_blocks.t() * exp(postparams.offset + postparams.X * x) );
      }
    }
    if(postparams.family == "binomial"){
      arma::vec sigmoid;
      
      if(postparams.which == "w"){
        // x=w
        sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.Z * x ));
        
      } else {
        // x=beta
        sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.X * x ));
      }
      
      // y and y1 are both zero when missing data
      loglike = arma::conv_to<double>::from( 
        postparams.y.t() * log(sigmoid) + postparams.y1.t() * log(1-sigmoid)
      );
      if(std::isnan(loglike)){
        loglike = -arma::datum::inf;
        //Rcpp::Rcout << "loglike new: " << loglike << "\n";
      }
    }
    
    if(postparams.which != "w"){
      // unless it's the latent spatial process, this is gaussian
      double logprior = arma::conv_to<double>::from(
        x.t() * postparams.mstar - .5 * x.t() * postparams.Vw_i * x);
      return ( loglike + logprior );
    } else {
      std::chrono::steady_clock::time_point start;
      std::chrono::steady_clock::time_point end;
      double timing = 0;
      
      //Rcpp::Rcout << "> loglike > U: " << postparams.u << "\n";
      
      //Rcpp::Rcout << "entering sauce \n";
      //Rcpp::Rcout << "-- Kxxi \n";
      // prior part : t process prior. here: full conditionals
      //arma::mat Kxxi = (*postparams.param_data).w_cond_prec_parents(postparams.u);
      //Rcpp::Rcout << "-- Kcx \n";
      //arma::mat Kcx = (*postparams.param_data).w_cond_mean_K(postparams.u);
      //Rcpp::Rcout << "-- Ri \n";
      //arma::mat Ri = (*postparams.param_data).w_cond_prec(postparams.u);
      //Rcpp::Rcout << "-- p_ix \n";
      //arma::uvec p_ix = (*postparams.parents_indexing)(postparams.u);
      //Rcpp::Rcout << "-- w_parents \n";
      //arma::vec w_parents = arma::vectorise( (*postparams.w_full).rows(p_ix) );
      
      //Rcpp::Rcout << "----> fwdcond_dmvt \n";
      //start = std::chrono::steady_clock::now();
      
      double logprior = fwdcond_dmvt(x, postparams.Ri, postparams.parKxxpar, 
                                     postparams.Kcxpar, postparams.nu,
                                     postparams.parents_dim);
      
      //end = std::chrono::steady_clock::now();
      //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
      //Rcpp::Rcout << "loglike1 " << llcount << " "
      //            << timing
      //            << "us.\n";
      
      //start = std::chrono::steady_clock::now();
      
      //Rcpp::Rcout << "-- children_of_this \n";
      //arma::uvec children_of_this = (*postparams.children)(postparams.u);
      
      for(int c=0; c<postparams.num_children; c++ ){
        //int child = children_of_this(c);
        
        //Rcpp::Rcout << "> loglike > child #" << c << "\n";
        //Rcpp::Rcout << "-- child [" << child << "]\n";
        //Rcpp::Rcout << "-- c_ix \n";
        //arma::uvec c_ix = (*postparams.indexing)(child);
        //Rcpp::Rcout << "-- " << arma::size(c_ix) << " \n";
        //Rcpp::Rcout << "-- pofc_ix \n";
        //arma::uvec pofc_ix = (*postparams.parents_indexing)(child);
        //Rcpp::Rcout << "-- pofc_ix_x \n";
        //arma::uvec pofc_ix_x = (*postparams.u_is_which_col_f_u)(c)(0);
        //Rcpp::Rcout << "-- pofc_ix_other \n";
        //arma::uvec pofc_ix_other = (*postparams.u_is_which_col_f_u)(c)(1);
        //Rcpp::Rcout << "-- w_childs_parents \n";
        //arma::vec w_childs_parents = arma::vectorise( (*postparams.w_full).rows(pofc_ix) );
        //Rcpp::Rcout << "-- w_child \n";
        //arma::vec w_child = arma::vectorise( (*postparams.w_full).rows(c_ix) );
        //Rcpp::Rcout << "-- w_otherparents \n";
        //arma::vec w_otherparents = w_childs_parents.rows(pofc_ix_other);
        //Rcpp::Rcout << "-- Kxxi_of_child \n";
        //arma::mat Kxxi_of_child = (*postparams.param_data).w_cond_prec_parents(child);
        //Rcpp::Rcout << "-- Kcx_of_child \n";
        //arma::mat Kcx_of_child = (*postparams.param_data).w_cond_mean_K(child);
        //Rcpp::Rcout << "-- Ri_of_child \n";
        //arma::mat Ri_of_child = (*postparams.param_data).w_cond_prec(child);
        
        //arma::mat Kcx_x = Kcx_of_child.cols(ix_x);
        //arma::mat Kcx_other = Kcx_of_child.cols(ix_otherparents);
        //arma::mat Kxxi_x = Kxxi_of_child(ix_x, ix_x);
        //arma::mat Kxxi_other = Kxxi_of_child(ix_otherparents, ix_otherparents);
        //arma::mat Kxxi_xo = Kxxi_of_child(ix_x, ix_otherparents);
        
        //Rcpp::Rcout << "----> bwdcond_dmvt \n";
        //int num_parents = (*postparams.parents_indexing)(child).n_elem;
        logprior += bwdcond_dmvt(x, postparams.w_child(c), postparams.Ri_of_child(c), 
                                 postparams.Kcx_x(c), postparams.Kxxi_x(c),
                                 postparams.Kxo_wo(c), postparams.Kco_wo(c), postparams.woKoowo(c), 
                                 postparams.nu, postparams.dim_of_pars_of_children(c));
        
      }
      
      //end = std::chrono::steady_clock::now();
      //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
      //Rcpp::Rcout << "loglike2 " << llcount << " "
      //            << timing
      //            << "us.\n";
      //llcount ++;
      //Rcpp::Rcout << "loglike: " << loglike << " logprior: " << logprior << "\n";
      return ( loglike + logprior );
    }
  }
  
}

// Gradient of the log posterior
inline arma::vec grad_loglike_cpp(const arma::vec& x, const DistParams& postparams){
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  start = std::chrono::steady_clock::now();
  
  if(postparams.latent == "gaussian"){
    arma::vec grad_loglike(x.n_elem);
    if(postparams.family == "gaussian"){
      if(postparams.which == "w"){
        grad_loglike = 1.0/postparams.tausq * (postparams.Zres - postparams.ZtZ * x);
      } else {
        grad_loglike = 1.0/postparams.tausq * (postparams.Zres - postparams.XtX * x);
      }
    }
    if(postparams.family == "poisson"){
      if(postparams.which == "w"){
        grad_loglike = postparams.Z.t() * (postparams.y - postparams.na_1_blocks % 
          exp(postparams.offset + postparams.Z * x));
      } else {
        grad_loglike = postparams.X.t() * (postparams.y - postparams.na_1_blocks % 
          exp(postparams.offset + postparams.X * x));
      }
    }
    if(postparams.family == "binomial"){
      arma::vec sigmoid;
      if(postparams.which == "w"){
        sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.Z * x));
        grad_loglike = postparams.Z.t() * (postparams.y - postparams.na_1_blocks % sigmoid );
      } else {
        sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.X * x));
        grad_loglike = postparams.X.t() * (postparams.y - postparams.na_1_blocks % sigmoid );
      }
      //Rcpp::Rcout << "gradient at: " << x.t() << "\n";
      //Rcpp::Rcout << "grad_loglike: " << grad_loglike.t() << "\n";
    }
    
    if(postparams.which != "w"){
      arma::vec grad_logprior = postparams.mstar - postparams.Vw_i * x;
      
      //end = std::chrono::steady_clock::now();
      //grad_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
      return grad_loglike + grad_logprior;
    } else {
      //Rcpp::Rcout << "> grad > U: " << postparams.u << "\n";
      
      // prior part : t process prior. here: full conditionals
      //arma::mat Kxxi = (*postparams.param_data).w_cond_prec_parents(postparams.u);
      //arma::mat Kcx = (*postparams.param_data).w_cond_mean_K(postparams.u);
      //arma::mat Ri = (*postparams.param_data).w_cond_prec(postparams.u);
      //arma::uvec p_ix = (*postparams.parents_indexing)(postparams.u);
      //arma::vec w_parents = arma::vectorise( (*postparams.w_full).rows(p_ix) );
      
      //Rcpp::Rcout << "grad_fwdcond_dmvt parents \n";
      //start = std::chrono::steady_clock::now();
      
      arma::vec grad_logprior_par = grad_fwdcond_dmvn(x, postparams.Ri, 
                                                      postparams.parKxxpar, postparams.Kcxpar, 
                                                      postparams.parents_dim);
      
      arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
      
      //end = std::chrono::steady_clock::now();
      //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      //Rcpp::Rcout << "grad1 " << glcount << " "
      //            << timing
      //            << "us.\n";
      
      //start = std::chrono::steady_clock::now();
      
      //Rcpp::Rcout << "grad step-1\n"; 
      //Rcpp::Rcout << grad_logprior << "\n";
      //arma::uvec children_of_this = (*postparams.children)(postparams.u);
      
      
      for(int c=0; c<postparams.num_children; c++ ){
        //Rcpp::Rcout << "> grad > child #" << c << "\n";
        //int child = children_of_this(c);
        //Rcpp::Rcout << "child [" << child << "]\n";
        //arma::uvec c_ix = (*postparams.indexing)(child);
        //arma::uvec pofc_ix = (*postparams.parents_indexing)(child);
        //arma::uvec pofc_ix_x = (*postparams.u_is_which_col_f_u)(c)(0);
        //arma::uvec pofc_ix_other = (*postparams.u_is_which_col_f_u)(c)(1);
        //arma::vec w_childs_parents = arma::vectorise( (*postparams.w_full).rows(pofc_ix) );
        
        //arma::vec w_otherparents = w_childs_parents.rows(pofc_ix_other);
        //arma::mat Kxxi_of_child = (*postparams.param_data).w_cond_prec_parents(child);
        //arma::mat Kcx_of_child = (*postparams.param_data).w_cond_mean_K(child);
        
        //arma::vec w_child = arma::vectorise( (*postparams.w_full).rows(c_ix) );
        //arma::mat Ri_of_child = (*postparams.param_data).w_cond_prec(child);
        
        //arma::mat Kcx_x = Kcx_of_child.cols(pofc_ix_x);
        //arma::mat Kcx_other = Kcx_of_child.cols(pofc_ix_other);
        //arma::mat Kxxi_x = Kxxi_of_child(pofc_ix_x, pofc_ix_x);
        //arma::mat Kxxi_other = Kxxi_of_child(pofc_ix_other, pofc_ix_other);
        //arma::mat Kxxi_xo = Kxxi_of_child(pofc_ix_x, pofc_ix_other);
        
        //Kxo_wo = Kxxi_xo * w_otherparents;
        //Kco_wo = Kcx_other*w_otherparents;
        //woKoowo = w_otherparents.t() * Kxxi_other * w_otherparents;
        
        //Rcpp::Rcout << "grad_bwdcond_dmvt child \n";
        //Rcpp::Rcout << "grad step-child " << c << "\n"; 
        //int num_parents = (*postparams.parents_indexing)(child).n_elem;
        
        grad_logprior_chi += grad_bwdcond_dmvn(x, postparams.w_child(c), postparams.Ri_of_child(c), 
                                               postparams.Kcx_x(c), postparams.Kxxi_x(c),
                                               postparams.Kxo_wo(c), postparams.Kco_wo(c), 
                                               postparams.woKoowo(c), 
                                               postparams.dim_of_pars_of_children(c));
        //Rcpp::Rcout << grad_logprior << "\n";
      }
      
      
      //Rcpp::Rcout << "gradient: \n";
      //Rcpp::Rcout << arma::join_horiz(grad_loglike, grad_logprior) << "\n";
      
      //end = std::chrono::steady_clock::now();
      //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      //Rcpp::Rcout << "grad2 " << glcount << " "
      //            << timing
      //            << "us.\n";
      //glcount ++;
      
      //Rcpp::Rcout << arma::join_horiz(grad_loglike, grad_logprior_par, grad_logprior_chi) <<"\n";
      //Rcpp::Rcout << "LL: " << grad_loglike << " LP: " << grad_logprior_par + grad_logprior_chi << "\n";
      return grad_loglike + grad_logprior_par + grad_logprior_chi;
    }
    
  }
  
  if(postparams.latent == "student"){
    arma::vec grad_loglike(x.n_elem);
    if(postparams.family == "gaussian"){
      if(postparams.which == "w"){
        grad_loglike = 1.0/postparams.tausq * (postparams.Zres - postparams.ZtZ * x);
      } else {
        grad_loglike = 1.0/postparams.tausq * (postparams.Zres - postparams.XtX * x);
      }
    }
    if(postparams.family == "poisson"){
      if(postparams.which == "w"){
        grad_loglike = postparams.Z.t() * (postparams.y - postparams.na_1_blocks % 
          exp(postparams.offset + postparams.Z * x));
      } else {
        grad_loglike = postparams.X.t() * (postparams.y - postparams.na_1_blocks % 
          exp(postparams.offset + postparams.X * x));
      }
    }
    if(postparams.family == "binomial"){
      arma::vec sigmoid;
      if(postparams.which == "w"){
        sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.Z * x));
        grad_loglike = postparams.Z.t() * (postparams.y - postparams.na_1_blocks % sigmoid );
      } else {
        sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.X * x));
        grad_loglike = postparams.X.t() * (postparams.y - postparams.na_1_blocks % sigmoid );
      }
    }
    
    if(postparams.which != "w"){
      arma::vec grad_logprior = postparams.mstar - postparams.Vw_i * x;
      return grad_loglike + grad_logprior;
    } else {
      //Rcpp::Rcout << "> grad > U: " << postparams.u << "\n";
      
      // prior part : t process prior. here: full conditionals
      //arma::mat Kxxi = (*postparams.param_data).w_cond_prec_parents(postparams.u);
      //arma::mat Kcx = (*postparams.param_data).w_cond_mean_K(postparams.u);
      //arma::mat Ri = (*postparams.param_data).w_cond_prec(postparams.u);
      //arma::uvec p_ix = (*postparams.parents_indexing)(postparams.u);
      //arma::vec w_parents = arma::vectorise( (*postparams.w_full).rows(p_ix) );
      
      //Rcpp::Rcout << "grad_fwdcond_dmvt parents \n";
      //start = std::chrono::steady_clock::now();
      
      arma::vec grad_logprior_par = grad_fwdcond_dmvt(x, postparams.Ri, 
                                                      postparams.parKxxpar, postparams.Kcxpar, 
                                                      postparams.nu, postparams.parents_dim);
      
      arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
      
      //end = std::chrono::steady_clock::now();
      //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      //Rcpp::Rcout << "grad1 " << glcount << " "
      //            << timing
      //            << "us.\n";
      
      //start = std::chrono::steady_clock::now();
      
      //Rcpp::Rcout << "grad step-1\n"; 
      //Rcpp::Rcout << grad_logprior << "\n";
      //arma::uvec children_of_this = (*postparams.children)(postparams.u);
      
      
      for(int c=0; c<postparams.num_children; c++ ){
        //Rcpp::Rcout << "> grad > child #" << c << "\n";
        //int child = children_of_this(c);
        //Rcpp::Rcout << "child [" << child << "]\n";
        //arma::uvec c_ix = (*postparams.indexing)(child);
        //arma::uvec pofc_ix = (*postparams.parents_indexing)(child);
        //arma::uvec pofc_ix_x = (*postparams.u_is_which_col_f_u)(c)(0);
        //arma::uvec pofc_ix_other = (*postparams.u_is_which_col_f_u)(c)(1);
        //arma::vec w_childs_parents = arma::vectorise( (*postparams.w_full).rows(pofc_ix) );
        
        //arma::vec w_otherparents = w_childs_parents.rows(pofc_ix_other);
        //arma::mat Kxxi_of_child = (*postparams.param_data).w_cond_prec_parents(child);
        //arma::mat Kcx_of_child = (*postparams.param_data).w_cond_mean_K(child);
        
        //arma::vec w_child = arma::vectorise( (*postparams.w_full).rows(c_ix) );
        //arma::mat Ri_of_child = (*postparams.param_data).w_cond_prec(child);
        
        //arma::mat Kcx_x = Kcx_of_child.cols(pofc_ix_x);
        //arma::mat Kcx_other = Kcx_of_child.cols(pofc_ix_other);
        //arma::mat Kxxi_x = Kxxi_of_child(pofc_ix_x, pofc_ix_x);
        //arma::mat Kxxi_other = Kxxi_of_child(pofc_ix_other, pofc_ix_other);
        //arma::mat Kxxi_xo = Kxxi_of_child(pofc_ix_x, pofc_ix_other);
        
        //Kxo_wo = Kxxi_xo * w_otherparents;
        //Kco_wo = Kcx_other*w_otherparents;
        //woKoowo = w_otherparents.t() * Kxxi_other * w_otherparents;
        
        //Rcpp::Rcout << "grad_bwdcond_dmvt child \n";
        //Rcpp::Rcout << "grad step-child " << c << "\n"; 
        //int num_parents = (*postparams.parents_indexing)(child).n_elem;
        
        grad_logprior_chi += grad_bwdcond_dmvt(x, postparams.w_child(c), postparams.Ri_of_child(c), 
                                               postparams.Kcx_x(c), postparams.Kxxi_x(c),
                                               postparams.Kxo_wo(c), postparams.Kco_wo(c), 
                                               postparams.woKoowo(c), postparams.nu,
                                               postparams.dim_of_pars_of_children(c));
        //Rcpp::Rcout << grad_logprior << "\n";
      }
      
      
      //Rcpp::Rcout << "gradient: \n";
      //Rcpp::Rcout << arma::join_horiz(grad_loglike, grad_logprior) << "\n";
      
      //end = std::chrono::steady_clock::now();
      //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      //Rcpp::Rcout << "grad2 " << glcount << " "
      //            << timing
      //            << "us.\n";
      //glcount ++;
      
      //Rcpp::Rcout << arma::join_horiz(grad_loglike, grad_logprior_par, grad_logprior_chi) <<"\n";
      //Rcpp::Rcout << "LL: " << grad_loglike << " LP: " << grad_logprior_par + grad_logprior_chi << "\n";
      return grad_loglike + grad_logprior_par + grad_logprior_chi;
    }
    
  }
}


// SPECIALIZED

// log posterior 
inline double loglike_cpp(const arma::vec& x, const TDistParams& postparams){
  // likelihood part
  double loglike = 0;
  if(postparams.family == "gaussian"){
    // simple example: normal-normal model
    // argument x here is w in y = Xb + Zw + e where e ~ N(0, tausq)
    // calculate unnormalized log posterior
    if(postparams.which == "w"){
      loglike = 1.0/postparams.tausq * arma::conv_to<double>::from(
        postparams.Zres.t() * x - .5 * x.t() * postparams.ZtZ * x);
      //Rcpp::Rcout << "gaussian loglike: " << loglike << "\n";
      //Rcpp::Rcout << x.t() * postparams.ZtZ * x << "\n";
      //Rcpp::Rcout << postparams.Zres.t() * x << "\n";
    } else {
      //beta
      loglike = 1.0/postparams.tausq * arma::conv_to<double>::from(
        postparams.Zres.t() * x - .5 * x.t() * postparams.XtX * x);
    }
    
  }
  if(postparams.family == "poisson"){
    // poisson
    if(postparams.which == "w"){
      loglike = arma::conv_to<double>::from( postparams.y.t() * postparams.Z * x - 
        postparams.na_1_blocks.t() * exp(postparams.offset + postparams.Z * x) );
    } else {
      //beta
      loglike = arma::conv_to<double>::from( postparams.y.t() * postparams.X * x - 
        postparams.na_1_blocks.t() * exp(postparams.offset + postparams.X * x) );
    }
  }
  if(postparams.family == "binomial"){
    arma::vec sigmoid;
    
    if(postparams.which == "w"){
      // x=w
      sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.Z * x ));
      
    } else {
      // x=beta
      sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.X * x ));
    }
    
    // y and y1 are both zero when missing data
    loglike = arma::conv_to<double>::from( 
      postparams.y.t() * log(sigmoid) + postparams.y1.t() * log(1-sigmoid)
    );
    if(std::isnan(loglike)){
      loglike = -arma::datum::inf;
      //Rcpp::Rcout << "loglike new: " << loglike << "\n";
    }
  }
  
  if(postparams.which != "w"){
    // unless it's the latent spatial process, this is gaussian
    double logprior = arma::conv_to<double>::from(
      x.t() * postparams.mstar - .5 * x.t() * postparams.Vw_i * x);
    return ( loglike + logprior );
  } else {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    double timing = 0;
    
    //Rcpp::Rcout << "> loglike > U: " << postparams.u << "\n";
    
    //Rcpp::Rcout << "entering sauce \n";
    //Rcpp::Rcout << "-- Kxxi \n";
    // prior part : t process prior. here: full conditionals
    //arma::mat Kxxi = (*postparams.param_data).w_cond_prec_parents(postparams.u);
    //Rcpp::Rcout << "-- Kcx \n";
    //arma::mat Kcx = (*postparams.param_data).w_cond_mean_K(postparams.u);
    //Rcpp::Rcout << "-- Ri \n";
    //arma::mat Ri = (*postparams.param_data).w_cond_prec(postparams.u);
    //Rcpp::Rcout << "-- p_ix \n";
    //arma::uvec p_ix = (*postparams.parents_indexing)(postparams.u);
    //Rcpp::Rcout << "-- w_parents \n";
    //arma::vec w_parents = arma::vectorise( (*postparams.w_full).rows(p_ix) );
    
    //Rcpp::Rcout << "----> fwdcond_dmvt \n";
    //start = std::chrono::steady_clock::now();
    
    double logprior = fwdcond_dmvt(x, postparams.Ri, postparams.parKxxpar, 
                                   postparams.Kcxpar, postparams.nu,
                                   postparams.parents_dim);
    
    //end = std::chrono::steady_clock::now();
    //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    //Rcpp::Rcout << "loglike1 " << llcount << " "
    //            << timing
    //            << "us.\n";
    
    //start = std::chrono::steady_clock::now();
    
    //Rcpp::Rcout << "-- children_of_this \n";
    //arma::uvec children_of_this = (*postparams.children)(postparams.u);
    
    for(int c=0; c<postparams.num_children; c++ ){
      //int child = children_of_this(c);
      
      //Rcpp::Rcout << "> loglike > child #" << c << "\n";
      //Rcpp::Rcout << "-- child [" << child << "]\n";
      //Rcpp::Rcout << "-- c_ix \n";
      //arma::uvec c_ix = (*postparams.indexing)(child);
      //Rcpp::Rcout << "-- " << arma::size(c_ix) << " \n";
      //Rcpp::Rcout << "-- pofc_ix \n";
      //arma::uvec pofc_ix = (*postparams.parents_indexing)(child);
      //Rcpp::Rcout << "-- pofc_ix_x \n";
      //arma::uvec pofc_ix_x = (*postparams.u_is_which_col_f_u)(c)(0);
      //Rcpp::Rcout << "-- pofc_ix_other \n";
      //arma::uvec pofc_ix_other = (*postparams.u_is_which_col_f_u)(c)(1);
      //Rcpp::Rcout << "-- w_childs_parents \n";
      //arma::vec w_childs_parents = arma::vectorise( (*postparams.w_full).rows(pofc_ix) );
      //Rcpp::Rcout << "-- w_child \n";
      //arma::vec w_child = arma::vectorise( (*postparams.w_full).rows(c_ix) );
      //Rcpp::Rcout << "-- w_otherparents \n";
      //arma::vec w_otherparents = w_childs_parents.rows(pofc_ix_other);
      //Rcpp::Rcout << "-- Kxxi_of_child \n";
      //arma::mat Kxxi_of_child = (*postparams.param_data).w_cond_prec_parents(child);
      //Rcpp::Rcout << "-- Kcx_of_child \n";
      //arma::mat Kcx_of_child = (*postparams.param_data).w_cond_mean_K(child);
      //Rcpp::Rcout << "-- Ri_of_child \n";
      //arma::mat Ri_of_child = (*postparams.param_data).w_cond_prec(child);
      
      //arma::mat Kcx_x = Kcx_of_child.cols(ix_x);
      //arma::mat Kcx_other = Kcx_of_child.cols(ix_otherparents);
      //arma::mat Kxxi_x = Kxxi_of_child(ix_x, ix_x);
      //arma::mat Kxxi_other = Kxxi_of_child(ix_otherparents, ix_otherparents);
      //arma::mat Kxxi_xo = Kxxi_of_child(ix_x, ix_otherparents);
      
      //Rcpp::Rcout << "----> bwdcond_dmvt \n";
      //int num_parents = (*postparams.parents_indexing)(child).n_elem;
      logprior += bwdcond_dmvt(x, postparams.w_child(c), postparams.Ri_of_child(c), 
                               postparams.Kcx_x(c), postparams.Kxxi_x(c),
                               postparams.Kxo_wo(c), postparams.Kco_wo(c), postparams.woKoowo(c), 
                               postparams.nu, postparams.dim_of_pars_of_children(c));
      
    }
    
    //end = std::chrono::steady_clock::now();
    //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    //Rcpp::Rcout << "loglike2 " << llcount << " "
    //            << timing
    //            << "us.\n";
    //llcount ++;
    //Rcpp::Rcout << "loglike: " << loglike << " logprior: " << logprior << "\n";
    return ( loglike + logprior );
  }
}

// Gradient of the log posterior
inline arma::vec grad_loglike_cpp(const arma::vec& x, const TDistParams& postparams){
  arma::vec grad_loglike(x.n_elem);
  if(postparams.family == "gaussian"){
    if(postparams.which == "w"){
      grad_loglike = 1.0/postparams.tausq * (postparams.Zres - postparams.ZtZ * x);
    } else {
      grad_loglike = 1.0/postparams.tausq * (postparams.Zres - postparams.XtX * x);
    }
  }
  if(postparams.family == "poisson"){
    if(postparams.which == "w"){
      grad_loglike = postparams.Z.t() * (postparams.y - postparams.na_1_blocks % 
        exp(postparams.offset + postparams.Z * x));
    } else {
      grad_loglike = postparams.X.t() * (postparams.y - postparams.na_1_blocks % 
        exp(postparams.offset + postparams.X * x));
    }
  }
  if(postparams.family == "binomial"){
    arma::vec sigmoid;
    if(postparams.which == "w"){
      sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.Z * x));
      grad_loglike = postparams.Z.t() * (postparams.y - postparams.na_1_blocks % sigmoid );
    } else {
      sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.X * x));
      grad_loglike = postparams.X.t() * (postparams.y - postparams.na_1_blocks % sigmoid );
    }
  }
  
  if(postparams.which != "w"){
    arma::vec grad_logprior = postparams.mstar - postparams.Vw_i * x;
    return grad_loglike + grad_logprior;
  } else {
    //Rcpp::Rcout << "> grad > U: " << postparams.u << "\n";
    
    // prior part : t process prior. here: full conditionals
    //arma::mat Kxxi = (*postparams.param_data).w_cond_prec_parents(postparams.u);
    //arma::mat Kcx = (*postparams.param_data).w_cond_mean_K(postparams.u);
    //arma::mat Ri = (*postparams.param_data).w_cond_prec(postparams.u);
    //arma::uvec p_ix = (*postparams.parents_indexing)(postparams.u);
    //arma::vec w_parents = arma::vectorise( (*postparams.w_full).rows(p_ix) );
    
    //Rcpp::Rcout << "grad_fwdcond_dmvt parents \n";
    //start = std::chrono::steady_clock::now();
    
    arma::vec grad_logprior_par = grad_fwdcond_dmvt(x, postparams.Ri, 
                                                    postparams.parKxxpar, postparams.Kcxpar, 
                                                    postparams.nu, postparams.parents_dim);
    
    arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
    
    //end = std::chrono::steady_clock::now();
    //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    //Rcpp::Rcout << "grad1 " << glcount << " "
    //            << timing
    //            << "us.\n";
    
    //start = std::chrono::steady_clock::now();
    
    //Rcpp::Rcout << "grad step-1\n"; 
    //Rcpp::Rcout << grad_logprior << "\n";
    //arma::uvec children_of_this = (*postparams.children)(postparams.u);
    
    
    for(int c=0; c<postparams.num_children; c++ ){
      //Rcpp::Rcout << "> grad > child #" << c << "\n";
      //int child = children_of_this(c);
      //Rcpp::Rcout << "child [" << child << "]\n";
      //arma::uvec c_ix = (*postparams.indexing)(child);
      //arma::uvec pofc_ix = (*postparams.parents_indexing)(child);
      //arma::uvec pofc_ix_x = (*postparams.u_is_which_col_f_u)(c)(0);
      //arma::uvec pofc_ix_other = (*postparams.u_is_which_col_f_u)(c)(1);
      //arma::vec w_childs_parents = arma::vectorise( (*postparams.w_full).rows(pofc_ix) );
      
      //arma::vec w_otherparents = w_childs_parents.rows(pofc_ix_other);
      //arma::mat Kxxi_of_child = (*postparams.param_data).w_cond_prec_parents(child);
      //arma::mat Kcx_of_child = (*postparams.param_data).w_cond_mean_K(child);
      
      //arma::vec w_child = arma::vectorise( (*postparams.w_full).rows(c_ix) );
      //arma::mat Ri_of_child = (*postparams.param_data).w_cond_prec(child);
      
      //arma::mat Kcx_x = Kcx_of_child.cols(pofc_ix_x);
      //arma::mat Kcx_other = Kcx_of_child.cols(pofc_ix_other);
      //arma::mat Kxxi_x = Kxxi_of_child(pofc_ix_x, pofc_ix_x);
      //arma::mat Kxxi_other = Kxxi_of_child(pofc_ix_other, pofc_ix_other);
      //arma::mat Kxxi_xo = Kxxi_of_child(pofc_ix_x, pofc_ix_other);
      
      //Kxo_wo = Kxxi_xo * w_otherparents;
      //Kco_wo = Kcx_other*w_otherparents;
      //woKoowo = w_otherparents.t() * Kxxi_other * w_otherparents;
      
      //Rcpp::Rcout << "grad_bwdcond_dmvt child \n";
      //Rcpp::Rcout << "grad step-child " << c << "\n"; 
      //int num_parents = (*postparams.parents_indexing)(child).n_elem;
      
      grad_logprior_chi += grad_bwdcond_dmvt(x, postparams.w_child(c), postparams.Ri_of_child(c), 
                                             postparams.Kcx_x(c), postparams.Kxxi_x(c),
                                             postparams.Kxo_wo(c), postparams.Kco_wo(c), 
                                             postparams.woKoowo(c), postparams.nu,
                                             postparams.dim_of_pars_of_children(c));
      //Rcpp::Rcout << grad_logprior << "\n";
    }
    
    
    //Rcpp::Rcout << "gradient: \n";
    //Rcpp::Rcout << arma::join_horiz(grad_loglike, grad_logprior) << "\n";
    
    //end = std::chrono::steady_clock::now();
    //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    //Rcpp::Rcout << "grad2 " << glcount << " "
    //            << timing
    //            << "us.\n";
    //glcount ++;
    
    //Rcpp::Rcout << arma::join_horiz(grad_loglike, grad_logprior_par, grad_logprior_chi) <<"\n";
    //Rcpp::Rcout << "LL: " << grad_loglike << " LP: " << grad_logprior_par + grad_logprior_chi << "\n";
    return grad_loglike + grad_logprior_par + grad_logprior_chi;
  }
  
}


// log posterior 
inline double loglike_cpp(const arma::vec& x, const GDistParams& postparams){
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  
  start = std::chrono::steady_clock::now();
  
  // likelihood part
  double loglike = 0;
  if(postparams.family == "gaussian"){
    // simple example: normal-normal model
    // argument x here is w in y = Xb + Zw + e where e ~ N(0, tausq)
    // calculate unnormalized log posterior
    if(postparams.which == "w"){
      loglike = 1.0/postparams.tausq * arma::conv_to<double>::from(
        postparams.Zres.t() * x - .5 * x.t() * postparams.ZtZ * x);
      //Rcpp::Rcout << "gaussian loglike: " << loglike << "\n";
      //Rcpp::Rcout << x.t() * postparams.ZtZ * x << "\n";
      //Rcpp::Rcout << postparams.Zres.t() * x << "\n";
    } else {
      //beta
      loglike = 1.0/postparams.tausq * arma::conv_to<double>::from(
        postparams.Zres.t() * x - .5 * x.t() * postparams.XtX * x);
    }
    
  }
  if(postparams.family == "poisson"){
    // poisson
    if(postparams.which == "w"){
      loglike = arma::conv_to<double>::from( postparams.y.t() * postparams.Z * x - 
        postparams.na_1_blocks.t() * exp(postparams.offset + postparams.Z * x) );
    } else {
      //beta
      loglike = arma::conv_to<double>::from( postparams.y.t() * postparams.X * x - 
        postparams.na_1_blocks.t() * exp(postparams.offset + postparams.X * x) );
    }
  }
  if(postparams.family == "binomial"){
    //arma::vec sigmoid;
    arma::vec XB;
    if(postparams.which == "w"){
      // x=w
      //sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.Z * x ));
      XB = postparams.offset + postparams.Z * x;
    } else {
      // x=beta
      //sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.X * x ));
      XB = postparams.offset + postparams.X * x;
    }
    
    // y and y1 are both zero when missing data
    //loglike = arma::conv_to<double>::from( 
    //  postparams.y.t() * log(sigmoid) + postparams.y1.t() * log(1-sigmoid)
    //);
    
    loglike = arma::conv_to<double>::from(-postparams.ones.t() * log1p(exp(-XB)) - postparams.y1.t() * XB);
    //Rcpp::Rcout << "v " << loglike << "\n";
    
    //Rcpp::Rcout << "loglike at: " << x.t() << "\n";
    //Rcpp::Rcout << "loglike: " << loglike << "\n";
    if(std::isnan(loglike)){
      loglike = -arma::datum::inf;
      //Rcpp::Rcout << "loglike new: " << loglike << "\n";
    }
  }
  
  if(postparams.which != "w"){
    // unless it's the latent spatial process, this is gaussian
    double logprior = arma::conv_to<double>::from(
      x.t() * postparams.mstar - .5 * x.t() * postparams.Vw_i * x);
    
    //end = std::chrono::steady_clock::now();
    //loglike_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    //Rcpp::Rcout << "intermediate: " << loglike + logprior << "\n";
    return ( loglike + logprior );
  } else {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    double timing = 0;
    
    //Rcpp::Rcout << "> loglike > U: " << postparams.u << "\n";
    
    //Rcpp::Rcout << "entering sauce \n";
    //Rcpp::Rcout << "-- Kxxi \n";
    // prior part : t process prior. here: full conditionals
    //arma::mat Kxxi = (*postparams.param_data).w_cond_prec_parents(postparams.u);
    //Rcpp::Rcout << "-- Kcx \n";
    //arma::mat Kcx = (*postparams.param_data).w_cond_mean_K(postparams.u);
    //Rcpp::Rcout << "-- Ri \n";
    //arma::mat Ri = (*postparams.param_data).w_cond_prec(postparams.u);
    //Rcpp::Rcout << "-- p_ix \n";
    //arma::uvec p_ix = (*postparams.parents_indexing)(postparams.u);
    //Rcpp::Rcout << "-- w_parents \n";
    //arma::vec w_parents = arma::vectorise( (*postparams.w_full).rows(p_ix) );
    
    //Rcpp::Rcout << "----> fwdcond_dmvt \n";
    //start = std::chrono::steady_clock::now();
    
    double logprior = fwdcond_dmvn(x, postparams.Ri, postparams.parKxxpar, 
                                   postparams.Kcxpar, 
                                   postparams.parents_dim);
    
    //end = std::chrono::steady_clock::now();
    //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    //Rcpp::Rcout << "loglike1 " << llcount << " "
    //            << timing
    //            << "us.\n";
    
    //start = std::chrono::steady_clock::now();
    
    //Rcpp::Rcout << "-- children_of_this \n";
    //arma::uvec children_of_this = (*postparams.children)(postparams.u);
    
    for(int c=0; c<postparams.num_children; c++ ){
      //int child = children_of_this(c);
      
      //Rcpp::Rcout << "> loglike > child #" << c << "\n";
      //Rcpp::Rcout << "-- child [" << child << "]\n";
      //Rcpp::Rcout << "-- c_ix \n";
      //arma::uvec c_ix = (*postparams.indexing)(child);
      //Rcpp::Rcout << "-- " << arma::size(c_ix) << " \n";
      //Rcpp::Rcout << "-- pofc_ix \n";
      //arma::uvec pofc_ix = (*postparams.parents_indexing)(child);
      //Rcpp::Rcout << "-- pofc_ix_x \n";
      //arma::uvec pofc_ix_x = (*postparams.u_is_which_col_f_u)(c)(0);
      //Rcpp::Rcout << "-- pofc_ix_other \n";
      //arma::uvec pofc_ix_other = (*postparams.u_is_which_col_f_u)(c)(1);
      //Rcpp::Rcout << "-- w_childs_parents \n";
      //arma::vec w_childs_parents = arma::vectorise( (*postparams.w_full).rows(pofc_ix) );
      //Rcpp::Rcout << "-- w_child \n";
      //arma::vec w_child = arma::vectorise( (*postparams.w_full).rows(c_ix) );
      //Rcpp::Rcout << "-- w_otherparents \n";
      //arma::vec w_otherparents = w_childs_parents.rows(pofc_ix_other);
      //Rcpp::Rcout << "-- Kxxi_of_child \n";
      //arma::mat Kxxi_of_child = (*postparams.param_data).w_cond_prec_parents(child);
      //Rcpp::Rcout << "-- Kcx_of_child \n";
      //arma::mat Kcx_of_child = (*postparams.param_data).w_cond_mean_K(child);
      //Rcpp::Rcout << "-- Ri_of_child \n";
      //arma::mat Ri_of_child = (*postparams.param_data).w_cond_prec(child);
      
      //arma::mat Kcx_x = Kcx_of_child.cols(ix_x);
      //arma::mat Kcx_other = Kcx_of_child.cols(ix_otherparents);
      //arma::mat Kxxi_x = Kxxi_of_child(ix_x, ix_x);
      //arma::mat Kxxi_other = Kxxi_of_child(ix_otherparents, ix_otherparents);
      //arma::mat Kxxi_xo = Kxxi_of_child(ix_x, ix_otherparents);
      
      //Rcpp::Rcout << "----> bwdcond_dmvt \n";
      //int num_parents = (*postparams.parents_indexing)(child).n_elem;
      logprior += bwdcond_dmvn(x, postparams.w_child(c), postparams.Ri_of_child(c), 
                               postparams.Kcx_x(c), postparams.Kxxi_x(c),
                               postparams.Kxo_wo(c), postparams.Kco_wo(c), postparams.woKoowo(c), 
                               postparams.dim_of_pars_of_children(c));
      
    }
    
    //end = std::chrono::steady_clock::now();
    //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    //Rcpp::Rcout << "loglike2 " << llcount << " "
    //            << timing
    //            << "us.\n";
    //llcount ++;
    //Rcpp::Rcout << "loglike: " << loglike << " logprior: " << logprior << "\n";
    return ( loglike + logprior );
  }
}

// Gradient of the log posterior
inline arma::vec grad_loglike_cpp(const arma::vec& x, const GDistParams& postparams){
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  
  start = std::chrono::steady_clock::now();
  
  arma::vec grad_loglike(x.n_elem);
  if(postparams.family == "gaussian"){
    if(postparams.which == "w"){
      grad_loglike = 1.0/postparams.tausq * (postparams.Zres - postparams.ZtZ * x);
    } else {
      grad_loglike = 1.0/postparams.tausq * (postparams.Zres - postparams.XtX * x);
    }
  }
  if(postparams.family == "poisson"){
    if(postparams.which == "w"){
      grad_loglike = postparams.Z.t() * (postparams.y - postparams.na_1_blocks % 
        exp(postparams.offset + postparams.Z * x));
    } else {
      grad_loglike = postparams.X.t() * (postparams.y - postparams.na_1_blocks % 
        exp(postparams.offset + postparams.X * x));
    }
  }
  if(postparams.family == "binomial"){
    arma::vec sigmoid;
    if(postparams.which == "w"){
      sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.Z * x));
      grad_loglike = postparams.Z.t() * (postparams.y - postparams.na_1_blocks % sigmoid );
    } else {
      sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.X * x));
      grad_loglike = postparams.X.t() * (postparams.y - postparams.na_1_blocks % sigmoid );
    }
    //Rcpp::Rcout << "gradient at: " << x.t() << "\n";
    //Rcpp::Rcout << "grad_loglike: " << grad_loglike.t() << "\n";
  }
  
  if(postparams.which != "w"){
    arma::vec grad_logprior = postparams.mstar - postparams.Vw_i * x;
    
    //end = std::chrono::steady_clock::now();
    //grad_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    return grad_loglike + grad_logprior;
  } else {
    //Rcpp::Rcout << "> grad > U: " << postparams.u << "\n";
    
    // prior part : t process prior. here: full conditionals
    //arma::mat Kxxi = (*postparams.param_data).w_cond_prec_parents(postparams.u);
    //arma::mat Kcx = (*postparams.param_data).w_cond_mean_K(postparams.u);
    //arma::mat Ri = (*postparams.param_data).w_cond_prec(postparams.u);
    //arma::uvec p_ix = (*postparams.parents_indexing)(postparams.u);
    //arma::vec w_parents = arma::vectorise( (*postparams.w_full).rows(p_ix) );
    
    //Rcpp::Rcout << "grad_fwdcond_dmvt parents \n";
    //start = std::chrono::steady_clock::now();
    
    arma::vec grad_logprior_par = grad_fwdcond_dmvn(x, postparams.Ri, 
                                                    postparams.parKxxpar, postparams.Kcxpar, 
                                                    postparams.parents_dim);
    
    arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
    
    //end = std::chrono::steady_clock::now();
    //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    //Rcpp::Rcout << "grad1 " << glcount << " "
    //            << timing
    //            << "us.\n";
    
    //start = std::chrono::steady_clock::now();
    
    //Rcpp::Rcout << "grad step-1\n"; 
    //Rcpp::Rcout << grad_logprior << "\n";
    //arma::uvec children_of_this = (*postparams.children)(postparams.u);
    
    
    for(int c=0; c<postparams.num_children; c++ ){
      //Rcpp::Rcout << "> grad > child #" << c << "\n";
      //int child = children_of_this(c);
      //Rcpp::Rcout << "child [" << child << "]\n";
      //arma::uvec c_ix = (*postparams.indexing)(child);
      //arma::uvec pofc_ix = (*postparams.parents_indexing)(child);
      //arma::uvec pofc_ix_x = (*postparams.u_is_which_col_f_u)(c)(0);
      //arma::uvec pofc_ix_other = (*postparams.u_is_which_col_f_u)(c)(1);
      //arma::vec w_childs_parents = arma::vectorise( (*postparams.w_full).rows(pofc_ix) );
      
      //arma::vec w_otherparents = w_childs_parents.rows(pofc_ix_other);
      //arma::mat Kxxi_of_child = (*postparams.param_data).w_cond_prec_parents(child);
      //arma::mat Kcx_of_child = (*postparams.param_data).w_cond_mean_K(child);
      
      //arma::vec w_child = arma::vectorise( (*postparams.w_full).rows(c_ix) );
      //arma::mat Ri_of_child = (*postparams.param_data).w_cond_prec(child);
      
      //arma::mat Kcx_x = Kcx_of_child.cols(pofc_ix_x);
      //arma::mat Kcx_other = Kcx_of_child.cols(pofc_ix_other);
      //arma::mat Kxxi_x = Kxxi_of_child(pofc_ix_x, pofc_ix_x);
      //arma::mat Kxxi_other = Kxxi_of_child(pofc_ix_other, pofc_ix_other);
      //arma::mat Kxxi_xo = Kxxi_of_child(pofc_ix_x, pofc_ix_other);
      
      //Kxo_wo = Kxxi_xo * w_otherparents;
      //Kco_wo = Kcx_other*w_otherparents;
      //woKoowo = w_otherparents.t() * Kxxi_other * w_otherparents;
      
      //Rcpp::Rcout << "grad_bwdcond_dmvt child \n";
      //Rcpp::Rcout << "grad step-child " << c << "\n"; 
      //int num_parents = (*postparams.parents_indexing)(child).n_elem;
      
      grad_logprior_chi += grad_bwdcond_dmvn(x, postparams.w_child(c), postparams.Ri_of_child(c), 
                                             postparams.Kcx_x(c), postparams.Kxxi_x(c),
                                             postparams.Kxo_wo(c), postparams.Kco_wo(c), 
                                             postparams.woKoowo(c), 
                                             postparams.dim_of_pars_of_children(c));
      //Rcpp::Rcout << grad_logprior << "\n";
    }
    
    
    //Rcpp::Rcout << "gradient: \n";
    //Rcpp::Rcout << arma::join_horiz(grad_loglike, grad_logprior) << "\n";
    
    //end = std::chrono::steady_clock::now();
    //timing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    //Rcpp::Rcout << "grad2 " << glcount << " "
    //            << timing
    //            << "us.\n";
    //glcount ++;
    
    //Rcpp::Rcout << arma::join_horiz(grad_loglike, grad_logprior_par, grad_logprior_chi) <<"\n";
    //Rcpp::Rcout << "LL: " << grad_loglike << " LP: " << grad_logprior_par + grad_logprior_chi << "\n";
    return grad_loglike + grad_logprior_par + grad_logprior_chi;
  }
  
}


/*
 inline double loglike_cpp(const arma::vec& x, const GDistParams& postparams){
 double loglike = 0;
 if(postparams.family == "gaussian"){
 
 // simple example: normal-normal model
 // argument x here is w in y = Xb + Zw + e
 // calculate unnormalized log posterior
 if(postparams.which == "w"){
 loglike = 1.0/postparams.tausq * arma::conv_to<double>::from(
 postparams.Zres.t() * x - .5 * x.t() * postparams.ZtZ * x);
 } else {
 //beta
 loglike = 1.0/postparams.tausq * arma::conv_to<double>::from(
 postparams.Zres.t() * x - .5 * x.t() * postparams.XtX * x);
 }
 
 }
 if(postparams.family == "poisson"){
 // poisson
 if(postparams.which == "w"){
 loglike = arma::conv_to<double>::from( postparams.y.t() * postparams.Z * x - 
 postparams.na_1_blocks.t() * exp(postparams.offset + postparams.Z * x) );
 } else {
 //beta
 loglike = arma::conv_to<double>::from( postparams.y.t() * postparams.X * x - 
 postparams.na_1_blocks.t() * exp(postparams.offset + postparams.X * x) );
 }
 }
 if(postparams.family == "binomial"){
 arma::vec sigmoid;
 
 if(postparams.which == "w"){
 // x=w
 sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.Z * x ));
 
 } else {
 // x=beta
 sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.X * x ));
 }
 
 // y and y1 are both zero when missing data
 loglike = arma::conv_to<double>::from( 
 postparams.y.t() * log(sigmoid) + postparams.y1.t() * log(1-sigmoid)
 );
 }
 
 double logprior = arma::conv_to<double>::from(
 x.t() * postparams.mstar - .5 * x.t() * postparams.Vw_i * x);
 
 return ( loglike + logprior );
 }
 
 // Gradient of the log posterior
 inline arma::vec grad_loglike_cpp(const arma::vec& x, const GDistParams& postparams){
 //glcount ++;
 
 //1/sigma.sq * ( -crossprod(Z, y - X %*% beta) +crossprod(Z) %*% ww ) + w_prior_prec %*% ww
 arma::vec grad_loglike(x.n_elem);
 if(postparams.family == "gaussian"){
 if(postparams.which == "w"){
 //Rcpp::Rcout << "ok" << endl;
 grad_loglike = 1.0/postparams.tausq * (postparams.Zres - postparams.ZtZ * x);
 } else {
 grad_loglike = 1.0/postparams.tausq * (postparams.Zres - postparams.XtX * x);
 }
 }
 if(postparams.family == "poisson"){
 if(postparams.which == "w"){
 grad_loglike = postparams.Z.t() * (postparams.y - postparams.na_1_blocks % 
 exp(postparams.offset + postparams.Z * x));
 } else {
 grad_loglike = postparams.X.t() * (postparams.y - postparams.na_1_blocks % 
 exp(postparams.offset + postparams.X * x));
 }
 }
 if(postparams.family == "binomial"){
 arma::vec sigmoid;
 if(postparams.which == "w"){
 sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.Z * x));
 grad_loglike = postparams.Z.t() * (postparams.y - postparams.na_1_blocks % sigmoid );
 } else {
 sigmoid = 1.0/(1.0 + exp(-postparams.offset - postparams.X * x));
 grad_loglike = postparams.X.t() * (postparams.y - postparams.na_1_blocks % sigmoid );
 }
 }
 
 arma::vec grad_prior = postparams.mstar - postparams.Vw_i * x;
 //Rcpp::Rcout << "LL: " << grad_loglike << " LP: " << grad_prior << "\n";
 return grad_loglike + grad_prior;
 }
 */
