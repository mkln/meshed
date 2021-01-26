#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>
//#include "tmesh_utils.h"
#include "../distributions/densities_gradients.h"

class MVDistParams {
public:
  // common stuff
  // latent process type
  std::string latent;
  arma::mat y; // output data
  
  arma::mat offset; // offset for this update
  int n;
  
  double logfullcondit(const arma::vec& x);
  arma::vec gradient_logfullcondit(const arma::vec& x);
  
  MVDistParams();
  //MVDistParams(const arma::vec& y, const arma::vec& offset, const arma::mat& X, std::string, std::string);
  //MVDistParams(const arma::vec& y, const arma::vec& offset, const arma::uvec& outtype, std::string);
  
};

inline MVDistParams::MVDistParams(){
  n=-1;
}

inline double MVDistParams::logfullcondit(const arma::vec& x){
  return 0;
}

inline arma::vec MVDistParams::gradient_logfullcondit(const arma::vec& x){
  return 0;
}

class MVDistParamsW : public MVDistParams {
public:
  arma::uvec family; // 0: gaussian, 1: poisson, 2: bernoulli, length q
  int k;
  arma::vec z;
  
  arma::mat Lambda_lmc;
  //arma::mat LambdaH;
  
  arma::umat na_mat;
  arma::vec tausq;
  
  //arma::vec na_1_blocks; // indicator vector by block
  int block_ct_obs; // number of not-na
  
  /// added
  arma::vec nu; // student
  
  arma::uvec indexing_target;
  
  bool fgrid;
  
  arma::cube Kxxi;
  arma::cube Kcx;
  arma::cube Ri;
  arma::cube Hproject;
  
  arma::vec parKxxpar;
  arma::mat Kcxpar;
  arma::mat w_parents;
  
  int num_children;
  double parents_dim;
  arma::vec dim_of_pars_of_children;
  
  arma::field<arma::cube> Kcx_x;//(c) = (*param_data).w_cond_mean_K(child).cols(pofc_ix_x);
  arma::field<arma::cube> Kxxi_x;//(c) = (*param_data).w_cond_prec_parents(child)(pofc_ix_x, pofc_ix_x);
  arma::field<arma::mat> w_child;//(c) = arma::vectorise( (*w_full).rows(c_ix) );
  arma::field<arma::cube> Ri_of_child;//(c) = (*param_data).w_cond_prec(child);
  arma::field<arma::mat> Kxo_wo;//(c) = Kxxi_xo * w_otherparents;
  arma::field<arma::mat> Kco_wo;//(c) = Kcx_other * w_otherparents;
  arma::mat woKoowo;//(c) = w_otherparents.t() * Kxxi_other * w_otherparents;
  
  
  void initialize();
  void update_mv(const arma::mat& new_offset, const arma::vec& tausq, const arma::mat& Lambda_lmc_in);
  double logfullcondit(const arma::mat& x);
  arma::vec gradient_logfullcondit(const arma::mat& x);
  arma::mat neghess_logfullcondit(const arma::mat& x);
  
  MVDistParamsW(const arma::mat& y_all, //const arma::mat& Z_in,
                const arma::umat& na_mat_all, const arma::mat& offset_all, 
                const arma::uvec& indexing_target,
                const arma::uvec& outtype, int k, std::string latent_in, 
                bool fgrid_in);
  MVDistParamsW();
  
  //using MVDistParams::logfullcondit;
  //using MVDistParams::gradient_logfullcondit;
};


inline MVDistParamsW::MVDistParamsW(){
  n=-1;
}

inline MVDistParamsW::MVDistParamsW(const arma::mat& y_all, //const arma::mat& Z_in,
                                    const arma::umat& na_mat_all, const arma::mat& offset_all, 
                                    const arma::uvec& indexing_target_in,
                                    const arma::uvec& outtype, int k, std::string latent_in, 
                                    bool fgrid_in){
  
  indexing_target = indexing_target_in;
  y = y_all.rows(indexing_target);
  offset = offset_all.rows(indexing_target);
  na_mat = na_mat_all.rows(indexing_target);
  
  // ----
  
  family = outtype; //= arma::vectorise(familymat);
  
  nu = arma::ones(k) * 2;
  latent = latent_in;
  //which = "w";
  
  n = y.n_rows;
  z = arma::ones(n); //Z_in.col(0);
  
  fgrid = fgrid_in;
}

inline void MVDistParamsW::initialize(){
  
}

inline void MVDistParamsW::update_mv(const arma::mat& offset_all, const arma::vec& tausq_in,
                                     const arma::mat& Lambda_lmc_in){
  // arma::mat tausqmat = arma::zeros<arma::umat>(arma::size(new_offset));
  // for(int i=0; i<tausqmat.n_cols; i++){
  //   tausqmat.col(i).fill(tausq(i));
  // }
  // tausq_long = arma::vectorise(tausqmat);
  Lambda_lmc = Lambda_lmc_in;
  
  
  tausq = tausq_in;
  offset = offset_all.rows(indexing_target);
}

// log posterior 
inline double MVDistParamsW::logfullcondit(const arma::mat& x){
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  start = std::chrono::steady_clock::now();
  
  //Rcpp::Rcout << "logfullcondit 0" << endl;
  
  double loglike = 0;
  //Rcpp::Rcout << arma::size(y) << " " << arma::size(offset) << " " << arma::size(xstar) << " " << arma::size(tausq) << endl;
  for(int i=0; i<y.n_rows; i++){
    arma::mat wloc;
    if(fgrid){
      wloc = arma::sum(arma::trans(Hproject.slice(i) % arma::trans(x)), 0);
    } else {
      wloc = x.row(i);
    }
    for(int j=0; j<y.n_cols; j++){
      //Rcpp::Rcout << i << " - " << j << endl;
      if(na_mat(i, j) > 0){
        double xstar = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        //double xz = x(i) * z(i);
        if(family(j) == 0){ //if(family == "gaussian"){
          double y_minus_mean = y(i, j) - offset(i, j) - xstar;
          loglike += gaussian_logdensity(y_minus_mean, tausq(j));
        } else {
          if(family(j) == 1){ //if(family=="poisson"){
            double lambda = exp(offset(i, j) + xstar);//xz);//x(i));
            loglike += poisson_logpmf(y(i, j), lambda);
          } else {
            if(family(j) == 2){ //if(family=="binomial"){
              double sigmoid = 1.0/(1.0 + exp(-offset(i, j) - xstar));//xz ));
              loglike += bernoulli_logpmf(y(i, j), sigmoid);
            }
          }
        } 
      }
    }
  }
  
  //Rcpp::Rcout << "logfullcondit 1" << endl;
  
  // prior
  if(latent == "gaussian"){
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    double timing = 0;
    
    double logprior = fwdcond_dmvn(x, Ri, parKxxpar, Kcxpar);
    
    
    for(int c=0; c<num_children; c++ ){
      
      logprior += bwdcond_dmvn(x, w_child(c), Ri_of_child(c), 
                               Kcx_x(c), Kxxi_x(c),
                               Kxo_wo(c), Kco_wo(c), woKoowo.col(c), 
                               dim_of_pars_of_children(c));
      
    }
    return ( loglike + logprior );
    
    
    
  }
  
  if(latent == "student"){
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    double timing = 0;
    
    double logprior = fwdcond_dmvt(x, Ri, parKxxpar, 
                                   Kcxpar, nu,
                                   parents_dim);
    
    for(int c=0; c<num_children; c++ ){
      logprior += bwdcond_dmvt(x, w_child(c), Ri_of_child(c), 
                               Kcx_x(c), Kxxi_x(c),
                               Kxo_wo(c), Kco_wo(c), woKoowo.col(c), 
                               nu, dim_of_pars_of_children(c));
      
    }
    
    return ( loglike + logprior );
    
  }
}

// Gradient of the log posterior
inline arma::vec MVDistParamsW::gradient_logfullcondit(const arma::mat& x){
  
  int q = y.n_cols;
  int k = x.n_cols;
  
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  start = std::chrono::steady_clock::now();
  
  arma::vec grad_loglike = arma::zeros(x.n_rows * x.n_cols);
  
  int nr = y.n_rows;
  int indxsize = x.n_rows;
  
  //Rcpp::Rcout << "gradient_logfullcondit 0" << endl;
  
  arma::mat xstar = x * Lambda_lmc.t();
  
  if(fgrid){
    
    for(int i=0; i<nr; i++){
      arma::mat wloc = arma::sum(arma::trans(Hproject.slice(i) % arma::trans(x)), 0);
      arma::mat LambdaH = arma::zeros(q, k*indxsize);  
      for(int j=0; j<q; j++){
        if(na_mat(i, j) == 1){
          arma::mat Hloc = Hproject.slice(i);
          for(int jx=0; jx<k; jx++){
            arma::mat Hsub = Hloc.row(jx); //data.Hproject(u).subcube(jx,0,ix,jx,indxsize-1, ix);
            // this outcome margin observed at this location
            LambdaH.submat(j, jx*indxsize, j, (jx+1)*indxsize-1) += Lambda_lmc(j, jx) * Hsub;
          }
        }
      }
        
      //Rcpp::Rcout << "part 1 " << i << endl;
      
      for(int j=0; j<y.n_cols; j++){
        if(na_mat(i, j) > 0){
          arma::vec gradloc;
          double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
          arma::mat LambdaHt = LambdaH.row(j).t();
          
          if(family(j) == 0){  // family=="gaussian"
            double y_minus_mean = y(i, j) - offset(i, j) - xij;
            gradloc = LambdaHt * gaussian_loggradient(y_minus_mean, tausq(j));
            grad_loglike += gradloc; 
          } else if(family(j) == 1){ //if(family == "poisson"){
            grad_loglike += LambdaHt * poisson_loggradient(y(i, j), offset(i, j), xij); //xz) * z(i);
          } else if(family(j) == 2){ //if(family == "binomial"){
            grad_loglike += LambdaHt * bernoulli_loggradient(y(i, j), offset(i, j), xij); //xz) * z(i);
          }
        }
      }
      //Rcpp::Rcout << "part 2 " << i << endl;
    }
    
  } else {
    //Rcpp::Rcout << "STEP Y " << endl;
    //arma::mat LambdaH;
    
    for(int i=0; i<nr; i++){
      arma::mat wloc = x.row(i);
      
      for(int j=0; j<y.n_cols; j++){
        if(na_mat(i, j) > 0){
          arma::vec gradloc = arma::zeros(k);
          arma::mat LambdaHt = Lambda_lmc.row(j).t();
          double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
          if(family(j) == 0){  // family=="gaussian"
            double y_minus_mean = y(i, j) - offset(i, j) - xij;
            gradloc = LambdaHt * gaussian_loggradient(y_minus_mean, tausq(j));
          } else if(family(j) == 1){ //if(family == "poisson"){
            //grad_loglike = Z.t() * (y - na_vec % 
            //  exp(offset + Z * x));
            gradloc = LambdaHt * poisson_loggradient(y(i, j), offset(i, j), xij); //xz) * z(i);
          } else if(family(j) == 2){ //if(family == "binomial"){
            gradloc = LambdaHt * bernoulli_loggradient(y(i, j), offset(i, j), xij); //xz) * z(i);
          }
          
          for(int s=0; s<k; s++){
            grad_loglike(s * indxsize + i) += gradloc(s);   
          }
        } 
      }
    }
  }
  

  if(latent == "gaussian"){
    //Rcpp::Rcout << "grad 1 " << endl;
    arma::vec grad_logprior_par = grad_fwdcond_dmvn(x, Ri, parKxxpar, Kcxpar);
    
    arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
    for(int c=0; c<num_children; c++ ){
      grad_logprior_chi += grad_bwdcond_dmvn(x, w_child(c), Ri_of_child(c), 
                                             Kcx_x(c), Kxxi_x(c),
                                             Kxo_wo(c), Kco_wo(c), 
                                             woKoowo.col(c), 
                                             dim_of_pars_of_children(c));
    }
    
    return grad_loglike + 
      grad_logprior_par + 
      grad_logprior_chi;
  }
  
  if(latent == "student"){
    arma::vec grad_logprior_par = grad_fwdcond_dmvt(x, Ri, 
                                                    parKxxpar, Kcxpar, 
                                                    nu, parents_dim);
    
    
    
    arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
    for(int c=0; c<num_children; c++ ){
      grad_logprior_chi += grad_bwdcond_dmvt(x, w_child(c), Ri_of_child(c), 
                                             Kcx_x(c), Kxxi_x(c),
                                             Kxo_wo(c), Kco_wo(c), 
                                             woKoowo.col(c), nu,
                                             dim_of_pars_of_children(c));
      //Rcpp::Rcout << grad_logprior << "\n";
    }
    
    
    
    return grad_loglike + 
      grad_logprior_par + 
      grad_logprior_chi;
  }
}

// Gradient of the log posterior
inline arma::mat MVDistParamsW::neghess_logfullcondit(const arma::mat& x){
  
  int q = y.n_cols;
  int k = x.n_cols;
  
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  start = std::chrono::steady_clock::now();
  
  if(latent == "student"){
    return arma::eye(x.n_rows * x.n_cols,
                       x.n_rows * x.n_cols);
  }
  
  arma::mat neghess_loglike = arma::zeros(x.n_rows * x.n_cols,
                                  x.n_rows * x.n_cols);
  
  int nr = y.n_rows;
  int indxsize = x.n_rows;
  
  //Rcpp::Rcout << "gradient_logfullcondit 0" << endl;
  
  arma::mat xstar = x * Lambda_lmc.t();
  
  if(fgrid){
    for(int i=0; i<nr; i++){
      arma::mat LambdaH = arma::zeros(q, k*indxsize);  
      for(int j=0; j<q; j++){
        if(na_mat(i, j) == 1){
          arma::mat Hloc = Hproject.slice(i);
          for(int jx=0; jx<k; jx++){
            arma::mat Hsub = Hloc.row(jx); //data.Hproject(u).subcube(jx,0,ix,jx,indxsize-1, ix);
            // this outcome margin observed at this location
            LambdaH.submat(j, jx*indxsize, j, (jx+1)*indxsize-1) += Lambda_lmc(j, jx) * Hsub;
          }
        }
      }
      
      //Rcpp::Rcout << "part 1 " << i << endl;
      
      for(int j=0; j<y.n_cols; j++){
        if(na_mat(i, j) > 0){
          if(family(j) == 0){  // family=="gaussian"
            arma::mat LambdaHt = LambdaH.row(j).t();
            arma::mat neghessloc = LambdaHt * LambdaHt.t() / tausq(j);
            neghess_loglike += neghessloc;
          } 
        }
      }
      //Rcpp::Rcout << "part 2 " << i << endl;
    }
    
  } else {
    
    for(int i=0; i<nr; i++){
      arma::mat wloc = x.row(i);
      for(int j=0; j<y.n_cols; j++){
        arma::mat LambdaHt = Lambda_lmc.row(j).t();
        if(na_mat(i, j) > 0){
          double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
          arma::mat neghessloc;
          if(family(j) == 0){  // family=="gaussian"
            neghessloc = LambdaHt * LambdaHt.t() / tausq(j);
          } else if(family(j) == 1){
            neghessloc = LambdaHt * LambdaHt.t() * exp(xij);
          } else if(family(j) == 2){
            double exij = exp(-xij);
            double opexij = (1.0 + exij);
            neghessloc = LambdaHt * LambdaHt.t() * exij / (opexij*opexij); 
          }
          
          for(int s1=0; s1<k; s1++){
            for(int s2=0; s2<k; s2++){
              neghess_loglike(s1 * indxsize + i, s2*indxsize + i) += neghessloc(s1, s2);
            }
          }
        }
      }
    }
    
  }
  
  if(latent == "gaussian"){
    arma::mat neghess_logprior_par = neghess_fwdcond_dmvn(x, Ri, parKxxpar, Kcxpar);
    
    arma::mat neghess_logprior_chi = arma::zeros(arma::size(neghess_logprior_par));
    for(int c=0; c<num_children; c++ ){
      neghess_logprior_chi += neghess_bwdcond_dmvn(x, w_child(c), Ri_of_child(c), 
                                             Kcx_x(c), Kxxi_x(c),
                                             Kxo_wo(c), Kco_wo(c), 
                                             woKoowo.col(c), 
                                             dim_of_pars_of_children(c));
      
    }
    
    return neghess_loglike + 
      neghess_logprior_par + 
      neghess_logprior_chi;
  }
  
  if(latent == "student"){
    arma::mat neghess_logprior_par = neghess_fwdcond_dmvt(x, Ri, parKxxpar, Kcxpar, nu, parents_dim);
    
    arma::mat neghess_logprior_chi = arma::zeros(arma::size(neghess_logprior_par));
    for(int c=0; c<num_children; c++ ){
      neghess_logprior_chi += neghess_bwdcond_dmvt(x, w_child(c), Ri_of_child(c),
                                                   Kcx_x(c), Kxxi_x(c),
                                                   Kxo_wo(c), Kco_wo(c),
                                                   woKoowo.col(c), nu,
                                                   dim_of_pars_of_children(c));

    }
    
    return neghess_loglike + 
      neghess_logprior_par + 
      neghess_logprior_chi;
  }
}


class MVDistParamsBeta : public MVDistParams {
public:
  int family; // for beta
  
  arma::mat X; //for updates of beta
  double tausq; // reg variance
  
  // gaussian
  arma::mat XtX;
  arma::vec Xres;
  
  // binom
  arma::vec ones;
  arma::vec y1; // 1-y
  
  // for beta updates in non-Gaussian y models
  arma::vec mstar;
  arma::mat Vw_i;
  
  // mass matrix
  arma::mat Sig;
  arma::mat Sig_i_tchol;
  arma::mat M;
  arma::mat Michol;
  
  void initialize();
  void update_mv(const arma::vec& new_offset, 
                 const double& tausq_in, const arma::vec& Smu_tot, const arma::mat& Sigi_tot);
  MVDistParamsBeta(const arma::vec& y_in, const arma::vec& offset_in, 
                   const arma::mat& X_in, int family_in, std::string latent_in);
  MVDistParamsBeta();
  
  double logfullcondit(const arma::vec& x);
  arma::vec gradient_logfullcondit(const arma::vec& x);
  arma::mat neghess_logfullcondit(const arma::vec& x);
  
  //using MVDistParams::MVDistParams;
  //using MVDistParams::logfullcondit;
  //using MVDistParams::gradient_logfullcondit;
};


inline MVDistParamsBeta::MVDistParamsBeta(){
  n=-1;
}

inline MVDistParamsBeta::MVDistParamsBeta(const arma::vec& y_in, const arma::vec& offset_in, 
                                          const arma::mat& X_in, int family_in, std::string latent_in){
  family = family_in;
  n = y_in.n_elem;
  y = y_in;
  offset = offset_in;
  X = X_in;
  //which = "beta";
  
  latent = latent_in;
  
  if(family != 0){ // gaussian
    ones = arma::ones(n);
  }
  
  if(family == 2){ // binomial
    y1 = 1-y;
  }
  
  initialize();
}

inline void MVDistParamsBeta::initialize(){
  mstar = arma::zeros(X.n_cols);
  Vw_i = arma::eye(X.n_cols, X.n_cols);
  
  if(family == 0){
    XtX = X.t() * X;
    Sig = arma::inv_sympd(Vw_i + XtX); //
    Sig_i_tchol = arma::trans( arma::inv(arma::trimatl(arma::chol(Sig, "lower"))) );
    M = arma::eye(n, n);//M = Sig;
    Michol = M;//Sig_i_tchol;
  } 
}

inline void MVDistParamsBeta::update_mv(const arma::vec& new_offset, const double& tausq_in, const arma::vec& Smu_tot, const arma::mat& Sigi_tot){
  tausq = tausq_in;
  offset = new_offset;
  
  mstar = Smu_tot;
  Vw_i = Sigi_tot;
  
  if(family == 0){
    Xres = X.t() * (y - offset);
    M = tausq * Sig;
    Michol = pow(tausq, -.5) * Sig_i_tchol;
  }
}

// log posterior 
inline double MVDistParamsBeta::logfullcondit(const arma::vec& x){
  //Rcpp::Rcout << "lik " << endl;
  //std::chrono::steady_clock::time_point start;
  //std::chrono::steady_clock::time_point end;
  //start = std::chrono::steady_clock::now();
  
  double loglike = 0;
  
  
  
  if(family==1){
    loglike = arma::conv_to<double>::from( y.t() * X * x - ones.t() * exp(offset + X * x) );
  } else if (family==2) {
    // x=beta
    arma::vec sigmoid = 1.0/(1.0 + exp(-offset - X * x ));
    // y and y1 are both zero when missing data
    loglike = arma::conv_to<double>::from( 
      y.t() * log(sigmoid) + y1.t() * log(1-sigmoid)
    );
    if(std::isnan(loglike)){
      loglike = -arma::datum::inf;
      //Rcpp::Rcout << "loglike new: " << loglike << "\n";
    }
  } else if(family == 0){
    loglike = 1.0/tausq * arma::conv_to<double>::from(
        Xres.t() * x - .5 * x.t() * XtX * x);
  }
  
  
  /*
  if(family == "binomial2"){
    //arma::vec sigmoid;
    arma::vec XB = offset + X * x;
  
    loglike = arma::conv_to<double>::from(-ones.t() * log1p(exp(-XB)) - y1.t() * XB);
    //Rcpp::Rcout << "v " << loglike << "\n";
    
    //Rcpp::Rcout << "loglike at: " << x.t() << "\n";
    //Rcpp::Rcout << "loglike: " << loglike << "\n";
    if(std::isnan(loglike)){
      loglike = -arma::datum::inf;
      //Rcpp::Rcout << "loglike new: " << loglike << "\n";
    }
  }*/
  
  double logprior = arma::conv_to<double>::from(
    x.t() * mstar - .5 * x.t() * Vw_i * x);
  
  return ( loglike + logprior );
  
}

// Gradient of the log posterior
inline arma::vec MVDistParamsBeta::gradient_logfullcondit(const arma::vec& x){
  //Rcpp::Rcout << "grad " << endl;
  //std::chrono::steady_clock::time_point start;
  //std::chrono::steady_clock::time_point end;
  //start = std::chrono::steady_clock::now();
  
  arma::vec grad_loglike = arma::zeros(x.n_elem);
  
  if(family==0){
    grad_loglike = 1.0/tausq * (Xres - XtX * x);
  } else if(family == 1){
    grad_loglike = X.t() * (y - //na_vec % 
      exp(offset + X * x));
  } else if(family == 2){
    arma::vec sigmoid = 1.0/(1.0 + exp(-offset - X * x));
    grad_loglike = X.t() * (y - //na_vec % 
      sigmoid );
  }
    
  
  arma::vec grad_logprior = mstar - Vw_i * x;
  //end = std::chrono::steady_clock::now();
  //grad_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return grad_loglike + grad_logprior;

}


inline arma::mat MVDistParamsBeta::neghess_logfullcondit(const arma::vec& x){
  return arma::eye(x.n_elem, x.n_elem);
}