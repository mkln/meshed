#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>
//#include "tmesh_utils.h"
#include "../distributions/densities_gradients.h"

class NodeData {
public:
  // common stuff
  // latent process type
  std::string latent;
  arma::mat y; // output data
  
  arma::mat ystar; // for binomial and beta outcomes
  
  arma::mat offset; // offset for this update
  int n;
  
  double logfullcondit(const arma::vec& x);
  arma::vec gradient_logfullcondit(const arma::vec& x);
  
  NodeData();
  
};

inline NodeData::NodeData(){
  n=-1;
}

inline double NodeData::logfullcondit(const arma::vec& x){
  return 0;
}

inline arma::vec NodeData::gradient_logfullcondit(const arma::vec& x){
  return 0;
}

class NodeDataW : public NodeData {
public:
  arma::uvec family; // 0: gaussian, 1: poisson, 2: bernoulli, 3: beta, length q
  int k;
  arma::vec z;
  
  arma::mat Lambda_lmc;
  
  arma::umat na_mat;
  arma::vec tausq;
  
  int block_ct_obs; // number of not-na
  
  arma::uvec indexing_target;
  
  bool fgrid;
  
  //arma::cube Kxxi;
  arma::cube * Kcx;
  arma::cube * Ri;
  arma::cube * Hproject;
  
  //arma::vec parKxxpar;
  arma::mat Kcxpar;
  //arma::mat w_parents;
  
  int num_children;
  double parents_dim;
  //arma::vec dim_of_pars_of_children;
  
  arma::field<arma::cube> Kcx_x;//(c) = (*param_data).w_cond_mean_K(child).cols(pofc_ix_x);
  //arma::field<arma::cube> Kxxi_x;//(c) = (*param_data).w_cond_prec_parents(child)(pofc_ix_x, pofc_ix_x);
  arma::field<arma::mat> w_child;//(c) = arma::vectorise( (*w_full).rows(c_ix) );
  arma::field<arma::cube *> Ri_of_child;//(c) = (*param_data).w_cond_prec(child);
  //arma::field<arma::mat> Kxo_wo;//(c) = Kxxi_xo * w_otherparents;
  arma::field<arma::mat> Kco_wo;//(c) = Kcx_other * w_otherparents;
  //arma::mat woKoowo;//(c) = w_otherparents.t() * Kxxi_other * w_otherparents;
  
  void initialize();
  void update_mv(const arma::mat& new_offset, const arma::vec& tausq, const arma::mat& Lambda_lmc_in);
  
  double logfullcondit(const arma::mat& x);
  arma::vec gradient_logfullcondit(const arma::mat& x);
  arma::mat neghess_logfullcondit(const arma::mat& x);
  
  void compute_dens_and_grad(double& xtarget, arma::vec& xgrad, const arma::mat& x);
  
  NodeDataW(const arma::mat& y_all, //const arma::mat& Z_in,
                const arma::umat& na_mat_all, const arma::mat& offset_all, 
                const arma::uvec& indexing_target,
                const arma::uvec& outtype, int k, 
                bool fgrid_in);
  
  NodeDataW();
  
};


inline NodeDataW::NodeDataW(){
  n=-1;
}

inline NodeDataW::NodeDataW(const arma::mat& y_all, //const arma::mat& Z_in,
                                    const arma::umat& na_mat_all, const arma::mat& offset_all, 
                                    const arma::uvec& indexing_target_in,
                                    const arma::uvec& outtype, int k, 
                                    bool fgrid_in){
  
  indexing_target = indexing_target_in;
  y = y_all.rows(indexing_target);
  offset = offset_all.rows(indexing_target);
  na_mat = na_mat_all.rows(indexing_target);
  
  // ----
  
  family = outtype; //= arma::vectorise(familymat);
  
  if(arma::any(family == 3)){
    ystar = arma::zeros(arma::size(y));
    for(int j=0; j<y.n_cols; j++){
      if(family(j) == 3){
        ystar.col(j) = log( y.col(j) / (1.0-y.col(j)) );
      }  
    }
  }
  
  
  n = y.n_rows;
  z = arma::ones(n); //Z_in.col(0);
  
  fgrid = fgrid_in;
}

inline void NodeDataW::initialize(){
  
}

inline void NodeDataW::update_mv(const arma::mat& offset_all, const arma::vec& tausq_in,
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

inline void NodeDataW::compute_dens_and_grad(double& xtarget, arma::vec& xgrad, const arma::mat& x){
  int nr = y.n_rows;
  int q = y.n_cols;
  int k = x.n_cols;
  
  arma::vec grad_loglike = arma::zeros(x.n_rows * x.n_cols);
  
  int indxsize = x.n_rows;
  
  double loglike = 0;
  for(int i=0; i<nr; i++){
    
    arma::mat wloc;
    if(fgrid){
      wloc = arma::sum(arma::trans((*Hproject).slice(i) % arma::trans(x)), 0);
    } else {
      wloc = x.row(i);
    }
    
    for(int j=0; j<q; j++){
      if(na_mat(i, j) > 0){
        arma::vec gradloc;
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        arma::mat LambdaHt;
        
        if(fgrid){
          LambdaHt = arma::zeros(k*indxsize, 1);  
          arma::mat Hloc = (*Hproject).slice(i);
          for(int jx=0; jx<k; jx++){
            arma::mat Hsub = Hloc.row(jx).t(); //data.Hproject(u).subcube(jx,0,ix,jx,indxsize-1, ix);
            // this outcome margin observed at this location
            LambdaHt.submat(jx*indxsize, 0, (jx+1)*indxsize-1, 0) += Lambda_lmc(j, jx) * Hsub;
          }
        } else {
          LambdaHt = Lambda_lmc.row(j).t();
        }
        
        if(family(j) == 0){  // family=="gaussian"
          double y_minus_mean = y(i, j) - offset(i, j) - xij;
          loglike += gaussian_logdensity(y_minus_mean, tausq(j));
          gradloc = LambdaHt * gaussian_loggradient(y_minus_mean, tausq(j));
        } else if(family(j) == 1){ //if(family == "poisson"){
          double lambda = exp(offset(i, j) + xij);//xz);//x(i));
          loglike += poisson_logpmf(y(i, j), lambda);
          gradloc = LambdaHt * poisson_loggradient(y(i, j), offset(i, j), xij); //xz) * z(i);
        } else if(family(j) == 2){ //if(family == "binomial"){
          double sigmoid = 1.0/(1.0 + exp(-offset(i, j) - xij));//xz ));
          loglike += bernoulli_logpmf(y(i, j), sigmoid);
          gradloc = LambdaHt * bernoulli_loggradient(y(i, j), offset(i, j), xij); //xz) * z(i);
        } else if(family(j) == 3){
          double sigmoid = 1.0/(1.0 + exp(-offset(i, j) - xij));
          loglike += betareg_logdens(y(i, j), sigmoid, 1.0/tausq(j));
          gradloc = LambdaHt * betareg_loggradient(ystar(i, j), sigmoid, 1.0/tausq(j));
        }
        
        if(fgrid){
          grad_loglike += gradloc;
        } else {
          for(int s=0; s<k; s++){
            grad_loglike(s * indxsize + i) += gradloc(s);   
          }
        }
      }
    }
  }
  
  // GP prior
  double logprior = 0;
  arma::vec grad_logprior_par;
  
  //double logprior = fwdcond_dmvn(x, Ri, Kcxpar);
  //arma::vec grad_logprior_par = grad_fwdcond_dmvn(x, Ri, Kcxpar);
  fwdconditional_mvn(logprior, grad_logprior_par, x, Ri, Kcxpar);
  
  double logprior_chi = 0;
  arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
  for(int c=0; c<num_children; c++ ){
    bwdconditional_mvn(logprior_chi, grad_logprior_chi, x, w_child(c), Ri_of_child(c), 
                             Kcx_x(c), Kco_wo(c));
  }
  
  xtarget = logprior + loglike;
  
  xgrad = grad_loglike + 
    grad_logprior_par + 
    grad_logprior_chi;

}

// log posterior 
inline double NodeDataW::logfullcondit(const arma::mat& x){
  double loglike = 0;
  
  for(int i=0; i<y.n_rows; i++){
    arma::mat wloc;
    if(fgrid){
      wloc = arma::sum(arma::trans((*Hproject).slice(i) % arma::trans(x)), 0);
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
        } else if(family(j) == 1){ //if(family=="poisson"){
          double lambda = exp(offset(i, j) + xstar);//xz);//x(i));
          loglike += poisson_logpmf(y(i, j), lambda);
        } else if(family(j) == 2){ //if(family=="binomial"){
          double sigmoid = 1.0/(1.0 + exp(-offset(i, j) - xstar));//xz ));
          loglike += bernoulli_logpmf(y(i, j), sigmoid);
        } else if(family(j) == 3){
          double sigmoid = 1.0/(1.0 + exp(-offset(i, j) - xstar));
          loglike += betareg_logdens(y(i, j), sigmoid, 1.0/tausq(j));
          
        }
         
      }
    }
  }
  
  // GP prior
  double logprior = fwdcond_dmvn(x, Ri, Kcxpar);
  for(int c=0; c<num_children; c++ ){
    logprior += bwdcond_dmvn(x, w_child(c), Ri_of_child(c), 
                             Kcx_x(c), Kco_wo(c));
    
  }
  return ( loglike + logprior );
}

// Gradient of the log posterior
inline arma::vec NodeDataW::gradient_logfullcondit(const arma::mat& x){
  int q = y.n_cols;
  int k = x.n_cols;
  
  arma::vec grad_loglike = arma::zeros(x.n_rows * x.n_cols);
  
  int nr = y.n_rows;
  int indxsize = x.n_rows;
  
  if(fgrid){
    
    for(int i=0; i<nr; i++){
      arma::mat wloc = arma::sum(arma::trans((*Hproject).slice(i) % arma::trans(x)), 0);
      arma::mat LambdaH = arma::zeros(q, k*indxsize);  
      for(int j=0; j<q; j++){
        if(na_mat(i, j) == 1){
          arma::mat Hloc = (*Hproject).slice(i);
          for(int jx=0; jx<k; jx++){
            arma::mat Hsub = Hloc.row(jx); //data.Hproject(u).subcube(jx,0,ix,jx,indxsize-1, ix);
            // this outcome margin observed at this location
            LambdaH.submat(j, jx*indxsize, j, (jx+1)*indxsize-1) += Lambda_lmc(j, jx) * Hsub;
          }
        }
      }
      
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
          } else if(family(j) == 3){
            double sigmoid = 1.0/(1.0 + exp(-offset(i, j) - xij));
            grad_loglike += LambdaHt * betareg_loggradient(ystar(i, j), sigmoid, 1.0/tausq(j));
          }
        }
      }
    }
    
  } else {
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
          } else if(family(j) == 3){
            double sigmoid = 1.0/(1.0 + exp(-offset(i, j) - xij));
            gradloc += LambdaHt * betareg_loggradient(ystar(i, j), sigmoid, 1.0/tausq(j));
          }
          
          for(int s=0; s<k; s++){
            grad_loglike(s * indxsize + i) += gradloc(s);   
          }
        } 
      }
    }
  }

  arma::vec grad_logprior_par = grad_fwdcond_dmvn(x, Ri, Kcxpar);
  
  arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
  for(int c=0; c<num_children; c++ ){
    grad_logprior_chi += grad_bwdcond_dmvn(x, w_child(c), Ri_of_child(c), 
                                           Kcx_x(c), Kco_wo(c));
  }
  
  return grad_loglike + 
    grad_logprior_par + 
    grad_logprior_chi;
}

// Gradient of the log posterior
inline arma::mat NodeDataW::neghess_logfullcondit(const arma::mat& x){
  int q = y.n_cols;
  int k = x.n_cols;
  
  arma::mat neghess_logtarg = arma::zeros(x.n_rows * x.n_cols,
                                  x.n_rows * x.n_cols);
  
  int nr = y.n_rows;
  int indxsize = x.n_rows;

  if(fgrid){
    
    for(int i=0; i<nr; i++){
      arma::mat wloc = arma::sum(arma::trans((*Hproject).slice(i) % arma::trans(x)), 0);
      for(int j=0; j<q; j++){
        if(na_mat(i, j) > 0){
          arma::mat Hloc = (*Hproject).slice(i);
          arma::rowvec LambdaHrowj = arma::zeros<arma::rowvec>(k*indxsize);  
          for(int jx=0; jx<k; jx++){
            arma::mat Hsub = Hloc.row(jx); //data.Hproject(u).subcube(jx,0,ix,jx,indxsize-1, ix);
            // this outcome margin observed at this location
            //LambdaH.submat(j, jx*indxsize, j, (jx+1)*indxsize-1) += Lambda_lmc(j, jx) * Hsub;
            double mult = 1;
            if (family(j) == 0){
              mult = pow(tausq(j), -0.5);
            } else {
              //arma::mat wloc = x.row(i);
              double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
              
              if (family(j) == 1){
                mult = pow(exp(xij), 0.5);
              } else if (family(j) == 2){
                double exij = exp(-xij);
                double opexij = (1.0 + exij);
                mult = pow(exij / (opexij*opexij), 0.5);
              } else if (family(j) == 3){
                double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
                double sigmoid = 1.0/(1.0 + exp(-offset(i, j) - xij));
                double tausq2 = tausq(j) * tausq(j);
                mult = - 1.0/tausq2 * (R::trigamma(sigmoid / tausq(j)) + 
                  R::trigamma( (1.0-sigmoid) / tausq(j) ) ) * 
                  pow(sigmoid * (1.0 - sigmoid), 2.0);  // notation of 
              }
            }
            LambdaHrowj.subvec(jx*indxsize, (jx+1)*indxsize-1) += (mult * Lambda_lmc(j, jx)) * Hsub;
          }
          
          neghess_logtarg += LambdaHrowj.t() * LambdaHrowj; 
        }
      }
    }
    
  } else {
    
    for(int i=0; i<nr; i++){
      
      for(int j=0; j<y.n_cols; j++){
        
        if(na_mat(i, j) > 0){
          double mult = 1;
          if (family(j) == 0){
            mult = pow(tausq(j), -0.5);
          } else {
            arma::mat wloc = x.row(i);
            double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
            
            if (family(j) == 1){
              mult = pow(exp(xij), 0.5);
            } else if (family(j) == 2){
              double exij = exp(-xij);
              double opexij = (1.0 + exij);
              mult = pow(exij / (opexij*opexij), 0.5);
            } else if (family(j) == 3){
              double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
              double sigmoid = 1.0/(1.0 + exp(-offset(i, j) - xij));
              double tausq2 = tausq(j) * tausq(j);
              mult = - 1.0/tausq2 * (R::trigamma( sigmoid / tausq(j) ) + 
                R::trigamma( (1.0-sigmoid) / tausq(j) ) ) * pow(sigmoid * (1.0 - sigmoid), 2.0);  // notation of 
            }
          }
          
          arma::mat LambdaHt = Lambda_lmc.row(j).t() * mult;
          arma::mat neghessloc = LambdaHt * LambdaHt.t();
          
          for(int s1=0; s1<k; s1++){
            for(int s2=0; s2<k; s2++){
              neghess_logtarg(s1 * indxsize + i, s2*indxsize + i) += neghessloc(s1, s2);
            }
          }
        }
      }
    }
    
  }
  
  neghess_fwdcond_dmvn(neghess_logtarg, x, Ri);
  
  //arma::mat neghess_logprior_chi = arma::zeros(arma::size(neghess_logprior_par));
  for(int c=0; c<num_children; c++ ){
    // adds to neghess_logprior_par
    neghess_bwdcond_dmvn(neghess_logtarg, x, w_child(c), Ri_of_child(c), Kcx_x(c));
  }
  
  
  
  return neghess_logtarg;// + 
    //neghess_logprior_par; // + 
    //neghess_logprior_chi;
}



class NodeDataB : public NodeData {
public:
  int family; // for beta
  
  arma::mat X; //for updates of beta
  double tausq; // reg variance
  
  // gaussian
  arma::mat XtX;
  arma::vec Xres;
  
  // binom
  arma::vec ones;
  
  // beta distrib outcomes
  arma::vec ystar;
    
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
  NodeDataB(const arma::vec& y_in, const arma::vec& offset_in, 
                   const arma::mat& X_in, int family_in);
  NodeDataB();
  
  double logfullcondit(const arma::vec& x);
  arma::vec gradient_logfullcondit(const arma::vec& x);
  arma::mat neghess_logfullcondit(const arma::vec& x);
  void compute_dens_and_grad(double& xtarget, arma::vec& xgrad, const arma::mat& x);
  //using MVDistParams::MVDistParams;
  //using MVDistParams::logfullcondit;
  //using MVDistParams::gradient_logfullcondit;
};

inline NodeDataB::NodeDataB(){
  n=-1;
}

inline NodeDataB::NodeDataB(const arma::vec& y_in, const arma::vec& offset_in, 
                                          const arma::mat& X_in, int family_in){
  family = family_in;
  n = y_in.n_elem;
  y = y_in;
  offset = offset_in;
  X = X_in;
  //which = "beta";
  
  if(family != 0){ // gaussian
    ones = arma::ones(n);
  }
  
  if(family == 2){ // binomial
    ystar = 1-y;
  }
  
  if(family == 3){
    ystar = log(y / (1.0 - y));
  }
  
  initialize();
}

inline void NodeDataB::initialize(){
  mstar = arma::zeros(X.n_cols);
  Vw_i = arma::eye(X.n_cols, X.n_cols);
  XtX = X.t() * X;
  
  if(family == 0){
    Sig = arma::inv_sympd(Vw_i + XtX); //
    Sig_i_tchol = arma::trans( arma::inv(arma::trimatl(arma::chol(Sig, "lower"))) );
    M = arma::eye(arma::size(Sig));//M = Sig;
    Michol = M;//Sig_i_tchol;
  } 
}

inline void NodeDataB::update_mv(const arma::vec& new_offset, const double& tausq_in, const arma::vec& Smu_tot, const arma::mat& Sigi_tot){
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
inline double NodeDataB::logfullcondit(const arma::vec& x){
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
      y.t() * log(sigmoid) + ystar.t() * log(1-sigmoid)
    );
    if(std::isnan(loglike)){
      loglike = -arma::datum::inf;
      //Rcpp::Rcout << "loglike new: " << loglike << "\n";
    }
  } else if(family == 0){
    loglike = 1.0/tausq * arma::conv_to<double>::from(
        Xres.t() * x - .5 * x.t() * XtX * x);
  } else if(family == 3){
    loglike = 0;
    double lgtsq = R::lgammafn(1.0/tausq);
    arma::vec sigmoid = 1.0/(1.0 + exp(-offset - X * x ));
    for(int i=0; i<y.n_elem; i++){
      loglike += lgtsq - R::lgammafn(sigmoid(i) / tausq) - R::lgammafn((1.0-sigmoid(i)) / tausq) +
        (sigmoid(i) / tausq - 1.0) * log(y(i)) + 
        ((1.0-sigmoid(i)) / tausq - 1.0) * log(1.0-y(i));
    }
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
inline arma::vec NodeDataB::gradient_logfullcondit(const arma::vec& x){
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
  } else if(family == 3){
    arma::vec sigmoid = 1.0/(1.0 + exp(-offset - X * x));
    arma::vec mustar = arma::zeros(y.n_elem);
    arma::vec Tym = arma::zeros(y.n_elem);
    for(int i=0; i<y.n_elem; i++){
      double oneminusmu = 1.0-sigmoid(i);
      mustar(i) = R::digamma(sigmoid(i) / tausq) - R::digamma(oneminusmu / tausq);
      Tym(i) = sigmoid(i) * (1-sigmoid(i)) * (ystar(i) - mustar(i));
    }
    grad_loglike = X.t() * Tym/ tausq;
  }
    
  
  arma::vec grad_logprior = mstar - Vw_i * x;
  //end = std::chrono::steady_clock::now();
  //grad_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return grad_loglike + grad_logprior;

}

inline arma::mat NodeDataB::neghess_logfullcondit(const arma::vec& x){
  return XtX + Vw_i;
}

inline void NodeDataB::compute_dens_and_grad(double& xtarget, arma::vec& xgrad, const arma::mat& x){
  xtarget = logfullcondit(x);
  xgrad = gradient_logfullcondit(x);
}