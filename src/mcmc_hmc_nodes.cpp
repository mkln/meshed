#include "mcmc_hmc_nodes.h"

NodeData::NodeData(){
  n=-1;
}

double NodeData::logfullcondit(const arma::vec& x){
  return 0;
}

arma::vec NodeData::gradient_logfullcondit(const arma::vec& x){
  return 0;
}


NodeDataW::NodeDataW(){
  n=-1;
  fgrid = false;
}

NodeDataW::NodeDataW(const arma::mat& y_all, //const arma::mat& Z_in,
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
    for(unsigned int j=0; j<y.n_cols; j++){
      if(family(j) == 3){
        ystar.col(j) = log( y.col(j) / (1.0-y.col(j)) );
      }  
    }
  }
  
  
  n = y.n_rows;
  z = arma::ones(n); //Z_in.col(0);
  
  fgrid = fgrid_in;
}

void NodeDataW::initialize(){
  
}

void NodeDataW::update_mv(const arma::mat& offset_all, const arma::vec& tausq_in,
                                     const arma::mat& Lambda_lmc_in){
  // arma::mat tausqmat = arma::zeros<arma::umat>(arma::size(new_offset));
  // for(unsigned int i=0; i<tausqmat.n_cols; i++){
  //   tausqmat.col(i).fill(tausq(i));
  // }
  // tausq_long = arma::vectorise(tausqmat);
  Lambda_lmc = Lambda_lmc_in;
  tausq = tausq_in;
  offset = offset_all.rows(indexing_target);
}


double NodeDataW::fwdcond_dmvn(const arma::mat& x, 
                    const arma::cube* Ri,
                    const arma::mat& Kcxpar){
  // conditional of x | parents
  
  double numer = 0;
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(Kcxpar.n_cols > 0){ // meaning w_parents.n_rows > 0
      xcentered -= Kcxpar.col(j);
    } 
    numer += arma::conv_to<double>::from( xcentered.t() * (*Ri).slice(j) * xcentered );
  }
  return -.5 * numer;//result;
}

arma::vec NodeDataW::grad_fwdcond_dmvn(const arma::mat& x){
  
  // gradient of conditional of x | parents
  arma::mat norm_grad = arma::zeros(arma::size(x));
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(Kcxpar.n_cols > 0){ // meaning w_parents.n_rows > 0
      xcentered -= Kcxpar.col(j);
    } 
    norm_grad.col(j) = -(*Ri).slice(j) * xcentered;
  }
  return arma::vectorise(norm_grad);//result;
}

void NodeDataW::fwdconditional_mvn(double& logtarget, arma::vec& gradient, 
                        const arma::mat& x){
  arma::mat norm_grad = arma::zeros(arma::size(x));
  double numer = 0;
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(Kcxpar.n_cols > 0){ // meaning w_parents.n_rows > 0
      xcentered -= Kcxpar.col(j);
    } 
    arma::vec Rix = (*Ri).slice(j) * xcentered;
    numer += arma::conv_to<double>::from( xcentered.t() * Rix );
    norm_grad.col(j) = - Rix;
  }
  logtarget = -.5 * numer;//result;
  gradient = arma::vectorise(norm_grad);//result;
}



double NodeDataW::bwdcond_dmvn(const arma::mat& x, int c){
  // conditional of Y | x, others
  
  double numer = 0;
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::vec xcentered = w_child(c).col(j) - Kcx_x(c).slice(j)*x.col(j);
    if(Kco_wo(c).n_cols > 0){
      xcentered -= Kco_wo(c).col(j);
    } 
    numer += arma::conv_to<double>::from(xcentered.t() * (*Ri_of_child(c)).slice(j) * xcentered);
  }
  
  return -0.5*numer;
}

arma::vec NodeDataW::grad_bwdcond_dmvn(const arma::mat& x, int c){
  // gradient of conditional of Y | x, others
  arma::mat result = arma::zeros(arma::size(x));
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::mat wccenter = w_child(c).col(j) - Kcx_x(c).slice(j) * x.col(j);
    if(Kco_wo(c).n_cols > 0){
      wccenter -= Kco_wo(c).col(j);
    } 
    result.col(j) = Kcx_x(c).slice(j).t() * (*Ri_of_child(c)).slice(j) * wccenter;
  }
  return arma::vectorise(result);
}

void NodeDataW::bwdconditional_mvn(double& xtarget, arma::vec& gradient, const arma::mat& x, int c){
  
  arma::mat result = arma::zeros(arma::size(x));
  double numer = 0;
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::vec xcentered = w_child(c).col(j) - Kcx_x(c).slice(j)*x.col(j);
    if(Kco_wo(c).n_cols > 0){
      xcentered -= Kco_wo(c).col(j);
    } 
    arma::vec Rix = (*Ri_of_child(c)).slice(j) * xcentered;
    numer += arma::conv_to<double>::from(xcentered.t() * Rix);
    result.col(j) = Kcx_x(c).slice(j).t() * Rix;
  }
  xtarget -= 0.5*numer;
  gradient += arma::vectorise(result);
}

void NodeDataW::neghess_fwdcond_dmvn(arma::mat& result, const arma::mat& x){
  
  int k = (*Ri).n_slices;
  int nr = (*Ri).n_rows;
  int nc = (*Ri).n_cols;
  
  //Rcpp::Rcout << "neghess_fwdcond_dmvn " << arma::size((*Ri)) << "\n";
  //arma::mat result = arma::zeros(nr * k, nc * k);
  for(int j=0; j<k; j++){
    result.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) += (*Ri).slice(j);
  }
  //return result;
}

void NodeDataW::neghess_bwdcond_dmvn(arma::mat& result,
                          const arma::mat& x, int c){
  
  int k = (*Ri_of_child(c)).n_slices;
  int nr = Kcx_x(c).n_cols; //(*Ri_of_child).n_rows;
  int nc = Kcx_x(c).n_cols; //(*Ri_of_child).n_cols;
  
  //Rcpp::Rcout << "neghess_bwdcond_dmvn " << arma::size(Kcx_x) << " " << arma::size((*Ri_of_child)) << "\n";
  
  //arma::mat result = arma::zeros(nr * k, nc * k);
  for(int j=0; j<k; j++){
    result.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) += Kcx_x(c).slice(j).t() * (*Ri_of_child(c)).slice(j) * Kcx_x(c).slice(j);
  }
  
  //return result;
}

void NodeDataW::mvn_dens_grad_neghess(double& xtarget, arma::vec& gradient, arma::mat& neghess,
                           const arma::mat& x, int c){
  
  
  int k = (*Ri_of_child(c)).n_slices;
  int nr = Kcx_x(c).n_cols; //(*Ri_of_child).n_rows;
  int nc = Kcx_x(c).n_cols; //(*Ri_of_child).n_cols;
  
  arma::mat result = arma::zeros(arma::size(x));
  double numer = 0;
  for(int j=0; j<k; j++){
    arma::vec xcentered = w_child(c).col(j) - Kcx_x(c).slice(j)*x.col(j);
    if(Kco_wo(c).n_cols > 0){
      xcentered -= Kco_wo(c).col(j);
    } 
    arma::mat KRichild = Kcx_x(c).slice(j).t() * (*Ri_of_child(c)).slice(j);
    numer += arma::conv_to<double>::from(xcentered.t() * (*Ri_of_child(c)).slice(j) * xcentered);
    result.col(j) = KRichild * xcentered;
    neghess.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) += KRichild * Kcx_x(c).slice(j);
  }
  xtarget -= 0.5*numer;
  gradient += arma::vectorise(result);
  
}


arma::vec NodeDataW::compute_dens_and_grad(double& xtarget, const arma::mat& x){
  unsigned int nr = y.n_rows;
  unsigned int q = y.n_cols;
  unsigned int k = x.n_cols;
  
  arma::vec grad_loglike = arma::zeros(x.n_rows * x.n_cols);
  
  int indxsize = x.n_rows;
  
  double loglike = 0;
  for(int i=0; i<nr; i++){
    arma::mat LambdaH, Hloc;
    arma::mat wloc;
    if(fgrid){
      Hloc = (*Hproject).slice(i);
      wloc = arma::sum(arma::trans((*Hproject).slice(i) % arma::trans(x)), 0);
    } else {
      wloc = x.row(i);
    }
    if(fgrid){
      LambdaH = arma::zeros(q, k*indxsize);
    }
    for(unsigned int j=0; j<q; j++){
      if(na_mat(i, j) > 0){
        
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        
        double ystarij = family(j) == 3? ystar(i, j) : 0;
        arma::vec gradloc = get_likdens_likgrad(loglike, y(i,j), ystarij, tausq(j), 
                                                offset(i, j), xij, family(j));
        
        if(fgrid){
          for(unsigned int jx=0; jx<k; jx++){
            arma::mat Hsub = Hloc.row(jx); //data.Hproject(u).subcube(jx,0,ix,jx,indxsize-1, ix);
            // this outcome margin observed at this location
            LambdaH.submat(j, jx*indxsize, j, (jx+1)*indxsize-1) += Lambda_lmc(j, jx) * Hsub;
          }
          arma::vec adding_grad = arma::trans(LambdaH.row(j)) * gradloc;
          grad_loglike += adding_grad;
        } else {
          arma::mat LambdaHt = Lambda_lmc.row(j).t();
          arma::vec Lgrad = LambdaHt * gradloc;
          for(unsigned int s1=0; s1<k; s1++){
            grad_loglike(s1 * indxsize + i) += Lgrad(s1);   
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
  fwdconditional_mvn(logprior, grad_logprior_par, x);
  
  double logprior_chi = 0;
  arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
  for(unsigned int c=0; c<num_children; c++ ){
    bwdconditional_mvn(logprior_chi, grad_logprior_chi, x, c);
  }
  
  xtarget = logprior + logprior_chi + loglike;
  
  return grad_loglike + 
    grad_logprior_par + 
    grad_logprior_chi;

}

// log posterior 
double NodeDataW::logfullcondit(const arma::mat& x){
  double loglike = 0;
  
  for(unsigned int i=0; i<y.n_rows; i++){
    arma::mat wloc;
    if(fgrid){
      wloc = arma::sum(arma::trans((*Hproject).slice(i) % arma::trans(x)), 0);
    } else {
      wloc = x.row(i);
    }
    for(unsigned int j=0; j<y.n_cols; j++){
      //Rcpp::Rcout << i << " - " << j << endl;
      if(na_mat(i, j) > 0){
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        double ystarij = family(j) == 3? ystar(i, j) : 0;
        arma::vec gradloc = get_likdens_likgrad(loglike, y(i,j), ystarij, tausq(j), 
                                                offset(i, j), xij, family(j), false);
      }
    }
  }
  
  // GP prior
  double logprior = fwdcond_dmvn(x, Ri, Kcxpar);
  for(unsigned int c=0; c<num_children; c++ ){
    logprior += bwdcond_dmvn(x, c);
    
  }
  return ( loglike + logprior );
}

// log posterior 
double NodeDataW::loglike(const arma::mat& x){
  double loglike = 0;
  
  for(unsigned int i=0; i<y.n_rows; i++){
    arma::mat wloc;
    if(fgrid){
      wloc = arma::sum(arma::trans((*Hproject).slice(i) % arma::trans(x)), 0);
    } else {
      wloc = x.row(i);
    }
    for(unsigned int j=0; j<y.n_cols; j++){
      //Rcpp::Rcout << i << " - " << j << endl;
      if(na_mat(i, j) > 0){
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        double ystarij = family(j) == 3? ystar(i, j) : 0;
        arma::vec gradloc = get_likdens_likgrad(loglike, y(i,j), ystarij, tausq(j), 
                                                offset(i, j), xij, family(j), false);
      }
    }
  }
  
  return ( loglike );
}

// Gradient of the log posterior
arma::vec NodeDataW::gradient_logfullcondit(const arma::mat& x){
  int q = y.n_cols;
  int k = x.n_cols;
  
  arma::vec grad_loglike = arma::zeros(x.n_rows * x.n_cols);
  
  int nr = y.n_rows;
  int indxsize = x.n_rows;
  
  if(fgrid){
    
    for(int i=0; i<nr; i++){
      arma::mat wloc = arma::sum(arma::trans((*Hproject).slice(i) % arma::trans(x)), 0);
      arma::mat LambdaH = arma::zeros(q, k*indxsize);  
      arma::mat Hloc = (*Hproject).slice(i);
      for(int j=0; j<q; j++){
        if(na_mat(i, j) == 1){
          for(int jx=0; jx<k; jx++){
            arma::mat Hsub = Hloc.row(jx); //data.Hproject(u).subcube(jx,0,ix,jx,indxsize-1, ix);
            // this outcome margin observed at this location
            LambdaH.submat(j, jx*indxsize, j, (jx+1)*indxsize-1) += Lambda_lmc(j, jx) * Hsub;
          }
        }
      }
      
      for(unsigned int j=0; j<y.n_cols; j++){
        if(na_mat(i, j) > 0){
          double loglike = 0;
          double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
          arma::mat LambdaHt = LambdaH.row(j).t();
          
          double ystarij = family(j) == 3? ystar(i, j) : 0;
          arma::vec gradloc = get_likdens_likgrad(loglike, y(i,j), ystarij, tausq(j), 
                                                  offset(i, j), xij, family(j));
          grad_loglike += LambdaHt * gradloc;
        }
      }
    }
    
  } else {
    for(int i=0; i<nr; i++){
      arma::mat wloc = x.row(i);
      
      for(unsigned int j=0; j<y.n_cols; j++){
        if(na_mat(i, j) > 0){
          double loglike = 0;
          arma::mat LambdaHt = Lambda_lmc.row(j).t();
          double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
          double ystarij = family(j) == 3? ystar(i, j) : 0;
          arma::vec gradloc = LambdaHt * get_likdens_likgrad(loglike, y(i,j), ystarij, tausq(j), 
                                                             offset(i, j), xij, family(j));
          
          for(int s=0; s<k; s++){
            grad_loglike(s * indxsize + i) += gradloc(s);   
          }
        } 
      }
    }
  }

  arma::vec grad_logprior_par = grad_fwdcond_dmvn(x);
  
  arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
  for(unsigned int c=0; c<num_children; c++ ){
    grad_logprior_chi += grad_bwdcond_dmvn(x, c);
  }
  
  return grad_loglike + 
    grad_logprior_par + 
    grad_logprior_chi;
}


arma::mat NodeDataW::compute_dens_grad_neghess(double& xtarget, arma::vec& xgrad, const arma::mat& x){
  int nr = y.n_rows;
  int q = y.n_cols;
  int k = x.n_cols;
  
  //Rcpp::Rcout << "compute_dens_grad_neghess start " << endl;
  
  arma::vec grad_loglike = arma::zeros(x.n_rows * x.n_cols);
  arma::mat neghess_logtarg = arma::zeros(x.n_rows * x.n_cols,
                                          x.n_rows * x.n_cols);
  
  int indxsize = x.n_rows;
  
  double loglike = 0;
  
  for(int i=0; i<nr; i++){
    arma::mat wloc, Hloc;
    arma::mat LambdaH;
    
    if(fgrid){
      Hloc = (*Hproject).slice(i);
      wloc = arma::sum(arma::trans(Hloc % arma::trans(x)), 0);
    } else {
      wloc = x.row(i);
    }
    
    // get mult
    arma::vec mult = arma::zeros(q);
    for(int j=0; j<q; j++){
      if(na_mat(i, j) > 0){
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        mult(j) = get_mult(y(i,j), tausq(j), offset(i,j), xij, family(j));
      }
    }
    
    
    if(fgrid){
      LambdaH = arma::zeros(q, k*indxsize);
    }
    
    for(int j=0; j<q; j++){
      if(na_mat(i, j) > 0){
        
        double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
        
        double ystarij = family(j) == 3? ystar(i, j) : 0;
        arma::vec gradloc = get_likdens_likgrad(loglike, y(i,j), ystarij, tausq(j), 
                                                offset(i, j), xij, family(j));

        if(fgrid){
          for(unsigned int jx=0; jx<k; jx++){
            arma::mat Hsub = Hloc.row(jx); //data.Hproject(u).subcube(jx,0,ix,jx,indxsize-1, ix);
            // this outcome margin observed at this location
            LambdaH.submat(j, jx*indxsize, j, (jx+1)*indxsize-1) += (mult(j) * Lambda_lmc(j, jx)) * Hsub;
          }
          arma::vec adding_grad = arma::trans(LambdaH.row(j)/mult(j)) * gradloc;
          grad_loglike += adding_grad;
      
        } else {
          arma::mat LambdaHt = Lambda_lmc.row(j).t();
          arma::vec Lgrad = LambdaHt * gradloc;
          
          arma::mat hess_LambdaH = Lambda_lmc.row(j).t() * mult(j);
          arma::mat neghessloc = hess_LambdaH * hess_LambdaH.t();
          
          for(int s1=0; s1<k; s1++){
            grad_loglike(s1 * indxsize + i) += Lgrad(s1);   
            for(int s2=0; s2<k; s2++){
              neghess_logtarg(s1 * indxsize + i, s2*indxsize + i) += neghessloc(s1, s2);
            }
          }  
          
        }
      }
    }
    
    if(fgrid){
      neghess_logtarg += LambdaH.t() * LambdaH;
    }
    
    
    
  }
  
  //Rcpp::Rcout << "compute_dens_grad_neghess finishing " << endl;
  // GP prior
  double logprior = 0;
  arma::vec grad_logprior_par = arma::zeros(x.n_elem);
  
  //double logprior = fwdcond_dmvn(x, Ri, Kcxpar);
  //arma::vec grad_logprior_par = grad_fwdcond_dmvn(x, Ri, Kcxpar);
  
  fwdconditional_mvn(logprior, grad_logprior_par, x); // ***
  neghess_fwdcond_dmvn(neghess_logtarg, x);
  
  double logprior_chi = 0;
  arma::vec grad_logprior_chi = arma::zeros(grad_logprior_par.n_elem);
  
  
  for(unsigned int c=0; c<num_children; c++ ){
    //bwdconditional_mvn(logprior_chi, grad_logprior_chi, x, w_child(c), Ri_of_child(c), 
    //                   Kcx_x(c), Kco_wo(c));
    //neghess_bwdcond_dmvn(neghess_logtarg, x, w_child(c), Ri_of_child(c), Kcx_x(c));
    
    // with pointer to cube -- memory leak?
    mvn_dens_grad_neghess(logprior_chi, grad_logprior_chi, neghess_logtarg, x, c);
  }
  
  xtarget = logprior + logprior_chi + loglike;
  
  xgrad = grad_loglike + 
    grad_logprior_par + 
    grad_logprior_chi;
  
  return neghess_logtarg;
  
}



// Gradient of the log posterior
arma::mat NodeDataW::neghess_logfullcondit(const arma::mat& x){
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
            double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
            double mult = get_mult(y(i,j), tausq(j), offset(i,j), xij, family(j));
            
            LambdaHrowj.subvec(jx*indxsize, (jx+1)*indxsize-1) += (mult * Lambda_lmc(j, jx)) * Hsub;
          }
          
          neghess_logtarg += LambdaHrowj.t() * LambdaHrowj; 
        }
      }
    }
    
  } else {
    
    for(int i=0; i<nr; i++){
      arma::mat wloc = x.row(i);
      for(unsigned int j=0; j<y.n_cols; j++){
        if(na_mat(i, j) > 0){
          double xij = arma::conv_to<double>::from(Lambda_lmc.row(j) * wloc.t());
          double mult = get_mult(y(i,j), tausq(j), offset(i,j), xij, family(j));
          
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
  
  neghess_fwdcond_dmvn(neghess_logtarg, x);
  
  //arma::mat neghess_logprior_chi = arma::zeros(arma::size(neghess_logprior_par));
  for(unsigned int c=0; c<num_children; c++ ){
    // adds to neghess_logprior_par
    neghess_bwdcond_dmvn(neghess_logtarg, x, c);
  }
  
  
  
  return neghess_logtarg;// + 
    //neghess_logprior_par; // + 
    //neghess_logprior_chi;
}


// Neghess of the log cond prior
arma::mat NodeDataW::neghess_prior(const arma::mat& x){
  
  arma::mat neghess_logtarg = arma::zeros(x.n_rows * x.n_cols,
                                          x.n_rows * x.n_cols);
  
  neghess_fwdcond_dmvn(neghess_logtarg, x);
  
  //arma::mat neghess_logprior_chi = arma::zeros(arma::size(neghess_logprior_par));
  for(unsigned int c=0; c<num_children; c++ ){
    // adds to neghess_logprior_par
    neghess_bwdcond_dmvn(neghess_logtarg, x, c);
  }
  
  return neghess_logtarg;// + 
  //neghess_logprior_par; // + 
  //neghess_logprior_chi;
}


NodeDataB::NodeDataB(){
  n=-1;
}

NodeDataB::NodeDataB(const arma::vec& y_in, const arma::vec& offset_in, 
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

void NodeDataB::initialize(){
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

void NodeDataB::update_mv(const arma::vec& new_offset, const double& tausq_in, const arma::vec& Smu_tot, const arma::mat& Sigi_tot){
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
double NodeDataB::logfullcondit(const arma::vec& x){
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
      y.t() * log(sigmoid + 1e-6) + ystar.t() * log(1-sigmoid + 1e-6)
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
    
    arma::vec logcomps = arma::zeros(y.n_elem);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(unsigned int i=0; i<y.n_elem; i++){
      logcomps(i) = lgtsq - R::lgammafn(sigmoid(i) / tausq) - R::lgammafn((1.0-sigmoid(i)) / tausq) +
        (sigmoid(i) / tausq - 1.0) * log(y(i)) + 
        ((1.0-sigmoid(i)) / tausq - 1.0) * log(1.0-y(i));
    }
    loglike += arma::accu(logcomps);
  } else if(family == 4){ // negative binomial
    arma::vec Xx = X * x;
    arma::vec logmuvec = offset + X * x;
    
    loglike = 0;
    arma::vec logcomps = arma::zeros(y.n_elem);
    for(unsigned int i=0; i<y.n_elem; i++){
      double logmu = logmuvec(i);
      double mu = exp(logmu);
      double alpha = tausq;
      logcomps(i) = negbin_logdens(y(i), mu, logmu, alpha);
    }
    loglike += arma::accu(logcomps);
  }
  
  double logprior = arma::conv_to<double>::from(
    x.t() * mstar - .5 * x.t() * Vw_i * x);
  
  return ( loglike + logprior );
  
}

// Gradient of the log posterior
arma::vec NodeDataB::gradient_logfullcondit(const arma::vec& x){
  //Rcpp::Rcout << "grad " << endl;
  //std::chrono::steady_clock::time_point start;
  //std::chrono::steady_clock::time_point end;
  //start = std::chrono::steady_clock::now();
  
  arma::vec grad_loglike = arma::zeros(x.n_elem);
  
  if(family==0){
    grad_loglike = 1.0/tausq * (Xres - XtX * x);
  } else if(family == 1){
    arma::vec Xx = X * x;
    arma::vec muvec = arma::zeros(offset.n_elem);
    for(unsigned int i=0; i<muvec.n_elem; i++){
      double logmu = offset(i) + Xx(i);
      if(logmu > TOL_LOG_HIGH){
        logmu = TOL_LOG_HIGH;
      }
      muvec(i) = exp(logmu);
    }
    grad_loglike = X.t() * (y - muvec);
  } else if(family == 2){
    arma::vec sigmoid = 1.0/(1.0 + exp(-offset - X * x));
    grad_loglike = X.t() * (y - //na_vec % 
      sigmoid );
  } else if(family == 3){
    arma::vec sigmoid = 1.0/(1.0 + exp(-offset - X * x));
    arma::vec mustar = arma::zeros(y.n_elem);
    arma::vec Tym = arma::zeros(y.n_elem);
    
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(unsigned int i=0; i<y.n_elem; i++){
      double oneminusmu = 1.0-sigmoid(i);
      mustar(i) = R::digamma(sigmoid(i) / tausq) - R::digamma(oneminusmu / tausq);
      Tym(i) = sigmoid(i) * (1-sigmoid(i)) * (ystar(i) - mustar(i));
    }
    grad_loglike = X.t() * Tym/ tausq;
  } else if(family == 4){ // negative binomial
    arma::vec Xx = X * x;
    arma::vec muvec = arma::zeros(offset.n_elem);
    for(unsigned int i=0; i<muvec.n_elem; i++){
      double logmu = offset(i) + Xx(i);
      if(logmu > 5){
        logmu = 5;
      }
      muvec(i) = exp(logmu);
    }
    double alpha = tausq;
    grad_loglike = X.t() * ((y-muvec)/(1.0 + alpha * muvec));
  }
    
  
  arma::vec grad_logprior = mstar - Vw_i * x;
  //end = std::chrono::steady_clock::now();
  //grad_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return grad_loglike + grad_logprior;

}

void NodeDataB::set_XtDX(const arma::vec& x){
  arma::vec Xb = X*x;
  arma::vec mult = arma::zeros(X.n_rows);
  arma::mat Xresult = X;
  for(unsigned int i=0; i<X.n_rows; i++){
    /*
    if(family == 0){  // family=="gaussian"
      mult(i) = pow(tausq, -0.5);
    } else if (family == 1){
      double mu = exp(offset(i) + Xb(i));
      mult(i) = poisson_neghess_mult_sqrt(mu);
    } else if (family == 2){
      double exij = exp(-offset(i) - Xb(i));
      mult(i) = bernoulli_neghess_mult_sqrt(exij);
    } else if (family == 3){
      double sigmoid = 1.0/(1.0 + exp(-offset(i) - Xb(i)));
      mult(i) = betareg_neghess_mult_sqrt(sigmoid, tausq);
    } else if(family == 4){
      double logmu = offset(i) + Xb(i);
      double alpha = tausq;
      mult(i) = negbin_neghess_mult_sqrt(y(i), logmu, alpha);
    }*/
    
    mult(i) = get_mult(y(i), tausq, offset(i), Xb(i), family);
    Xresult.row(i) = X.row(i) * mult(i);
  }
  
  XtX = Xresult.t() * Xresult;
}

arma::mat NodeDataB::neghess_logfullcondit(const arma::vec& x){
  set_XtDX(x);
  return XtX + Vw_i;
}

arma::vec NodeDataB::compute_dens_and_grad(double& xtarget, const arma::mat& x){
  xtarget = logfullcondit(x);
  return gradient_logfullcondit(x);
}

arma::mat NodeDataB::compute_dens_grad_neghess(double& xtarget, arma::vec& xgrad, const arma::mat& x){
  xtarget = logfullcondit(x);
  xgrad = gradient_logfullcondit(x);
  return neghess_logfullcondit(x);
}
