#include "covariance_lmc.h"

using namespace std;

// Attention: this REQUIRES a change in the indexing 
// because we are outputing block diagonal matrices
// ie sorting by variable first, then coordinates
// which is NOT what the other code is currently doing!


// exponential covariance
arma::mat expcorrel(const arma::mat& x, const arma::mat& y, const double& phi, bool same){
  // 0 based indexing
  if(same){
    arma::mat pmag = arma::sum(x % x, 1);
    int np = x.n_rows;
    arma::mat K = exp(-phi * sqrt(abs(arma::repmat(pmag.t(), np, 1) + arma::repmat(pmag, 1, np) - 2 * x * x.t())));
    return K;
  } else {
    arma::mat pmag = arma::sum(x % x, 1);
    arma::mat qmag = arma::sum(y % y, 1);
    int np = x.n_rows;
    int nq = y.n_rows;
    arma::mat K = exp(-phi * sqrt(abs(arma::repmat(qmag.t(), np, 1) + arma::repmat(pmag, 1, nq) - 2 * x * y.t())));
    return K;
  }
}

arma::mat Correlationf(const arma::mat& x, const arma::mat& y, const arma::vec& theta, bool same){
  return expcorrel(x, y, theta(0), same);
}


arma::mat CviaKron(const arma::mat& coords, 
                   const arma::uvec& indx, const arma::uvec& indy,
                   int k, const arma::mat& theta, bool same){
  arma::mat res = arma::zeros(indx.n_rows * k, indy.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int j=0; j<k; j++){
    arma::mat CC = Correlationf(coordsx, coordsy, theta.col(j), same);
    res += arma::kron(arma::diagmat(Iselect.col(j)), CC);
  }
  return res;
}


arma::mat CviaKron_invsympd(const arma::mat& coords, const arma::uvec& indx, 
                            int k, const arma::mat& theta){
  arma::mat res = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  for(int j=0; j<k; j++){
    arma::mat CCi = arma::inv_sympd( Correlationf(coordsx, coordsx, theta.col(j), true) );
    res += arma::kron(arma::diagmat(Iselect.col(j)), CCi);
  }
  return res;
}


arma::mat CviaKron_chol(const arma::mat& coords, const arma::uvec& indx, 
                        int k, const arma::mat& theta){
  arma::mat res = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  
  for(int j=0; j<k; j++){
    arma::mat CC_chol = arma::chol( arma::symmatu( 
      Correlationf(coordsx, coordsx, theta.col(j), true) ), "lower");
    res += arma::kron(arma::diagmat(Iselect.col(j)), CC_chol);
  }
  return res;
}


double CviaKron_invchol(arma::mat& res,
    const arma::mat& coords, const arma::uvec& indx, 
    int k, const arma::mat& theta){
  res = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  double logdet = 0;
  for(int j=0; j<k; j++){
    arma::mat CC_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( 
      Correlationf(coordsx, coordsx, theta.col(j), true) ), "lower")));
    res += arma::kron(arma::diagmat(Iselect.col(j)), CC_chol);
    logdet += arma::accu(log(CC_chol.diag()));
  }
  return logdet;
}



arma::mat CviaKron_H(const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta){
  arma::mat res = arma::zeros(indx.n_rows * k, indy.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int j=0; j<k; j++){
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), false);
    arma::mat Cyy_i = arma::inv_sympd( arma::symmatu(
      Correlationf(coordsy, coordsy, theta.col(j), true)) );
    arma::mat H = Cxy * Cyy_i;
    res += arma::kron(arma::diagmat(Iselect.col(j)), H);
  }
  return res;
}


arma::mat CviaKron_R(const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta){
  arma::mat res = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int j=0; j<k; j++){
    arma::mat Cxx = Correlationf(coordsx, coordsx, theta.col(j), true);
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), false);
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), true) );
    arma::mat R = Cxx - Cxy * Cyy_i * Cxy.t();
    res += arma::kron(arma::diagmat(Iselect.col(j)), R);
  }
  return res;
}

void CviaKron_HRj_bdiag(
    arma::field<arma::mat>& Hj, arma::field<arma::mat>& Rj, 
    const arma::field<arma::mat>& Kxxi_cache,
    const arma::mat& coords, const arma::uvec& indx, 
    const arma::uvec& naix, const arma::uvec& indy, 
    int k, const arma::mat& theta){
  
  Rj = arma::field<arma::mat> (indx.n_elem);
  Hj = arma::field<arma::mat> (indx.n_elem);
  
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int ix=0; ix<indx.n_rows; ix++){
    Rj(ix) = arma::zeros(k, k);
    Hj(ix) = arma::zeros(k, k*indy.n_rows);
  }
  for(int j=0; j<k; j++){
    arma::mat Cyy_i = Kxxi_cache(j);// arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), true) );
    for(int ix=0; ix<indx.n_rows; ix++){
      if(naix(ix) == 1){
        arma::mat Cxx = Correlationf(coordsx.row(ix), coordsx.row(ix), theta.col(j), true);
        arma::mat Cxy = Correlationf(coordsx.row(ix), coordsy, theta.col(j), false);
        arma::mat Hloc = Cxy * Cyy_i;
        Hj(ix) += arma::kron(arma::diagmat(Iselect.col(j)), Hloc);
        arma::mat R = Cxx - Hloc * Cxy.t();
        Rj(ix) += arma::kron(arma::diagmat(Iselect.col(j)), R);
      }
    }
  }
  
}


void CviaKron_HRj_chol_bdiag(
    arma::cube& Hj, arma::mat& Rjchol, const arma::uvec& naix,
    const arma::mat& coords, const arma::uvec& indx, const arma::uvec& indy, 
    int k, const arma::mat& theta){
  
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  
  for(int j=0; j<k; j++){
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), true) );
    for(int ix=0; ix<indx.n_rows; ix++){
      if(naix(ix) == 0){ // otherwise it's not missing
        arma::mat Cxx = Correlationf(coordsx.row(ix), coordsx.row(ix), theta.col(j), true);
        arma::mat Cxy = Correlationf(coordsx.row(ix), coordsy, theta.col(j), false);
        arma::mat Hloc = Cxy * Cyy_i;
        
        Hj.slice(ix).row(j) = Hloc;//+=arma::kron(arma::diagmat(Iselect.col(j)), Hloc);
        double Rcholtemp = arma::conv_to<double>::from(
          Cxx - Hloc * Cxy.t() );
        Rjchol(j, ix) = pow(abs(Rcholtemp), .5); // 0 could be numerically negative
      }
      
    }
  }
}

void CviaKron_HRj_chol_bdiag_wcache(
    arma::cube& Hj, arma::mat& Rjchol, 
    const arma::cube& Kxxi_cache, const arma::uvec& naix,
    const arma::mat& coords, const arma::uvec& indx, const arma::uvec& indy, 
    int k, const arma::mat& theta){
  
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  
  for(int j=0; j<k; j++){
    arma::mat Cyy_i = Kxxi_cache.slice(j);//arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), true) );
    for(int ix=0; ix<indx.n_rows; ix++){
      if(naix(ix) == 0){ // otherwise it's not missing
        arma::mat Cxx = Correlationf(coordsx.row(ix), coordsx.row(ix), theta.col(j), true);
        arma::mat Cxy = Correlationf(coordsx.row(ix), coordsy, theta.col(j), false);
        arma::mat Hloc = Cxy * Cyy_i;
        
        Hj.slice(ix).row(j) = Hloc;//+=arma::kron(arma::diagmat(Iselect.col(j)), Hloc);
        double Rcholtemp = arma::conv_to<double>::from(
          Cxx - Hloc * Cxy.t() );
        Rjchol(j,ix) = pow(abs(Rcholtemp), .5); // 0 could be numerically negative
      }
    }
  }
}

double CviaKron_HRi(arma::mat& H, arma::mat& Ri,
    const arma::mat& coords, 
    const arma::uvec& indx, const arma::uvec& indy, 
    int k, const arma::mat& theta){
  H = arma::zeros(indx.n_rows * k, indy.n_rows * k);
  Ri = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  double logdet=0;
  for(int j=0; j<k; j++){
    arma::mat Cxx = Correlationf(coordsx, coordsx, theta.col(j), true);
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), false);
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), true) );
    arma::mat Hloc = Cxy * Cyy_i;
    arma::mat Rloc_ichol = arma::inv(arma::trimatl(arma::chol( arma::symmatu(
                    Cxx - Hloc * Cxy.t()) , "lower")));
    logdet += arma::accu(log(Rloc_ichol.diag()));
    H += arma::kron(arma::diagmat(Iselect.col(j)), Hloc);
    Ri += arma::kron(arma::diagmat(Iselect.col(j)), Rloc_ichol.t() * Rloc_ichol);
  }
  return logdet;
}


arma::mat CviaKron_Ri(const arma::mat& coords, 
                      const arma::uvec& indx, const arma::uvec& indy,  
                      int k, const arma::mat& theta){
  arma::mat res = arma::zeros(indx.n_rows * k, indy.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int j=0; j<k; j++){
    arma::mat Cxx = Correlationf(coordsx, coordsx, theta.col(j), true);
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), false);
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), true) );
    arma::mat Ri = arma::inv_sympd( Cxx - Cxy * Cyy_i * Cxy.t() );
    res += arma::kron(arma::diagmat(Iselect.col(j)), Ri);
  }
  return res;
}


arma::mat CviaKron_Rchol(const arma::mat& coords, 
                         const arma::uvec& indx, const arma::uvec& indy,  
                         int k, const arma::mat& theta){
  arma::mat res = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int j=0; j<k; j++){
    arma::mat Cxx = Correlationf(coordsx, coordsx, theta.col(j), true);
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), false);
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), true) );
    arma::mat Rchol = arma::chol(arma::symmatu( Cxx - Cxy * Cyy_i * Cxy.t() ), "lower");
    res += arma::kron(arma::diagmat(Iselect.col(j)), Rchol);
  }
  return res;
}


arma::mat CviaKron_Rcholinv(const arma::mat& coords, 
                            const arma::uvec& indx, const arma::uvec& indy,  
                            int k, const arma::mat& theta){
  arma::mat res = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int j=0; j<k; j++){
    arma::mat Cxx = Correlationf(coordsx, coordsx, theta.col(j), true);
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), false);
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), true) );
    arma::mat Rcholinv = arma::inv(arma::trimatl(
      arma::chol(arma::symmatu( Cxx - Cxy * Cyy_i * Cxy.t() ), "lower")));
    res += arma::kron(arma::diagmat(Iselect.col(j)), Rcholinv);
  }
  return res;
}


// inplace functions


double CviaKron_HRi_(arma::cube& H, arma::cube& Ri,
                     const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta){
  // inplace version of CviaKron_HRi
  int dimx = indx.n_elem;
  int dimy = indy.n_elem;
  
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  double logdet=0;
  for(int j=0; j<k; j++){
    arma::mat Cxx = Correlationf(coordsx, coordsx, theta.col(j), true);
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), false);
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), true) );
    arma::mat Hloc = Cxy * Cyy_i;
    arma::mat Rloc_ichol = arma::inv(arma::trimatl(arma::chol( arma::symmatu(
      Cxx - Hloc * Cxy.t()) , "lower")));
    logdet += arma::accu(log(Rloc_ichol.diag()));
    
    int firstrow = j*dimx;
    int lastrow = (j+1)*dimx-1;
    int firstcol = j*dimy;
    int lastcol = (j+1)*dimy-1;
    if(indy.n_elem > 0){
      H.slice(j) = Hloc;//H.submat(firstrow, firstcol, lastrow, lastcol) = Hloc;
    }
    Ri.slice(j) = Rloc_ichol.t() * Rloc_ichol;//Ri.submat(firstrow, firstrow, lastrow, lastrow) = Rloc_ichol.t() * Rloc_ichol; // symmetric
  }
  return logdet;
}

double CviaKron_invsympd_wdet_(arma::cube& res,
                         const arma::mat& coords, const arma::uvec& indx, 
                         int k, const arma::mat& theta){
  // inplace
  int dimx = indx.n_elem;
  arma::mat coordsx = coords.rows(indx);
  double logdet = 0;
  for(int j=0; j<k; j++){
    arma::mat CC_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( 
      Correlationf(coordsx, coordsx, theta.col(j), true) ), "lower")));
    int firstrow = j*dimx;
    int lastrow = (j+1)*dimx-1;
    res.slice(j) = CC_chol.t()*CC_chol;//res.submat(firstrow, firstrow, lastrow, lastrow) = CC_chol.t()*CC_chol;
    logdet += arma::accu(log(CC_chol.diag()));
  }
  return logdet;
}

void CviaKron_HRj_bdiag_(
    arma::cube& Hj, arma::cube& Rj, 
    const arma::cube& Kxxi_cache,
    const arma::mat& coords, const arma::uvec& indx, 
    const arma::uvec& naix, const arma::uvec& indy, 
    int k, const arma::mat& theta){
  //Rcpp::Rcout << "CviaKron_HRj_bdiag_ " << endl;
  //Rcpp::Rcout << indx.n_elem << " " << indy.n_elem << " k=" << k << endl;
  // inplace version of CviaKron_HRj_bdiag
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  int dimx = indx.n_elem;
  int dimy = indy.n_elem;
  for(int j=0; j<k; j++){
    arma::mat Cyy_i = Kxxi_cache.slice(j);// 
      //arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), true) );
    
    int firstcol = j*dimy;
    int lastcol = (j+1)*dimy-1;
    for(int ix=0; ix<indx.n_rows; ix++){
      if(naix(ix) == 1){
        arma::mat Cxx = Correlationf(coordsx.row(ix), coordsx.row(ix), theta.col(j), true);
        arma::mat Cxy = Correlationf(coordsx.row(ix), coordsy, theta.col(j), false);
        arma::mat Hloc = Cxy * Cyy_i;
        arma::mat R = Cxx - Hloc * Cxy.t();
        
        
        //Rcpp::Rcout << "Hj: " << arma::size(Hj(ix)) << " Rj: " << arma::size(Rj(ix)) << endl;
        //Rcpp::Rcout << j << " " << firstcol << " " << lastcol << endl;
        //Rcpp::Rcout << "Hloc " << arma::size(Hloc) << endl;
        //Rcpp::Rcout << "j: " << j << " firstcol: " << firstcol << " lastcol: " << lastcol << " ix: " << ix << endl;
        //Rcpp::Rcout << arma::size(Hj) << endl;
        //Hj.subcube(j, firstcol, ix, j, lastcol, ix) = Hloc; //******** diag(1:2) %x% matrix(rnorm(2), ncol=2)
        Hj.subcube(j, 0, ix, j, Hj.n_cols-1, ix) = Hloc;
        //Rcpp::Rcout << "R " << arma::size(R) << endl;
        Rj(j, j, ix) = abs(R(0,0));
      }
    }
  }
}


