#ifndef RCPP_bmtruncn
#define RCPP_bmtruncn

#include "RcppArmadillo.h"

using namespace std;

inline arma::uvec usetdiff(const arma::uvec& x, const arma::uvec& y) {
  std::vector<int> a = arma::conv_to< std::vector<int> >::from(arma::sort(x));
  std::vector<int> b = arma::conv_to< std::vector<int> >::from(arma::sort(y));
  std::vector<int> out;
  
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::inserter(out, out.end()));
  return arma::conv_to<arma::uvec>::from(out);
}

inline arma::vec pnorm01_vec(const arma::vec& x, int lower=1, int logged=0){
  /*arma::vec p(x.n_elem);
   for(unsigned int i = 0; i<x.n_elem; i++){
   p(i) = R::pnorm(x(i), 0, 1, lower, logged);
   }
   return(p);*/
  Rcpp::NumericVector xn = Rcpp::wrap(x);
  return Rcpp::pnorm(xn, 0.0, 1.0, lower, logged);
}

inline arma::vec qnorm01_vec(const arma::vec& x, int lower=1, int logged=0){
  /*arma::vec q(x.n_elem);
   for(unsigned int i = 0; i<x.n_elem; i++){
   q(i) = R::qnorm(x(i), 0, 1, lower, logged);
   }*/
  Rcpp::NumericVector xn = Rcpp::wrap(x);
  return Rcpp::qnorm(xn, 0.0, 1.0);
}

inline arma::vec rndpp_runif(int n){
  Rcpp::RNGScope scope;
  return Rcpp::runif(n, 0, 1);
}

inline arma::vec lnNpr_cpp(const arma::vec& a, const arma::vec& b){
  arma::vec p = arma::zeros(a.n_elem);
  arma::uvec I = arma::find(a > 0);
  arma::uvec idx = arma::find(b < 0);
  int siz = a.n_elem;
  arma::uvec all_idx = arma::regspace<arma::uvec>(0, siz-1);
  arma::uvec I2 = usetdiff(usetdiff(all_idx, I), idx);
  //clog << I2 << endl;
  
  arma::vec pa;
  arma::vec pb; 
  if(I.n_elem > 0){
    pa = a(I);
    pb = b(I);
    pa = pnorm01_vec(pa, 0, 1);
    pb = pnorm01_vec(pb, 0, 1);
    arma::vec eba = -exp(pb-pa);
    p(I) = pa + log1p(eba);
  }
  if(idx.n_elem > 0){
    pa = a(idx);
    pb = b(idx);
    pa = pnorm01_vec(pa, 1, 1);
    pb = pnorm01_vec(pb, 1, 1);
    arma::vec eba = -exp(pa-pb);
    p(idx) = pb + log1p(eba);
  }
  if(I2.n_elem > 0){
    pa = a(I2);
    pb = b(I2);
    pa = pnorm01_vec(pa, 1, 0);
    pb = pnorm01_vec(pb, 0, 0);
    arma::vec eba = -pa-pb;
    p(I2) = log1p(eba);
  }
  return p;
}

inline arma::field<arma::mat> cholperm_cpp(arma::mat Sig, arma::vec l, arma::vec u){ 
  //arma::vec& l_out, arma::vec& u_out, arma::vec& perm_out){
  
  //clog << arma::size(Sig) << " " << arma::size(l) << " " << arma::size(u) << endl;
  arma::field<arma::mat> returning(4);
  
  double eps = pow(10, -10);
  int d = l.n_elem;
  arma::vec perm = arma::regspace<arma::vec>(0, d-1);
  arma::mat L = arma::zeros(d, d);
  arma::vec z = arma::zeros(d);
  //arma::mat Sigcopy = Sig;
  
  for(unsigned int j=0; j<d; j++){
    arma::vec pr(d);
    pr.fill(arma::datum::inf);
    
    arma::uvec I = arma::regspace<arma::uvec>(j, d-1);
    arma::uvec I_c = arma::regspace<arma::uvec>(0, j-1);
    arma::uvec zerocol = arma::zeros<arma::uvec>(1);
    arma::vec D = Sig.diag();
    
    arma:: vec s = D.elem(I);
    if(j > 1){
      s = s - pow(L.submat(I, I_c), 2) * arma::ones(j); // &
    } else {
      if(j == 1){
        s = s - pow(L.submat(I, zerocol), 2);
      }
    }
    s.elem(arma::find(s<0)).fill(eps);
    s = pow(s, 0.5);
    
    arma::vec cols = arma::zeros(I.n_elem);
    if(j > 1){
      cols = L.submat(I, I_c) * z.elem(I_c);
    } else {
      if(j == 1){
        cols = L.submat(I, zerocol) * z(0);
      }
    }
    
    arma::vec tl = (l.elem(I) - cols ) / s;
    arma::vec tu = (u.elem(I) - cols ) / s;
    
    pr.elem(I) = lnNpr_cpp(tl, tu);
    
    int k = pr.index_min();
    
    arma::uvec jk(2), kj(2);
    jk(0) = j;
    jk(1) = k;
    kj(0) = k;
    kj(1) = j;
    
    Sig.rows(jk) = Sig.rows(kj);
    Sig.cols(jk) = Sig.cols(kj);
    L.rows(jk) = L.rows(kj);
    l.elem(jk) = l.elem(kj);
    u.elem(jk) = u.elem(kj);
    perm.elem(jk) = perm.elem(kj);
    
    arma::uvec jj = arma::ones<arma::uvec>(1) * j;
    
    double s2 = Sig(j, j);
    if(j>0){
      s2 -=arma::accu(pow(L.submat(jj, I_c), 2));
    }
    if(s2<0){
      s2 = eps;
    }
    L(j,j) = pow(s2, 0.5);
    
    arma::uvec sel = arma::regspace<arma::uvec>(j+1, d-1);
    arma::uvec sel_c = arma::regspace<arma::uvec>(0, j-1);
    
    if(j < d-1){
      if(j > 1){
        L.submat(sel, jj) = (Sig.submat(sel, jj) - L.submat(sel, I_c) * L.submat(jj, I_c).t()) / L(j,j);
      } else {
        if(j == 1){
          // L[(j + 1):d, j] = (Sig[(j + 1):d, j] - L[(j + 1):d, 1] * L[j, 1]) / L[j, j]
          L.submat(sel, jj) = (Sig.submat(sel, jj) - L.submat(sel, zerocol) * L(j, 0)) / L(j,j);
        } else {
          if(j == 0){
            //L[(j + 1):d, j] = Sig[(j + 1):d, j] / L[j, j]
            L.submat(sel, jj) = Sig.submat(sel, jj) / L(j,j);
          }
        }
      }
    }
    
    arma::uvec sel_n = arma::regspace<arma::uvec>(0, j);
    arma::vec tl2 = (l(j) - L.submat(jj, sel_n) * z.elem(sel_n)) / L(j,j);
    arma::vec tu2 = (u(j) - L.submat(jj, sel_n) * z.elem(sel_n)) / L(j,j);
    
    arma::vec w = lnNpr_cpp(tl2, tu2);
    z(j) = arma::conv_to<double>::from((exp(-.5 * pow(tl2, 2) - w) - 
      exp(-.5 * pow(tu2, 2) - w)) / pow(2*M_PI, .5));
  }
  
  //l_out = l;
  //u_out = u;
  //perm_out = perm;
  returning(0) = L;
  returning(1) = l;
  returning(2) = u;
  returning(3) = perm;
  return returning;
}

inline arma::mat gradpsi_cpp(const arma::vec& y, const arma::mat& L, 
                             const arma::vec& l, const arma::vec& u, arma::vec& grad){
  int d = u.n_elem;
  arma::vec c = arma::zeros(d);
  
  arma::vec x = c;
  arma::vec mu = c;
  
  arma::uvec sel = arma::regspace<arma::uvec>(0, d-2);
  x.elem(sel) = y.elem(sel);
  mu.elem(sel) = y.elem(sel+d-1);
  
  arma::uvec no1 = arma::regspace<arma::uvec>(1, d-1);
  arma::uvec nod = arma::regspace<arma::uvec>(0, d-2);
  
  c.elem(no1) = L.rows(no1) * x;
  
  arma::vec lt = l - mu - c;
  arma::vec ut = u - mu - c;
  
  arma::vec w = lnNpr_cpp(lt, ut);
  
  arma::vec pl = exp(-.5 * pow(lt, 2) - w) / pow(2*M_PI, .5);
  arma::vec pu = exp(-.5 * pow(ut, 2) - w) / pow(2*M_PI, .5);
  
  arma::vec P = pl - pu;
  arma::vec dfdx = -mu.elem(nod) + (P.t() * L.cols(nod)).t();
  arma::vec dfdm = mu - x + P;
  grad = arma::join_vert(dfdx, dfdm.elem(nod));
  lt.elem(arma::find_nonfinite(lt)).fill(0.0);
  ut.elem(arma::find_nonfinite(ut)).fill(0.0);
  
  arma::vec dP = - pow(P, 2) + lt % pl - ut % pu;
  arma::mat DL = arma::repmat(dP.t(), L.n_rows, 1) % L;
  
  arma::mat mx = - arma::eye(d, d) + DL;
  arma::mat xx = L.t() * DL;
  mx = mx.submat(sel, sel);
  xx = xx.submat(sel, sel);
  
  arma::mat Jac;
  if(d > 2){
    Jac = arma::join_vert( arma::join_horiz(xx, mx.t()),
                           arma::join_horiz(mx, arma::diagmat(1 + dP.elem(sel))) );
  } else {
    Jac = arma::join_vert( arma::join_horiz(xx, mx.t()),
                           arma::join_horiz(mx, 1 + dP.elem(sel)) );
  }
  
  return Jac;
}


inline arma::vec armasolve(const arma::mat& A, 
                           arma::vec grad){
  return arma::solve(A, -grad);
}

inline arma::vec nleq(const arma::vec& l, const arma::vec& u, const arma::mat& L){
  int d = l.n_elem;
  arma::vec x = arma::zeros(2*d-2);
  double err = arma::datum::inf;
  int iter = 0;
  double eps = pow(10, -10);
  while(err > eps){
    arma::vec grad;
    arma::mat Jac = gradpsi_cpp(x, L, l, u, grad);
    arma::vec del = arma::solve(Jac, -grad);
    x = x + del;
    err = arma::accu(pow(grad, 2));
    iter ++;
    if(iter > 100){
      break;
    }
  }
  return x;
}

inline arma::vec ntail_cpp(const arma::vec& l, const arma::vec& u){
  arma::vec c = pow(l, 2)/2.0;
  int n = l.n_elem;
  arma::vec f = exp(c - pow(u, 2)/2.0) - 1.0;
  arma::vec x = c - log1p(rndpp_runif(n) % f);
  arma::uvec I = arma::find(pow(rndpp_runif(n), 2) % x > c);
  int d = I.n_elem;
  while(d > 0){
    arma::vec cy = c.elem(I);
    arma::vec y = cy - log1p(rndpp_runif(d) % f.elem(I));
    arma::vec ruextract2 = pow(rndpp_runif(d),2);
    arma::uvec idx = arma::find(ruextract2 % y < cy);
    arma::uvec idx_c = arma::find(ruextract2 % y >= cy);
    x.elem(I.elem(idx)) = y.elem(idx);
    I = I.elem(idx_c);
    d = I.n_elem;
  }
  return pow(x*2, .5);
}


inline arma::vec trnd_cpp(const arma::vec& l, const arma::vec& u){
  arma::vec x = Rcpp::rnorm(l.n_elem, 0, 1);
  arma::uvec I = arma::find((x < l) + (x > u));
  int d = I.n_elem;
  while(d > 0){
    arma::vec ly = l.elem(I);
    arma::vec uy = u.elem(I);
    arma::vec y = Rcpp::rnorm(ly.n_elem, 0, 1);
    arma::uvec idx = arma::find((y > ly)%(y < uy));
    arma::uvec idx_c = arma::find(1-(y > ly)%(y < uy));
    x.elem(I.elem(idx)) = y.elem(idx);
    I = I.elem(idx_c);
    d = I.n_elem;
  }
  return x;
}

inline arma::vec tn_cpp(const arma::vec& l, const arma::vec& u){
  double tol = 2.05;
  arma::vec x = l;
  arma::uvec I = arma::find(abs(u-l) > tol);
  if(I.n_elem > 0){
    arma::vec tl = l.elem(I);
    arma::vec tu = u.elem(I);
    x.elem(I) = trnd_cpp(tl, tu);
  }
  arma::uvec I_c = arma::find(abs(u-l) <= tol);
  if(I_c.n_elem > 0){
    arma::vec tl = l.elem(I_c);
    arma::vec tu = u.elem(I_c);
    arma::vec pl = pnorm01_vec(tl);
    arma::vec pu = pnorm01_vec(tu);
    x.elem(I_c) = qnorm01_vec(pl + (pu - pl) % rndpp_runif(tl.n_elem));
  }
  return x;
}


inline arma::vec trandn_cpp(const arma::vec& l, const arma::vec& u){
  // this will NOT check l>u
  
  arma::vec x = arma::zeros(l.n_elem);
  double a = .4;
  arma::uvec I = arma::find(l > a);
  if(I.n_elem > 0){
    arma::vec tl = l.elem(I);
    arma::vec tu = u.elem(I);
    x.elem(I) = ntail_cpp(tl, tu);
  }
  arma::uvec J = arma::find(u < (-a));
  if(J.n_elem > 0){
    arma::vec tl = -u.elem(J);
    arma::vec tu = -l.elem(J);
    x.elem(J) = -ntail_cpp(tl, tu);
  }
  arma::uvec IJ_c = arma::find((l <= a) % (u >= (-a)));
  if(IJ_c.n_elem > 0){
    arma::vec tl = l.elem(IJ_c);
    arma::vec tu = u.elem(IJ_c);
    x.elem(IJ_c) = tn_cpp(tl, tu);
  }
  return x;
}


inline arma::mat mvnrnd_cpp(int n, const arma::mat& L, 
                            const arma::vec& l, const arma::vec& u, 
                            arma::vec mu, arma::vec& logpr){
  int d = l.n_elem;
  mu = arma::join_vert(mu, arma::zeros(1,1));
  arma::mat Z = arma::zeros(d, n);
  arma::vec p = arma::zeros(n);
  arma::vec col = arma::zeros(n, 1);
  arma::vec tl = arma::zeros(n, 1);
  arma::vec tu = arma::zeros(n, 1);
  for(int k = 0; k<d; k++){
    col = (L.submat(k, 0, k, k) * Z.rows(0, k)).t();
    tl = l(k) - mu(k) - col;
    tu = u(k) - mu(k) - col;
    Z.row(k) = (mu(k) + trandn_cpp(tl, tu)).t();
    p = p + lnNpr_cpp(tl, tu) + .5*pow(mu(k), 2) - mu(k) * Z.row(k).t();
  }
  logpr = p;
  return Z;
}


inline double psy_cpp(arma::vec x, const arma::mat& L, 
                      arma::vec l, arma::vec u, arma::vec mu){
  //int d = u.n_elem;
  x = arma::join_vert(x, arma::zeros(1,1));
  mu = arma::join_vert(mu, arma::zeros(1,1));
  
  arma::vec c = L * x;
  l = l - mu - c;
  u = u - mu - c;
  return arma::accu(lnNpr_cpp(l, u) + .5 * pow(mu, 2) - x % mu);
}


inline arma::mat mvrandn_cpp(const arma::vec& l_in, const arma::vec& u_in, 
                             const arma::mat& Sig, int n){
  int d = l_in.n_elem;
  int accept=0;
  // no checks
  
  arma::vec l, u, perm;
  arma::field<arma::mat> chperm_results = cholperm_cpp(Sig, l_in, u_in); //, l, u, perm);
  //clog << "getting after " << endl;
  arma::mat Lfull = chperm_results(0);
  l = chperm_results(1).col(0);
  u = chperm_results(2).col(0);
  perm = chperm_results(3).col(0);
  
  arma::vec D = Lfull.diag();
  arma::mat Z = arma::zeros(d, n);
  arma::vec logpr;
  //
  if(abs(D).min() < pow(10, -10)){
    clog << "Method may fail as covariance matrix is singular!" << endl;
  }
  arma::mat Dm = arma::repmat(D, 1, d);
  arma::mat L = Lfull / Dm;
  u = u / D;
  l = l / D;
  L = L - arma::eye(d,d);
  //clog << "here " << L << endl;
  arma::vec xmu = nleq(l, u, L);
  //clog << "here 2" << endl;
  //clog << xmu << endl;
  
  arma::vec x = xmu.subvec(0, d-2);
  arma::vec mu = xmu.subvec(d-1, 2*d-3);
  
  double psistar = psy_cpp(x, L, l, u, mu);
  int iter = 0;
  arma::mat rv = arma::zeros(d,0);
  
  do {
    Z = mvnrnd_cpp(n, L, l, u, mu, logpr);
    arma::uvec idx = arma::find(-log(rndpp_runif(n)) > (psistar-logpr));
    rv = arma::join_horiz(rv, Z.cols(idx));
    accept = rv.n_cols;
    iter++;
    
    if(iter == 1000){
      clog << "Acceptance prob. smaller than 0.001" << endl;
    } else { 
      if(iter > 10000){
        accept = n;
        rv = arma::join_horiz(rv, Z);
        clog << "Sample is only approximately distributed" << endl;
      }
    }
  } while (accept < n);
  arma::uvec order = arma::sort_index(perm);
  rv = rv.cols(0, n-1);
  rv = Lfull * rv;
  rv = rv.rows(order);
  return rv;
}

inline arma::mat mvtruncnormal(const arma::vec& mean, 
                        const arma::vec& l_in, const arma::vec& u_in, 
                        const arma::mat& Sig, int n){
  if(mean.n_elem > 1){
    arma::mat meanmat = arma::zeros(mean.n_elem, n);
    arma::mat truncraw = mvrandn_cpp(l_in-mean, u_in-mean, Sig, n);
    for(unsigned int i=0; i<n; i++){
      meanmat.col(i) = mean + truncraw.col(i);
    }
    return meanmat;
  } else {
    arma::mat result = arma::zeros(1, n);
    double sig = sqrt(Sig(0,0));
    arma::vec lbar = (l_in - mean)/sig;
    arma::vec ncdf = pnorm01_vec(lbar);
    
    for(unsigned int i=0; i<n; i++){
      double u = R::runif(ncdf(0), 1);
      arma::vec uv = arma::ones(1) * u;
      arma::vec xbar = qnorm01_vec(uv);
      result.col(i) = sig * xbar + mean;
    }
    return result;
  }
}


#endif
