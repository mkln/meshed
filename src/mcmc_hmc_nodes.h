#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>
//#include "tmesh_utils.h"
#include "distrib_densities_gradients.h"

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
  
  unsigned int num_children;
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
  
  // **
  
  
  double fwdcond_dmvn(const arma::mat& x, 
                      const arma::cube* Ri,
                      const arma::mat& Kcxpar);
  arma::vec grad_fwdcond_dmvn(const arma::mat& x);
  
  void fwdconditional_mvn(double& logtarget, arma::vec& gradient, 
                          const arma::mat& x);
  
  double bwdcond_dmvn(const arma::mat& x, int c);
  
  arma::vec grad_bwdcond_dmvn(const arma::mat& x, int c);
  void bwdconditional_mvn(double& xtarget, arma::vec& gradient, const arma::mat& x, int c);
  
  void neghess_fwdcond_dmvn(arma::mat& result, const arma::mat& x);
  void neghess_bwdcond_dmvn(arma::mat& result, const arma::mat& x, int c);
  void mvn_dens_grad_neghess(double& xtarget, arma::vec& gradient, arma::mat& neghess,
                             const arma::mat& x, int c);
  
  // **
  double logfullcondit(const arma::mat& x);
  double loglike(const arma::mat& x);
  
  arma::vec gradient_logfullcondit(const arma::mat& x);
  arma::mat neghess_logfullcondit(const arma::mat& x);
  arma::mat neghess_prior(const arma::mat& x);
  
  arma::vec compute_dens_and_grad(double& xtarget, const arma::mat& x);
  arma::mat compute_dens_grad_neghess(double& xtarget, arma::vec& xgrad, const arma::mat& x);
  arma::mat compute_dens_grad_neghess2(double& xtarget, arma::vec& xgrad, const arma::mat& x);
  
  NodeDataW(const arma::mat& y_all, //const arma::mat& Z_in,
            const arma::umat& na_mat_all, const arma::mat& offset_all, 
            const arma::uvec& indexing_target,
            const arma::uvec& outtype, int k, 
            bool fgrid_in);
  
  NodeDataW();
  
};


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
  void set_XtDX(const arma::vec& x);
  
  NodeDataB(const arma::vec& y_in, const arma::vec& offset_in, 
            const arma::mat& X_in, int family_in);
  NodeDataB();
  
  double logfullcondit(const arma::vec& x);
  
  arma::vec gradient_logfullcondit(const arma::vec& x);
  arma::mat neghess_logfullcondit(const arma::vec& x);
  arma::vec compute_dens_and_grad(double& xtarget, const arma::mat& x);
  arma::mat compute_dens_grad_neghess(double& xtarget, arma::vec& xgrad, const arma::mat& x);
  //using MVDistParams::MVDistParams;
  //using MVDistParams::logfullcondit;
  //using MVDistParams::gradient_logfullcondit;
};
