// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// Cov_matern
arma::mat Cov_matern(const arma::mat& x, const arma::mat& y, const double& sigmasq, const double& phi, const double& nu, const double& tausq, bool same, int nThreads);
RcppExport SEXP _spmeshed_Cov_matern(SEXP xSEXP, SEXP ySEXP, SEXP sigmasqSEXP, SEXP phiSEXP, SEXP nuSEXP, SEXP tausqSEXP, SEXP sameSEXP, SEXP nThreadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const double& >::type sigmasq(sigmasqSEXP);
    Rcpp::traits::input_parameter< const double& >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< const double& >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< const double& >::type tausq(tausqSEXP);
    Rcpp::traits::input_parameter< bool >::type same(sameSEXP);
    Rcpp::traits::input_parameter< int >::type nThreads(nThreadsSEXP);
    rcpp_result_gen = Rcpp::wrap(Cov_matern(x, y, sigmasq, phi, nu, tausq, same, nThreads));
    return rcpp_result_gen;
END_RCPP
}
// Cov_matern2
arma::mat Cov_matern2(const arma::mat& x, const arma::mat& y, const double& phi, bool same, int twonu);
RcppExport SEXP _spmeshed_Cov_matern2(SEXP xSEXP, SEXP ySEXP, SEXP phiSEXP, SEXP sameSEXP, SEXP twonuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const double& >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< bool >::type same(sameSEXP);
    Rcpp::traits::input_parameter< int >::type twonu(twonuSEXP);
    rcpp_result_gen = Rcpp::wrap(Cov_matern2(x, y, phi, same, twonu));
    return rcpp_result_gen;
END_RCPP
}
// Cov_matern_h
double Cov_matern_h(const double& h, const double& sigmasq, const double& phi, const double& nu, const double& tausq);
RcppExport SEXP _spmeshed_Cov_matern_h(SEXP hSEXP, SEXP sigmasqSEXP, SEXP phiSEXP, SEXP nuSEXP, SEXP tausqSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const double& >::type h(hSEXP);
    Rcpp::traits::input_parameter< const double& >::type sigmasq(sigmasqSEXP);
    Rcpp::traits::input_parameter< const double& >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< const double& >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< const double& >::type tausq(tausqSEXP);
    rcpp_result_gen = Rcpp::wrap(Cov_matern_h(h, sigmasq, phi, nu, tausq));
    return rcpp_result_gen;
END_RCPP
}
// Cov_powexp_h
double Cov_powexp_h(const double& h, const double& sigmasq, const double& phi, const double& nu, const double& tausq);
RcppExport SEXP _spmeshed_Cov_powexp_h(SEXP hSEXP, SEXP sigmasqSEXP, SEXP phiSEXP, SEXP nuSEXP, SEXP tausqSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const double& >::type h(hSEXP);
    Rcpp::traits::input_parameter< const double& >::type sigmasq(sigmasqSEXP);
    Rcpp::traits::input_parameter< const double& >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< const double& >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< const double& >::type tausq(tausqSEXP);
    rcpp_result_gen = Rcpp::wrap(Cov_powexp_h(h, sigmasq, phi, nu, tausq));
    return rcpp_result_gen;
END_RCPP
}
// blanket
arma::field<arma::uvec> blanket(const arma::field<arma::uvec>& parents, const arma::field<arma::uvec>& children, const arma::uvec& names, const arma::uvec& block_ct_obs);
RcppExport SEXP _spmeshed_blanket(SEXP parentsSEXP, SEXP childrenSEXP, SEXP namesSEXP, SEXP block_ct_obsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type parents(parentsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type children(childrenSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type names(namesSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type block_ct_obs(block_ct_obsSEXP);
    rcpp_result_gen = Rcpp::wrap(blanket(parents, children, names, block_ct_obs));
    return rcpp_result_gen;
END_RCPP
}
// coloring
arma::ivec coloring(const arma::field<arma::uvec>& blanket, const arma::uvec& block_names, const arma::uvec& block_ct_obs);
RcppExport SEXP _spmeshed_coloring(SEXP blanketSEXP, SEXP block_namesSEXP, SEXP block_ct_obsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type blanket(blanketSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type block_names(block_namesSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type block_ct_obs(block_ct_obsSEXP);
    rcpp_result_gen = Rcpp::wrap(coloring(blanket, block_names, block_ct_obs));
    return rcpp_result_gen;
END_RCPP
}
// kthresholdscp
arma::vec kthresholdscp(arma::vec x, int k);
RcppExport SEXP _spmeshed_kthresholdscp(SEXP xSEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(kthresholdscp(x, k));
    return rcpp_result_gen;
END_RCPP
}
// part_axis_parallel
arma::mat part_axis_parallel(const arma::mat& coords, const arma::vec& Mv, int n_threads, bool verbose);
RcppExport SEXP _spmeshed_part_axis_parallel(SEXP coordsSEXP, SEXP MvSEXP, SEXP n_threadsSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Mv(MvSEXP);
    Rcpp::traits::input_parameter< int >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(part_axis_parallel(coords, Mv, n_threads, verbose));
    return rcpp_result_gen;
END_RCPP
}
// part_axis_parallel_fixed
arma::mat part_axis_parallel_fixed(const arma::mat& coords, const arma::field<arma::vec>& thresholds, int n_threads);
RcppExport SEXP _spmeshed_part_axis_parallel_fixed(SEXP coordsSEXP, SEXP thresholdsSEXP, SEXP n_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::vec>& >::type thresholds(thresholdsSEXP);
    Rcpp::traits::input_parameter< int >::type n_threads(n_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(part_axis_parallel_fixed(coords, thresholds, n_threads));
    return rcpp_result_gen;
END_RCPP
}
// mesh_graph_cpp
Rcpp::List mesh_graph_cpp(const arma::mat& layers_descr, const arma::uvec& Mv, bool verbose, bool both_spatial_axes);
RcppExport SEXP _spmeshed_mesh_graph_cpp(SEXP layers_descrSEXP, SEXP MvSEXP, SEXP verboseSEXP, SEXP both_spatial_axesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type layers_descr(layers_descrSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type Mv(MvSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type both_spatial_axes(both_spatial_axesSEXP);
    rcpp_result_gen = Rcpp::wrap(mesh_graph_cpp(layers_descr, Mv, verbose, both_spatial_axes));
    return rcpp_result_gen;
END_RCPP
}
// knn_naive
arma::umat knn_naive(const arma::mat& x, const arma::mat& search_here, const arma::vec& weights);
RcppExport SEXP _spmeshed_knn_naive(SEXP xSEXP, SEXP search_hereSEXP, SEXP weightsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type search_here(search_hereSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type weights(weightsSEXP);
    rcpp_result_gen = Rcpp::wrap(knn_naive(x, search_here, weights));
    return rcpp_result_gen;
END_RCPP
}
// mesh_graph_hyper
Rcpp::List mesh_graph_hyper(const arma::umat& bucbl, const arma::umat& bavail, const arma::vec& na_which, const arma::mat& centroids, const arma::mat& avcentroids, const arma::uvec& avblocks, int k);
RcppExport SEXP _spmeshed_mesh_graph_hyper(SEXP bucblSEXP, SEXP bavailSEXP, SEXP na_whichSEXP, SEXP centroidsSEXP, SEXP avcentroidsSEXP, SEXP avblocksSEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::umat& >::type bucbl(bucblSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type bavail(bavailSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type na_which(na_whichSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type centroids(centroidsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type avcentroids(avcentroidsSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type avblocks(avblocksSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(mesh_graph_hyper(bucbl, bavail, na_which, centroids, avcentroids, avblocks, k));
    return rcpp_result_gen;
END_RCPP
}
// meshed_mcmc
Rcpp::List meshed_mcmc(const arma::mat& y, const arma::uvec& family, const arma::mat& X, const arma::mat& coords, int k, const arma::field<arma::uvec>& parents, const arma::field<arma::uvec>& children, const arma::vec& layer_names, const arma::vec& layer_gibbs_group, const arma::field<arma::uvec>& indexing, const arma::field<arma::uvec>& indexing_obs, const arma::mat& set_unif_bounds_in, const arma::mat& beta_Vi, const arma::vec& sigmasq_ab, const arma::vec& tausq_ab, int matern_twonu, const arma::mat& start_w, const arma::mat& lambda, const arma::umat& lambda_mask, const arma::mat& theta, const arma::mat& beta, const arma::vec& tausq, const arma::mat& mcmcsd, int mcmc_keep, int mcmc_burn, int mcmc_thin, int mcmc_startfrom, int num_threads, bool adapting, bool use_cache, bool forced_grid, bool use_ps, bool verbose, bool debug, int print_every, bool sample_beta, bool sample_tausq, bool sample_lambda, bool sample_theta, bool sample_w);
RcppExport SEXP _spmeshed_meshed_mcmc(SEXP ySEXP, SEXP familySEXP, SEXP XSEXP, SEXP coordsSEXP, SEXP kSEXP, SEXP parentsSEXP, SEXP childrenSEXP, SEXP layer_namesSEXP, SEXP layer_gibbs_groupSEXP, SEXP indexingSEXP, SEXP indexing_obsSEXP, SEXP set_unif_bounds_inSEXP, SEXP beta_ViSEXP, SEXP sigmasq_abSEXP, SEXP tausq_abSEXP, SEXP matern_twonuSEXP, SEXP start_wSEXP, SEXP lambdaSEXP, SEXP lambda_maskSEXP, SEXP thetaSEXP, SEXP betaSEXP, SEXP tausqSEXP, SEXP mcmcsdSEXP, SEXP mcmc_keepSEXP, SEXP mcmc_burnSEXP, SEXP mcmc_thinSEXP, SEXP mcmc_startfromSEXP, SEXP num_threadsSEXP, SEXP adaptingSEXP, SEXP use_cacheSEXP, SEXP forced_gridSEXP, SEXP use_psSEXP, SEXP verboseSEXP, SEXP debugSEXP, SEXP print_everySEXP, SEXP sample_betaSEXP, SEXP sample_tausqSEXP, SEXP sample_lambdaSEXP, SEXP sample_thetaSEXP, SEXP sample_wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type family(familySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type parents(parentsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type children(childrenSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type layer_names(layer_namesSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type layer_gibbs_group(layer_gibbs_groupSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type indexing(indexingSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type indexing_obs(indexing_obsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type set_unif_bounds_in(set_unif_bounds_inSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_Vi(beta_ViSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type sigmasq_ab(sigmasq_abSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type tausq_ab(tausq_abSEXP);
    Rcpp::traits::input_parameter< int >::type matern_twonu(matern_twonuSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type start_w(start_wSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type lambda_mask(lambda_maskSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type tausq(tausqSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type mcmcsd(mcmcsdSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc_keep(mcmc_keepSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc_burn(mcmc_burnSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc_thin(mcmc_thinSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc_startfrom(mcmc_startfromSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< bool >::type adapting(adaptingSEXP);
    Rcpp::traits::input_parameter< bool >::type use_cache(use_cacheSEXP);
    Rcpp::traits::input_parameter< bool >::type forced_grid(forced_gridSEXP);
    Rcpp::traits::input_parameter< bool >::type use_ps(use_psSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type debug(debugSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    Rcpp::traits::input_parameter< bool >::type sample_beta(sample_betaSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_tausq(sample_tausqSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_lambda(sample_lambdaSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_theta(sample_thetaSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_w(sample_wSEXP);
    rcpp_result_gen = Rcpp::wrap(meshed_mcmc(y, family, X, coords, k, parents, children, layer_names, layer_gibbs_group, indexing, indexing_obs, set_unif_bounds_in, beta_Vi, sigmasq_ab, tausq_ab, matern_twonu, start_w, lambda, lambda_mask, theta, beta, tausq, mcmcsd, mcmc_keep, mcmc_burn, mcmc_thin, mcmc_startfrom, num_threads, adapting, use_cache, forced_grid, use_ps, verbose, debug, print_every, sample_beta, sample_tausq, sample_lambda, sample_theta, sample_w));
    return rcpp_result_gen;
END_RCPP
}
// cube_tcrossprod
arma::cube cube_tcrossprod(const arma::cube& x);
RcppExport SEXP _spmeshed_cube_tcrossprod(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cube& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(cube_tcrossprod(x));
    return rcpp_result_gen;
END_RCPP
}
// summary_list_mean
arma::mat summary_list_mean(const arma::field<arma::mat>& x, int num_threads);
RcppExport SEXP _spmeshed_summary_list_mean(SEXP xSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::mat>& >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(summary_list_mean(x, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// summary_list_q
arma::mat summary_list_q(const arma::field<arma::mat>& x, double q, int num_threads);
RcppExport SEXP _spmeshed_summary_list_q(SEXP xSEXP, SEXP qSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::mat>& >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type q(qSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(summary_list_q(x, q, num_threads));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_spmeshed_Cov_matern", (DL_FUNC) &_spmeshed_Cov_matern, 8},
    {"_spmeshed_Cov_matern2", (DL_FUNC) &_spmeshed_Cov_matern2, 5},
    {"_spmeshed_Cov_matern_h", (DL_FUNC) &_spmeshed_Cov_matern_h, 5},
    {"_spmeshed_Cov_powexp_h", (DL_FUNC) &_spmeshed_Cov_powexp_h, 5},
    {"_spmeshed_blanket", (DL_FUNC) &_spmeshed_blanket, 4},
    {"_spmeshed_coloring", (DL_FUNC) &_spmeshed_coloring, 3},
    {"_spmeshed_kthresholdscp", (DL_FUNC) &_spmeshed_kthresholdscp, 2},
    {"_spmeshed_part_axis_parallel", (DL_FUNC) &_spmeshed_part_axis_parallel, 4},
    {"_spmeshed_part_axis_parallel_fixed", (DL_FUNC) &_spmeshed_part_axis_parallel_fixed, 3},
    {"_spmeshed_mesh_graph_cpp", (DL_FUNC) &_spmeshed_mesh_graph_cpp, 4},
    {"_spmeshed_knn_naive", (DL_FUNC) &_spmeshed_knn_naive, 3},
    {"_spmeshed_mesh_graph_hyper", (DL_FUNC) &_spmeshed_mesh_graph_hyper, 7},
    {"_spmeshed_meshed_mcmc", (DL_FUNC) &_spmeshed_meshed_mcmc, 40},
    {"_spmeshed_cube_tcrossprod", (DL_FUNC) &_spmeshed_cube_tcrossprod, 1},
    {"_spmeshed_summary_list_mean", (DL_FUNC) &_spmeshed_summary_list_mean, 2},
    {"_spmeshed_summary_list_q", (DL_FUNC) &_spmeshed_summary_list_q, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_spmeshed(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
