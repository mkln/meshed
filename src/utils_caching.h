#include <RcppArmadillo.h>

using namespace std;

arma::vec caching_pairwise_compare_u(const arma::field<arma::mat>& blocks,
                                     const arma::vec& names,
                                     const arma::vec& block_ct_obs);


arma::uvec caching_pairwise_compare_uc(const arma::field<arma::mat>& blocks,
                                       const arma::vec& names,
                                       const arma::vec& ct_obs, bool cached);


arma::uvec caching_pairwise_compare_uci(const arma::mat& coords,
                                        const arma::field<arma::uvec>& indexing,
                                        const arma::vec& names,
                                        const arma::vec& ct_obs, bool cached);

arma::uvec caching_pairwise_compare_m(const arma::field<arma::mat>& blocks,
                                       const arma::vec& ct_obs, bool cached);

arma::uvec caching_pairwise_compare_mi(const arma::mat& coords,
                                        const arma::field<arma::uvec>& indexing,
                                        const arma::vec& ct_obs, bool cached);