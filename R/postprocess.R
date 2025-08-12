cross_covariance_matrix_function_h <- function(spmeshed_object, distance=0, correl=FALSE, num_threads=1){
  
  dd <- sum(grepl("Var", colnames(spmeshed_object$savedata$coords_blocking)))
  twonu_in <- spmeshed_object$savedata$matern_fix_twonu
  
  result <- with(spmeshed_object, crosscov_matfun_h(distance, 
                    lambda_mcmc, theta_mcmc, correl, 
                    num_threads, dd, twonu_in))
  
  return(result)
}