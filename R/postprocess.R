cross_covariance_matrix_function_h <- function(spmeshed_object, distance=0, correl=FALSE, num_threads=1){
  
  dd <- (ncol(meshed_out$coordsdata)-3)/2
  twonu_in <- spmeshed_object$savedata$matern_fix_twonu

  result <- with(spmeshed_object, crosscov_matfun_h(distance, 
                    lambda_mcmc, theta_mcmc, correl, 
                    num_threads, dd, twonu_in))
  
  return(result)
}

recover_W <- function(spmeshed_object){
  with(spmeshed_object, recover_W_cpp(savedata$v_mcmc, savedata$lambda_raw_mcmc, mcmc_ix))
}

recover_linear_predictor <- function(spmeshed_object){
  with(spmeshed_object, recover_linear_predictor_cpp(savedata$x[savedata$osix,], 
                                                     beta_mcmc, savedata$v_mcmc, savedata$lambda_raw_mcmc, mcmc_ix))
}