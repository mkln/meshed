predict.spmeshed <- function(object,
                             newx,
                             newcoords, 
                             n_threads=4,
                             verbose=FALSE, ...){
  
  if(is.null(object$savedata)){
    stop("Data not saved from spmeshed output.")
  }
  if(any(object$savedata$family != "gaussian")){
    stop("Currently not implemented. Insert prediction locations into main spmeshed functions.")
  }
  
  if(object$success == FALSE){
    warning("MCMC was unsuccessful, predictions likely invalid.")
  }
  
  dd <- ncol(newcoords)
  colnames(newcoords) <- cname <- paste0("Var", 1:dd)
  
  all_coords <- object$coordsdata %>% 
    mutate(preds=0) %>%
    bind_rows(newcoords %>% 
                as.data.frame() %>% 
                mutate(forced_grid = 0, preds=1)) %>%
    mutate(predix=1:n()) #%>%
    #dplyr::arrange(!!!rlang::syms(cname)) 
  
  fixed_thresholds <- object$savedata$fixed_thresholds

  # redo domain partitioning with the new coords using the same thresholds
  # this just assigns the new coords to the correct partition number
  suppressMessages(coords_blocking <- all_coords %>% 
    dplyr::select(!!!rlang::syms(cname)) %>%
                as.matrix() %>%
                tessellation_axis_parallel_fix(fixed_thresholds, 1) %>%
    left_join(all_coords))
  
  coords <- object$coordsdata %>% dplyr::select(!!!rlang::syms(cname)) %>% as.matrix()
    
    
  # restore DAG
  parents                      <- object$savedata$parents
  children                     <- object$savedata$children
  block_names                  <- object$savedata$block_names
  block_groups                 <- object$savedata$block_groups
  
  mcmc_thin <- object$savedata$mcmc_thin
  mcmc_burn <- object$savedata$mcmc_burn
  mcmc_keep <- object$savedata$mcmc_keep
  
  thinned_mcmc <- seq(1, dim(object$theta_mcmc)[3], mcmc_thin)
  theta_mcmc <- object$theta_mcmc[,,thinned_mcmc, drop=FALSE]
  lambda_mcmc <- object$lambda_mcmc[,,thinned_mcmc, drop=FALSE]
  beta_mcmc <- object$beta_mcmc[,,thinned_mcmc, drop=FALSE]
  tausq_mcmc <- object$tausq_mcmc[,thinned_mcmc, drop=FALSE]
  
  indexing_grid <- object$savedata$indexing_grid
  indexing_obs <- object$savedata$indexing_obs
  use_forced_grid <- object$savedata$use_forced_grid

  pred_coords <- coords_blocking %>% dplyr::filter(.data$preds==1) %>% dplyr::select(-.data$forced_grid)
  
  twonu <- object$savedata$matern_fix_twonu
  use_ps <- object$savedata$use_ps
  
  returning <- spmeshed_predict(
            newx[order(pred_coords$predix),,drop=FALSE],
            pred_coords %>% dplyr::select(!!!rlang::syms(cname)) %>% as.matrix(), 
            pred_coords %>% dplyr::pull(.data$block), 
            coords, 
            parents,
            block_names, 
            indexing_grid,
            
            object$v_mcmc,
            theta_mcmc, 
            lambda_mcmc, 
            beta_mcmc,
            tausq_mcmc,
            twonu, 
            use_ps,
            verbose,
            n_threads)
  
  colnames(returning$coords_out) <- cname
  returning$coords_out %<>% as.data.frame()
  
  return(returning) 
  
}


