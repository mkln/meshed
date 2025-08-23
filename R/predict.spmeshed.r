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
  
  savedata <- object$savedata

  ord <- savedata$osix
  coordsdata <- object$coordsdata[, paste0("Var", 1:dd)]
  coords <- object$coordsdata %>% dplyr::select(!!!rlang::syms(cname)) %>% as.matrix()
  # fitting ordering for v
  v_mcmc <- savedata$v_mcmc[order(ord),,,drop=FALSE]
  
  all_coords <- coordsdata %>% 
    mutate(preds=0) %>%
    bind_rows(newcoords %>% 
                as.data.frame() %>% 
                mutate(preds=1)) %>%
    mutate(predix=1:n()) #%>%
    #dplyr::arrange(!!!rlang::syms(cname)) 
  
  fixed_thresholds <- savedata$fixed_thresholds

  # redo domain partitioning with the new coords using the same thresholds
  # this just assigns the new coords to the correct partition number
  suppressMessages(coords_blocking <- all_coords %>% 
    dplyr::select(!!!rlang::syms(cname)) %>%
                as.matrix() %>%
                meshed:::tessellation_axis_parallel_fix(fixed_thresholds, 1) %>%
    left_join(all_coords))
  
  pred_coords <- coords_blocking %>% dplyr::filter(.data$preds==1)
  
  
    
    
  # restore DAG
  parents                      <- savedata$parents
  children                     <- savedata$children
  block_names                  <- savedata$block_names
  block_groups                 <- savedata$block_groups
  
  mcmc_thin <- savedata$mcmc_thin
  mcmc_burn <- savedata$mcmc_burn
  mcmc_keep <- savedata$mcmc_keep
  
  thinned_mcmc <- object$mcmc_ix
  theta_mcmc <- savedata$theta_raw_mcmc[,,thinned_mcmc, drop=FALSE]
  lambda_mcmc <- savedata$lambda_raw_mcmc[,,thinned_mcmc, drop=FALSE]
  beta_mcmc <- object$beta_mcmc[,,thinned_mcmc, drop=FALSE]
  tausq_mcmc <- object$tausq_mcmc[,thinned_mcmc, drop=FALSE]
  
  twonu <- savedata$matern_fix_twonu
  use_ps <- savedata$use_ps
  
  returning <- spmeshed_predict(
            newx[order(pred_coords$predix),,drop=FALSE],
            pred_coords %>% dplyr::select(!!!rlang::syms(cname)) %>% as.matrix(), 
            pred_coords %>% dplyr::pull(.data$block), 
            coords, 
            parents,
            block_names, 
            savedata$indexing,
            
            v_mcmc,
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


