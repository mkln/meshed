gramar <- function(y, x, z, k=NULL, 
                       family = "gaussian",
                       block_size = 30,
                        proj_method = "lapeig",
                       n_samples = 1000,
                       n_burnin = 100,
                       n_thin = 1,
                       n_threads = 4,
                       verbose = 0,
                       settings    = list(adapting=TRUE, ps=TRUE, 
                                          scale_coords=TRUE,
                                          cache=FALSE, saving=FALSE),
                       prior       = list(beta=NULL, tausq=NULL, sigmasq = NULL,
                                          toplim = NULL, btmlim = NULL, set_unif_bounds=NULL),
                       starting    = list(beta=NULL, tausq=NULL, theta=NULL, lambda=NULL, v=NULL, 
                                          mcmcsd=.05, mcmc_startfrom=0),
                       debug       = list(sample_beta=TRUE, sample_tausq=TRUE, 
                                          sample_theta=TRUE, sample_w=TRUE, sample_lambda=TRUE,
                                          verbose=FALSE, debug=FALSE),
                       indpart = FALSE
){
  
  
  if(verbose > 0){
    cat("Bayesian Graph Machine regression model fit via Markov chain Monte Carlo\n")
  }
  # init
  model_tag <- "Bayesian Graph Machine regression\n
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o\n\n"
  
  set_default <- function(x, default_val){
    return(if(is.null(x)){
      default_val} else {
        x
      })}
  
  # data management pt 1
  if(1){
    mcmc_keep <- n_samples
    mcmc_burn <- n_burnin
    mcmc_thin <- n_thin
    
    mcmc_adaptive    <- settings$adapting %>% set_default(TRUE)
    mcmc_verbose     <- debug$verbose %>% set_default(FALSE)
    mcmc_debug       <- debug$debug %>% set_default(FALSE)
    saving <- settings$saving %>% set_default(FALSE)
    use_ps <- settings$ps %>% set_default(TRUE)
    which_hmc <- 4
    low_mem <- FALSE
    proj_dim <- 2
    #proj_method <- "lapeig"
    
    # data management part 0 - reshape/rename
    if(is.null(dim(y))){
      y <- matrix(y, ncol=1)
      orig_y_colnames <- colnames(y) <- "Y_1"
    } else {
      if(is.null(colnames(y))){
        orig_y_colnames <- colnames(y) <- paste0('Y_', 1:ncol(y))
      } else {
        orig_y_colnames <- colnames(y)
        colnames(y)     <- paste0('Y_', 1:ncol(y))
      }
    }
    
    if(verbose == 0){
      mcmc_print_every <- 0
    } else {
      if(verbose <= 20){
        mcmc_tot <- mcmc_burn + mcmc_thin * mcmc_keep
        mcmc_print_every <- 1+round(mcmc_tot / verbose)
      } else {
        if(is.infinite(verbose)){
          mcmc_print_every <- 1
        } else {
          mcmc_print_every <- verbose
        }
      }
    }
    
    if(proj_method == "pca"){
      X_pca <- prcomp(x)
      coords <- X_pca$x[,1:proj_dim] %>% as.matrix()
    } else if(proj_method == "lapeig"){
      X_lapeig <- Rdimtools::do.lapeig(x)
      coords <- X_lapeig$Y
    } else if(proj_method == "mds"){
      X_mds <- Rdimtools::do.mds(x)
      coords <- X_mds$Y
    } else { 
      coords <- x[,1:proj_dim]
    }
    
    
    
    colnames(coords) <- paste0("Var", 1:ncol(coords))
    dd <- ncol(coords)
    
    # covariates/confounders
    
    if(is.null(colnames(x))){
      orig_X_colnames <- colnames(x) <- paste0('X_', 1:ncol(x))
    } else {
      orig_X_colnames <- colnames(x)
      colnames(x)     <- paste0('X_', 1:ncol(x))
    }
    
    if(is.null(colnames(z))){
      orig_Z_colnames <- colnames(z) <- paste0('Z_', 1:ncol(z))
    } else {
      orig_Z_colnames <- colnames(z)
      colnames(z)     <- paste0('Z_', 1:ncol(z))
    }
    
    nr             <- nrow(x)
    q              <- ncol(y)
    k              <- ifelse(is.null(k), q, k)
    p              <- ncol(x)
    
    # family id 
    family <- if(length(family)==1){rep(family, q)} else {family}
    family_in <- data.frame(family=family)
    available_families <- data.frame(id=0:4, family=c("gaussian", "poisson", "binomial", "beta", "negbinomial"))
    family_id <- family_in %>% left_join(available_families, by=c("family"="family")) %>% pull(.data$id)
    
    latent <- "gaussian"
    if(!(latent %in% c("gaussian"))){
      stop("Latent process not recognized. Choose 'gaussian'")
    }
    
    # for spatial data: matern or expon, for spacetime: gneiting 2002 
    #n_par_each_process <- ifelse(dd==2, 1, 3) 
    
    axis_partition <- rep(round((nr/block_size)^(1/dd)), dd)
    
    use_forced_grid <- FALSE
    use_cache <- settings$cache %>% set_default(FALSE)
    scale_coords <- settings$scale_coords %>% set_default(TRUE)
    
    # what are we sampling
    sample_w       <- debug$sample_w %>% set_default(TRUE)
    sample_beta    <- debug$sample_beta %>% set_default(TRUE)
    sample_tausq   <- debug$sample_tausq %>% set_default(TRUE)
    sample_theta   <- debug$sample_theta %>% set_default(TRUE)
    sample_lambda  <- debug$sample_lambda %>% set_default(TRUE)
    
  }
  
  # data management pt 2
  if(1){
    yrownas <- apply(y, 1, function(i) ifelse(sum(is.na(i))==q, NA, 1))
    na_which <- ifelse(!is.na(yrownas), 1, NA)
    simdata <- data.frame(ix=1:nrow(coords)) %>% 
      cbind(coords, y, na_which, x, z) %>% 
      as.data.frame()
    
    #####
    #cat("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    #cat("{q} outcome variables" %>% glue::glue())
    #cat("\n")
    
    simdata %<>% 
      dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
    
    sort_ix     <- simdata$ix
    
    
    # take the coordinates and scale them
    coords <- simdata %>% 
      dplyr::select(dplyr::contains("Var")) %>% 
      as.matrix() 
    
    nclust <- ceiling(nrow(coords)/block_size)
    kmclust <- kmeans(coords, nclust)
    
    
    # make this into a while loop
    sizes <- coords %>% cbind(data.frame(gg=kmclust$cluster)) %>%
      group_by(gg) %>% 
      summarise(size=n()) %>% 
      mutate(ctr1=kmclust$centers[,1],
             ctr2=kmclust$centers[,2],
             reduction_factor=ifelse(ceiling(size/block_size)>2, 2, ceiling(size/block_size)))
    
    centroid_locs <- repeat_centroid_perturb(
      sizes %>% dplyr::select(ctr1, ctr2) %>% as.matrix(),
      sizes$reduction_factor)
    
    voronoided <- FNN::get.knnx(centroid_locs, 
                                coords[,1:proj_dim], k=1)
    
    coords_blocking <- coords %>%
      as.data.frame() %>%
      mutate(block = voronoided$nn.index[,1]) %>% 
      dplyr::mutate(na_which = simdata$na_which, sort_ix=sort_ix)
    #############################3
    
    
    
    if(F) {
      coords_blocking %>% mutate(gg=kmclust$cluster) %>%
        ggplot(aes(Var1, Var2, color=factor(block))) + 
        #geom_point() + 
        theme_minimal() +
        geom_text(aes(label=gg))
    }
    
    coords_blocking %<>% 
      rename(ix=sort_ix)
    
  }
  nr_full <- nrow(coords_blocking)
  
  parents_children <- mesh_graph_build_nn(coords_blocking, centroid_locs)
  
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- parents_children[["names"]] 
  block_groups                 <- parents_children[["groups"]]#[order(block_names)]
  
  
  if(indpart){
    parents %<>% lapply(function(x) numeric(0))
    children %<>% lapply(function(x) numeric(0))
    block_groups %<>% rep(0, length(block_groups))
  }
  
  suppressMessages(simdata_in <- coords_blocking %>% #cbind(data.frame(ix=cbix)) %>% 
                     dplyr::select(-na_which) %>% dplyr::left_join(simdata))
  
  simdata_in %<>% 
    dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
  blocking <- simdata_in$block %>% 
    factor() %>% as.integer()
  indexing <- (1:nrow(simdata_in)-1) %>% 
    split(blocking)
  
  indexing_grid <- indexing
  indexing_obs <- indexing_grid
  
  if(1){
    # prior and starting values for mcmc
    
    matern_nu <- FALSE
    if(is.null(starting$nu)){
      start_nu <- 0.5
      matern_fix_twonu <- 1
    } else {
      start_nu <- starting$nu
      if(start_nu %in% c(0.5, 1.5, 2.5)){
        matern_fix_twonu <- 2 * start_nu
      }
    }
    
    if(is.null(prior$beta)){
      beta_Vi <- diag(ncol(z)) * 1/100
    } else {
      beta_Vi <- prior$beta
    }
    
    if(is.null(prior$tausq)){
      tausq_ab <- c(2, 1)
    } else {
      tausq_ab <- prior$tausq
      if(length(tausq_ab) == 1){
        tausq_ab <- c(tausq_ab[1], 0)
      }
    }
    
    if(is.null(prior$sigmasq)){
      sigmasq_ab <- c(2, 1)
    } else {
      sigmasq_ab <- prior$sigmasq
    }
    
    # starting values
    if(is.null(starting$beta)){
      start_beta   <- matrix(0, nrow=ncol(z), ncol=q)
    } else {
      start_beta   <- starting$beta
    }
    
    theta_names <- c("sigmasq", paste0("phi_", 1:p))
    npar <- (p+1) # (sigma + p variables) for each of the processes
    set_unif_bounds <- matrix(0, nrow=npar*k, ncol=2)
    btmlim <- prior$btmlim %>% set_default(1e-3)
    toplim <- prior$toplim %>% set_default(1e3)
    set_unif_bounds[,1] <- btmlim
    set_unif_bounds[,2] <- toplim
    
    start_theta <- matrix(1, ncol=k, nrow=npar) 
    
    # override defaults if prior bounds values are provided
    if(!is.null(prior$set_unif_bounds)){
      set_unif_bounds <- prior$set_unif_bounds
    } 
    # override defaults if starting values are provided
    if(!is.null(starting$theta)){
      start_theta <- starting$theta
    }
    
    if(is.null(starting$mcmcsd)){
      mcmc_mh_sd <- diag(k * npar) * 0.01
    } else {
      if(length(starting$mcmcsd) == 1){
        mcmc_mh_sd <- diag(k * npar) * starting$mcmcsd
      } else {
        mcmc_mh_sd <- starting$mcmcsd
      }
    }
    
    if(is.null(starting$tausq)){
      start_tausq  <- family %>% sapply(function(ff) if(ff == "gaussian"){.1} else {1})
    } else {
      start_tausq  <- starting$tausq
    }
    
    if(is.null(starting$lambda)){
      start_lambda <- matrix(0, nrow=q, ncol=k)
      diag(start_lambda) <- 2 #
    } else {
      start_lambda <- starting$lambda
    }
    
    if(is.null(starting$lambda_mask)){
      lambda_mask <- matrix(0, nrow=q, ncol=k)
      lambda_mask[lower.tri(lambda_mask)] <- 1
      diag(lambda_mask) <- 1 #***
    } else {
      lambda_mask <- starting$lambda_mask
    }
    
    if(is.null(starting$mcmc_startfrom)){
      mcmc_startfrom <- 0
    } else {
      mcmc_startfrom <- starting$mcmc_startfrom
    }
    
    if(is.null(starting$w)){
      start_v <- matrix(0, nrow = nrow(simdata_in), ncol = k)
    } else {
      # this is used to restart MCMC
      # assumes the ordering and the sizing is correct, 
      # so no change is necessary and will be input directly to mcmc
      start_v <- starting$v #%>% matrix(ncol=q)
    }
  }
  
  # finally prepare data
  sort_ix <- simdata_in$ix
  
  y <- simdata_in %>% 
    dplyr::select(dplyr::contains("Y_")) %>% 
    as.matrix()
  colnames(y) <- orig_y_colnames
  
  x <- simdata_in %>% 
    dplyr::select(dplyr::contains("X_")) %>% 
    as.matrix()
  colnames(x) <- orig_X_colnames
  x[is.na(x)] <- 0 # NAs if added coords due to empty blocks
  
  z <- simdata_in %>% 
    dplyr::select(dplyr::contains("Z_")) %>% 
    as.matrix()
  colnames(z) <- orig_Z_colnames
  if(any(is.na(z))){
    print(head(z))
    return(z)
    stop("Cannot have NA in Z matrix")
  }
  
  na_which <- simdata_in$na_which
  
  coords <- simdata_in %>% 
    dplyr::select(dplyr::contains("Var")) %>% 
    as.matrix()
  
  coordsdata <- simdata_in %>% 
    dplyr::select(1:dd) 
  cat("Sending to MCMC >\n ")
  
  mcmc_run <- meshed_mcmc
  
  comp_time <- system.time({
    results <- mcmc_run(y, family_id, z, x, k,
                        
                        parents, children, 
                        block_names, block_groups,
                        
                        indexing_grid, indexing_obs,
                        
                        set_unif_bounds,
                        beta_Vi, 
                        
                        
                        sigmasq_ab,
                        tausq_ab,
                        
                        matern_fix_twonu,
                        
                        start_v, 
                        
                        start_lambda,
                        lambda_mask,
                        
                        start_theta,
                        start_beta,
                        start_tausq,
                        
                        mcmc_mh_sd,
                        
                        mcmc_keep, mcmc_burn, mcmc_thin,
                        
                        mcmc_startfrom,
                        
                        n_threads,
                        
                        which_hmc,
                        mcmc_adaptive, # adapting
                        
                        use_cache,
                        use_forced_grid,
                        use_ps,
                        
                        mcmc_verbose, mcmc_debug, # verbose, debug
                        mcmc_print_every, # print all iter
                        low_mem,
                        # sampling of:
                        # beta tausq sigmasq theta w
                        sample_beta, sample_tausq, 
                        sample_lambda,
                        sample_theta, sample_w) 
  })
  
  if(saving){
    
    listN <- function(...){
      anonList <- list(...)
      names(anonList) <- as.character(substitute(list(...)))[-1]
      anonList
    }
    
    saved <- listN(y, x, z, coords, k,
                   
                   parents, children, 
                   block_names, block_groups,
                   
                   indexing_grid, indexing_obs,
                   
                   set_unif_bounds,
                   beta_Vi, 
                   
                   tausq_ab,
                   
                   start_v, 
                   
                   start_lambda,
                   lambda_mask,
                   
                   start_theta,
                   start_beta,
                   start_tausq,
                   
                   mcmc_mh_sd,
                   
                   mcmc_keep, mcmc_burn, mcmc_thin,
                   
                   mcmc_startfrom,
                   
                   n_threads,
                   
                   mcmc_adaptive, # adapting
                   
                   use_forced_grid,
                   
                   mcmc_verbose, mcmc_debug, # verbose, debug
                   mcmc_print_every, # print all iter
                   # sampling of:
                   # beta tausq sigmasq theta w
                   sample_beta, sample_tausq, 
                   sample_lambda,
                   sample_theta, sample_w)
  } else {
    saved <- "Model data not saved."
  }
  
  returning <- list(data = simdata_in,
                    savedata = saved,
                    block_names = block_names,
                    block_groups = block_groups,
                    parents = parents,
                    children = children,
                    coordsblocking = coords_blocking) %>% 
    c(results)
  return(returning)
  
}
