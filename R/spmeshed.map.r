
spmeshed.map <- function(y, x, coords, k=NULL,
                         family = "gaussian",
                         axis_partition = NULL, 
                         block_size = 30,
                         grid_size=NULL,
                         grid_custom = NULL,
                         pars = list(phi=NULL, tausq=NULL),
                         theta = NULL,
                         tausq = NULL,
                         maxit = 1000,
                         n_threads = 4,
                         verbose = FALSE,
                         predict_everywhere = FALSE,
                         settings = list(forced_grid=NULL, cache=NULL),
                         debug = list(map_beta=TRUE, map_w=TRUE,
                                verbose=FALSE, debug=FALSE)
){
  if(verbose){
    cat("Bayesian Meshed GP regression model via Maximum a Posteriori\n")
  }
  model_tag <- "Bayesian Meshed GP regression model\n
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o\n(maximum a posteriori)\n"
  
  set_default <- function(x, default_val){
    return(if(is.null(x)){
      default_val} else {
        x
      })}
  
  # data management pt 1
  if(1){
    
    map_w <- debug$map_w %>% set_default(TRUE)
    map_beta <- debug$map_beta %>% set_default(FALSE)
    
    verbose <- verbose | (debug$verbose %>% set_default(FALSE))
    debug <- debug$debug %>% set_default(FALSE)
    
    coords %<>% as.matrix()
    
    
    q              <- ncol(y)
    k <- ifelse(is.null(k), q, k)
    
    
    dd             <- ncol(coords)
    
    if(dd > 2){
      stop("Not implemented for spacetime data.")
    }
    
    p              <- ncol(x)
    
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
    
    print_every <- 0
  
    if(is.null(colnames(x))){
      orig_X_colnames <- colnames(x) <- paste0('X_', 1:ncol(x))
    } else {
      orig_X_colnames <- colnames(x)
      colnames(x)     <- paste0('X_', 1:ncol(x))
    }
    
    if(is.null(colnames(coords))){
      orig_coords_colnames <- colnames(coords) <- paste0('Var', 1:dd)
    } else {
      orig_coords_colnames <- colnames(coords)
      colnames(coords)     <- paste0('Var', 1:dd)
    }

    # family id 
    family <- if(length(family)==1){rep(family, q)} else {family}
    family_in <- data.frame(family=family)
    available_families <- data.frame(id=0:4, family=c("gaussian", "poisson", "binomial", "beta", "negbinomial"))
    family_id <- family_in %>% left_join(available_families, by=c("family"="family")) %>% pull(.data$id)
    
    nr <- nrow(x)
    
    if(length(axis_partition) == 1){
      axis_partition <- rep(axis_partition, dd)
    }
    if(is.null(axis_partition)){
      axis_partition <- rep(round((nr/block_size)^(1/dd)), dd)
    }
    
    # -- heuristics for gridded data --
    # if we observed all unique combinations of coordinates, this is how many rows we'd have
    heuristic_gridded <- prod( coords %>% apply(2, function(x) length(unique(x))) )
    # if data are not gridded, then the above should be MUCH larger than the number of rows
    # if we're not too far off maybe the dataset is actually gridded
    if(heuristic_gridded*0.5 < nrow(coords)){
      data_likely_gridded <- TRUE
    } else {
      data_likely_gridded <- FALSE
    }
    if(ncol(coords) == 3){
      # with time, check if there's equal spacing
      timepoints <- coords[,3] %>% unique()
      time_spacings <- timepoints %>% sort() %>% diff() %>% round(5) %>% unique()
      if(length(time_spacings) < .1 * length(timepoints)){
        data_likely_gridded <- TRUE
      }
    }
    
    #cat("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    
    if(is.null(settings$forced_grid)){
      if(data_likely_gridded){
        #cat("I think the data look gridded so I'm setting forced_grid=F.\n")
        use_forced_grid <- FALSE
      } else {
        #cat("I think the data don't look gridded so I'm setting forced_grid=T.\n")
        use_forced_grid <- TRUE
      }
    } else {
      use_forced_grid <- settings$forced_grid %>% set_default(TRUE)
      if(!use_forced_grid & !data_likely_gridded){
        warning("Data look not gridded: force a grid with settings$forced_grid=T.")
      }
    }
    
    use_cache <- settings$cache %>% set_default(TRUE)
    if(use_forced_grid & (!use_cache)){
      warning("Using a forced grid with no cache is a waste of resources.")
    }
    
  }
  
  # data management pt 2
  if(1){
    yrownas <- apply(y, 1, function(i) ifelse(sum(is.na(i))==q, NA, 1))
    na_which <- ifelse(!is.na(yrownas), 1, NA)
    simdata <- data.frame(ix=1:nrow(coords)) %>% 
      cbind(coords, y, na_which, x) %>% 
      as.data.frame()
    
    #####
    #cat("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    #cat("{q} outcome variables on {nrow(unique(coords))} unique locations." %>% glue::glue())
    #cat("\n")
    if(use_forced_grid){ 
      # user showed intention to use fixed grid
      if(!is.null(grid_custom$grid)){
        grid <- grid_custom$grid
        gridcoords_lmc <- grid %>% as.data.frame()
        colnames(gridcoords_lmc)[1:dd] <- colnames(coords)
        if(ncol(grid) == dd + p){
          # we have the covariate values at the reference grid, so let's use them to make predictions
          colnames(gridcoords_lmc)[-(1:dd)] <- colnames(x)
        }
      } else {
        gs <- round(nrow(coords)^(1/ncol(coords)))
        gsize <- if(is.null(grid_size)){ rep(gs, ncol(coords)) } else { grid_size }
        
        xgrids <- list()
        for(j in 1:dd){
          xgrids[[j]] <- seq(min(coords[,j]), max(coords[,j]), length.out=gsize[j])
        }
        
        gridcoords_lmc <- expand.grid(xgrids)
      }
      
      #cat("Forced grid built with {nrow(gridcoords_lmc)} locations." %>% glue::glue())
      #cat("\n")
      simdata <- dplyr::bind_rows(simdata %>% dplyr::mutate(thegrid=0), 
                                  gridcoords_lmc %>% dplyr::mutate(thegrid=1))
      
      absize <- round(nrow(gridcoords_lmc)/prod(axis_partition))
    } else {
      if(length(axis_partition) < ncol(coords)){
        stop("Error: axis_partition not specified for all axes.")
      }
      simdata %<>% 
        dplyr::mutate(thegrid = 0)
      absize <- round(nrow(simdata)/prod(axis_partition))
    }
    #cat("Partitioning grid axes into {paste0(axis_partition, collapse=', ')} intervals. Approx block size {absize}" %>% glue::glue())
    #cat("\n")
    
    #cat("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    
    simdata %<>% 
      dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
    
    
    coords <- simdata %>% 
      dplyr::select(dplyr::contains("Var")) %>% 
      as.matrix()
    sort_ix     <- simdata$ix
    
    # Domain partitioning and gibbs groups
    if(use_forced_grid){
      if(!is.null(grid_custom$axis_interval_partition)){
        fixed_thresholds <- grid_custom$axis_interval_partition
        axis_partition <- sapply(fixed_thresholds, function(x) length(x) + 1)
      } else {
        gridded_coords <- simdata %>% dplyr::filter(.data$thegrid==1) %>% dplyr::select(dplyr::contains("Var")) %>% as.matrix()
        fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(gridded_coords[,i], axis_partition[i])) 
      }
    } else {
      fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(coords[,i], axis_partition[i])) 
    }
    
    
    # guaranteed to produce blocks using Mv
    system.time(fake_coords_blocking <- coords %>% 
                  as.matrix() %>% 
                  gen_fake_coords(fixed_thresholds, 1) )
    
    # Domain partitioning and gibbs groups
    system.time(coords_blocking <- coords %>% 
                  as.matrix() %>%
                  tessellation_axis_parallel_fix(fixed_thresholds, 1) %>% 
                  dplyr::mutate(na_which = simdata$na_which, sort_ix=sort_ix) )
    
    
    coords_blocking %<>% dplyr::rename(ix=sort_ix)
    
    # check if some blocks come up totally empty
    blocks_prop <- coords_blocking[,paste0("L", 1:dd)] %>% unique()
    blocks_fake <- fake_coords_blocking[,paste0("L", 1:dd)] %>% unique()
    if(nrow(blocks_fake) != nrow(blocks_prop)){
      #cat("Adding fake coords to avoid empty blocks ~ don't like? Use lower [axis_partition]\n")
      # with current Mv, some blocks are completely empty
      # this messes with indexing. so we add completely unobserved coords
      suppressMessages(adding_blocks <- blocks_fake %>% dplyr::setdiff(blocks_prop) %>%
                         dplyr::left_join(fake_coords_blocking))
      coords_blocking <- dplyr::bind_rows(coords_blocking, adding_blocks)
      
      coords_blocking %<>% dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
    }
    
  }
  nr_full <- nrow(coords_blocking)

  
  # DAG
  if(dd < 4){
    suppressMessages(parents_children <- mesh_graph_build(coords_blocking %>% dplyr::select(-.data$ix), axis_partition, FALSE, n_threads))
  } else {
    suppressMessages(parents_children <- mesh_graph_build_hypercube(coords_blocking %>% dplyr::select(-.data$ix)))
  }
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- parents_children[["names"]] 
  block_groups                 <- parents_children[["groups"]]#[order(block_names)]
  
  suppressMessages(simdata_in <- coords_blocking %>% #cbind(data.frame(ix=cbix)) %>% 
                     dplyr::select(-na_which) %>% dplyr::left_join(simdata))
  
  simdata_in %<>% 
    dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
  blocking <- simdata_in$block %>% 
    factor() %>% as.integer()
  indexing <- (1:nrow(simdata_in)-1) %>% 
    split(blocking)
  
  if(use_forced_grid){
    indexing_grid_ids <- simdata_in$thegrid %>% split(blocking)
    indexing_grid <- list()
    indexing_obs <- list()
    for(i in 1:length(indexing)){
      indexing_grid[[i]] <- indexing[[i]][which(indexing_grid_ids[[i]] == 1)]
      if(predict_everywhere){
        indexing_obs[[i]] <- indexing[[i]]
      } else {
        indexing_obs[[i]] <- indexing[[i]][which(indexing_grid_ids[[i]] == 0)]
      }
    }
  } else {
    indexing_grid <- indexing
    indexing_obs <- indexing_grid
  }
  
  if(TRUE){
    # prior and starting values for mcmc
    matern_nu <- FALSE
    start_nu <- 0.5
    matern_fix_twonu <- 1
    
    if(!is.null(pars$phi)){
      if(family %in% c("gaussian", "beta", "negbinomial")){
        if(is.null(pars$tausq)){
          stop("Must specify pars$tausq for this family.")
        }
      } else {
        pars$tausq <- 1
      }
      
      all_values <- expand.grid(as.list(pars[c("phi", "tausq")]))
      colnames(all_values)[1:2] <- c("phi", "tausq")
      
      theta_values <- cbind(all_values[,1,drop=F],1) %>% 
        t() %>% as.data.frame() %>% as.list() %>% 
        lapply(function(x) matrix(x, ncol=1))
      tausq_values <- all_values[,2] %>% matrix(nrow=q)   
    } else {
      theta_values <- theta %>% lapply(function(x) rbind(x, 1))
      tausq_values <- tausq
      all_values <- list(theta = theta_values, tausq = tausq_values)
    }
  
    beta_Vi <- diag(ncol(x)) * 1/100
    
    start_beta   <- matrix(0, nrow=p, ncol=q)
    
    #if(is.null(starting$lambda)){
    lambda_values <- matrix(0, nrow=q, ncol=k)
    diag(lambda_values) <- 1 #*** 10
    
    lambda_mask <- matrix(0, nrow=q, ncol=k)
    lambda_mask[lower.tri(lambda_mask)] <- 1
    diag(lambda_mask) <- 1 #*** 
    
    lambda_values <- lambda_mask
    
    start_w <- matrix(0, nrow = nrow(simdata_in), ncol = k)
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
  
  na_which <- simdata_in$na_which
  
  coords <- simdata_in %>% 
    dplyr::select(dplyr::contains("Var")) %>% 
    as.matrix()
  
  coords_renamer <- colnames(coords)
  names(coords_renamer) <- orig_coords_colnames
  
  coordsdata <- simdata_in %>% 
    dplyr::select(1:dd, .data$thegrid) %>%
    dplyr::rename(!!!coords_renamer,
                  forced_grid=.data$thegrid)
  
  ## checking
  if(use_forced_grid){
    suppressMessages(checking <- coordsdata %>% left_join(coords_blocking) %>% 
      group_by(.data$block) %>% summarise(nfg = sum(.data$forced_grid)) %>% filter(.data$nfg==0))
    if(nrow(checking) > 0){
      stop("Partition is too fine for the current reference set. ")
    }
  }
  adaptive <- FALSE
  
  comp_time <- system.time({
    results <- meshed_casc(y, family_id, k, x, coords, 
                           
                           parents, children, 
                           block_names, block_groups,
                           
                           indexing_grid, indexing_obs,
                           
                           beta_Vi, 
                           
                           matern_fix_twonu,
                           
                           start_w, 
                           
                           lambda_values,
                           lambda_mask,
                           
                           theta_values,
                           start_beta,
                           tausq_values,
                           
                           maxit,
                           n_threads,
                           
                           adaptive, # adapting
                           
                           use_cache,
                           use_forced_grid,
                           
                           verbose, debug, # verbose, debug
                           print_every,
                           
                           map_beta, map_w) 
  })
  
  returning <- list(coordsdata = coordsdata,
                    pardf = all_values
                    #block_names = block_names,
                    #block_groups = block_groups,
                    #parents = parents,
                    #children = children,
                    #coordsblocking = coords_blocking
                    ) %>% 
    c(results)
  class(returning) <- "spmeshed.map"
  return(returning) 
  
}



