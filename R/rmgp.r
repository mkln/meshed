
rmeshedgp <- function(coords, theta, axis_partition=NULL, block_size=100, 
                      n_threads=1, cache=TRUE, verbose=FALSE, debug=FALSE){
  
  dd             <- ncol(coords)
  nr             <- nrow(coords)
  orig_coords_colnames <- colnames(coords)
  
  if(is.null(axis_partition)){
    axis_partition <- rep(round((nr/block_size)^(1/dd)), dd)
  } else {
    if(length(axis_partition) == 1){
      axis_partition <- rep(axis_partition, dd)
    }
  }
  
  use_cache <- cache
  
  y <- rep(1, nr)
  na_which <- rep(1, nr)
  simdata <- data.frame(ix=1:nr) %>% 
    cbind(coords, y, na_which) %>% 
    as.data.frame()
  
  if(length(axis_partition) < ncol(coords)){
    stop("Error: axis_partition not specified for all axes.")
  }
  simdata %<>% 
    dplyr::mutate(thegrid = 0)
  absize <- round(nrow(simdata)/prod(axis_partition))
  
  if(verbose & debug){
    cat("Partitioning grid axes into {paste0(axis_partition, collapse=', ')} intervals. Approx block size {absize}" %>% glue::glue())
    cat("\n")
  }
  
  simdata %<>% 
    dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
  
  coords <- simdata %>% 
    dplyr::select(dplyr::contains("Var")) %>% 
    as.matrix()
  sort_ix     <- simdata$ix
  
  if(verbose & debug){
    cat("Finding thresholds...\n")
  }
  fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(coords[,i], axis_partition[i])) 
  
  if(verbose & debug){
    cat("Domain partitioning...\n")
  }
  # Domain partitioning and gibbs groups
  system.time(coords_blocking <- coords %>% 
                as.matrix() %>%
                tessellation_axis_parallel_fix(fixed_thresholds, n_threads) %>% 
                dplyr::mutate(na_which = simdata$na_which, sort_ix=sort_ix) )
  
  coords_blocking %<>% dplyr::rename(ix=sort_ix)
  
  if(verbose & debug){
    cat("Building DAG...\n")
  }
  # DAG
  if(dd < 4){
    suppressMessages(parents_children <- 
                       mesh_graph_build(coords_blocking %>% dplyr::select(-.data$ix), 
                                        axis_partition, FALSE, n_threads))
  } else {
    suppressMessages(parents_children <- 
                       mesh_graph_build_hypercube(coords_blocking %>% dplyr::select(-.data$ix)))
  }
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- parents_children[["names"]] 
  block_groups                 <- parents_children[["groups"]]#[order(block_names)]
  
  # these two lines remove the DAG and make all blocks independent
  #parents %<>% lapply(function(x) x[x==-1]) 
  #children %<>% lapply(function(x) x[x==-1])
  
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
  
  matern_nu <- T
  matern_fix_twonu <- 1
  
  # override defaults if starting values are provided
  if((dd==3) & (length(theta)==5)){
    matern_fix_twonu <- 2 * theta[5]
    if(!(matern_fix_twonu %in% c(1, 3, 5))){
      stop("Value of nu (input at theta[5]) must be 0.5, 1.5, or 2.5.")
    }
    theta <- theta[1:4]
  } 
  theta %<>% matrix(ncol=1)
  
  # finally prepare data
  sort_ix <- simdata_in$ix
  
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
  
  if(verbose & debug){
    cat("Sending to C++ for sampling.\n")
  }
  w <- rmeshedgp_internal(coords, parents, children,
                                     block_names, block_groups,
                                     indexing_grid, indexing_obs,
                                     matern_fix_twonu,
                                     theta,
                                     n_threads,
                                     use_cache,
                                     verbose, debug)
  simulated_data <- coords %>% cbind(w) %>% as.data.frame()
  colnames(simulated_data)[dd+1] <- "w"
  return(simulated_data)
}