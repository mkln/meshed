

tessellation_axis_parallel <- function(coordsmat, Mv, n_threads=1){
  
  blocks_by_coord <- part_axis_parallel(coordsmat, Mv, n_threads) %>% apply(2, factor)
  colnames(blocks_by_coord) <- paste0("L", 1:ncol(coordsmat))
  
  block <- blocks_by_coord %>% 
    as.data.frame() %>% as.list() %>% interaction()
  blockdf <- data.frame(blocks_by_coord %>% apply(2, as.numeric), block=as.numeric(block))
  
  if(ncol(coordsmat)==2){
    result <- cbind(coordsmat, blockdf) %>% 
      mutate(color = ((L1-1)*2+(L2-1)) %% 4)
    return(result)
  } else {
    result <- cbind(coordsmat, blockdf) %>% 
      mutate(color = 4*(L3 %% 2) + (((L1-1)*2+(L2-1)) %% 4))
    return(result)
  }
}

gen_fake_coords <- function(coordsmat, thresholds, n_threads=1){
  # if some blocks become empty later, add coords at empty blocks
  # these will have no covariates and y=NA so they will be "predicted" in MCMC
  # with no effect on the estimates other than to make life easier with coding
  # some tiny slowdown will happen
  
  dd <- ncol(coordsmat)
  fake_coords <- 1:dd %>% lapply(function(i) c(min(coordsmat[,i]), 
                                               thresholds[[i]],
                                               max(coordsmat[,i]))) %>% 
    lapply(function(vec) vec[-length(vec)] + diff(vec) / 2) %>%
    expand.grid() %>% as.matrix()
  
  blocks_by_fakecoord <- part_axis_parallel_fixed(fake_coords, thresholds, n_threads) %>% apply(2, factor)
  colnames(blocks_by_fakecoord) <- paste0("L", 1:ncol(coordsmat))
  
  block <- blocks_by_fakecoord %>% 
    as.data.frame() %>% as.list() %>% interaction()
  blockdf <- data.frame(blocks_by_fakecoord %>% apply(2, as.numeric), block=as.numeric(block))
  
  if(ncol(coordsmat)==2){
    result <- cbind(fake_coords, blockdf) #%>% 
      #mutate(color = ((L1-1)*2+(L2-1)) %% 4)
    return(result)
  } else {
    result <- cbind(fake_coords, blockdf) #%>% 
      #mutate(color = 4*(L3 %% 2) + (((L1-1)*2+(L2-1)) %% 4))
    return(result)
  }
}

tessellation_axis_parallel_fix <- function(coordsmat, thresholds, n_threads){
  
  blocks_by_coord <- part_axis_parallel_fixed(coordsmat, thresholds, n_threads) %>% apply(2, factor)
  colnames(blocks_by_coord) <- paste0("L", 1:ncol(coordsmat))
  
  block <- blocks_by_coord %>% 
    as.data.frame() %>% as.list() %>% interaction()
  blockdf <- data.frame(blocks_by_coord %>% apply(2, as.numeric), block=as.numeric(block))
  result <- cbind(coordsmat, blockdf)# %>% 
    #mutate(color = (L1+L2) %% 2)
  
  if(ncol(coordsmat)==2){
    result <- cbind(coordsmat, blockdf) #%>% 
      #mutate(color = ((L1-1)*2+(L2-1)) %% 4)
    return(result)
  } else {
    result <- cbind(coordsmat, blockdf) #%>% 
      #mutate(color = 4*(L3 %% 2) + (((L1-1)*2+(L2-1)) %% 4))
    return(result)
  }
}

