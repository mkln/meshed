library(ggplot2)
library(magrittr)
library(meshed)

# spatial domain (we choose a grid to make a nice image later)
# this generates a dataset of size 32400
xx <- seq(0, 1, length.out=180)
coords <- expand.grid(xx, xx) %>%
  as.matrix()

raster_plot <- function(df){
  ggplot(df, aes(Var1, Var2, fill=w)) +
    geom_raster() +
    scale_fill_viridis_c() +
    theme_minimal() }

# spatial data, exponential covariance
# phi=14, sigma^2=2
simdata <- rmeshedgp(coords, c(14, 2))
raster_plot(simdata)

# spatial data, matern covariance
# phi=14, nu=1, sigma^2=2
simdata <- rmeshedgp(coords, c(14, 1, 2))
raster_plot(simdata)


# spacetime data, gneiting's covariance
# 324000 locations
stcoords <- expand.grid(xx, xx, seq(0, 1, length.out=10))

# it should take less than a couple of seconds
simdata <- rmeshedgp(stcoords, c(1, 14, .5, 2))
# plot data at 7th time period
raster_plot(simdata %>% dplyr::filter(Var3==unique(Var3)[7])) 
