## Meshed Gaussian Process Regression

This package provides functions for fitting big data Bayesian geostatistics models using latent Meshed Gaussian Processes (MGPs). In particular, any combination of the following is allowed:

 - data at irregular spatial locations;
 - spatiotemporal data;
 - multivariate outcomes;
 - spatial misalignment of multivariate outcomes;
 - spatial or spatiotemporal factor models;
 - outcomes of different types.
 
All these use-cases are implemented via the `spmeshed` function. See vignettes for some examples.
The package also provides a function for sampling MGPs a priori via the `rmeshedgp` function, which allows to simulate smooth correlated data at millions of spatial or spatiotemporal locations with minimal resources.
MGPs are introduced in [Peruzzi et al. (2020)](https://doi.org/10.1080/01621459.2020.1833889), [arXiv](https://arxiv.org/abs/2003.11208). This package implements cubic MGPs (QMGPs). Posterior sampling of all unknowns can be performed via MCMC-GriPS as detailed in [Peruzzi et al. (2021)](https://arxiv.org/abs/2101.03579). For non-Gaussian outcomes, QMGPs are fit via Langevin-SiMPA as detailed in [Peruzzi & Dunson (2022)](https://arxiv.org/abs/2201.10080).


### Install from CRAN: `install.packages("meshed")`

Alternatively, `devtools::install_github("mkln/meshed")` installs from GitHub.

#### Tips for best performance:

 - `meshed` works best with OpenMP and OpenBLAS or Intel MKL. 
 - [Dirk Eddelbuettel has a great guide on installing Intel MKL on Debian/Ubuntu systems](http://dirk.eddelbuettel.com/blog/2018/04/15/#018_mkl_for_debian_ubuntu). In that case it is important to add `MKL_THREADING_LAYER=GNU` to `~/.Renviron`. 
 - On systems with AMD CPUs, it may be best to install `intel-mkl-2019.5-075` and then also add the line `MKL_DEBUG_CPU_TYPE=5` to `~/.Renviron`. I have not tested more recent versions of Intel MKL.
 - If using OpenBLAS, it might be important to let OpenMP do *all* the parallelization when running `meshed`. I think this can be done with the [RhpcBLASctl](https://CRAN.R-project.org/package=RhpcBLASctl) package. YMMV.

### Vignettes

Some super toy examples just to get a feel of how `meshed::spmeshed` works are available [at the CRAN page for `meshed`](https://CRAN.R-project.org/package=meshed).

### Poster
![](img/poster.jpg?raw=true)


### Highly Scalable Bayesian Geostatistical Modeling via Meshed Gaussian Processes on Partitioned Domains
M Peruzzi, S Banerjee, AO Finley (2020). [JASA](https://doi.org/10.1080/01621459.2020.1833889), [arXiv](https://arxiv.org/abs/2003.11208).

We introduce a class of scalable Bayesian hierarchical models for the analysis of massive geostatistical datasets. The underlying idea combines ideas on high-dimensional geostatistics by partitioning the spatial domain and modeling the regions in the partition using a sparsity-inducing directed acyclic graph (DAG). We extend the model over the DAG to a well-defined spatial process, which we call the Meshed Gaussian Process (MGP). A major contribution is the development of a MGPs on tessellated domains, accompanied by a Gibbs sampler for the efficient recovery of spatial random effects. In particular, the cubic MGP (Q-MGP) can harness high-performance computing resources by executing all large-scale operations in parallel within the Gibbs sampler, improving mixing and computing time compared to sequential updating schemes. Unlike some existing models for large spatial data, a Q-MGP facilitates massive caching of expensive matrix operations, making it particularly apt in dealing with spatiotemporal remote-sensing data. We compare Q-MGPs with large synthetic and real world data against state-of-the-art methods. We also illustrate using Normalized Difference Vegetation Index (NDVI) data from the Serengeti park region to recover latent multivariate spatiotemporal random effects at millions of locations. 

### Spatial Meshing for General Bayesian Multivariate Models
M Peruzzi & DB Dunson (2022). [arXiv](https://arxiv.org/abs/2201.10080).

Quantifying spatial and/or temporal associations in multivariate geolocated data of different types is achievable via spatial random effects in a Bayesian hierarchical model, but severe computational bottlenecks arise when spatial dependence is encoded as a latent Gaussian process (GP) in the increasingly common large scale data settings on which we focus. The scenario worsens in non-Gaussian models because the reduced analytical tractability leads to additional hurdles to computational efficiency. In this article, we introduce Bayesian models of spatially referenced data in which the likelihood or the latent process (or both) are not Gaussian. First, we exploit the advantages of spatial processes built via directed acyclic graphs, in which case the spatial nodes enter the Bayesian hierarchy and lead to posterior sampling via routine Markov chain Monte Carlo (MCMC) methods. Second, motivated by the possible inefficiencies of popular gradient-based sampling approaches in the multivariate contexts on which we focus, we introduce the simplified manifold preconditioner adaptation (SiMPA) algorithm which uses second order information about the target but avoids expensive matrix operations. We demostrate the performance and efficiency improvements of our methods relative to alternatives in extensive synthetic and real world remote sensing and community ecology applications with large scale data at up to hundreds of thousands of spatial locations and up to tens of outcomes. 

### Grid-Parametrize-Split (GriPS) for Improved Scalable Inference in Spatial Big Data Analysis
M Peruzzi, S Banerjee, DB Dunson, AO Finley (2021). [arXiv](https://arxiv.org/abs/2101.03579).

Rapid advancements in spatial technologies including Geographic Information Systems (GIS) and remote sensing have generated massive amounts of spatially referenced data in a variety of scientific applications. These advancements have led to a substantial, and still expanding, literature on the modeling and analysis of spatially oriented big data. 
Massively scalable spatial processes, in particular Gaussian processes (GPs), are being explored extensively for the pervasive big data settings. Recent developments include GPs constructed from sparse Directed Acyclic Graphs (DAGs) with a limited number of neighbors (parents) to characterize spatial dependence. The DAG can be used to devise fast algorithms for posterior sampling of the latent process, but these may exhibit pathological behavior in estimating covariance parameters. While these issues are mitigated by considering marginalized samplers that exploit the underlying sparse precision matrix, these algorithms are slower, less flexible, and oblivious of structure in the data. The current article introduces the Grid-Parametrize-Split (GriPS) approach for conducting Bayesian inference in spatially oriented big data settings by a combination of careful model construction and algorithm design to effectuate substantial improvements in MCMC efficiency. We demonstrate the effectiveness of our proposed methods through simulation experiments and subsequently model remotely sensed variables from NASA's Goddard LiDAR, Hyper-Spectral, and Thermal imager (G-LiHT).
