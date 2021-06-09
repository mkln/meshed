## Meshed Gaussian Process Regression

This package provides functions for fitting big data Bayesian geostatistics models using latent Meshed Gaussian Processes (MGPs). In particular, any combination of the following is allowed:

 - data at irregular spatial locations;
 - spatiotemporal data;
 - multivariate outcomes;
 - spatial misalignment of multivariate outcomes;
 - spatial or spatiotemporal factor models;
 - outcomes of different types.
 
All these use-cases are implemented via the `spmeshed` function. See vignettes.
The package also provides a function for sampling MGPs a priori: the `rmeshedgp` function allows to simulate smooth correlated data at millions of spatial or spatiotemporal locations with minimal resources.
General MGPs are described in Peruzzi et al (2020). This package implements cubic MGPs (QMGPs). Posterior sampling of all unknowns is performed via MCMC-GriPS as detailed in Peruzzi et al (2021).


### Highly Scalable Bayesian Geostatistical Modeling via Meshed Gaussian Processes on Partitioned Domains
M Peruzzi, S Banerjee, AO Finley (2020). [JASA](https://doi.org/10.1080/01621459.2020.1833889), [arXiv](https://arxiv.org/abs/2003.11208).

We introduce a class of scalable Bayesian hierarchical models for the analysis of massive geostatistical datasets. The underlying idea combines ideas on high-dimensional geostatistics by partitioning the spatial domain and modeling the regions in the partition using a sparsity-inducing directed acyclic graph (DAG). We extend the model over the DAG to a well-defined spatial process, which we call the Meshed Gaussian Process (MGP). A major contribution is the development of a MGPs on tessellated domains, accompanied by a Gibbs sampler for the efficient recovery of spatial random effects. In particular, the cubic MGP (Q-MGP) can harness high-performance computing resources by executing all large-scale operations in parallel within the Gibbs sampler, improving mixing and computing time compared to sequential updating schemes. Unlike some existing models for large spatial data, a Q-MGP facilitates massive caching of expensive matrix operations, making it particularly apt in dealing with spatiotemporal remote-sensing data. We compare Q-MGPs with large synthetic and real world data against state-of-the-art methods. We also illustrate using Normalized Difference Vegetation Index (NDVI) data from the Serengeti park region to recover latent multivariate spatiotemporal random effects at millions of locations. 

### Grid-Parametrize-Split (GriPS) for Improved Scalable Inference in Spatial Big Data Analysis
M Peruzzi, S Banerjee, DB Dunson, AO Finley (2021). [arXiv](https://arxiv.org/abs/2101.03579).

Rapid advancements in spatial technologies including Geographic Information Systems (GIS) and remote sensing have generated massive amounts of spatially referenced data in a variety of scientific and data-driven industrial applications. These advancements have led to a substantial, and still expanding, literature on the modeling and analysis of spatially oriented big data. In particular, Bayesian inferences for high-dimensional spatial processes are being sought in a variety of remote-sensing applications including, but not limited to, modeling next generation Light Detection and Ranging (LiDAR) systems and other remotely sensed data. Massively scalable spatial processes, in particular Gaussian processes (GPs), are being explored extensively for the increasingly encountered big data settings. Recent developments include GPs constructed from sparse Directed Acyclic Graphs (DAGs) with a limited number of neighbors (parents) to characterize dependence across the spatial domain. The DAG can be used to devise fast algorithms for posterior sampling of the latent process, but these may exhibit pathological behavior in estimating covariance parameters. While these issues are mitigated by considering marginalized samplers that exploit the underlying sparse precision matrix, these algorithms are slower, less flexible, and oblivious of structure in the data. The current article introduces the Grid-Parametrize-Split (GriPS) approach for conducting Bayesian inference in spatially oriented big data settings by a combination of careful model construction and algorithm design to effectuate substantial improvements in MCMC convergence. We demonstrate the effectiveness of our proposed methods through simulation experiments and subsequently undertake the modeling of LiDAR outcomes and production of their predictive maps using G-LiHT and other remotely sensed variables. 
