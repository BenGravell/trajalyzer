https://github.com/BenGravell/trajalyzer/blob/main/trajectory_manifold/clustering.py

Can using faiss with e-means provide a speedup or better clustering results?







Create a new module in the codebase for manifold learning/fitting. The goal of this module is to identify promising native alternative/transformed representations of trajectories, in terms of "compressed" or "low dimensional" action sequence primitives. As an example, the family of low degree polynomials e.g. cubics. Or likewise in a Cauchy polynomial basis. Or a Fourier basis with only low frequency components. Or wavelet (inverse) transform coefficients. Or learned exemplar basis. Or learned auto encoding/latent space.

It would be lovely if we found a <10 dim rep.

Automate the search over such candidate representations, then create a new expansion/plug-in in the viewer for exploring the bases/manifolds.

