# D-HYDROPT: Differentiable HYDROPT for fast gradient-based inversion of ocean colour satellite datasets.

**An experimental fork of HYDROPT implementing automatic differentiation using JAX**

[![license](https://img.shields.io/github/license/tadz-io/hydropt?label=license)](https://github.com/tadz-io/hydropt/blob/master/LICENSE)

## Description

D-HYDROPT is an experimental extension of the HYDROPT radiative transfer framework that leverages automatic differentiation (AD) for gradient-based inversion of satellite ocean colour observations. Built on [JAX](https://github.com/google/jax), D-HYDROPT enables efficient computation of sensitivities (Jacobian matrices) and gradient-based parameter optimisation without manual derivative implementation.

**Key differences from original HYDROPT:**
- Fully differentiable forward model using JAX
- Automatic computation of Jacobians ∂R<sub>rs</sub>/∂θ for sensitivity analysis
- Gradient-based optimisation (gradient descent with line search)
- Direct parameterisation of absorption a(λ) and backscattering b<sub>b</sub>(λ) spectra
- Support for spectral response functions (sensor band integration)

This implementation uses the HYDROPT polynomial approximation of radiative transfer in the 400-700 nm range, valid for nadir viewing geometry and 30° solar zenith angle.

### Original HYDROPT Framework

D-HYDROPT builds on the established HYDROPT framework developed by Van Der Woerd & Pasterkamp (2008) and Holtrop & Van Der Woerd (2021). Please cite the original work:

> Holtrop, T., & Van Der Woerd, H. J. (**2021**). HYDROPT: An Open-Source Framework for Fast Inverse Modelling of Multi- and Hyperspectral Observations from Oceans, Coastal and Inland Waters. *Remote Sensing*, 13(15), 3006. [doi:10.3390/rs13153006](https://www.mdpi.com/2072-4292/13/15/3006)

> Van Der Woerd, H.J. & Pasterkamp, R. (**2008**). HYDROPT: A fast and flexible method to retrieve chlorophyll-a from multispectral satellite observations of optically complex coastal waters. *Remote Sensing of Environment*, 112, 1795–1807. [doi:10.1016/j.rse.2007.09.001](https://www.sciencedirect.com/science/article/abs/pii/S003442570700421X?via%3Dihub)

```bibtex
@article{Holtrop_2021,
    title={HYDROPT: An Open-Source Framework for Fast Inverse Modelling of Multi- and Hyperspectral Observations from Oceans, Coastal and Inland Waters},
    author={Holtrop, Tadzio and Van Der Woerd, Hendrik Jan},
    journal={Remote Sensing}, 
    volume={13},
    number={15}, 
    month={Jul}, 
    pages={3006},
    year={2021}, 
    DOI={10.3390/rs13153006}, 
    publisher={MDPI AG}
}
```

## Features

- **Automatic differentiation**: Compute exact gradients through the radiative transfer model using JAX
- **Jacobian analysis**: Sensitivity maps showing how each sensor band responds to optical parameters
- **Gradient-based inversion**: Efficient parameter estimation using gradient descent with line search
- **Spectral response integration**: Native support for satellite band spectral response functions (Gaussian approximation)
- **Vectorised computation**: Fast batch processing of multiple spectra using `jax.vmap`
- **Wavelength range**: 400-700 nm (polynomial coefficients)
- **Direct spectral parameterisation**: Optimise full a(λ) and b<sub>b</sub>(λ) spectra with optional smoothness constraints

## Installation

Clone this repository and install dependencies:

```bash
git clone <repository-url>
cd D-HYDROPT
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- JAX (with GPU support optional)
- NumPy
- Pandas
- Matplotlib

## Getting Started

### Basic Forward Model

```python
import jax.numpy as jnp
from hydropt_jax.core import rrs_poly, monomial_powers_2d
from hydropt_jax.data import load_pace_poly

# Load HYDROPT polynomial coefficients
fine_wl, coeffs = load_pace_poly()
powers = monomial_powers_2d(degree=4)

# Define optical properties
a = jnp.ones_like(fine_wl) * 0.05    # absorption [m^-1]
bb = jnp.ones_like(fine_wl) * 0.005  # backscattering [m^-1]

# Compute Rrs
rrs = rrs_poly(a, bb, jnp.asarray(coeffs), jnp.asarray(powers))
```

### Gradient Computation

```python
import jax

# Define forward model in log-space for numerical stability
def forward(log_params):
    log_a, log_bb = log_params
    a = jnp.exp(log_a)
    bb = jnp.exp(log_bb)
    return rrs_poly(a, bb, coeffs, powers)

# Compute Jacobian automatically
jacobian_fn = jax.jacrev(forward)
log_params = jnp.stack([jnp.log(a), jnp.log(bb)])
J = jacobian_fn(log_params)  # Shape: (n_wavelengths, 2, n_wavelengths)
```

### Satellite Band Integration

```python
from hydropt_jax.core import build_gaussian_B, msi_sigma_nm
import numpy as np

# Sentinel-2 MSI band centres
band_wl = np.array([443, 492, 560, 665, 704, 740, 780])

# Build spectral response functions (Gaussian approximation)
sigma = msi_sigma_nm(band_wl)  # Per-band FWHM
B = build_gaussian_B(fine_wl, band_wl, sigma)

# Band-integrated Rrs
rrs_bands = rrs @ B.T  # Matrix multiplication
```

### Gradient-Based Inversion

See `examples/dhydropt_example_msi_final_plot.py` for a complete example including:
- Jacobian computation and visualisation
- Gradient descent optimisation with line search
- Pixel-level parameter retrieval from MSI observations

```python
# Example: Simple gradient descent
def loss_fn(log_params, target):
    pred = forward(log_params) @ B.T
    return jnp.mean((pred - target)**2)

grad_fn = jax.grad(loss_fn)

# Optimisation loop
for step in range(100):
    g = grad_fn(log_params, target_rrs)
    log_params = log_params - learning_rate * g
```

## Examples

Run the MSI sensitivity analysis and inversion example:

```bash
python examples/dhydropt_example_msi_final_plot.py --use_band_srf --gd_steps 120
```

This will:
1. Compute Jacobian statistics across an ensemble of optical states
2. Select a representative pixel
3. Perform gradient-based optimisation
4. Generate a comprehensive visualisation showing sensitivities and convergence

## Project Structure

```
D-HYDROPT/
├── hydropt/              # Original HYDROPT modules (retained for data/utilities)
│   ├── data/            # Polynomial coefficients, IOP data
│   └── bio_optics.py    # IOP definitions
├── hydropt_jax/         # JAX-based differentiable implementation
│   ├── core.py          # Forward model and polynomial evaluation
│   └── data.py          # Data loading utilities
├── examples/            # Usage examples and visualisation scripts
└── tests/               # Unit tests and validation
```

## Limitations

- **Geometry**: Nadir viewing, 30° solar zenith angle only
- **Wavelength range**: 400-700 nm (polynomial training domain)
- **Approximation**: Uses polynomial fit to full radiative transfer (see original papers for accuracy assessment)
- **Experimental**: This is research code; API may change

## Licence

[AGPL-3.0](./LICENSE)

## Acknowledgements

This implementation by Mortimer Werther extends the original HYDROPT framework. We thank Tadzio Holtrop and Hendrik Jan van der Woerd for developing and openly sharing HYDROPT.
