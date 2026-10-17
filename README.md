# D-HYDROPT

**Differentiable HYDROPT: inversion of remote-sensing reflectance by gradient descent, built on [JAX](https://github.com/jax-ml/jax).**

[![license](https://img.shields.io/github/license/mowerther/D-HYDROPT?label=license)](./LICENSE)

D-HYDROPT is an experimental fork of the [HYDROPT](https://github.com/tadz-io/hydropt)
radiative transfer framework. It reimplements the polynomial forward model of HYDROPT in JAX.
Automatic differentiation therefore provides the Jacobians ∂R<sub>rs</sub>/∂a and
∂R<sub>rs</sub>/∂b<sub>b</sub> to machine precision, for sensitivity analysis and retrieval by
gradient descent.

The original NumPy implementation remains in `hydropt/` as the reference. The tests compare the
JAX model with it, and both agree within a relative tolerance of 1e-10 for the forward model
and its gradient.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Validity domain of the polynomial](#validity-domain-of-the-polynomial)
- [Satellite band integration](#satellite-band-integration)
- [Worked example](#worked-example)
- [Repository structure](#repository-structure)
- [Limitations](#limitations)
- [References](#references)
- [Citing HYDROPT](#citing-hydropt)
- [Licence](#licence)

---

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/mowerther/D-HYDROPT.git
cd D-HYDROPT
pip install -e .
```

The installation includes NumPy, pandas, matplotlib, scikit-learn, xarray and JAX. For GPU or
TPU support, install the matching JAX wheel first, as described in the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

Run the tests:

```bash
pytest
```

`hydropt_jax` sets JAX to double precision (x64) when imported. The polynomial is fitted in
log space with coefficients up to about 700. In float32, `exp()` underflows to zero for
log(R<sub>rs</sub>) below about −103, and R<sub>rs</sub> can underflow to zero for absorbing
water. Double precision moves this limit to about −745. Set `HYDROPT_JAX_DISABLE_X64=1` to
override the setting.

---

## Quick start

```python
import jax.numpy as jnp
from hydropt_jax import load_pace_poly, load_water_iops, monomial_powers_2d, rrs_from_excess

# HYDROPT polynomial coefficients, on a 5 nm grid from 400 to 710 nm.
wavelength, coeffs = load_pace_poly()
powers = monomial_powers_2d(4)

# Pure water sets the physical floor on the IOPs. Parameterise what is on top of it.
a_water, bb_water = load_water_iops(wavelength)

log_excess_a = jnp.log(jnp.full(wavelength.shape, 0.05))    # m^-1 above pure water
log_excess_bb = jnp.log(jnp.full(wavelength.shape, 0.005))  # m^-1 above pure water

rrs = rrs_from_excess(log_excess_a, log_excess_bb, a_water, bb_water, coeffs, powers)
```

JAX differentiates the forward model directly:

```python
import jax

def forward(state):
    return rrs_from_excess(state[0], state[1], a_water, bb_water, coeffs, powers)

state = jnp.stack([log_excess_a, log_excess_bb])

jacobian = jax.jacrev(forward)(state)   # (n_wavelengths, 2, n_wavelengths)
```

`jax.jit` and `jax.vmap` apply to these functions. `rrs_poly_batched` applies `vmap` over many
states.

---

## Validity domain of the polynomial

HYDROPT approximates radiative transfer with a polynomial of degree 4 in log(a) and
log(b<sub>b</sub>), fitted separately at each wavelength. The polynomial is valid within the
range of IOPs it was fitted to. Outside that range, R<sub>rs</sub> = exp(log R<sub>rs</sub>)
can underflow to zero, and its gradient with it. An optimiser then receives a zero gradient
from those bands and no error.

Two limits bound the valid region.

**1. Pure water sets the lower bound of the IOPs.** Natural waters absorb and backscatter at
least as much as pure water, which absorbs 0.827 m⁻¹ at 710 nm (Pope and Fry, 1997).
`iops_from_excess()` expresses the IOPs as pure water plus a non-negative excess. Each finite
state and each gradient step therefore satisfy this bound:

```python
from hydropt_jax import iops_from_excess

a, bb = iops_from_excess(log_excess_a, log_excess_bb, a_water, bb_water)
# a >= a_water and bb >= bb_water for finite states and after each gradient step
```

`load_water_iops()` takes pure water absorption from Mason et al. (2016) at 400–550 nm and
from Pope and Fry (1997) at 555–710 nm, and pure water backscattering from Morel (1974).

**2. The coefficients grow above 680 nm.** The largest coefficient at each wavelength stays
near 3.0 from 400 to 685 nm and then grows steeply:

| wavelength | 400–680 nm | 685 nm | 690 nm | 695 nm | 700 nm | 705 nm | 710 nm |
|---|---|---|---|---|---|---|---|
| max &#124;coefficient&#124; | ≈ 3 | ≈ 3 | 15 | 66 | 172 | 328 | 707 |
| largest excess absorption with log(R<sub>rs</sub>) in [−30, 0], excess backscattering 0.005 m⁻¹ | > 6 m⁻¹ | 5.9 | 1.6 | 0.84 | 0.61 | 0.55 | 0.50 |

Over excess absorption up to 10 m⁻¹ and excess backscattering up to 0.5 m⁻¹,
log(R<sub>rs</sub>) stays above −30 from 400 to 680 nm. At 685 nm it reaches about −87,
although the coefficients there are still near 3. `RELIABLE_WAVELENGTH_RANGE` is therefore
`(400.0, 680.0)`. Above 680 nm the polynomial holds for low excess absorption, and
`check_domain()` verifies whether a given state satisfies the limits:

```python
from hydropt_jax import DomainError, check_domain, reliable_band_mask

# Collect problems without raising: returns [] for a state inside the domain.
problems = check_domain(a, bb, wavelength, coeffs, powers, error=False)

# Or restrict the state to RELIABLE_WAVELENGTH_RANGE.
mask = reliable_band_mask(wavelength)                  # 57 of the 63 bands
check_domain(a[mask], bb[mask], wavelength[mask], coeffs[mask], powers)   # passes
```

By default it raises `DomainError` with a message that names each problem and its
wavelengths:

```python
try:
    check_domain(jnp.full(wavelength.shape, 0.05),     # below pure water in the red
                 jnp.full(wavelength.shape, 0.005),
                 wavelength, coeffs, powers)
except DomainError as error:
    print(error)
```

```text
state is outside the HYDROPT polynomial domain:
  - absorption is below pure water at 34 of 63 wavelengths (worst: a = 0.05 m^-1 at 710 nm, where pure water alone is 0.827 m^-1). Parameterise with iops_from_excess() to prevent this.
  - 6 of 63 wavelengths lie outside the reliable range 400-680 nm (685, 690, 695, 700, 705, 710). Above 680 nm the coefficients grow by two orders of magnitude, and Rrs can underflow to zero for absorbing water.
  - log(Rrs) falls outside [-30, 0] at 6 of 63 wavelengths (largest excursion: -inf at 695 nm, i.e. Rrs = 0 sr^-1). Values outside this range arise from extrapolation of the polynomial. Dark water gives log(Rrs) within the range.
```

`check_domain()` evaluates eagerly, before or after an optimisation, outside `jit` and
`grad`. Inside the optimisation loop, `iops_from_excess()` keeps the state at or above the
pure water floor.

---

## Satellite band integration

D-HYDROPT approximates the spectral response functions of Sentinel-2 MSI with Gaussians built
from the nominal widths of Drusch et al. (2012):

```python
import numpy as np
from hydropt_jax import build_gaussian_srf, classify_bands, msi_sigma_nm

band_wl = np.array([443, 492, 560, 665, 704, 740, 783])
sigma = msi_sigma_nm(band_wl)                      # Gaussian from nominal MSI FWHM
srf = build_gaussian_srf(wavelength, band_wl, sigma)

rrs_bands = rrs @ srf.T
```

The matrix includes trapezoidal wavelength weights, so this product preserves
reflectance units and a constant spectrum on uniform or non-uniform grids.

**Verify the band coverage before using band values.** `build_gaussian_srf` normalises each
row over the coefficient grid. For a band with most of its response above 710 nm, the row
places its weight on the last wavelengths of the grid. The resulting value approximates
R<sub>rs</sub>(710) under the label of that band. For Sentinel-2 MSI:

| band | response inside 400–710 nm | class |
|---|---|---|
| 443, 492, 560, 665 nm | ≥ 99.9 % | `full` |
| 704 nm | 82.7 % | `clipped`, with a truncation bias |
| 740 nm | 0.0001 % | `unusable`, approximates R<sub>rs</sub>(710) |
| 783 nm | 0.0000 % | `unusable`, approximates R<sub>rs</sub>(710) |

The rows for 740 and 783 nm correlate at 0.99967 because both approximate R<sub>rs</sub> at the
end of the grid. `classify_bands()` reports these fractions and labels each band as `full`,
`clipped` or `unusable`.

---

## Worked example

```bash
python examples/msi_gradient_inversion.py
```

Loads Sentinel-2 MSI R<sub>rs</sub>, labels the bands, computes Jacobian statistics over an
ensemble of optical states, inverts one pixel by gradient descent, and writes
`examples/figures/msi_gradient_inversion.png`.

![MSI sensitivity and gradient inversion](examples/figures/msi_gradient_inversion.png)

**(A, B)** Median ∂R<sub>rs</sub><sup>band</sup>/∂ln Δa and ∂R<sub>rs</sub><sup>band</sup>/∂ln Δb<sub>b</sub>
across the ensemble, where Δa = a − a<sub>water</sub> and Δb<sub>b</sub> = b<sub>b</sub> −
b<sub>b,water</sub> are the excess IOPs above pure water. The colour axis is symmetric
logarithmic and depicts sensitivities that span two orders of magnitude from the blue bands
to the red bands. **(C)** MSI R<sub>rs</sub>, the initial state and the fit. The ringed point
is the 704 nm band, whose response the coefficient grid cuts short. **(D)** Convergence.

In the shaded region above 680 nm, the polynomial holds as long as excess absorption stays
low: up to 0.84 m⁻¹ at 695 nm, falling to 0.50 m⁻¹ at 710 nm. The 704 nm band lies in this
region. For the retrieved state of this pixel, `check_domain()` finds log(R<sub>rs</sub>)
within [−30, 0] at all 63 wavelengths and reports only the six wavelengths above 680 nm.

Useful flags: `--select {best,worst,median,improve,index}` selects the pixel to display,
`--gd-steps` and `--gd-lr` control the optimiser, and `--msi-csv` reads your own data.
Run with `--help` for the rest.

The inversion is underdetermined: the state has two free parameters at each of 63
wavelengths, constrained by five MSI bands. Many states fit these bands equally well. The
example illustrates exact gradients and stable convergence within the physical domain. A
well-posed retrieval reduces the state to a few bio-optical parameters, as
`hydropt.bio_optics` does.

---

## Repository structure

```
hydropt/            Original HYDROPT (NumPy/lmfit), retained as the reference implementation
  data/             Polynomial coefficients, pure water IOPs, phytoplankton SIOPs
  hydropt.py        BioOpticalModel, PolynomialReflectance, InversionModel
  bio_optics.py     IOP models for water, phytoplankton, NAP, CDOM
hydropt_jax/        Differentiable implementation
  core.py           Polynomial features and the rrs_poly kernel
  data.py           Coefficient tables and pure water IOPs
  domain.py         Water floor, domain validation, reliable wavelength range
  bands.py          Spectral response functions and band coverage classification
examples/           Worked example and sample data
tests/              Unit tests, equivalence of the JAX and NumPy implementations, domain guards
```

`rrs_poly` contains only array operations, so JAX traces it under `jit`, `grad` and `vmap`.
Functions that branch on values are in `hydropt_jax.domain`.

---

## Limitations

- **Geometry:** nadir view, 30° solar zenith angle only. The coefficients encode it.
- **Wavelengths:** the coefficients are tabulated at 400–710 nm in 5 nm steps. The range
  400–680 nm is reliable over the envelope of IOPs described above.
- **Approximation:** a polynomial fitted to HydroLight simulations of radiative transfer
  (Holtrop and Van der Woerd, 2021). See the papers below for its accuracy.
- **Table of degree 5:** `PACE_polynom_05.csv` ships with the package and loads with
  `load_pace_poly(5)`. It is a separate fit, and the tests compare only the table of degree 4
  with the reference implementation. Treat it as experimental.
- **Experimental:** D-HYDROPT is research code, and the API may change.

---

## References

- Drusch, M., et al. (2012). Sentinel-2: ESA's Optical High-Resolution Mission for GMES
  Operational Services. *Remote Sensing of Environment*, 120, 25–36.
  [doi:10.1016/j.rse.2011.11.026](https://doi.org/10.1016/j.rse.2011.11.026)
- Mason, J. D., Cone, M. T., & Fry, E. S. (2016). Ultraviolet (250–550 nm) absorption
  spectrum of pure water. *Applied Optics*, 55(25), 7163–7172.
  [doi:10.1364/AO.55.007163](https://doi.org/10.1364/AO.55.007163)
- Morel, A. (1974). Optical properties of pure water and pure sea water. In N. G. Jerlov &
  E. Steemann Nielsen (Eds.), *Optical Aspects of Oceanography* (pp. 1–24). Academic Press.
- Pope, R. M., & Fry, E. S. (1997). Absorption spectrum (380–700 nm) of pure water. II.
  Integrating cavity measurements. *Applied Optics*, 36(33), 8710–8723.
  [doi:10.1364/AO.36.008710](https://doi.org/10.1364/AO.36.008710)

---

## Citing HYDROPT

D-HYDROPT builds on HYDROPT, developed by Van der Woerd & Pasterkamp (2008) and
Holtrop & Van der Woerd (2021). Please cite the original work:

> Holtrop, T., & Van Der Woerd, H. J. (**2021**). HYDROPT: An Open-Source Framework for Fast
> Inverse Modelling of Multi- and Hyperspectral Observations from Oceans, Coastal and Inland
> Waters. *Remote Sensing*, 13(15), 3006.
> [doi:10.3390/rs13153006](https://www.mdpi.com/2072-4292/13/15/3006)

> Van Der Woerd, H. J., & Pasterkamp, R. (**2008**). HYDROPT: A fast and flexible method to
> retrieve chlorophyll-a from multispectral satellite observations of optically complex
> coastal waters. *Remote Sensing of Environment*, 112, 1795–1807.
> [doi:10.1016/j.rse.2007.09.001](https://www.sciencedirect.com/science/article/abs/pii/S003442570700421X)

```bibtex
@article{Holtrop_2021,
    title={HYDROPT: An Open-Source Framework for Fast Inverse Modelling of Multi- and
           Hyperspectral Observations from Oceans, Coastal and Inland Waters},
    author={Holtrop, Tadzio and Van Der Woerd, Hendrik Jan},
    journal={Remote Sensing},
    volume={13}, number={15}, pages={3006}, month={Jul}, year={2021},
    DOI={10.3390/rs13153006}, publisher={MDPI AG}
}
```

---

## Licence

[AGPL-3.0](./LICENSE)

## Acknowledgements

This implementation by Mortimer Werther extends the original HYDROPT framework. Thanks to
Tadzio Holtrop and Hendrik Jan Van der Woerd for developing and openly sharing HYDROPT.
