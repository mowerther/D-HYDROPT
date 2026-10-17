"""
Spectral response functions and band selection.

build_gaussian_srf() normalises each row over the coefficient grid it
receives. For a band with most of its response *outside* that grid, the row
places its weight on the last wavelengths of the grid, and the resulting value
approximates the reflectance at the end of the grid under the label of that
band. For Sentinel-2 MSI, the Gaussian approximations of bands 740 and 783
have 1e-6 and 4e-18 of their response inside the 400-710 nm table of HYDROPT.
Their rows correlate at 0.99967 because both approximate Rrs at 710 nm.

classify_bands() reports the coverage of each band. Verify it before using
reflectance averaged over a band.
"""
import math

import numpy as np
import pandas as pd
import jax.numpy as jnp

from .domain import RELIABLE_WAVELENGTH_RANGE

__all__ = [
    'CLIPPED_SRF_FRACTION',
    'FULL_SRF_FRACTION',
    'MSI_FWHM_NM',
    'build_gaussian_srf',
    'classify_bands',
    'msi_sigma_nm',
    'srf_coverage',
]


#: Nominal Sentinel-2 MSI bandwidths (full width at half maximum) in nm, by
#: band centre (Drusch et al., 2012, doi:10.1016/j.rse.2011.11.026). The widths
#: measured for S2A and S2B differ from these by up to 2 nm for B1-B7 and by
#: about 10 nm for B8 (ESA, Sentinel-2 MSI spectral resolution). A Gaussian
#: built from these widths approximates the tabulated spectral response
#: function of each satellite.
MSI_FWHM_NM = {443: 20, 490: 65, 560: 35, 665: 30, 705: 15, 740: 15, 783: 20, 842: 115}

#: Coverage of the grid at or above which classify_bands labels a band 'full'.
FULL_SRF_FRACTION = 0.95

#: Bands with coverage from this value up to FULL_SRF_FRACTION are labelled
#: 'clipped': the end of the grid cuts their response, and their modelled
#: value carries a truncation bias. Bands below it are labelled 'unusable'.
#: Callers fit 'full' and 'clipped' bands and discard 'unusable' bands.
CLIPPED_SRF_FRACTION = 0.10


# --------------------------------------------------------------------------
# Spectral response
# --------------------------------------------------------------------------

def msi_sigma_nm(band_wl):
    '''
    Gaussian sigma in nm for each MSI band centre, from its nominal FWHM.

    Each band centre maps to the nearest entry in MSI_FWHM_NM. The centre
    492 nm therefore takes the 65 nm width of B2, and 704 nm the 15 nm width
    of B5.
    '''
    centres = [min(MSI_FWHM_NM, key=lambda c: abs(c - float(b))) for b in np.asarray(band_wl)]
    fwhm = np.array([MSI_FWHM_NM[c] for c in centres], float)

    return fwhm / np.sqrt(8.0 * np.log(2.0))


def build_gaussian_srf(fine_wl, band_wl, sigma_nm):
    '''
    Matrix B of shape (n_bands, n_wavelengths) for integration over each band.

    Each row includes the trapezoidal wavelength weights and sums to one.
    Rrs @ B.T therefore gives the reflectance averaged over each band, and
    preserves a constant spectrum on uniform and non-uniform wavelength grids.

    The normalisation covers fine_wl only. A band with most of its response
    outside that grid therefore produces a normalised row that averages the
    small part of the response inside the grid. srf_coverage() reports the
    size of that part.
    '''
    fine = jnp.asarray(fine_wl, jnp.float64)
    band = jnp.asarray(band_wl, jnp.float64)
    sigma = jnp.asarray(np.broadcast_to(np.asarray(sigma_nm, float), band.shape), jnp.float64)
    response = jnp.exp(-0.5 * ((fine[None, :] - band[:, None]) / sigma[:, None]) ** 2)

    # Include the integration measure in B itself: callers use a dot product,
    # not a second wavelength integral. Endpoints receive half the interval as
    # weight.
    spacing = jnp.diff(fine)
    weights = jnp.concatenate((spacing[:1], spacing[:-1] + spacing[1:], spacing[-1:])) / 2
    weighted_response = response * weights[None, :]

    return weighted_response / jnp.sum(weighted_response, axis=1, keepdims=True)


def _gaussian_mass(lo, hi, centre, sigma):
    '''
    Mass of a unit Gaussian between lo and hi.

    Uses erfc when the interval lies on one side of the centre. The difference
    of erf values cancels in the far tail. MSI 783 nm lies 8.6 sigma above
    710 nm, where erf(hi) - erf(lo) rounds to 0 and the erfc difference gives
    4e-18.
    '''
    scale = sigma * math.sqrt(2.0)
    if centre > hi:
        return 0.5 * (math.erfc((centre - hi) / scale) - math.erfc((centre - lo) / scale))
    if centre < lo:
        return 0.5 * (math.erfc((lo - centre) / scale) - math.erfc((hi - centre) / scale))

    return 0.5 * (math.erf((hi - centre) / scale) - math.erf((lo - centre) / scale))


def srf_coverage(band_wl, sigma_nm, wavelength_range):
    '''
    Fraction of the Gaussian approximation of each band response inside
    wavelength_range.

    Computed analytically from the error function. The fraction therefore
    covers the whole Gaussian, including the part outside the sampled grid.
    The tabulated response functions of S2A and S2B differ from the Gaussian,
    so the fractions are approximate.

    sigma_nm may be a scalar width shared by all bands.
    '''
    band = np.asarray(band_wl, float)
    sigma = np.broadcast_to(np.asarray(sigma_nm, float), band.shape)
    lo, hi = wavelength_range

    return np.array([_gaussian_mass(lo, hi, b, s) for b, s in zip(band, sigma)])


def classify_bands(band_wl, sigma_nm, grid_wl):
    '''
    Label bands 'full', 'clipped' or 'unusable' by spectral coverage.

    The function reports two fractions of the Gaussian approximation of each
    band response:

      coverage - fraction inside the coefficient grid. For a band with almost
                 no response inside the grid, integration over the grid
                 returns the reflectance at the red end of the grid.
      reliable - fraction inside RELIABLE_WAVELENGTH_RANGE, where log(Rrs)
                 stays above LOG_RRS_FLOOR over the envelope of excess IOPs
                 described in hydropt_jax.domain. A band with high coverage
                 and low reliable coverage can be modelled as long as excess
                 absorption stays low. check_domain() on the retrieved state
                 verifies this.

    The label follows from coverage: 'full' at or above FULL_SRF_FRACTION,
    'clipped' at or above CLIPPED_SRF_FRACTION, and 'unusable' below it.
    Callers fit 'full' and 'clipped' bands and discard 'unusable' bands. A
    clipped band averages Rrs over the part of its response inside the grid.

    Returns a DataFrame with one row per band.
    '''
    grid_range = (float(np.min(grid_wl)), float(np.max(grid_wl)))
    coverage = srf_coverage(band_wl, sigma_nm, grid_range)
    reliable = srf_coverage(band_wl, sigma_nm, RELIABLE_WAVELENGTH_RANGE)
    label = np.where(coverage >= FULL_SRF_FRACTION, 'full',
                     np.where(coverage >= CLIPPED_SRF_FRACTION, 'clipped', 'unusable'))

    return pd.DataFrame({'centre_nm': np.asarray(band_wl, float),
                         'coverage': coverage,
                         'reliable': reliable,
                         'class': label})
