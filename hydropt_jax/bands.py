"""
Spectral response functions and band selection.

Band integration is where a validity domain quietly stops being enforced.
build_gaussian_srf() normalises each row over the coefficient grid it is given,
so a band whose response lies mostly *outside* that grid still comes back as a
tidy normalised row -- and the reflectance it produces is not that band at all,
it is the reflectance at the edge of the grid wearing that band's label. For
Sentinel-2 MSI, bands 740 and 783 have 1e-6 and 4e-18 of their response inside
HYDROPT's 400-710 nm table, and the two rows correlate at 0.99967 because both
are the same delta function at 710 nm.

classify_bands() is the check that catches this. Consult it before trusting any
band-integrated result.
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


#: Published Sentinel-2 MSI full width at half maximum, in nm, by band centre.
MSI_FWHM_NM = {443: 20, 490: 65, 560: 35, 665: 30, 705: 15, 740: 15, 783: 20, 842: 115}

#: A band's response is treated as complete above this coverage of the grid.
FULL_SRF_FRACTION = 0.95

#: Between this and FULL_SRF_FRACTION the band is clipped by the end of the
#: grid: still modellable, but with a truncation bias. Below it, band
#: integration does not measure that band at all and it is dropped.
CLIPPED_SRF_FRACTION = 0.10


# --------------------------------------------------------------------------
# Spectral response
# --------------------------------------------------------------------------

def msi_sigma_nm(band_wl):
    '''
    Gaussian sigma in nm for each MSI band centre, from its published FWHM.

    Band centres are snapped to the nearest entry in MSI_FWHM_NM, so 492 picks
    up B2's 65 nm width and 704 picks up B5's 15 nm.
    '''
    centres = [min(MSI_FWHM_NM, key=lambda c: abs(c - float(b))) for b in np.asarray(band_wl)]
    fwhm = np.array([MSI_FWHM_NM[c] for c in centres], float)

    return fwhm / np.sqrt(8.0 * np.log(2.0))


def build_gaussian_srf(fine_wl, band_wl, sigma_nm):
    '''
    Band-integration matrix B of shape (n_bands, n_wavelengths).

    Each row includes the trapezoidal wavelength weights and sums to one.
    Rrs @ B.T therefore gives band-averaged reflectance, preserving a constant
    spectrum on both uniform and non-uniform wavelength grids.

    Note that the normalisation is over fine_wl only, so a band whose response
    mostly falls outside that grid still produces a normalised row. It will be
    a confident-looking average of whatever sliver happens to be in range --
    which is exactly why srf_coverage() below has to be consulted separately.
    '''
    fine = jnp.asarray(fine_wl, jnp.float64)
    band = jnp.asarray(band_wl, jnp.float64)
    sigma = jnp.asarray(np.broadcast_to(np.asarray(sigma_nm, float), band.shape), jnp.float64)
    response = jnp.exp(-0.5 * ((fine[None, :] - band[:, None]) / sigma[:, None]) ** 2)

    # Include the integration measure in B itself: callers use a dot product,
    # not a second wavelength integral. Endpoints receive half-interval weights.
    spacing = jnp.diff(fine)
    weights = jnp.concatenate((spacing[:1], spacing[:-1] + spacing[1:], spacing[-1:])) / 2
    weighted_response = response * weights[None, :]

    return weighted_response / jnp.sum(weighted_response, axis=1, keepdims=True)


def _gaussian_mass(lo, hi, centre, sigma):
    '''
    Mass of a unit Gaussian between lo and hi.

    Uses erfc when the interval lies wholly to one side of the centre. The
    plain erf difference cancels in the far tail -- MSI 783 nm sits 8.6 sigma
    above 710 nm, and erf(hi) - erf(lo) rounds that to exactly zero rather
    than 4e-18.
    '''
    scale = sigma * math.sqrt(2.0)
    if centre > hi:
        return 0.5 * (math.erfc((centre - hi) / scale) - math.erfc((centre - lo) / scale))
    if centre < lo:
        return 0.5 * (math.erfc((lo - centre) / scale) - math.erfc((hi - centre) / scale))

    return 0.5 * (math.erf((hi - centre) / scale) - math.erf((lo - centre) / scale))


def srf_coverage(band_wl, sigma_nm, wavelength_range):
    '''
    Fraction of each band's true Gaussian response inside wavelength_range.

    Computed analytically from the error function rather than from the sampled
    matrix, so it measures the real instrument response rather than the part
    that happened to be sampled.

    sigma_nm may be a scalar width shared by every band.
    '''
    band = np.asarray(band_wl, float)
    sigma = np.broadcast_to(np.asarray(sigma_nm, float), band.shape)
    lo, hi = wavelength_range

    return np.array([_gaussian_mass(lo, hi, b, s) for b, s in zip(band, sigma)])


def classify_bands(band_wl, sigma_nm, grid_wl):
    '''
    Sort bands into 'full', 'clipped' and 'unusable' by spectral coverage.

    Two coverage fractions are reported, because two different things can go
    wrong with a band:

      coverage - fraction of the band's response inside the coefficient grid.
                 This is the hard limit: a band with no response in the table
                 cannot be modelled at all, and band-integrating over the table
                 anyway just re-reports the reflectance at the table's red end.
      reliable - fraction inside RELIABLE_WAVELENGTH_RANGE, the sub-range whose
                 coefficients hold for any plausible IOP state. A band with
                 high coverage but low reliable coverage is modellable, but
                 only while excess absorption stays low; check_domain() on the
                 retrieved state is what confirms that after the fact.

    Bands classed 'full' or 'clipped' are both fitted -- a clipped band is a
    real measurement of a real part of the spectrum, just averaged over less of
    it than the instrument saw. Only 'unusable' bands are discarded.

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
