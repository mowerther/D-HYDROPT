"""
Spectral response functions and band selection.

The module represents the response of each Sentinel-2 MSI band in two ways:

  * Gaussians from the nominal widths in MSI_FWHM_NM, with msi_sigma_nm(),
    build_gaussian_srf(), srf_coverage() and classify_bands().
  * The ESA tables of the measured response of S2A, S2B and S2C, with
    load_msi_srf(), build_tabulated_srf(), tabulated_srf_coverage() and
    classify_tabulated_bands().

build_gaussian_srf() normalises each row over the coefficient grid it
receives. For a band with most of its response *outside* that grid, the row
places its weight on the last wavelengths of the grid, and the resulting value
approximates the reflectance at the end of the grid under the label of that
band. For Sentinel-2 MSI, the Gaussian approximations of bands 740 and 783
have 1e-6 and 4e-18 of their response inside the 400-710 nm table of HYDROPT.
Their rows correlate at 0.99967 because both approximate Rrs at 710 nm.

classify_bands() and classify_tabulated_bands() report the coverage of each
band. Verify it before using reflectance averaged over a band.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp

from .domain import RELIABLE_WAVELENGTH_RANGE

__all__ = [
    'CLIPPED_SRF_FRACTION',
    'FULL_SRF_FRACTION',
    'MSI_FWHM_NM',
    'build_gaussian_srf',
    'build_tabulated_srf',
    'classify_bands',
    'classify_tabulated_bands',
    'load_msi_srf',
    'msi_sigma_nm',
    'srf_coverage',
    'tabulated_srf_coverage',
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

# Tabulated response of bands B1 to B8A, at 1 nm, from the ESA Sentinel-2
# spectral response functions, version 4.0 (COPE-GSEG-EOPG-TN-15-0007, 2024).
_MSI_SRF_FILE = Path(__file__).with_name('msi_srf.csv')
_MSI_SATELLITES = ('S2A', 'S2B', 'S2C')
_MSI_BANDS = {443: 'B1', 490: 'B2', 560: 'B3', 665: 'B4', 705: 'B5', 740: 'B6',
              783: 'B7', 842: 'B8', 865: 'B8A'}


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
    The tabulated response functions of S2A, S2B and S2C differ from the
    Gaussian, so the fractions are approximate. tabulated_srf_coverage()
    gives the fractions of the tables.

    sigma_nm may be a scalar width shared by all bands.
    '''
    band = np.asarray(band_wl, float)
    sigma = np.broadcast_to(np.asarray(sigma_nm, float), band.shape)
    lo, hi = wavelength_range

    return np.array([_gaussian_mass(lo, hi, b, s) for b, s in zip(band, sigma)])


# --------------------------------------------------------------------------
# Tabulated spectral response
# --------------------------------------------------------------------------

def load_msi_srf(band_wl, satellite='S2A'):
    '''
    Tabulated spectral response functions of Sentinel-2 MSI.

    band_wl   - band centre in nm, or a sequence of them. Each centre maps to
                the nearest MSI band from B1 (443 nm) to B8A (865 nm).
    satellite - 'S2A', 'S2B' or 'S2C'

    Returns (srf_wl, response): wavelengths in nm at 1 nm spacing, shape
    (n_srf,), and the response of each band with a peak of 1, shape
    (n_bands, n_srf).

    The table holds bands B1 to B8A of the ESA Sentinel-2 spectral response
    functions, version 4.0 (COPE-GSEG-EOPG-TN-15-0007, 2024). ESA derived
    them from ground measurements of each instrument before launch, averaged
    over the detectors.
    '''
    if satellite not in _MSI_SATELLITES:
        raise ValueError('satellite must be one of {}, not {!r}'
                         .format(', '.join(_MSI_SATELLITES), satellite))
    table = pd.read_csv(_MSI_SRF_FILE)
    centres = [min(_MSI_BANDS, key=lambda c: abs(c - float(b))) for b in np.atleast_1d(band_wl)]
    columns = ['{}_{}'.format(satellite, _MSI_BANDS[c]) for c in centres]

    return table['wavelength'].to_numpy(float), table[columns].to_numpy(float).T


def _nodes(srf_wl, wavelength_range, extra=()):
    '''Tabulated wavelengths inside wavelength_range, plus its limits and extra.'''
    lo, hi = wavelength_range
    inside = srf_wl[(srf_wl > lo) & (srf_wl < hi)]

    return np.union1d(inside, np.concatenate(([lo, hi], np.asarray(extra, float))))


def _interpolate_rows(x, srf_wl, response):
    '''Each row of response, linearly interpolated onto x and zero outside srf_wl.'''
    return np.stack([np.interp(x, srf_wl, row, left=0.0, right=0.0) for row in response])


def build_tabulated_srf(fine_wl, srf_wl, response):
    '''
    Matrix B of shape (n_bands, n_wavelengths) for integration over tabulated
    spectral response functions.

    fine_wl  - wavelengths of the coefficient grid in nm, increasing
    srf_wl   - wavelengths of the tabulated response in nm, increasing
    response - tabulated response of each band, shape (n_bands, n_srf)

    Rrs @ B.T integrates each response against Rrs, linearly interpolated
    between the wavelengths of fine_wl, and divides by the integral of the
    response over the same range. Both factors are linear between nodes, so
    the integral of their product is exact. Each row sums to one and
    preserves a constant spectrum on uniform and non-uniform wavelength grids.

    The normalisation covers fine_wl only, as in build_gaussian_srf(). A band
    with response only outside fine_wl gives a row of NaN.
    tabulated_srf_coverage() reports the part of each response inside the
    grid.
    '''
    fine = np.asarray(fine_wl, float)
    srf_wl = np.asarray(srf_wl, float)
    response = np.atleast_2d(np.asarray(response, float))

    # Node k of the grid weighs Rrs with a hat function: 1 at fine[k], falling
    # linearly to 0 at the neighbouring nodes.
    x = _nodes(srf_wl, (fine[0], fine[-1]), extra=fine)
    hats = np.stack([np.interp(x, fine, row) for row in np.eye(fine.size)])
    values = _interpolate_rows(x, srf_wl, response)

    # Integral of the product of two functions that are linear on each
    # interval of x.
    f, g = values[:, None, :], hats[None, :, :]
    weights = np.sum(np.diff(x) * (2 * f[..., :-1] * g[..., :-1] + f[..., :-1] * g[..., 1:]
                                   + f[..., 1:] * g[..., :-1] + 2 * f[..., 1:] * g[..., 1:]),
                     axis=-1) / 6
    with np.errstate(invalid='ignore'):
        matrix = weights / np.sum(weights, axis=1, keepdims=True)

    return jnp.asarray(matrix, jnp.float64)


def tabulated_srf_coverage(srf_wl, response, wavelength_range):
    '''
    Fraction of each tabulated band response inside wavelength_range.

    srf_wl   - wavelengths of the tabulated response in nm, increasing
    response - tabulated response of each band, shape (n_bands, n_srf)

    The response is linear between the tabulated wavelengths, as in
    build_tabulated_srf().
    '''
    srf_wl = np.asarray(srf_wl, float)
    response = np.atleast_2d(np.asarray(response, float))
    x = _nodes(srf_wl, wavelength_range)
    inside = np.trapezoid(_interpolate_rows(x, srf_wl, response), x, axis=1)

    return inside / np.trapezoid(response, srf_wl, axis=1)


# --------------------------------------------------------------------------
# Band classification
# --------------------------------------------------------------------------

def _band_table(band_wl, coverage, reliable):
    '''One row per band with its coverage fractions and label.'''
    label = np.where(coverage >= FULL_SRF_FRACTION, 'full',
                     np.where(coverage >= CLIPPED_SRF_FRACTION, 'clipped', 'unusable'))

    return pd.DataFrame({'centre_nm': np.asarray(band_wl, float),
                         'coverage': coverage,
                         'reliable': reliable,
                         'class': label})


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

    return _band_table(band_wl, coverage, reliable)


def classify_tabulated_bands(band_wl, srf_wl, response, grid_wl):
    '''
    Label bands 'full', 'clipped' or 'unusable' by tabulated spectral coverage.

    band_wl  - band centres in nm, reported in the 'centre_nm' column
    srf_wl   - wavelengths of the tabulated response in nm, increasing
    response - tabulated response of each band, shape (n_bands, n_srf)
    grid_wl  - wavelengths of the coefficient grid in nm

    The columns and labels match those of classify_bands(), with fractions
    from tabulated_srf_coverage().

    Returns a DataFrame with one row per band.
    '''
    grid_range = (float(np.min(grid_wl)), float(np.max(grid_wl)))
    coverage = tabulated_srf_coverage(srf_wl, response, grid_range)
    reliable = tabulated_srf_coverage(srf_wl, response, RELIABLE_WAVELENGTH_RANGE)

    return _band_table(band_wl, coverage, reliable)
