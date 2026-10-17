"""Loading and resampling of the HYDROPT polynomial coefficient tables."""
import numpy as np
import jax.numpy as jnp
import pandas as pd

from hydropt.data import data_path

from .core import monomial_powers_2d

#: Coefficient tables shipped with HYDROPT, keyed by the total polynomial degree.
#:
#: Degree 4 is the reference table used throughout HYDROPT, and
#: tests/test_jax_equivalence.py compares it with the original NumPy
#: implementation. The degree 5 table ships with the package as a separate fit.
#: Its name lacks the '_h2o' suffix, and it predicts different Rrs for the same
#: a and bb. The tests cover degree 4 only, so treat degree 5 as experimental.
COEFFICIENT_FILES = {
    4: 'PACE_polynom_04_h2o.csv',
    5: 'PACE_polynom_05.csv',
}


def _term_name(i, j):
    '''Column label for the monomial a^i * bb^j, as used in the CSV headers.'''
    if i == 0 and j == 0:
        return '1'
    parts = []
    if i:
        parts.append('a' if i == 1 else 'a^{}'.format(i))
    if j:
        parts.append('bb' if j == 1 else 'bb^{}'.format(j))

    return ' '.join(parts)


def term_names(degree=4):
    '''
    Coefficient column names in monomial_powers_2d order.

    Derived from the exponents, not hard-coded, so the ordering of the
    loaded coefficients is guaranteed to match the ordering of the polynomial
    features they are multiplied by.
    '''
    return [_term_name(int(i), int(j)) for i, j in np.asarray(monomial_powers_2d(degree))]


def load_pace_poly(degree=4):
    '''
    Load a HYDROPT polynomial coefficient table.

    degree - total degree of the polynomial, as listed in COEFFICIENT_FILES.
             Degree 4 is validated and degree 5 is experimental.

    Returns (wavelength, coefficients), where wavelength has shape (n_bands,)
    in nm and coefficients has shape (n_bands, n_terms), with terms ordered to
    match monomial_powers_2d(degree).
    '''
    try:
        filename = COEFFICIENT_FILES[degree]
    except KeyError:
        raise ValueError(
            'no coefficient table for degree {}. Available degrees are {}'
            .format(degree, sorted(COEFFICIENT_FILES))) from None

    df = pd.read_csv(data_path(filename))
    lam = df['wavelength'].to_numpy()
    coeffs = df[term_names(degree)].to_numpy()

    return jnp.asarray(lam, jnp.float64), jnp.asarray(coeffs, jnp.float64)


def load_water_iops(wavelength=None):
    '''
    Pure water absorption and backscattering from the HYDROPT table.

    wavelength - wavelengths in nm to interpolate onto. None returns the table
                 on its native grid of 400-710 nm at 5 nm.

    Returns (a_water, bb_water), both in m^-1.

    Absorption follows Mason et al. (2016) at 400-550 nm and Pope and Fry
    (1997) at 555-710 nm. Backscattering follows Morel (1974), proportional
    to wavelength^-4.32.

    These values set the lower bound of the total IOPs, because natural waters
    absorb and backscatter at least as much as pure water. The polynomial of
    HYDROPT was fitted with water included, so its evaluation below this bound
    is extrapolation. See hydropt_jax.domain.
    '''
    df = pd.read_csv(data_path('water_mason016.csv'), index_col='wavelength')
    lam = df.index.to_numpy(float)
    a = df['a'].to_numpy(float)
    bb = df['bb'].to_numpy(float)
    if wavelength is None:
        return jnp.asarray(a, jnp.float64), jnp.asarray(bb, jnp.float64)

    target = np.asarray(wavelength, float)

    return (jnp.asarray(np.interp(target, lam, a), jnp.float64),
            jnp.asarray(np.interp(target, lam, bb), jnp.float64))


def resample_coeffs(lam_src, coeffs_src, lam_tgt):
    '''
    Linearly interpolate polynomial coefficients onto new wavelengths.

    lam_src    - source wavelengths, shape (n_src,)
    coeffs_src - source coefficients, shape (n_src, n_terms)
    lam_tgt    - target wavelengths, shape (n_tgt,)

    The function interpolates the *coefficients* of a fitted polynomial, not
    the reflectance it predicts. The largest coefficient stays near 3 from 400
    to 685 nm and grows to 15 at 690 nm and 707 at 710 nm. Coefficients
    interpolated between 685 and 710 nm therefore do not form a valid fit.
    Resample within 400-680 nm, the RELIABLE_WAVELENGTH_RANGE.
    '''
    lam_src = np.asarray(lam_src, float)
    lam_tgt = np.asarray(lam_tgt, float)
    coeffs_src = np.asarray(coeffs_src, float)
    out = np.empty((lam_tgt.size, coeffs_src.shape[1]), float)
    for k in range(coeffs_src.shape[1]):
        out[:, k] = np.interp(lam_tgt, lam_src, coeffs_src[:, k])

    return jnp.asarray(out, dtype=jnp.float64)
