"""Loading and resampling of the HYDROPT polynomial coefficient tables."""
import numpy as np
import jax.numpy as jnp
import pandas as pd

from hydropt.data import data_path

from .core import monomial_powers_2d

#: Coefficient tables shipped with HYDROPT, keyed by the total polynomial degree.
#:
#: Degree 4 is the reference table used throughout HYDROPT and validated against
#: the original NumPy implementation in tests/test_jax_equivalence.py. The
#: degree-5 table ships with the package but is a separate fit (note the missing
#: '_h2o' in its name -- it predicts noticeably different Rrs for the same a and
#: bb) and has not been validated here. Treat it as experimental.
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

    Derived from the exponents rather than hard-coded, so the ordering of the
    loaded coefficients is guaranteed to match the ordering of the polynomial
    features they are multiplied by.
    '''
    return [_term_name(int(i), int(j)) for i, j in np.asarray(monomial_powers_2d(degree))]


def load_pace_poly(degree=4):
    '''
    Load a HYDROPT polynomial coefficient table.

    degree - total degree of the polynomial; see COEFFICIENT_FILES. Only
             degree 4 is validated; degree 5 is experimental.

    Returns (wavelength, coefficients), where wavelength has shape (n_bands,)
    in nm and coefficients has shape (n_bands, n_terms), with terms ordered to
    match monomial_powers_2d(degree).
    '''
    try:
        filename = COEFFICIENT_FILES[degree]
    except KeyError:
        raise ValueError(
            'no coefficient table for degree {}; available degrees are {}'
            .format(degree, sorted(COEFFICIENT_FILES))) from None

    df = pd.read_csv(data_path(filename))
    lam = df['wavelength'].to_numpy()
    coeffs = df[term_names(degree)].to_numpy()

    return jnp.asarray(lam, jnp.float64), jnp.asarray(coeffs, jnp.float64)


def resample_coeffs(lam_src, coeffs_src, lam_tgt):
    '''
    Linearly interpolate polynomial coefficients onto new wavelengths.

    lam_src    - source wavelengths, shape (n_src,)
    coeffs_src - source coefficients, shape (n_src, n_terms)
    lam_tgt    - target wavelengths, shape (n_tgt,)

    Note that this interpolates the *coefficients* of a fitted polynomial, not
    the reflectance it predicts. The coefficients are smooth across most of the
    400-690 nm range but grow by an order of magnitude between 690 and 710 nm
    so interpolating across the red edge
    does not yield a valid fit. Resample within 400-690 nm only.
    '''
    lam_src = np.asarray(lam_src, float)
    lam_tgt = np.asarray(lam_tgt, float)
    coeffs_src = np.asarray(coeffs_src, float)
    out = np.empty((lam_tgt.size, coeffs_src.shape[1]), float)
    for k in range(coeffs_src.shape[1]):
        out[:, k] = np.interp(lam_tgt, lam_src, coeffs_src[:, k])

    return jnp.asarray(out, dtype=jnp.float64)
