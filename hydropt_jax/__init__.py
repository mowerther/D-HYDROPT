"""Differentiable HYDROPT: the polynomial radiative transfer model in JAX."""
import os

import jax

# HYDROPT's polynomial is fitted in log space, and its coefficients grow to
# ~700 on the a^4 term at the red edge of the domain. log(Rrs) there
# legitimately reaches -100 and below. float32 exp() underflows to exactly
# zero at log(Rrs) < -103, float64 only at < -745, so single precision
# silently zeroes the red end of every spectrum -- value *and* gradient.
# Set HYDROPT_JAX_DISABLE_X64=1 to opt out.
if os.environ.get('HYDROPT_JAX_DISABLE_X64', '') not in ('1', 'true', 'True'):
    jax.config.update('jax_enable_x64', True)

from .core import (  # noqa: E402
    monomial_powers_2d,
    polynomial_features_2d,
    rrs_poly,
    rrs_poly_batched,
)
from .data import (  # noqa: E402
    COEFFICIENT_FILES,
    load_pace_poly,
    resample_coeffs,
    term_names,
)

__all__ = [
    'COEFFICIENT_FILES',
    'load_pace_poly',
    'monomial_powers_2d',
    'polynomial_features_2d',
    'resample_coeffs',
    'rrs_poly',
    'rrs_poly_batched',
    'term_names',
]
