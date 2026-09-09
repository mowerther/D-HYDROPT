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
    load_water_iops,
    resample_coeffs,
    term_names,
)
from .bands import (  # noqa: E402
    MSI_FWHM_NM,
    build_gaussian_srf,
    classify_bands,
    msi_sigma_nm,
    srf_coverage,
)
from .domain import (  # noqa: E402
    RELIABLE_WAVELENGTH_RANGE,
    DomainError,
    check_domain,
    iops_from_excess,
    reliable_band_mask,
    rrs_from_excess,
)

__all__ = [
    'COEFFICIENT_FILES',
    'DomainError',
    'MSI_FWHM_NM',
    'RELIABLE_WAVELENGTH_RANGE',
    'build_gaussian_srf',
    'check_domain',
    'classify_bands',
    'iops_from_excess',
    'load_pace_poly',
    'load_water_iops',
    'reliable_band_mask',
    'monomial_powers_2d',
    'msi_sigma_nm',
    'polynomial_features_2d',
    'resample_coeffs',
    'rrs_from_excess',
    'rrs_poly',
    'rrs_poly_batched',
    'srf_coverage',
    'term_names',
]
