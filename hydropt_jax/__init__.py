"""Differentiable HYDROPT: the polynomial radiative transfer model in JAX."""
import os

import jax

# The polynomial of HYDROPT is fitted in log space, and its coefficients grow
# to about 700 on the a^4 term at the red end of the grid. Where the
# polynomial extrapolates there, log(Rrs) reaches -100 and below. exp()
# underflows to zero below log(Rrs) = -103 in float32 and below -745 in
# float64. In single precision, Rrs and its gradient can therefore underflow
# to zero for absorbing water. Set HYDROPT_JAX_DISABLE_X64=1 to opt out.
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
    build_tabulated_srf,
    classify_bands,
    classify_tabulated_bands,
    load_msi_srf,
    msi_sigma_nm,
    srf_coverage,
    tabulated_srf_coverage,
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
    'build_tabulated_srf',
    'check_domain',
    'classify_bands',
    'classify_tabulated_bands',
    'iops_from_excess',
    'load_msi_srf',
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
    'tabulated_srf_coverage',
    'term_names',
]
