"""
Validity domain of the HYDROPT polynomial.

HYDROPT approximates radiative transfer with a degree-4 polynomial in
``log(a)`` and ``log(bb)``, fitted per wavelength. Like any polynomial fit it
is only meaningful inside the region it was trained on, and ``rrs_poly`` has no
way to tell you when you have left it: it happily evaluates the polynomial
anywhere and returns a number.

Two things go wrong outside the fitted region, and both are silent.

**Below the pure-water floor.** No natural water body absorbs or backscatters
less than pure water. The original HYDROPT could not violate this because
``BioOpticalModel.sum_iop(incl_water=True)`` always added the water IOPs before
calling the forward model. The JAX kernel takes total ``a`` and ``bb``
directly, so that guarantee was lost. Use :func:`iops_from_excess` to get it
back: it parameterises the state as water plus a non-negative excess.

**Beyond 680 nm.** The shipped degree-4 coefficients degenerate at the red edge
of the table. The largest coefficient per wavelength sits near 3.0 from 400 nm
all the way to 685 nm, then grows to 15 at 690 nm, 66 at 695, 172 at 700, 328
at 705 and 707 at 710. Sweeping a wide envelope of physically plausible excess
IOPs (``da`` up to 10 m^-1, ``dbb`` up to 0.5 m^-1) over the water floor,
``log(Rrs)`` stays above -30 everywhere from 400 to 680 nm, but reaches -87 at
685 nm, -665 at 690 nm and -31000 at 710 nm. Since ``rrs_poly`` returns
``exp(log_rrs)``, that underflows to *exactly zero* -- value and gradient
alike -- from 685 nm in float32 and from 695 nm in float64. A gradient-based
inversion given those bands receives no signal from them at all and cannot
report that anything went wrong.

:data:`RELIABLE_WAVELENGTH_RANGE` records the 400-680 nm conclusion, and
:func:`check_domain` turns both failure modes into a loud error instead of a
silent zero.
"""
import warnings

import numpy as np
import jax.numpy as jnp

from .core import rrs_poly
from .data import load_water_iops

#: Wavelength range (nm) over which the degree-4 coefficients are numerically
#: reliable. Derived empirically; see the module docstring.
RELIABLE_WAVELENGTH_RANGE = (400.0, 680.0)

#: log(Rrs) below this is treated as the polynomial having run away rather than
#: as a genuinely dark spectrum. Rrs = exp(-30) is 1e-13 sr^-1, some seven
#: orders of magnitude below the noise floor of any ocean colour sensor.
LOG_RRS_FLOOR = -30.0

#: log(Rrs) above this is unphysical in the other direction: Rrs = 1 sr^-1 is
#: well above the brightest natural water.
LOG_RRS_CEILING = 0.0


class DomainError(ValueError):
    """Raised when IOPs or wavelengths fall outside the model's fitted domain."""


def reliable_band_mask(wavelength, wavelength_range=RELIABLE_WAVELENGTH_RANGE):
    '''
    Boolean mask selecting wavelengths where the coefficients are reliable.

    wavelength       - wavelengths in nm
    wavelength_range - (min, max) in nm; defaults to RELIABLE_WAVELENGTH_RANGE
    '''
    lam = np.asarray(wavelength, float)
    lo, hi = wavelength_range

    return (lam >= lo) & (lam <= hi)


def iops_from_excess(log_da, log_dbb, a_water, bb_water):
    '''
    Total IOPs from log excess IOPs, with the pure-water floor enforced.

    log_da   - log of absorption in excess of pure water, shape (n_bands,)
    log_dbb  - log of backscattering in excess of pure water, shape (n_bands,)
    a_water  - pure water absorption at the same wavelengths, m^-1
    bb_water - pure water backscattering at the same wavelengths, m^-1

    Returns (a, bb) in m^-1, guaranteed to satisfy a > a_water and
    bb > bb_water for any finite input.

    This is the differentiable parameterisation to optimise in. Because
    exp() is positive everywhere, no choice of log_da or log_dbb -- and no
    gradient step, however large -- can push the state below the physical
    floor. Compare optimising log(a) directly, which lets the optimiser walk
    straight out of the fitted domain.
    '''
    a = jnp.asarray(a_water) + jnp.exp(jnp.asarray(log_da))
    bb = jnp.asarray(bb_water) + jnp.exp(jnp.asarray(log_dbb))

    return a, bb


def rrs_from_excess(log_da, log_dbb, a_water, bb_water, coeffs, powers):
    '''
    Forward model parameterised by log excess IOPs over the pure-water floor.

    A convenience wrapper over iops_from_excess() and rrs_poly(); see those for
    the arguments. Fully differentiable, and safe to jit and vmap.
    '''
    a, bb = iops_from_excess(log_da, log_dbb, a_water, bb_water)

    return rrs_poly(a, bb, coeffs, powers)


def check_domain(a, bb, wavelength, coeffs=None, powers=None, error=True):
    '''
    Check that a state lies inside the model's fitted domain.

    a, bb      - total IOPs in m^-1, shape (n_bands,)
    wavelength - wavelengths in nm, shape (n_bands,)
    coeffs     - coefficient table; if given together with powers, the
                 resulting log(Rrs) is checked for runaway values too
    powers     - monomial exponents matching coeffs
    error      - raise DomainError on failure (default); if False, emit a
                 UserWarning and return the list of problems instead

    Returns a list of human-readable problem descriptions, empty if the state
    is fine.

    This is an eager NumPy check for use before or after an optimisation, not
    inside one. It cannot be called on traced values under jit or grad, where
    data-dependent branching is not allowed -- use iops_from_excess() there to
    make the failure impossible in the first place.
    '''
    a = np.asarray(a, float)
    bb = np.asarray(bb, float)
    lam = np.asarray(wavelength, float)
    problems = []

    finite_a, finite_bb = np.isfinite(a), np.isfinite(bb)
    for name, finite in (('absorption', finite_a), ('backscattering', finite_bb)):
        if not finite.all():
            problems.append(
                '{} is non-finite at {} of {} wavelengths ({} nm). '
                'All IOPs must be finite before evaluating the forward model.'
                .format(name, int((~finite).sum()), lam.size,
                        ', '.join('{:.0f}'.format(v) for v in lam[~finite])))

    a_water, bb_water = (np.asarray(v, float) for v in load_water_iops(lam))

    below_a = finite_a & (a < a_water)
    if below_a.any():
        worst = int(np.argmin(np.where(below_a, a / a_water, np.inf)))
        problems.append(
            'absorption is below pure water at {} of {} wavelengths '
            '(worst: a = {:.4g} m^-1 at {:.0f} nm, where pure water alone is '
            '{:.4g} m^-1). Parameterise with iops_from_excess() to prevent this.'
            .format(int(below_a.sum()), lam.size, a[worst], lam[worst], a_water[worst]))

    below_bb = finite_bb & (bb < bb_water)
    if below_bb.any():
        worst = int(np.argmin(np.where(below_bb, bb / bb_water, np.inf)))
        problems.append(
            'backscattering is below pure water at {} of {} wavelengths '
            '(worst: bb = {:.4g} m^-1 at {:.0f} nm, where pure water alone is '
            '{:.4g} m^-1). Parameterise with iops_from_excess() to prevent this.'
            .format(int(below_bb.sum()), lam.size, bb[worst], lam[worst], bb_water[worst]))

    outside = ~reliable_band_mask(lam)
    if outside.any():
        lo, hi = RELIABLE_WAVELENGTH_RANGE
        problems.append(
            '{} of {} wavelengths lie outside the reliable range {:.0f}-{:.0f} nm '
            '({}). The degree-4 coefficients degenerate past {:.0f} nm and Rrs '
            'underflows to exactly zero there.'
            .format(int(outside.sum()), lam.size, lo, hi,
                    ', '.join('{:.0f}'.format(v) for v in lam[outside][:6])
                    + (' ...' if outside.sum() > 6 else ''), hi))

    if coeffs is not None and powers is not None and finite_a.all() and finite_bb.all():
        rrs = np.asarray(rrs_poly(jnp.asarray(a), jnp.asarray(bb), coeffs, powers), float)
        with np.errstate(divide='ignore'):
            log_rrs = np.log(rrs)
        runaway = ~np.isfinite(log_rrs) | (log_rrs < LOG_RRS_FLOOR) | (log_rrs > LOG_RRS_CEILING)
        if runaway.any():
            excursion = np.maximum(LOG_RRS_FLOOR - log_rrs, log_rrs - LOG_RRS_CEILING)
            worst = int(np.argmax(np.where(np.isfinite(log_rrs), excursion, np.inf)))
            problems.append(
                'the polynomial has run away at {} of {} wavelengths: log(Rrs) '
                'leaves [{:.0f}, {:.0f}] (worst: {:.1f} at {:.0f} nm, i.e. Rrs = {:.3g} sr^-1). '
                'This means the polynomial is extrapolating, not that the water is dark.'
                .format(int(runaway.sum()), lam.size, LOG_RRS_FLOOR, LOG_RRS_CEILING,
                        log_rrs[worst], lam[worst], rrs[worst]))

    if problems:
        message = 'state is outside the HYDROPT polynomial domain:\n  - ' + '\n  - '.join(problems)
        if error:
            raise DomainError(message)
        warnings.warn(message, stacklevel=2)

    return problems
