"""
Validity domain of the HYDROPT polynomial.

HYDROPT approximates radiative transfer with a polynomial of degree 4 in
``log(a)`` and ``log(bb)``, fitted separately at each wavelength. The
polynomial is valid within the range of IOPs it was fitted to. ``rrs_poly``
evaluates the polynomial at the IOPs it receives and returns a number,
whether or not the IOPs lie in that range.

Two conditions bound the valid region.

**Pure water floor.** Natural waters absorb and backscatter at least as much
as pure water. ``BioOpticalModel.sum_iop(incl_water=True)`` adds the water
IOPs in the original. The JAX kernel takes total IOPs, and
:func:`iops_from_excess` provides the same guarantee: it parameterises the
state as water plus a non-negative excess.

**Above 680 nm.** The degree 4 coefficients grow at the red end of the table.
The largest coefficient at each wavelength stays near 3.0 from 400 to 685 nm,
then grows to 15 at 690 nm, 66 at 695, 172 at 700, 328 at 705 and 707 at 710.
Over a wide envelope of physically plausible excess IOPs above the water floor
(``da`` up to 10 m^-1, ``dbb`` up to 0.5 m^-1), ``log(Rrs)`` stays above -30
from 400 to 680 nm and reaches about -87 at 685 nm, -665 at 690 nm and -31000
at 710 nm. ``rrs_poly`` returns ``exp(log_rrs)``, which underflows to zero in
value and gradient from 685 nm in float32 and from 695 nm in float64. An
inversion by gradient descent then receives a zero gradient from those bands
and no error.

:data:`RELIABLE_WAVELENGTH_RANGE` records the range 400-680 nm.
:func:`check_domain` detects both conditions and raises :class:`DomainError`
with a message that names each problem.
"""
import warnings

import numpy as np
import jax.numpy as jnp

from .core import rrs_poly
from .data import load_water_iops

#: Wavelength range (nm) over which the coefficients of degree 4 are
#: numerically reliable, derived empirically as described in the module
#: docstring.
RELIABLE_WAVELENGTH_RANGE = (400.0, 680.0)

#: log(Rrs) below this value indicates extrapolation of the polynomial. Rrs =
#: exp(-30) is about 1e-13 sr^-1, at least seven orders of magnitude below the
#: noise of ocean colour sensors.
LOG_RRS_FLOOR = -30.0

#: log(Rrs) above this is unphysical in the other direction: Rrs = 1 sr^-1 is
#: well above the brightest natural water.
LOG_RRS_CEILING = 0.0


class DomainError(ValueError):
    """Raised when IOPs or wavelengths fall outside the fitted domain of the model."""


def reliable_band_mask(wavelength, wavelength_range=RELIABLE_WAVELENGTH_RANGE):
    '''
    Boolean mask selecting wavelengths where the coefficients are reliable.

    wavelength       - wavelengths in nm
    wavelength_range - (min, max) in nm, by default RELIABLE_WAVELENGTH_RANGE
    '''
    lam = np.asarray(wavelength, float)
    lo, hi = wavelength_range

    return (lam >= lo) & (lam <= hi)


def iops_from_excess(log_da, log_dbb, a_water, bb_water):
    '''
    Total IOPs from log excess IOPs, with the pure water floor enforced.

    log_da   - log of absorption in excess of pure water, shape (n_bands,)
    log_dbb  - log of backscattering in excess of pure water, shape (n_bands,)
    a_water  - pure water absorption at the same wavelengths, m^-1
    bb_water - pure water backscattering at the same wavelengths, m^-1

    Returns (a, bb) in m^-1, which satisfy a >= a_water and bb >= bb_water for
    finite input. The bound includes equality because exp(log_da) underflows
    to 0 for log_da below about -745.

    Optimise in this differentiable parameterisation. exp() is non-negative,
    so the state stays at or above the physical floor for finite log_da and
    log_dbb and after each gradient step, however large. An optimiser that
    works on log(a) directly can move the state below pure water and out of
    the fitted domain.
    '''
    a = jnp.asarray(a_water) + jnp.exp(jnp.asarray(log_da))
    bb = jnp.asarray(bb_water) + jnp.exp(jnp.asarray(log_dbb))

    return a, bb


def rrs_from_excess(log_da, log_dbb, a_water, bb_water, coeffs, powers):
    '''
    Forward model parameterised by log excess IOPs over the pure water floor.

    A convenience wrapper over iops_from_excess() and rrs_poly(). See those
    for the arguments. Fully differentiable, and safe to jit and vmap.
    '''
    a, bb = iops_from_excess(log_da, log_dbb, a_water, bb_water)

    return rrs_poly(a, bb, coeffs, powers)


def check_domain(a, bb, wavelength, coeffs=None, powers=None, error=True):
    '''
    Verify that a state lies inside the fitted domain of the model.

    a, bb      - total IOPs in m^-1, shape (n_bands,)
    wavelength - wavelengths in nm, shape (n_bands,)
    coeffs     - coefficient table. If given together with powers,
                 check_domain also verifies that log(Rrs) lies within
                 [LOG_RRS_FLOOR, LOG_RRS_CEILING]
    powers     - monomial exponents matching coeffs
    error      - raise DomainError on failure (default). If False, emit a
                 UserWarning and return the list of problems

    Returns a list of messages that describe each problem, empty if the state
    lies inside the domain.

    check_domain evaluates eagerly in NumPy, before or after an optimisation.
    Under jit and grad, JAX traces the inputs, and branching on traced values
    raises an error. Inside an optimisation, iops_from_excess() keeps the
    state at or above the water floor by construction.
    '''
    a = np.asarray(a, float)
    bb = np.asarray(bb, float)
    lam = np.asarray(wavelength, float)
    problems = []

    finite_a, finite_bb = np.isfinite(a), np.isfinite(bb)
    for name, finite in (('absorption', finite_a), ('backscattering', finite_bb)):
        if not finite.all():
            problems.append(
                '{} is NaN or infinite at {} of {} wavelengths ({} nm). '
                'Evaluate the forward model only on finite IOPs.'
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
            '({}). Above {:.0f} nm the coefficients grow by two orders of '
            'magnitude, and Rrs can underflow to zero for absorbing water.'
            .format(int(outside.sum()), lam.size, lo, hi,
                    ', '.join('{:.0f}'.format(v) for v in lam[outside][:6])
                    + (' ...' if outside.sum() > 6 else ''), hi))

    if coeffs is not None and powers is not None and finite_a.all() and finite_bb.all():
        rrs = np.asarray(rrs_poly(jnp.asarray(a), jnp.asarray(bb), coeffs, powers), float)
        with np.errstate(divide='ignore'):
            log_rrs = np.log(rrs)
        out_of_range = (~np.isfinite(log_rrs) | (log_rrs < LOG_RRS_FLOOR)
                        | (log_rrs > LOG_RRS_CEILING))
        if out_of_range.any():
            excursion = np.maximum(LOG_RRS_FLOOR - log_rrs, log_rrs - LOG_RRS_CEILING)
            worst = int(np.argmax(np.where(np.isfinite(log_rrs), excursion, np.inf)))
            problems.append(
                'log(Rrs) falls outside [{:.0f}, {:.0f}] at {} of {} wavelengths '
                '(largest excursion: {:.1f} at {:.0f} nm, i.e. Rrs = {:.3g} sr^-1). '
                'Values outside this range arise from extrapolation of the '
                'polynomial. Dark water gives log(Rrs) within the range.'
                .format(LOG_RRS_FLOOR, LOG_RRS_CEILING, int(out_of_range.sum()), lam.size,
                        log_rrs[worst], lam[worst], rrs[worst]))

    if problems:
        message = 'state is outside the HYDROPT polynomial domain:\n  - ' + '\n  - '.join(problems)
        if error:
            raise DomainError(message)
        warnings.warn(message, stacklevel=2)

    return problems
