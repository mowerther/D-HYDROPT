"""
The JAX fork must reproduce the original HYDROPT, exactly.

D-HYDROPT's whole claim is 'same radiative transfer, now differentiable'. That
claim is only worth anything if it is checked, so these tests pin the JAX
implementation to the NumPy one it was ported from -- coefficient ordering,
forward model, and gradient -- and pin the gradient to finite differences so
that a change agreed on by both implementations still gets caught.
"""
import unittest

import numpy as np
import jax
import jax.numpy as jnp

from hydropt.hydropt import PolynomialReflectance
from hydropt_jax import (
    load_pace_poly,
    load_water_iops,
    monomial_powers_2d,
    reliable_band_mask,
    rrs_poly,
    rrs_poly_batched,
)

# Both implementations are double precision, so agreement should be at the
# level of floating point reassociation, not of physics.
RTOL = 1e-10


def plausible_spectra(seed=0, n=25):
    '''
    Smooth spectra inside the fitted domain, on the full coefficient grid.

    Built as pure water plus a non-negative excess, so the water floor holds by
    construction. The excess absorption decays exponentially with wavelength
    and backscattering more gently, which is the broad shape of real coastal
    water; magnitudes span clear ocean through to turbid inland water.

    The comparisons below are restricted to reliable_band_mask(). Past 680 nm
    the coefficients degenerate and Rrs underflows into the subnormal range,
    where NumPy and JAX round differently -- the two implementations disagree
    there only because neither has anything meaningful to say. See
    tests/test_domain.py, which covers that region directly.
    '''
    lam, _ = load_pace_poly()
    lam = np.asarray(lam)
    a_water, bb_water = (np.asarray(v) for v in load_water_iops(lam))
    rng = np.random.default_rng(seed)
    for _ in range(n):
        excess_a = np.exp(rng.uniform(np.log(0.02), np.log(2.0))
                          - 0.004 * (lam - 400) + 0.05 * rng.standard_normal())
        excess_bb = np.exp(rng.uniform(np.log(2e-3), np.log(5e-2)) - 0.002 * (lam - 400))
        yield a_water + excess_a, bb_water + excess_bb


def reliable(array):
    '''Select the wavelengths where the coefficients are trustworthy.'''
    lam, _ = load_pace_poly()

    return np.asarray(array)[..., reliable_band_mask(lam)]


class TestCoefficientOrdering(unittest.TestCase):
    '''The two implementations must agree on what column means what.'''

    def test_powers_match_sklearn(self):
        # The NumPy model gets its exponents from sklearn; the JAX model
        # generates them by hand. If those ever diverge, every coefficient is
        # silently paired with the wrong monomial.
        np.testing.assert_array_equal(
            np.asarray(monomial_powers_2d(4)),
            np.asarray(PolynomialReflectance()._powers))

    def test_coefficient_tables_are_identical(self):
        _, coeffs = load_pace_poly()
        np.testing.assert_array_equal(
            np.asarray(coeffs),
            np.asarray(PolynomialReflectance()._parameters.values))


class TestForwardModel(unittest.TestCase):

    def setUp(self):
        self.reference = PolynomialReflectance()
        self.lam, self.coeffs = load_pace_poly()
        self.powers = monomial_powers_2d(4)

    def test_matches_numpy_implementation(self):
        for a, bb in plausible_spectra(seed=1):
            expected = self.reference.forward(np.vstack([a, bb]))
            actual = np.asarray(rrs_poly(
                jnp.asarray(a), jnp.asarray(bb), self.coeffs, self.powers))
            np.testing.assert_allclose(reliable(actual), reliable(expected), rtol=RTOL)

    def test_batched_matches_single(self):
        states = [np.vstack([a, bb]) for a, bb in plausible_spectra(seed=2, n=8)]
        batched = np.asarray(rrs_poly_batched(
            jnp.asarray(np.stack(states)), self.coeffs, self.powers))
        for i, state in enumerate(states):
            single = np.asarray(rrs_poly(
                jnp.asarray(state[0]), jnp.asarray(state[1]), self.coeffs, self.powers))
            np.testing.assert_allclose(reliable(batched[i]), reliable(single), rtol=RTOL)


class TestGradient(unittest.TestCase):

    def setUp(self):
        self.reference = PolynomialReflectance()
        self.lam, self.coeffs = load_pace_poly()
        self.powers = monomial_powers_2d(4)

    def _autodiff_gradient(self, a, bb):
        '''dRrs/da and dRrs/dbb at each wavelength, via reverse-mode AD.'''
        def forward(state):
            return rrs_poly(state[0], state[1], self.coeffs, self.powers)

        jacobian = np.asarray(jax.jacrev(forward)(jnp.asarray(np.vstack([a, bb]))))

        # Rrs at one wavelength depends only on the IOPs at that wavelength, so
        # the per-wavelength derivative is the diagonal of each block.
        return np.stack([np.diagonal(jacobian[:, 0, :]), np.diagonal(jacobian[:, 1, :])])

    def test_matches_analytic_gradient(self):
        for a, bb in plausible_spectra(seed=3):
            expected = reliable(self.reference.gradient(np.vstack([a, bb])))
            actual = reliable(self._autodiff_gradient(a, bb))
            # Compare relative to the larger of the two, so wavelengths where
            # the gradient is negligible do not blow the relative error up.
            scale = np.maximum(np.abs(expected), np.abs(actual))
            significant = scale > scale.max() * 1e-12
            np.testing.assert_array_less(
                np.abs(expected - actual)[significant] / scale[significant], RTOL)

    def test_matches_finite_differences(self):
        step = 1e-7
        for a, bb in plausible_spectra(seed=4, n=5):
            def forward(a_, bb_):
                return np.asarray(rrs_poly(
                    jnp.asarray(a_), jnp.asarray(bb_), self.coeffs, self.powers))

            expected = reliable(np.stack([
                (forward(a + step, bb) - forward(a - step, bb)) / (2 * step),
                (forward(a, bb + step) - forward(a, bb - step)) / (2 * step)]))
            actual = reliable(self._autodiff_gradient(a, bb))

            scale = np.maximum(np.abs(expected), np.abs(actual))
            significant = scale > scale.max() * 1e-6
            np.testing.assert_allclose(
                actual[significant], expected[significant], rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
