"""
Tests for the validity domain guard.

The failure these protect against is not a crash but a silence: outside the
fitted domain rrs_poly returns exactly zero with an exactly zero gradient, and
an optimiser given those bands simply gets nothing from them. The regression to
watch for is that going out of bounds stops being noisy again.
"""
import unittest

import numpy as np
import jax
import jax.numpy as jnp

from hydropt_jax import (
    DomainError,
    RELIABLE_WAVELENGTH_RANGE,
    check_domain,
    iops_from_excess,
    load_pace_poly,
    load_water_iops,
    monomial_powers_2d,
    reliable_band_mask,
    rrs_from_excess,
    rrs_poly,
)


class TestWaterFloor(unittest.TestCase):

    def setUp(self):
        self.lam, self.coeffs = load_pace_poly()
        self.powers = monomial_powers_2d(4)
        self.a_water, self.bb_water = load_water_iops(self.lam)

    def test_water_iops_interpolate_onto_requested_grid(self):
        a_water, bb_water = load_water_iops(self.lam)
        self.assertEqual(a_water.shape, self.lam.shape)
        self.assertEqual(bb_water.shape, self.lam.shape)
        # Water absorption rises steeply into the red; this is the value that
        # makes the old a = 0.05 m^-1 starting guess unphysical at 710 nm.
        self.assertGreater(float(a_water[-1]), 0.8)

    def test_excess_parameterisation_never_breaches_the_floor(self):
        # Even an absurd optimiser step cannot push the state below pure water,
        # because exp() is positive everywhere.
        for log_excess in (-1e6, -100.0, 0.0, 10.0):
            a, bb = iops_from_excess(
                jnp.full(self.lam.shape, log_excess),
                jnp.full(self.lam.shape, log_excess),
                self.a_water, self.bb_water)
            self.assertTrue(np.all(np.asarray(a) >= np.asarray(self.a_water)))
            self.assertTrue(np.all(np.asarray(bb) >= np.asarray(self.bb_water)))

    def test_excess_parameterisation_is_differentiable_and_jittable(self):
        mask = reliable_band_mask(self.lam)
        a_water, bb_water = self.a_water[mask], self.bb_water[mask]
        coeffs = self.coeffs[mask]
        state = jnp.stack([jnp.full(a_water.shape, np.log(0.05)),
                           jnp.full(a_water.shape, np.log(0.005))])

        @jax.jit
        def loss(z):
            return jnp.sum(rrs_from_excess(z[0], z[1], a_water, bb_water, coeffs, self.powers))

        gradient = np.asarray(jax.grad(loss)(state))
        self.assertTrue(np.all(np.isfinite(gradient)))
        # The point of the fix: no band contributes a dead zero gradient.
        self.assertFalse(np.any(gradient == 0.0))


class TestSilentZeros(unittest.TestCase):
    '''Document the failure mode the guard exists to catch.'''

    def setUp(self):
        self.lam, self.coeffs = load_pace_poly()
        self.powers = monomial_powers_2d(4)

    def test_out_of_domain_state_underflows_to_exact_zero(self):
        # This is the old example's starting guess: 0.05 m^-1 at every
        # wavelength, sixteen times below pure water's absorption at 710 nm.
        a = jnp.full(self.lam.shape, 0.05)
        bb = jnp.full(self.lam.shape, 0.005)
        rrs = np.asarray(rrs_poly(a, bb, self.coeffs, self.powers))

        self.assertTrue(np.any(rrs == 0.0),
                        'expected exact zeros past the red edge')
        # ... and the gradient dies with it, which is why the optimiser could
        # never recover those bands.

        def forward(state):
            return jnp.sum(rrs_poly(state[0], state[1], self.coeffs, self.powers))

        gradient = np.asarray(jax.grad(forward)(jnp.stack([a, bb])))
        self.assertTrue(np.any(gradient == 0.0))

    def test_in_domain_state_has_no_zeros(self):
        mask = reliable_band_mask(self.lam)
        a_water, bb_water = load_water_iops(self.lam[mask])
        rrs = np.asarray(rrs_from_excess(
            jnp.full(a_water.shape, np.log(0.05)),
            jnp.full(a_water.shape, np.log(0.005)),
            a_water, bb_water, self.coeffs[mask], self.powers))

        self.assertTrue(np.all(rrs > 0.0))
        self.assertTrue(np.all(np.isfinite(rrs)))
        # Physically plausible remote sensing reflectance.
        self.assertGreater(rrs.min(), 1e-6)
        self.assertLess(rrs.max(), 0.1)


class TestCheckDomain(unittest.TestCase):

    def setUp(self):
        self.lam, self.coeffs = load_pace_poly()
        self.powers = monomial_powers_2d(4)

    def test_reliable_range_excludes_the_degenerate_red_edge(self):
        lo, hi = RELIABLE_WAVELENGTH_RANGE
        self.assertEqual((lo, hi), (400.0, 680.0))
        mask = reliable_band_mask(self.lam)
        self.assertEqual(int(mask.sum()), 57)
        self.assertTrue(np.all(np.asarray(self.lam)[~mask] > hi))

    def test_raises_on_the_old_example_state(self):
        with self.assertRaises(DomainError) as raised:
            check_domain(np.full(self.lam.shape, 0.05),
                         np.full(self.lam.shape, 0.005),
                         self.lam, self.coeffs, self.powers)
        message = str(raised.exception)
        # The message has to name the actual problem, not just fail.
        self.assertIn('below pure water', message)
        self.assertIn('710 nm', message)
        self.assertIn('run away', message)

    def test_warns_instead_of_raising_when_asked(self):
        with self.assertWarns(UserWarning):
            problems = check_domain(np.full(self.lam.shape, 0.05),
                                    np.full(self.lam.shape, 0.005),
                                    self.lam, error=False)
        self.assertTrue(problems)

    def test_accepts_a_valid_state(self):
        mask = reliable_band_mask(self.lam)
        lam = np.asarray(self.lam)[mask]
        a_water, bb_water = load_water_iops(lam)
        a, bb = iops_from_excess(
            jnp.full(a_water.shape, np.log(0.05)),
            jnp.full(a_water.shape, np.log(0.005)), a_water, bb_water)

        self.assertEqual(check_domain(a, bb, lam, self.coeffs[mask], self.powers), [])

    def test_flags_wavelengths_outside_the_reliable_range(self):
        lam = np.asarray(self.lam)
        a_water, bb_water = load_water_iops(lam)
        with self.assertWarns(UserWarning):
            problems = check_domain(np.asarray(a_water) + 0.05,
                                    np.asarray(bb_water) + 0.005, lam, error=False)
        self.assertTrue(any('outside the reliable range' in p for p in problems))

    def test_rejects_nonfinite_iops_with_or_without_coefficients(self):
        mask = reliable_band_mask(self.lam)
        lam = self.lam[mask]
        water = [np.asarray(v) for v in load_water_iops(lam)]
        for variable, name in enumerate(('absorption', 'backscattering')):
            for value in (np.nan, np.inf, -np.inf):
                for include_coeffs in (False, True):
                    with self.subTest(variable=name, value=value, coeffs=include_coeffs):
                        state = [water[0] + 0.05, water[1] + 0.005]
                        state[variable][0] = value
                        optional = (self.coeffs[mask], self.powers) if include_coeffs else ()
                        with self.assertRaises(DomainError) as raised:
                            check_domain(*state, lam, *optional)
                        self.assertIn(name + ' is non-finite', str(raised.exception))
                        self.assertIn('400 nm', str(raised.exception))

    def test_warns_for_nonfinite_iops_without_coefficients(self):
        lam = self.lam[reliable_band_mask(self.lam)]
        a_water, bb_water = load_water_iops(lam)
        a = np.asarray(a_water) + 0.05
        a[0] = np.nan
        with self.assertWarns(UserWarning):
            problems = check_domain(a, np.asarray(bb_water) + 0.005, lam, error=False)
        self.assertTrue(any('absorption is non-finite' in p for p in problems))

    def test_runaway_diagnostic_identifies_the_failing_wavelength(self):
        lam = self.lam[reliable_band_mask(self.lam)]
        a_water, bb_water = load_water_iops(lam)
        # Constant polynomials isolate an overflow/underflow from valid bands.
        for log_rrs in (-800., 2., 800.):
            with self.subTest(log_rrs=log_rrs):
                coeffs = np.zeros((len(lam), len(self.powers)))
                coeffs[:, 0] = -5.
                coeffs[0, 0] = log_rrs
                with self.assertRaises(DomainError) as raised:
                    check_domain(np.asarray(a_water) + 0.05,
                                 np.asarray(bb_water) + 0.005, lam, coeffs, self.powers)
                self.assertIn('at 400 nm, i.e. Rrs', str(raised.exception))


if __name__ == '__main__':
    unittest.main()
