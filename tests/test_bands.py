"""Band averages preserve reflectance units on uniform and non-uniform grids."""
import unittest

import numpy as np
import jax
import jax.numpy as jnp

from hydropt_jax import (
    build_gaussian_srf,
    build_tabulated_srf,
    classify_bands,
    classify_tabulated_bands,
    load_msi_srf,
    load_pace_poly,
    load_water_iops,
    monomial_powers_2d,
    msi_sigma_nm,
    rrs_from_excess,
    srf_coverage,
    tabulated_srf_coverage,
)

MSI_BANDS = np.array([443., 492., 560., 665., 704., 740., 783.])
SATELLITES = ('S2A', 'S2B', 'S2C')
#: Nominal centres of B1 to B8A.
ALL_MSI_BANDS = np.array([443., 490., 560., 665., 705., 740., 783., 842., 865.])


class TestBandIntegration(unittest.TestCase):

    def test_constant_spectrum_is_preserved_including_clipped_bands(self):
        bands = np.array([443., 492., 560., 665., 704.])
        grids = [np.arange(400., 710.1, step) for step in (1., 5., 10.)]
        grids.append(np.array([400., 410., 443., 490., 550., 665., 700., 710.]))
        for grid in grids:
            with self.subTest(grid=grid):
                matrix = np.asarray(build_gaussian_srf(grid, bands, msi_sigma_nm(bands)))
                np.testing.assert_allclose(
                    np.full(grid.shape, 0.01) @ matrix.T, 0.01, rtol=1e-13)

    def test_nonconstant_spectrum_matches_trapezoidal_band_average(self):
        grid = np.array([400., 420., 443., 480., 500., 560., 620., 665., 695., 710.])
        bands = np.array([443., 492., 560., 665., 704.])
        sigma = msi_sigma_nm(bands)
        spectrum = 0.002 + 0.01 * np.exp(-(grid - 450.) ** 2 / 5000.)
        response = np.exp(-0.5 * ((grid[None, :] - bands[:, None]) / sigma[:, None]) ** 2)
        expected = (np.trapezoid(response * spectrum, grid, axis=1)
                    / np.trapezoid(response, grid, axis=1))
        matrix = np.asarray(build_gaussian_srf(grid, bands, sigma))
        np.testing.assert_allclose(spectrum @ matrix.T, expected, rtol=1e-13)

    def test_coverage_resolves_the_far_tail(self):
        # MSI 783 nm lies 8.6 sigma above 710 nm. An erf difference cancels to
        # zero there, and the mass inside the interval is 4.16e-18.
        coverage = srf_coverage(np.array([783.]), np.array([20. / np.sqrt(8 * np.log(2))]),
                                (400., 710.))
        np.testing.assert_allclose(coverage, 4.16e-18, rtol=1e-2)

    def test_coverage_accepts_a_shared_scalar_width(self):
        bands = np.array([443., 665., 704.])
        np.testing.assert_allclose(
            srf_coverage(bands, 10., (400., 710.)),
            srf_coverage(bands, np.full(bands.shape, 10.), (400., 710.)))


class TestTabulatedResponse(unittest.TestCase):

    def test_band_centres_map_to_the_nearest_msi_band(self):
        srf_wl, response = load_msi_srf([443., 492., 704.], 'S2B')
        self.assertEqual(response.shape, (3, srf_wl.size))
        np.testing.assert_array_equal(np.diff(srf_wl), 1.)
        np.testing.assert_allclose(response.max(axis=1), 1.)
        # ESA lists centres of 442.2, 492.3 and 703.8 nm for B1, B2 and B5 of S2B.
        centroid = (np.trapezoid(response * srf_wl, srf_wl, axis=1)
                    / np.trapezoid(response, srf_wl, axis=1))
        np.testing.assert_allclose(centroid, [442.2, 492.3, 703.8], atol=0.1)

    def test_rejects_an_unknown_satellite(self):
        with self.assertRaises(ValueError):
            load_msi_srf([443.], 'S2D')

    def test_constant_spectrum_is_preserved_including_clipped_bands(self):
        grids = [np.arange(400., 710.1, step) for step in (1., 5., 10.)]
        grids.append(np.array([400., 410., 443., 490., 550., 665., 700., 710.]))
        for satellite in SATELLITES:
            srf_wl, response = load_msi_srf(MSI_BANDS[:5], satellite)
            for grid in grids:
                with self.subTest(satellite=satellite, grid=grid):
                    matrix = np.asarray(build_tabulated_srf(grid, srf_wl, response))
                    np.testing.assert_allclose(
                        np.full(grid.shape, 0.01) @ matrix.T, 0.01, rtol=1e-13)

    def test_matches_dense_integration_of_the_interpolated_spectrum(self):
        grid = np.arange(400., 710.1, 5.)
        spectrum = 0.002 + 0.01 * np.exp(-(grid - 450.) ** 2 / 5000.)
        dense = np.linspace(400., 710., 310001)
        for satellite in SATELLITES:
            with self.subTest(satellite=satellite):
                srf_wl, response = load_msi_srf(MSI_BANDS[:5], satellite)
                weight = np.stack([np.interp(dense, srf_wl, row) for row in response])
                expected = (np.trapezoid(weight * np.interp(dense, grid, spectrum), dense, axis=1)
                            / np.trapezoid(weight, dense, axis=1))
                matrix = np.asarray(build_tabulated_srf(grid, srf_wl, response))
                np.testing.assert_allclose(spectrum @ matrix.T, expected, rtol=1e-9)

    def test_band_without_response_in_the_grid_gives_nan(self):
        srf_wl, response = load_msi_srf([665., 740.], 'S2A')
        matrix = np.asarray(build_tabulated_srf(np.arange(400., 710.1, 5.), srf_wl, response))
        self.assertTrue(np.all(np.isfinite(matrix[0])))
        self.assertTrue(np.all(np.isnan(matrix[1])))

    def test_coverage_of_the_coefficient_grid(self):
        for satellite in SATELLITES:
            with self.subTest(satellite=satellite):
                srf_wl, response = load_msi_srf(MSI_BANDS, satellite)
                coverage = tabulated_srf_coverage(srf_wl, response, (400., 710.))
                np.testing.assert_allclose(coverage[:4], 1., rtol=1e-12)
                self.assertTrue(0.10 < coverage[4] < 0.95)
                np.testing.assert_array_equal(coverage[5:], 0.)

    def test_classifies_msi_bands(self):
        for satellite in SATELLITES:
            with self.subTest(satellite=satellite):
                srf_wl, response = load_msi_srf(MSI_BANDS, satellite)
                bands = classify_tabulated_bands(
                    MSI_BANDS, srf_wl, response, np.arange(400., 710.1, 5.))
                self.assertEqual(list(bands['class']), ['full'] * 4 + ['clipped']
                                 + ['unusable'] * 2)
                np.testing.assert_array_equal(bands['centre_nm'], MSI_BANDS)

    def test_table_holds_nine_distinct_normalised_bands(self):
        for satellite in SATELLITES:
            with self.subTest(satellite=satellite):
                srf_wl, response = load_msi_srf(ALL_MSI_BANDS, satellite)
                self.assertEqual(response.shape, (9, srf_wl.size))
                self.assertTrue(np.all((response >= 0.) & (response <= 1.)))
                np.testing.assert_allclose(response.max(axis=1), 1.)
                # The table starts and ends with a row of zeros, so each response
                # is complete inside it.
                np.testing.assert_array_equal(response[:, [0, -1]], 0.)
                self.assertEqual(len({row.tobytes() for row in response}), 9)

    def test_accepts_a_scalar_band_centre(self):
        srf_wl, response = load_msi_srf(665., 'S2A')
        np.testing.assert_array_equal(response, load_msi_srf([665.], 'S2A')[1])
        self.assertEqual(response.shape, (1, srf_wl.size))

    def test_centres_of_s2b_map_to_the_same_bands_as_the_nominal_centres(self):
        # S2B lists 442.2, 558.9, 739.1 and 779.7 nm for B1, B3, B6 and B7.
        _, nominal = load_msi_srf([443., 560., 740., 783.], 'S2B')
        _, measured = load_msi_srf([442., 559., 739., 780.], 'S2B')
        np.testing.assert_array_equal(nominal, measured)

    def test_matrix_on_a_grid_between_the_table_wavelengths(self):
        # Grid nodes at x.5 nm fall between the 1 nm table wavelengths.
        grid = np.arange(400.5, 708., 5.)
        spectrum = 0.002 + 0.01 * np.exp(-(grid - 450.) ** 2 / 5000.)
        dense = np.linspace(grid[0], grid[-1], 307001)
        srf_wl, response = load_msi_srf(MSI_BANDS[:5], 'S2C')
        matrix = np.asarray(build_tabulated_srf(grid, srf_wl, response))
        np.testing.assert_allclose(matrix.sum(axis=1), 1., rtol=1e-13)
        weight = np.stack([np.interp(dense, srf_wl, row) for row in response])
        expected = (np.trapezoid(weight * np.interp(dense, grid, spectrum), dense, axis=1)
                    / np.trapezoid(weight, dense, axis=1))
        np.testing.assert_allclose(spectrum @ matrix.T, expected, rtol=1e-9)

    def test_coverage_adds_up_over_adjacent_ranges(self):
        srf_wl, response = load_msi_srf(MSI_BANDS, 'S2A')
        for split in (602.5, 680., 707.3):
            with self.subTest(split=split):
                np.testing.assert_allclose(
                    tabulated_srf_coverage(srf_wl, response, (400., split))
                    + tabulated_srf_coverage(srf_wl, response, (split, 710.)),
                    tabulated_srf_coverage(srf_wl, response, (400., 710.)), atol=1e-14)

    def test_coverage_matches_dense_integration_with_a_limit_inside_a_band(self):
        srf_wl, response = load_msi_srf([704.], 'S2B')
        dense = np.linspace(srf_wl[0], srf_wl[-1], 514001)
        weight = np.interp(dense, srf_wl, response[0])
        inside = dense <= 707.5
        expected = np.trapezoid(weight[inside], dense[inside]) / np.trapezoid(weight, dense)
        np.testing.assert_allclose(
            tabulated_srf_coverage(srf_wl, response, (400., 707.5)), expected, rtol=1e-6)

    def test_reliable_fraction_is_cut_at_680_nm(self):
        for satellite in SATELLITES:
            with self.subTest(satellite=satellite):
                srf_wl, response = load_msi_srf(MSI_BANDS, satellite)
                bands = classify_tabulated_bands(
                    MSI_BANDS, srf_wl, response, np.arange(400., 710.1, 5.))
                self.assertTrue(np.all(bands['reliable'] <= bands['coverage'] + 1e-15))
                # B4 extends past 680 nm, and B5 lies almost entirely above it.
                self.assertTrue(0.9 < bands['reliable'][3] < 1.)
                self.assertLess(bands['reliable'][4], 1e-3)

    def test_classifications_share_their_columns(self):
        grid = np.arange(400., 710.1, 5.)
        srf_wl, response = load_msi_srf(MSI_BANDS, 'S2A')
        gaussian = classify_bands(MSI_BANDS, msi_sigma_nm(MSI_BANDS), grid)
        tabulated = classify_tabulated_bands(MSI_BANDS, srf_wl, response, grid)
        self.assertEqual(list(gaussian.columns), list(tabulated.columns))
        self.assertEqual(list(gaussian['class']), list(tabulated['class']))

    def test_band_values_agree_with_the_gaussian_approximation(self):
        # The Gaussian approximation differs most for B2 of S2C, by 5.4% on this
        # spectrum. A mismatch of bands would differ by far more.
        grid = np.arange(400., 710.1, 5.)
        spectrum = 0.002 + 0.01 * np.exp(-(grid - 450.) ** 2 / 5000.)
        bands = MSI_BANDS[:5]
        gaussian = spectrum @ np.asarray(build_gaussian_srf(grid, bands, msi_sigma_nm(bands))).T
        for satellite in SATELLITES:
            with self.subTest(satellite=satellite):
                srf_wl, response = load_msi_srf(bands, satellite)
                tabulated = spectrum @ np.asarray(build_tabulated_srf(grid, srf_wl, response)).T
                np.testing.assert_allclose(tabulated, gaussian, rtol=0.06)

    def test_forward_model_differentiates_through_the_matrix_under_jit(self):
        lam, coeffs = load_pace_poly()
        powers = monomial_powers_2d(4)
        a_water, bb_water = load_water_iops(lam)
        srf_wl, response = load_msi_srf(MSI_BANDS[:5], 'S2A')
        matrix = build_tabulated_srf(lam, srf_wl, response)

        @jax.jit
        def loss(z):
            return jnp.sum(rrs_from_excess(z[0], z[1], a_water, bb_water, coeffs, powers)
                           @ matrix.T)

        state = jnp.stack([jnp.full(lam.shape, np.log(0.05)), jnp.full(lam.shape, np.log(0.005))])
        gradient = np.asarray(jax.grad(loss)(state))
        self.assertTrue(np.all(np.isfinite(gradient)))
        # Wavelengths with weight in a band receive a gradient, the others none.
        weighted = np.asarray(matrix).sum(axis=0) > 0
        self.assertTrue(np.all(gradient[:, weighted] != 0.))
        np.testing.assert_array_equal(gradient[:, ~weighted], 0.)


if __name__ == '__main__':
    unittest.main()
