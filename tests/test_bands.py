"""Band averages preserve reflectance units on uniform and non-uniform grids."""
import unittest

import numpy as np

from hydropt_jax import build_gaussian_srf, msi_sigma_nm, srf_coverage


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


if __name__ == '__main__':
    unittest.main()
