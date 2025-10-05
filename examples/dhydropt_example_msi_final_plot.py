"""
D-HYDROPT gradient visualisation and pixel correction example

This example demonstrates the differentiable HYDROPT radiative transfer model for aquatic remote sensing,
specifically focusing on gradient-based optimisation and sensitivity analysis for Sentinel-2 MSI pixels.

Purpose:
--------
The code showcases how automatic differentiation enables efficient gradient-based optimisation of aquatic optical
parameters (absorption and backscattering coefficients) from satellite-observed remote sensing reflectance (Rrs).
It visualises the Jacobian matrices that describe how changes in fine-resolution optical parameters affect
band-integrated satellite measurements, providing insights into parameter sensitivity and optimisation behaviour.

Code structure:
------------
1. Data loading: loads MSI Rrs observations from CSV files with wavelength-labelled columns
2. Forward model: uses polynomial approximation of HYDROPT radiative transfer model to predict Rrs from optical parameters
3. Spectral response: applies Gaussian spectral response functions to map fine wavelengths to MSI bands
4. Gradient computation: uses JAX automatic differentiation to compute Jacobians ∂Rrs_band/∂log_parameter
5. Statistical analysis: computes median sensitivities and coefficient of variation across parameter ensembles
6. Pixel selection: optimises multiple pixels and selects representative cases for visualisation
7. Visualisation: creates heatmaps showing parameter sensitivity patterns and optimisation convergence

Key assumptions and simplifications:
-----------------------------------
- Aquatic waters: assumes Case-1 waters dominated by phytoplankton, CDOM, and mineral particles
- Single scattering approximation: uses polynomial approximation rather than full radiative transfer solution
- Gaussian SRF: assumes Gaussian spectral response functions for MSI bands (approximates real instrument response)
- Log-space parameters: works in log-space for absorption (a) and backscattering (bb) for numerical stability
- RBF smoothness: assumes spectral smoothness using radial basis function perturbations
- Homogeneous water column: assumes vertically uniform optical properties (no depth-dependent effects)
- Clear sky conditions: neglects atmospheric effects (assumes pre-processed Rrs data)
- Linear mixing: assumes linear combination of optical components without complex interactions
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.patheffects as mpe
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from hydropt_jax.core import monomial_powers_2d, rrs_poly
from hydropt_jax.data import load_pace_poly

def build_gaussian_B(fine_wl, band_wl, sigma_nm=12.0):
    """
    Build a Gaussian spectral response function matrix that maps fine wavelengths to band wavelengths.

    Args:
        fine_wl: Array of fine-resolution wavelengths in nanometres.
        band_wl: Array of band centre wavelengths in nanometres.
        sigma_nm: Standard deviation of Gaussian kernels in nanometres. Can be scalar or array.

    Returns:
        Matrix B of shape (n_bands, n_wavelengths) where each row sums to 1.
    """
    fine = jnp.asarray(fine_wl, jnp.float64)
    band = jnp.asarray(band_wl, jnp.float64)
    sigma = jnp.ones_like(band) * sigma_nm if np.isscalar(sigma_nm) else jnp.asarray(sigma_nm, jnp.float64)
    d = fine[None, :] - band[:, None]
    w = jnp.exp(-0.5 * (d / sigma[:, None]) ** 2)
    Z = jnp.trapezoid(w, fine, axis=1)[:, None]
    return w / Z

def msi_sigma_nm(band_wl):
    """
    Compute Gaussian sigma values for MSI (Multispectral Instrument) band centres based on full width at half maximum (FWHM).

    Args:
        band_wl: Array of band centre wavelengths in nanometres.

    Returns:
        Array of sigma values in nanometres, converted from FWHM using the relationship sigma = FWHM / (2 * sqrt(2 * ln(2))).
    """
    fwhm_map = {443: 20, 490: 65, 560: 35, 665: 30, 705: 15, 740: 15, 783: 20, 842: 115}
    centers = np.array([min(fwhm_map, key=lambda c: abs(c - float(b))) for b in np.asarray(band_wl)], float)
    fwhm = np.array([fwhm_map[int(c)] for c in centers], float)
    return jnp.asarray(fwhm / np.sqrt(8.0 * np.log(2.0)), jnp.float64)

def load_rrs_csv(path):
    """
    Load remote sensing reflectance (Rrs) data from a CSV file with wavelength columns.

    Args:
        path: Path to CSV file where column headers are wavelengths (as numbers or strings containing numbers).

    Returns:
        Tuple of (X, wl) where X is a 2D array of Rrs values (n_samples, n_bands) and wl is a 1D array of wavelengths in nanometres.
        Rows with non-finite values are filtered out.
    """
    df = pd.read_csv(path, sep=None, engine='python')
    for idxcol in ("index", "Index", "Unnamed: 0", "Unnamed: 0.1"):
        if idxcol in df.columns:
            df = df.drop(columns=[idxcol])
    cols = []
    for c in df.columns:
        name = str(c)
        try:
            v = float(name)
            cols.append((v, c))
            continue
        except:
            pass
        import re
        m = re.search(r"(\d+(\.\d+)?)", name)
        if m:
            cols.append((float(m.group(1)), c))
    if not cols:
        raise ValueError("No numeric band columns found in CSV headers.")
    cols = sorted(cols, key=lambda t: t[0])
    wl = np.array([v for v, _ in cols], dtype=np.float32)
    X = df[[c for _, c in cols]].values.astype(np.float32)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    return X, wl

def rbf_basis(wl, K=6, sigma_nm=40.0):
    """
    Construct a radial basis function (RBF) matrix for spectral smoothness constraints.

    Args:
        wl: Array of wavelengths in nanometres.
        K: Number of RBF centres evenly spaced across the wavelength range.
        sigma_nm: Width of each RBF kernel in nanometres.

    Returns:
        Matrix Phi of shape (n_wavelengths, K) where each column is a normalised RBF kernel. Row sums approximate 1.
    """
    wl = jnp.asarray(wl, jnp.float64)
    c = jnp.linspace(float(wl[0]), float(wl[-1]), K)
    Phi = jnp.exp(-0.5 * ((wl[:, None] - c[None, :]) / sigma_nm) ** 2)
    s = jnp.sum(Phi, 0, keepdims=True)
    return Phi / (s + 1e-12)

def sample_Z_from_basis(key, N, fine_wl, la0, lb0, K=6, sigma_nm=40.0, amp=0.35):
    """
    Sample N log-scale absorption and backscattering spectra using RBF basis perturbations around a mean state.

    Args:
        key: JAX PRNG key for reproducible random sampling.
        N: Number of samples to generate.
        fine_wl: Array of fine-resolution wavelengths in nanometres.
        la0: Mean log-absorption spectrum.
        lb0: Mean log-backscattering spectrum.
        K: Number of RBF basis functions.
        sigma_nm: RBF kernel width in nanometres.
        amp: Amplitude of random perturbations.

    Returns:
        Tuple of (Z, Phi) where Z has shape (N, 2, n_wavelengths) with [log_a, log_bb] pairs, and Phi is the RBF basis matrix.
    """
    Phi = rbf_basis(fine_wl, K, sigma_nm)
    k1, k2 = jax.random.split(key)
    A = amp * jax.random.normal(k1, (N, K))
    Bc = amp * jax.random.normal(k2, (N, K))
    la = la0[None, :] + A @ Phi.T
    lb = lb0[None, :] + Bc @ Phi.T
    return jnp.stack([la, lb], 1), Phi

def build_jacobian_functions(coeffs, powers, B):
    """
    Build forward model and Jacobian functions for computing Rrs from log-scale absorption and backscattering.

    Args:
        coeffs: Polynomial coefficients for the HYDROPT radiative transfer model.
        powers: Monomial powers for the polynomial expansion.
        B: Spectral response matrix that maps fine wavelengths to band wavelengths.

    Returns:
        Tuple of (f_from_log, jf) where f_from_log computes band-integrated Rrs from [log_a, log_bb],
        and jf is its Jacobian (reverse-mode automatic differentiation).
    """
    coeffs = jnp.asarray(coeffs, jnp.float64)
    powers = jnp.asarray(powers, jnp.int32)
    B = jnp.asarray(B, jnp.float64)

    def f_from_log(z):
        a = jnp.exp(z[0])
        bb = jnp.exp(z[1])
        rrs_fine = rrs_poly(a, bb, coeffs, powers)
        return rrs_fine @ B.T

    jf = jax.jacrev(f_from_log)
    return f_from_log, jf

def compute_jacobian_stats(jfun, Z, B, fine_wl, n_samp=64, support_thresh=1e-3):
    """
    Compute median Jacobian statistics across a sample of state vectors, including sensitivity and coefficient of variation.

    Args:
        jfun: Tuple of (forward_function, jacobian_function) from build_jacobian_functions.
        Z: State array of shape (N, 2, n_wavelengths) with [log_a, log_bb] pairs.
        B: Spectral response matrix.
        fine_wl: Array of fine-resolution wavelengths in nanometres.
        n_samp: Maximum number of samples to use for computing statistics.
        support_thresh: Threshold for determining spectral support regions (relative to peak response).

    Returns:
        Tuple of (Jm_plot, cov_plot, vmax, amp_a, amp_bb, support, med_abs_a, med_abs_bb, med_cov_a, med_cov_bb)
        containing median Jacobians, coefficient of variation, colour scale limits, integrated amplitudes,
        support mask, and median statistics.
    """
    f_from_log, jf = jfun
    Zs = Z[:n_samp] if Z.shape[0] > n_samp else Z
    J = jax.vmap(jf)(Zs)
    M = int(B.shape[0])
    W = int(fine_wl.shape[0])
    sh = tuple(J.shape)
    axis2 = int(np.argwhere(np.array(sh) == 2).ravel()[0])
    axisM = [i for i, s in enumerate(sh) if s == M][0]
    axisW = [i for i, s in enumerate(sh) if s == W][0]
    J = jnp.transpose(J, (0, axis2, axisM, axisW))
    B = jnp.asarray(B, jnp.float64)
    support = B > (jnp.max(B, axis=1, keepdims=True) * support_thresh)
    Jm = jnp.median(J, axis=0)
    Js = jnp.std(J, axis=0)
    Jm_plot = jnp.where(support[None, :, :], Jm, 0.0)
    cov_plot = jnp.where(support[None, :, :], Js / (jnp.abs(Jm) + 1e-12), 0.0)
    amp_a = jnp.trapezoid(jnp.abs(Jm_plot[0]), fine_wl, axis=1)
    amp_bb = jnp.trapezoid(jnp.abs(Jm_plot[1]), fine_wl, axis=1)
    vals = jnp.where(support[None, None, :, :], jnp.abs(J), 0.0)
    vmax = float(jnp.quantile(vals, 0.98))
    med_abs_a = float(jnp.median(jnp.abs(Jm[0])[support]))
    med_abs_bb = float(jnp.median(jnp.abs(Jm[1])[support]))
    med_cov_a = float(jnp.median(cov_plot[0][support]))
    med_cov_bb = float(jnp.median(cov_plot[1][support]))
    return Jm_plot, cov_plot, vmax, amp_a, amp_bb, support, med_abs_a, med_abs_bb, med_cov_a, med_cov_bb

def select_pixel(Z, Y, jfun, steps=60, lr=0.2, mode="improve", sample=512, seed=0):
    """
    Select a representative pixel from a dataset by optimising all samples and choosing according to a selection criterion.

    Args:
        Z: Initial state array of shape (N, 2, n_wavelengths).
        Y: Target band Rrs values of shape (N, n_bands).
        jfun: Tuple of (forward_function, jacobian_function).
        steps: Number of gradient descent steps to perform.
        lr: Learning rate for gradient descent.
        mode: Selection criterion: "improve" (best improvement), "best" (lowest final loss), "worst" (highest final loss), or "median".
        sample: Maximum number of pixels to optimise (for computational efficiency).
        seed: Random seed for sampling.

    Returns:
        Tuple of (idx, z_init, z_final, loss_init, loss_final) for the selected pixel.
    """
    f, _ = jfun

    def loss_z(z, y):
        e = f(z) - y
        return jnp.mean(e * e)

    grad_fn = jax.jit(jax.vmap(jax.grad(loss_z)))
    loss_vec = jax.jit(jax.vmap(loss_z))
    N = Z.shape[0]
    if sample and N > sample:
        rng = np.random.default_rng(seed)
        sub_idx = np.asarray(rng.choice(N, size=sample, replace=False))
        Zs, Ys = Z[sub_idx], Y[sub_idx]
    else:
        sub_idx = np.arange(N)
        Zs, Ys = Z, Y
    Z0 = Zs
    for _ in range(steps):
        g = grad_fn(Zs, Ys)
        g_rms = jnp.sqrt(jnp.mean(g * g, axis=(1, 2), keepdims=True)) + 1e-8
        dirn = -g / g_rms
        scales = jnp.asarray([1.0, 0.5, 0.2, 0.1, 0.05])

        def try_scales(z, d, y):
            def L_at(s):
                z1 = z + lr * s * d
                e = f(z1) - y
                return jnp.mean(e * e)
            return jax.vmap(L_at)(scales)

        cand = jax.vmap(try_scales)(Zs, dirn, Ys)
        k = jnp.argmin(cand, axis=1)
        Zs = Zs + lr * scales[k][:, None, None] * dirn
    L0 = loss_vec(Z0, Ys)
    Lf = loss_vec(Zs, Ys)
    if mode == "best":
        j = int(jnp.argmin(Lf))
    elif mode == "worst":
        j = int(jnp.argmax(Lf))
    elif mode == "median":
        order = jnp.argsort(Lf)
        j = int(order[Lf.shape[0] // 2])
    else:
        j = int(jnp.argmax(L0 - Lf))
    idx = int(sub_idx[j])
    return idx, Z0[j], Zs[j], float(L0[j]), float(Lf[j])

def pixel_gradient_path(jfun, z0, y, steps=80, lr=0.2):
    """
    Compute the gradient descent trajectory for a single pixel using line search for step size selection.

    Args:
        jfun: Tuple of (forward_function, jacobian_function).
        z0: Initial state [log_a, log_bb] of shape (2, n_wavelengths).
        y: Target band Rrs values.
        steps: Number of optimisation steps.
        lr: Base learning rate, scaled by line search.

    Returns:
        Tuple of (zs, vals) where zs is the trajectory of states and vals is the trajectory of loss values.
    """
    f, _ = jfun

    def L(z):
        e = f(z) - y
        return jnp.mean(e * e)

    zs = [z0]
    vals = [float(L(z0))]
    z = z0
    for _ in range(steps):
        v, g = jax.value_and_grad(L)(z)
        g_rms = float(jnp.sqrt(jnp.mean(g * g))) + 1e-12
        d = -g / g_rms
        scales = jnp.asarray([1.0, 0.5, 0.25, 0.1, 0.05])
        cand = jax.vmap(lambda s: L(z + lr * s * d))(scales)
        k = int(jnp.argmin(cand))
        z = z + lr * float(scales[k]) * d
        zs.append(z)
        vals.append(float(cand[k]))
    return jnp.stack(zs), np.asarray(vals)

def pixel_baseline_path(jfun, z0, y, steps=80, lr=0.2, seed=0):
    """
    Compute a baseline optimisation trajectory using shuffled gradients to test importance of gradient structure.

    Args:
        jfun: Tuple of (forward_function, jacobian_function).
        z0: Initial state [log_a, log_bb] of shape (2, n_wavelengths).
        y: Target band Rrs values.
        steps: Number of optimisation steps.
        lr: Learning rate.
        seed: Random seed for gradient shuffling.

    Returns:
        Tuple of (zs, vals) where zs is the trajectory and vals is the loss trajectory. Used as a control experiment.
    """
    f, _ = jfun

    def L(z):
        e = f(z) - y
        return jnp.mean(e * e)

    z = z0
    zs = [z0]
    vals = [float(L(z0))]
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        g = jax.grad(L)(z)
        g_np = np.array(g).reshape(-1)
        rng.shuffle(g_np)
        g_shuf = jnp.asarray(g_np).reshape(g.shape)
        g_rms = float(jnp.sqrt(jnp.mean(g_shuf * g_shuf))) + 1e-12
        z = z - lr * (g_shuf / g_rms)
        zs.append(z)
        vals.append(float(L(z)))
    return jnp.stack(zs), np.asarray(vals)

def plot_compact(Jm, support, vmax, fine_wl, band_wl, y, yhat0, yhatT, vals, out_path):
    """
    Create a comprehensive visualisation showing Jacobian heatmaps, pixel fit quality, and optimisation convergence.

    Args:
        Jm: Median Jacobian array of shape (2, n_bands, n_wavelengths) for [log_a, log_bb].
        support: Boolean mask indicating spectral support regions.
        vmax: Maximum value for colour scale.
        fine_wl: Array of fine-resolution wavelengths in nanometres.
        band_wl: Array of band centre wavelengths in nanometres.
        y: Observed Rrs band values.
        yhat0: Initial predicted Rrs band values.
        yhatT: Final predicted Rrs band values after optimisation.
        vals: Loss values over optimisation trajectory.
        out_path: Path object or string for saving the figure.
    """
    import matplotlib as mpl
    from matplotlib.colors import TwoSlopeNorm
    mpl.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(15, 4))
    # 5 columns: heatmap, heatmap, colorbar, spacer (whitespace), right plots
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[1.1, 1.1, 0.05, 0.55, 1.6],
        height_ratios=[1.0, 1.0],
        wspace=0.15, hspace=0.4,
        top=0.96, bottom=0.13
    )

    cmap = plt.get_cmap('seismic').copy()
    cmap.set_bad(color='none', alpha=0.0)

    M = len(band_wl)

    # Find index of last band <= 700 nm
    bands_below_700 = [i for i, b in enumerate(band_wl) if b <= 700]
    last_band_idx = bands_below_700[-1] if bands_below_700 else M - 1

    # Find wavelength indices for 400-700 nm range
    wl_mask = (fine_wl >= 400) & (fine_wl <= 700)
    band_mask = np.arange(M) <= last_band_idx

    # Set fixed colorbar limits instead of computing from data
    vmax_fixed = 0.0003
    norm = TwoSlopeNorm(vmin=-vmax_fixed, vcenter=0.0, vmax=vmax_fixed)

    def draw_heat(ax, J, name, xlim_range=(None, None), ylim_range=(None, None), show_xlabel=True):
        A = np.asarray(J)
        mask = ~np.asarray(support)
        # Set unsupported regions to NaN (not masked array)
        A_plot = A.copy()
        A_plot[mask] = np.nan
        # White background
        ax.set_facecolor('white')
        ymin, ymax = M-0.5, -0.5
        im = ax.imshow(A_plot, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
                      extent=[float(fine_wl[0]), float(fine_wl[-1]), ymin, ymax], zorder=1)
        # Light grey vertical markers for band centers
        for b in band_wl:
            ax.axvline(float(b), lw=0.6, color='grey', alpha=0.25, zorder=0)
        yt = np.arange(M)
        yl = [f"{int(round(b))}" for b in band_wl]

        # Mark bands on y-axis
        ax.set_yticks(yt, minor=False)
        ax.set_yticklabels(yl, minor=False)
        ax.tick_params(axis="y", which="major", length=0, rotation=90)
        ax.set_yticks(np.linspace(ymin, ymax, M+1), minor=True)
        ax.grid(True, axis="y", which="minor")

        # Title / xticks
        ax.set_title(name, fontsize=14)
        if not show_xlabel:
            ax.set_xticklabels([])

        # Draw line on top of heatmap
        n_insets = last_band_idx+1
        for j in range(n_insets):
            inset = ax.inset_axes([0, 1-(j+1)/n_insets, 1, 1/n_insets],
                                  sharex=ax, ylim=(-vmax_fixed, vmax_fixed))
            inset.set_axis_off()
            inset.plot(fine_wl, A_plot[j], color="black", linewidth=1,
                       path_effects=[mpe.Stroke(linewidth=3, foreground="white"), mpe.Normal()])

        # Place +/- markers only where supported and within x/y limits if provided
        xmin, xmax = xlim_range if xlim_range[0] is not None else (float(fine_wl[0]), float(fine_wl[-1]))
        ymin, ymax = ylim_range if ylim_range[0] is not None else (-0.5, M - 0.5)
        ylow, yhigh = min(ymin, ymax), max(ymin, ymax)
        for m in range(M):
            if not (ylow <= m <= yhigh):
                continue
            if not np.any(~mask[m]):
                continue
            jj = int(np.nanargmax(np.abs(A[m]) * (~mask[m])))
            wl_pos = float(fine_wl[jj])
            if wl_pos < xmin or wl_pos > xmax:
                continue
            s = '+' if A[m, jj] >= 0 else '−'
            ax.text(
                wl_pos, m, s, ha='center', va='center', fontsize=8, fontweight='bold',
                color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5, pad=0.5)
            )
        return im

    # Heatmaps (columns 0 and 1)
    ax1 = fig.add_subplot(gs[:, 0])
    im1 = draw_heat(ax1, Jm[0], r"$\partial R_{rs}^{\mathrm{band}} / \partial \ln a(\lambda)$", xlim_range=(400, 700),
                   ylim_range=(last_band_idx + 0.5, -0.5), show_xlabel=True)
    ax1.set_ylabel("MSI band [nm]", fontsize=14, fontweight='bold')
    ax1.set_xlim(400, 700)
    ax1.set_ylim(last_band_idx + 0.5, -0.5)
    # Add subplot label (A)
    ax1.text(0.98, 0.98, '(A)', transform=ax1.transAxes, fontsize=14, fontweight='bold',
             va='top', ha='right', color='black')

    ax2 = fig.add_subplot(gs[:, 1])
    im2 = draw_heat(ax2, Jm[1], r"$\partial R_{rs}^{\mathrm{band}} / \partial \ln b_b(\lambda)$", xlim_range=(400, 700),
                   ylim_range=(last_band_idx + 0.5, -0.5), show_xlabel=True)
    ax2.set_yticklabels([])
    ax2.set_ylabel("")
    ax2.set_xlim(400, 700)
    ax2.set_ylim(last_band_idx + 0.5, -0.5)
    # Add subplot label (B)
    ax2.text(0.98, 0.98, '(B)', transform=ax2.transAxes, fontsize=14, fontweight='bold',
             va='top', ha='right', color='black')

    # Dedicated colorbar column (2) and a whitespace spacer column (3)
    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(im1, cax=cax, extend='both')
    cbar.set_label("Sensitivity", fontsize=14, fontweight='bold')
    ticks = np.linspace(-vmax_fixed, vmax_fixed, 7)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.4f}" for t in ticks])

    fig.add_subplot(gs[:, 3]).axis("off")  # spacer for clean white space

    # Right-hand plots (column 4)
    ax3 = fig.add_subplot(gs[0, 4])
    ax3.plot(band_wl, np.asarray(y), 'ko-', label='Observed', zorder=3)
    ax3.plot(band_wl, np.asarray(yhat0), 'o-', label='Initial', zorder=3)
    ax3.plot(band_wl, np.asarray(yhatT), 'o-', color='green', label='After gradient', zorder=3)

    # Add vertical line at 700 nm and filled area
    ylim = ax3.get_ylim()
    ax3.axvline(700, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax3.axvspan(700, max(band_wl) * 1.05, alpha=0.15, color='grey', zorder=0)
    ax3.text(700 + (max(band_wl) - 700) * 0.5, ylim[0] + (ylim[1] - ylim[0]) * 0.95,
            'HYDROPT\nlimit', ha='center', va='top', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', linewidth=1))

    ax3.set_xlabel('MSI band [nm]', fontsize=14, fontweight='bold')
    ax3.set_ylabel(r'$\mathbf{R_{rs}}$ [sr$^{-1}$]', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(framealpha=0.9)
    # Add subplot label (C)
    ax3.text(0.98, 0.98, '(C)', transform=ax3.transAxes, fontsize=14, fontweight='bold',
             va='top', ha='right', color='black')

    ax4 = fig.add_subplot(gs[1, 4])
    ax4.plot(np.arange(len(vals)), np.asarray(vals), linewidth=2, color='green')
    ax4.set_xlabel("Optimisation step", fontsize=14, fontweight='bold')
    ax4.set_ylabel("MSE", fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    # Add subplot label (D)
    ax4.text(0.98, 0.98, '(D)', transform=ax4.transAxes, fontsize=14, fontweight='bold',
             va='top', ha='right', color='black')

    # Shared x-label for both heatmaps - centered between both plots
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    center_x = (pos1.x0 + pos2.x1) / 2
    fig.text(center_x, 0.02, "Hydropt λ [nm]", ha='center', fontsize=14, fontweight='bold')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=500, bbox_inches="tight")

def main():
    """
    Main entry point for D-HYDROPT gradient visualisation and pixel correction demo.

    Loads MSI Rrs data, computes Jacobian statistics, selects a representative pixel, performs gradient-based optimisation,
    and generates a comprehensive visualisation showing sensitivity patterns and optimisation performance.
    """
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--msi_csv", default=str(here / "example_rrs_dataset_msi.csv"))
    p.add_argument("--use_band_srf", action="store_true")
    p.add_argument("--sigma_nm", type=float, default=12.0)
    p.add_argument("--subset", type=int, default=2000)
    p.add_argument("--jac_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--select", choices=["improve", "best", "median", "worst", "index"], default="best")
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--sample", type=int, default=512)
    p.add_argument("--gd_steps", type=int, default=120)
    p.add_argument("--gd_lr", type=float, default=0.1)
    p.add_argument("--basis_k", type=int, default=6)
    p.add_argument("--basis_sigma_nm", type=float, default=40.0)
    p.add_argument("--prior_amp", type=float, default=0.35)
    p.add_argument("--out", default=str(here / "figures" / "dhydropt_gradviz.png"))
    args = p.parse_args()

    X, band_wl = load_rrs_csv(args.msi_csv)
    if X.shape[0] > args.subset:
        X = X[:args.subset]

    fine_wl, coeffs = load_pace_poly()
    powers = monomial_powers_2d(4)

    a0 = jnp.ones_like(fine_wl) * 0.05
    bb0 = jnp.ones_like(fine_wl) * 0.005
    rrs0 = rrs_poly(a0, bb0, jnp.asarray(coeffs, jnp.float64), jnp.asarray(powers, jnp.int32))

    sigma = msi_sigma_nm(band_wl) if args.use_band_srf else args.sigma_nm
    B = build_gaussian_B(fine_wl, band_wl, sigma)

    key = jax.random.PRNGKey(args.seed)
    Xj = jnp.asarray(X)
    loga0 = jnp.log(jnp.ones_like(fine_wl) * 0.05)
    logbb0 = jnp.log(jnp.ones_like(fine_wl) * 0.005)
    K = args.basis_k if args.basis_k > 0 else max(4, int(len(band_wl) // 2))
    Z, _ = sample_Z_from_basis(key, int(Xj.shape[0]), fine_wl, loga0, logbb0, K=K, sigma_nm=args.basis_sigma_nm, amp=args.prior_amp)

    jfun = build_jacobian_functions(jnp.asarray(coeffs), jnp.asarray(powers), B)
    Jm, cov, vmax, amp_a, amp_bb, support, med_abs_a, med_abs_bb, med_cov_a, med_cov_bb = \
        compute_jacobian_stats(jfun, Z, B, fine_wl, args.jac_samples)

    # After line 578 where you compute Jm
    m_665 = int(np.argmin(np.abs(band_wl - 665)))
    idx_665_fine = int(np.argmin(np.abs(fine_wl - 665)))

    print(f"\nDiagnostics for 665 nm band:")
    print(f"Band index: {m_665}, Band center: {band_wl[m_665]} nm")
    print(f"Fine-wl closest to 665: {fine_wl[idx_665_fine]} nm")
    print(f"J_a[665 band, 665 wl] = {float(Jm[0, m_665, idx_665_fine]):.8f}")
    print(f"J_bb[665 band, 665 wl] = {float(Jm[1, m_665, idx_665_fine]):.8f}")
    print(f"SRF B[665 band, 665 wl] = {float(B[m_665, idx_665_fine]):.8f}")

    print("Median |∂Rrs/∂ln a| on support:", med_abs_a)
    print("Median |∂Rrs/∂ln b_b| on support:", med_abs_bb)
    print("Median CoV (a) on support:", med_cov_a)
    print("Median CoV (bb) on support:", med_cov_bb)

    if args.select == "index" and args.index is not None:
        idx = args.index
        z_init = Z[idx]
        y = Xj[idx]
    else:
        idx, z_init, _, L0, Lf = select_pixel(Z, Xj, jfun, steps=args.gd_steps, lr=args.gd_lr,
                                             mode=args.select, sample=args.sample, seed=args.seed + 1)
        y = Xj[idx]
    print(f"Selected pixel index: {idx}")

    zs, vals = pixel_gradient_path(jfun, z_init, y, steps=args.gd_steps, lr=args.gd_lr)
    _, vals_b = pixel_baseline_path(jfun, z_init, y, steps=args.gd_steps, lr=args.gd_lr, seed=args.seed + 2)
    improve = 100.0 * (vals[0] - vals[-1]) / max(vals[0], 1e-12)
    print(f"Pixel loss reduction along RT gradient: {improve:.1f}%")

    f, _ = jfun
    yhat0 = f(zs[0])
    yhatT = f(zs[-1])

    out_path = Path(args.out)
    plot_compact(Jm, support, vmax, fine_wl, band_wl, y, yhat0, yhatT, vals, out_path)
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
