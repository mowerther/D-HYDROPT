"""
Gradient-based inversion of Sentinel-2 MSI reflectance with D-HYDROPT.

What this demonstrates
----------------------
Automatic differentiation through the HYDROPT polynomial radiative transfer
model, used for two things: reading off how sensitive each satellite band is to
absorption and backscattering at each wavelength (the Jacobian), and inverting
observed Rrs for the optical state by gradient descent.

How the state is parameterised
------------------------------
Not as log(a) and log(bb) directly. Pure water sets a hard floor on the IOPs of
any natural water body -- at 710 nm water alone absorbs 0.827 m^-1 -- and
HYDROPT's polynomial was fitted with water always present, so a state below
that floor is extrapolation. The polynomial does not fail loudly there; it
returns exp() of a large negative number, which underflows to exactly zero in
both value and gradient, and the optimiser simply receives nothing.

So the free parameters here are the *excess* IOPs over pure water, in log
space, via hydropt_jax.domain.iops_from_excess(). Since exp() is positive
everywhere, no gradient step of any size can leave the physical domain.

Which bands are used
--------------------
Only bands whose spectral response lies inside the range where the coefficients
are trustworthy. Two separate limits apply:

  * The coefficient table stops at 710 nm. MSI bands 740 and 783 have
    essentially none of their response inside it (1e-6 and 4e-7 of their
    Gaussian mass), so band-integrating over the table produces a number that
    is not a 740 or 783 nm measurement at all -- it is Rrs at 710 nm, twice.
  * The coefficients degenerate before that stop. The largest coefficient per
    wavelength holds near 3.0 from 400 to 685 nm, then grows to 707 at 710 nm.
    Past 680 nm the model only holds for low excess absorption: at 685 nm it
    tolerates 6.1 m^-1 of excess absorption, at 700 nm 0.62, at 710 nm 0.51.

Bands are therefore sorted into three classes. 'Full' bands lie entirely
inside the grid and are fitted normally. 'Clipped' bands are cut by the end of
the grid -- MSI 704 keeps 83% of its response -- so they are fitted too, but
their modelled value is the average over the covered part only and carries a
known truncation bias; the figure marks them. 'Unusable' bands are dropped
outright.

What this is not
----------------
A well-posed retrieval. The state has two free parameters at each of 63
wavelengths and there are four usable MSI bands, so the inversion is
under-determined by more than an order of magnitude and the fit it reaches is
one of infinitely many that reproduce the observations exactly. The point here
is the gradient machinery -- that the Jacobian is exact, that the optimiser
converges, and that the state stays physical throughout. A real retrieval needs
the state reduced to a few bio-optical parameters, as in hydropt.bio_optics.

Other assumptions
-----------------
- Nadir view, 30 degree solar zenith angle (the polynomial's geometry).
- Gaussian spectral response functions, from published MSI FWHM values.
- Vertically homogeneous water column, Rrs already atmospherically corrected.
- Spectral smoothness imposed through a radial basis function expansion.

Usage
-----
    python examples/msi_gradient_inversion.py
    python examples/msi_gradient_inversion.py --gd-steps 800 --select worst
"""
import argparse
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import jax
import jax.numpy as jnp

from hydropt_jax import (
    RELIABLE_WAVELENGTH_RANGE,
    check_domain,
    iops_from_excess,
    load_pace_poly,
    load_water_iops,
    monomial_powers_2d,
    rrs_poly,
)

#: Published Sentinel-2 MSI full width at half maximum, in nm, by band centre.
MSI_FWHM_NM = {443: 20, 490: 65, 560: 35, 665: 30, 705: 15, 740: 15, 783: 20, 842: 115}

#: A band's response is treated as complete above this coverage of the grid.
FULL_SRF_FRACTION = 0.95

#: Between this and FULL_SRF_FRACTION the band is clipped by the end of the
#: grid: still modellable, but with a truncation bias. Below it, band
#: integration does not measure that band at all and it is dropped.
CLIPPED_SRF_FRACTION = 0.10


# --------------------------------------------------------------------------
# Spectral response
# --------------------------------------------------------------------------

def msi_sigma_nm(band_wl):
    '''
    Gaussian sigma in nm for each MSI band centre, from its published FWHM.

    Band centres are snapped to the nearest entry in MSI_FWHM_NM, so 492 picks
    up B2's 65 nm width and 704 picks up B5's 15 nm.
    '''
    centres = [min(MSI_FWHM_NM, key=lambda c: abs(c - float(b))) for b in np.asarray(band_wl)]
    fwhm = np.array([MSI_FWHM_NM[c] for c in centres], float)

    return fwhm / np.sqrt(8.0 * np.log(2.0))


def build_gaussian_srf(fine_wl, band_wl, sigma_nm):
    '''
    Band-integration matrix B of shape (n_bands, n_wavelengths).

    Each row is a Gaussian spectral response sampled on fine_wl and normalised
    so that it integrates to one over fine_wl. Rrs @ B.T then gives
    band-averaged reflectance.

    Note that the normalisation is over fine_wl only, so a band whose response
    mostly falls outside that grid still produces a normalised row. It will be
    a confident-looking average of whatever sliver happens to be in range --
    which is exactly why srf_coverage() below has to be consulted separately.
    '''
    fine = jnp.asarray(fine_wl, jnp.float64)
    band = jnp.asarray(band_wl, jnp.float64)
    sigma = jnp.asarray(np.broadcast_to(np.asarray(sigma_nm, float), band.shape), jnp.float64)
    response = jnp.exp(-0.5 * ((fine[None, :] - band[:, None]) / sigma[:, None]) ** 2)

    return response / jnp.trapezoid(response, fine, axis=1)[:, None]


def srf_coverage(band_wl, sigma_nm, wavelength_range):
    '''
    Fraction of each band's true Gaussian response inside wavelength_range.

    Computed analytically from the error function rather than from the sampled
    matrix, so it measures the real instrument response rather than the part
    that happened to be sampled.
    '''
    band = np.asarray(band_wl, float)
    sigma = np.asarray(sigma_nm, float)
    lo, hi = wavelength_range
    scale = np.sqrt(2.0) * sigma

    return np.array([0.5 * (math.erf((hi - b) / s) - math.erf((lo - b) / s))
                     for b, s in zip(band, scale)])


def classify_bands(band_wl, sigma_nm, grid_wl):
    '''
    Sort bands into 'full', 'clipped' and 'unusable' by spectral coverage.

    Two coverage fractions are reported, because two different things can go
    wrong with a band:

      coverage - fraction of the band's response inside the coefficient grid.
                 This is the hard limit: a band with no response in the table
                 cannot be modelled at all, and band-integrating over the table
                 anyway just re-reports the reflectance at the table's red end.
      reliable - fraction inside RELIABLE_WAVELENGTH_RANGE, the sub-range whose
                 coefficients hold for any plausible IOP state. A band with
                 high coverage but low reliable coverage is modellable, but
                 only while excess absorption stays low; check_domain() on the
                 retrieved state is what confirms that after the fact.

    Bands classed 'full' or 'clipped' are both fitted -- a clipped band is a
    real measurement of a real part of the spectrum, just averaged over less of
    it than the instrument saw. Only 'unusable' bands are discarded.

    Returns a DataFrame with one row per band.
    '''
    grid_range = (float(np.min(grid_wl)), float(np.max(grid_wl)))
    coverage = srf_coverage(band_wl, sigma_nm, grid_range)
    reliable = srf_coverage(band_wl, sigma_nm, RELIABLE_WAVELENGTH_RANGE)
    label = np.where(coverage >= FULL_SRF_FRACTION, 'full',
                     np.where(coverage >= CLIPPED_SRF_FRACTION, 'clipped', 'unusable'))

    return pd.DataFrame({'centre_nm': np.asarray(band_wl, float),
                         'coverage': coverage,
                         'reliable': reliable,
                         'class': label})


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_rrs_csv(path):
    '''
    Load Rrs from a CSV whose column headers are wavelengths.

    Returns (X, wl) with X of shape (n_pixels, n_bands). Rows containing any
    non-finite value are dropped.
    '''
    df = pd.read_csv(path, sep=None, engine='python')
    df = df.drop(columns=[c for c in df.columns
                          if str(c).startswith('Unnamed') or str(c).lower() == 'index'])

    columns = []
    for column in df.columns:
        match = re.search(r'\d+(\.\d+)?', str(column))
        if match:
            columns.append((float(match.group(0)), column))
    if not columns:
        raise ValueError('no wavelength-like column headers found in {}'.format(path))

    columns.sort()
    wl = np.array([w for w, _ in columns], float)
    X = df[[c for _, c in columns]].to_numpy(float)

    return X[np.isfinite(X).all(axis=1)], wl


# --------------------------------------------------------------------------
# Forward model and priors
# --------------------------------------------------------------------------

def build_forward(coeffs, powers, srf, a_water, bb_water):
    '''
    Band-integrated forward model and its Jacobian, over log excess IOPs.

    Returns (forward, jacobian). forward maps a state of shape
    (2, n_wavelengths) -- [log excess a, log excess bb] -- to band Rrs of shape
    (n_bands,). The pure-water floor is built in, so no state is out of domain.
    '''
    coeffs = jnp.asarray(coeffs, jnp.float64)
    powers = jnp.asarray(powers, jnp.int32)
    srf = jnp.asarray(srf, jnp.float64)

    def forward(state):
        a, bb = iops_from_excess(state[0], state[1], a_water, bb_water)

        return rrs_poly(a, bb, coeffs, powers) @ srf.T

    return forward, jax.jacrev(forward)


def rbf_basis(wl, n_centres=6, sigma_nm=40.0):
    '''
    Radial basis functions used to keep sampled spectra smooth.

    Returns a matrix of shape (n_wavelengths, n_centres), column-normalised.
    '''
    wl = jnp.asarray(wl, jnp.float64)
    centres = jnp.linspace(float(wl[0]), float(wl[-1]), n_centres)
    basis = jnp.exp(-0.5 * ((wl[:, None] - centres[None, :]) / sigma_nm) ** 2)

    return basis / (jnp.sum(basis, 0, keepdims=True) + 1e-12)


def sample_states(key, n, wl, log_da0, log_dbb0, n_centres=6, sigma_nm=40.0, amplitude=0.35):
    '''
    Draw n smooth states around a mean, perturbed in the RBF basis.

    Returns an array of shape (n, 2, n_wavelengths) of [log excess a,
    log excess bb] pairs.
    '''
    basis = rbf_basis(wl, n_centres, sigma_nm)
    key_a, key_bb = jax.random.split(key)
    log_da = log_da0[None, :] + amplitude * jax.random.normal(key_a, (n, n_centres)) @ basis.T
    log_dbb = log_dbb0[None, :] + amplitude * jax.random.normal(key_bb, (n, n_centres)) @ basis.T

    return jnp.stack([log_da, log_dbb], 1)


# --------------------------------------------------------------------------
# Sensitivity analysis
# --------------------------------------------------------------------------

def jacobian_statistics(jacobian_fn, states, srf, wl, n_samples=64, support_threshold=1e-3):
    '''
    Median Jacobian and coefficient of variation across an ensemble of states.

    Returns (median, coefficient_of_variation, support), each of shape
    (2, n_bands, n_wavelengths) for the median and CoV and (n_bands,
    n_wavelengths) for the support mask. Entries outside a band's spectral
    support are zeroed, since the Jacobian there is meaningless.
    '''
    J = jax.vmap(jacobian_fn)(states[:n_samples])
    # (n_states, n_bands, 2, n_wavelengths) -> (n_states, 2, n_bands, n_wavelengths)
    J = jnp.transpose(J, (0, 2, 1, 3))

    srf = jnp.asarray(srf, jnp.float64)
    support = srf > jnp.max(srf, axis=1, keepdims=True) * support_threshold
    median = jnp.median(J, axis=0)
    spread = jnp.std(J, axis=0)

    return (jnp.where(support[None, :, :], median, 0.0),
            jnp.where(support[None, :, :], spread / (jnp.abs(median) + 1e-12), 0.0),
            support)


# --------------------------------------------------------------------------
# Inversion
# --------------------------------------------------------------------------

def adam_step(state, gradient, moment, velocity, t, lr,
              beta1=0.9, beta2=0.999, eps=1e-8):
    '''One Adam update. Returns (state, moment, velocity).'''
    moment = beta1 * moment + (1.0 - beta1) * gradient
    velocity = beta2 * velocity + (1.0 - beta2) * gradient ** 2
    moment_hat = moment / (1.0 - beta1 ** t)
    velocity_hat = velocity / (1.0 - beta2 ** t)

    return state - lr * moment_hat / (jnp.sqrt(velocity_hat) + eps), moment, velocity


def descend(forward, state, target, steps=400, lr=0.05):
    '''
    Fit one pixel by Adam. Returns (trajectory, losses).

    Adam rather than plain normalised gradient descent, because the problem is
    badly scaled across wavelength. Rrs at 443 nm is roughly twenty times Rrs at
    665 nm for these spectra, and the Jacobian scales with Rrs, so a single
    step size normalised by the global gradient RMS is set almost entirely by
    the blue bands: the red bands receive a step far too small to move them.
    That is what left 665 nm sitting near zero in earlier versions of this
    example -- an optimiser artefact, not a limit of the model. Adam's
    per-parameter scaling removes it, and 665 nm converges along with the rest.
    '''
    @jax.jit
    def loss(z):
        residual = forward(z) - target

        return jnp.mean(residual * residual)

    grad_fn = jax.jit(jax.grad(loss))
    moment = jnp.zeros_like(state)
    velocity = jnp.zeros_like(state)
    trajectory = [state]
    losses = [float(loss(state))]
    for t in range(1, steps + 1):
        state, moment, velocity = adam_step(
            state, grad_fn(state), moment, velocity, t, lr)
        trajectory.append(state)
        losses.append(float(loss(state)))

    return jnp.stack(trajectory), np.asarray(losses)


def descend_batch(forward, states, targets, steps=400, lr=0.05):
    '''Vectorised Adam over many pixels; returns (states, initial, final).'''
    def loss(z, y):
        residual = forward(z) - y

        return jnp.mean(residual * residual)

    loss_fn = jax.jit(jax.vmap(loss))
    grad_fn = jax.jit(jax.vmap(jax.grad(loss)))

    @jax.jit
    def step(carry, t):
        z, moment, velocity = carry

        return adam_step(z, grad_fn(z, targets), moment, velocity, t, lr)

    initial = loss_fn(states, targets)
    carry = (states, jnp.zeros_like(states), jnp.zeros_like(states))
    for t in range(1, steps + 1):
        carry = step(carry, t)

    return carry[0], initial, loss_fn(carry[0], targets)


def select_pixel(forward, states, targets, mode='best', steps=400, lr=0.05,
                 sample=512, seed=0):
    '''
    Optimise a subset of pixels and pick one to show in detail.

    mode - 'best' lowest final loss, 'worst' highest, 'median', or 'improve'
           for the largest reduction.

    Returns (index_into_original_array, initial_loss, final_loss).
    '''
    n = states.shape[0]
    if sample and n > sample:
        subset = np.asarray(np.random.default_rng(seed).choice(n, size=sample, replace=False))
    else:
        subset = np.arange(n)

    _, initial, final = descend_batch(
        forward, states[subset], targets[subset], steps=steps, lr=lr)

    if mode == 'best':
        chosen = int(jnp.argmin(final))
    elif mode == 'worst':
        chosen = int(jnp.argmax(final))
    elif mode == 'median':
        chosen = int(jnp.argsort(final)[final.shape[0] // 2])
    else:
        chosen = int(jnp.argmax(initial - final))

    return int(subset[chosen]), float(initial[chosen]), float(final[chosen])


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def plot_results(median_jacobian, support, fine_wl, bands, observed,
                 predicted_initial, predicted_final, losses, out_path,
                 colour_limit=3e-4, linear_threshold=1e-7):
    '''
    Four-panel summary: two Jacobian heatmaps, the pixel fit, and convergence.

    bands is the DataFrame from classify_bands(), restricted to the bands that
    were carried through the inversion. Bands whose spectral response is
    clipped by the end of the coefficient grid are marked, so the figure states
    where the modelled band value carries a truncation bias.
    '''
    import matplotlib as mpl
    from matplotlib.colors import SymLogNorm

    mpl.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

    band_wl = bands['centre_nm'].to_numpy()
    is_clipped = (bands['class'] == 'clipped').to_numpy()
    n_bands = len(band_wl)
    lo, hi = RELIABLE_WAVELENGTH_RANGE

    figure = plt.figure(figsize=(15, 4))
    grid = figure.add_gridspec(
        2, 5, width_ratios=[1.1, 1.1, 0.05, 0.75, 1.6], height_ratios=[1.0, 1.0],
        wspace=0.15, hspace=0.4, top=0.96, bottom=0.13)

    colours = plt.get_cmap('seismic').copy()
    colours.set_bad(color='none', alpha=0.0)
    # Sensitivity scales with Rrs itself, which falls by two orders of magnitude
    # from the blue bands to the red ones. On a linear scale the red rows are
    # uniformly white and the panel says nothing about them, so the colour axis
    # is symmetric-log: linear through zero, logarithmic outside it, sign kept.
    norm = SymLogNorm(linthresh=linear_threshold, vmin=-colour_limit,
                      vmax=colour_limit, base=10)

    def draw_heatmap(axis, values, title):
        panel = np.asarray(values).copy()
        panel[~np.asarray(support)] = np.nan
        axis.set_facecolor('white')
        image = axis.imshow(
            panel, aspect='auto', cmap=colours, norm=norm, interpolation='nearest',
            extent=[float(fine_wl[0]), float(fine_wl[-1]), n_bands - 0.5, -0.5], zorder=1)

        for centre in band_wl:
            axis.axvline(float(centre), lw=0.6, color='grey', alpha=0.25, zorder=0)

        axis.set_yticks(np.arange(n_bands))
        axis.set_yticklabels(
            ['{:.0f}{}'.format(w, '*' if c else '')
             for w, c in zip(band_wl, is_clipped)])
        axis.tick_params(axis='y', which='major', length=0, rotation=90)
        axis.set_yticks(np.linspace(n_bands - 0.5, -0.5, n_bands + 1), minor=True)
        axis.grid(True, axis='y', which='minor')
        axis.set_title(title, fontsize=14)

        for row in range(n_bands):
            # Each row gets its own vertical scale, so the shape of a red band's
            # response is as legible as a blue band's. The colour underneath
            # still carries the absolute magnitude.
            peak = np.nanmax(np.abs(panel[row]))
            peak = float(peak) if np.isfinite(peak) and peak > 0 else 1.0
            inset = axis.inset_axes([0, 1 - (row + 1) / n_bands, 1, 1 / n_bands],
                                    sharex=axis, ylim=(-1.35 * peak, 1.35 * peak))
            inset.set_axis_off()
            inset.plot(fine_wl, panel[row], color='black', linewidth=1,
                       path_effects=[mpe.Stroke(linewidth=3, foreground='white'), mpe.Normal()])

        axis.set_xlim(float(fine_wl[0]), float(fine_wl[-1]))
        axis.set_ylim(n_bands - 0.5, -0.5)

        return image

    axis_a = figure.add_subplot(grid[:, 0])
    image = draw_heatmap(axis_a, median_jacobian[0],
                         r'$\partial R_{rs}^{\mathrm{band}} / \partial \ln a(\lambda)$')
    axis_a.set_ylabel('MSI band [nm]', fontsize=14, fontweight='bold')
    axis_a.text(0.98, 0.98, '(A)', transform=axis_a.transAxes, fontsize=14,
                fontweight='bold', va='top', ha='right')

    axis_b = figure.add_subplot(grid[:, 1])
    draw_heatmap(axis_b, median_jacobian[1],
                 r'$\partial R_{rs}^{\mathrm{band}} / \partial \ln b_b(\lambda)$')
    axis_b.set_yticklabels([])
    axis_b.text(0.98, 0.98, '(B)', transform=axis_b.transAxes, fontsize=14,
                fontweight='bold', va='top', ha='right')

    colour_axis = figure.add_subplot(grid[:, 2])
    bar = figure.colorbar(image, cax=colour_axis, extend='both')
    bar.set_label('Sensitivity [sr$^{-1}$]', fontsize=13, fontweight='bold')
    decades = [colour_limit / 10 ** k for k in range(3)]
    bar.set_ticks([-t for t in decades] + [0.0] + decades[::-1])

    figure.add_subplot(grid[:, 3]).axis('off')

    axis_fit = figure.add_subplot(grid[0, 4])
    axis_fit.plot(band_wl, np.asarray(observed), 'o-', color='black', markersize=9,
                  linewidth=2.5, label='Observed', zorder=3)
    axis_fit.plot(band_wl, np.asarray(predicted_initial), 'o--', color='tab:blue',
                  markersize=5, label='Initial guess', zorder=4)
    axis_fit.plot(band_wl, np.asarray(predicted_final), 'o--', color='limegreen',
                  markersize=5, linewidth=1.8, label='After gradient descent', zorder=6)

    if is_clipped.any():
        # Shade the range where the coefficients only hold for low absorption,
        # and ring the bands whose response the grid cuts short.
        axis_fit.axvspan(hi, max(band_wl) * 1.03, alpha=0.12, color='grey', zorder=0)
        axis_fit.axvline(hi, color='firebrick', linestyle='--', linewidth=1.2,
                         alpha=0.8, zorder=1)
        axis_fit.scatter(band_wl[is_clipped], np.asarray(predicted_final)[is_clipped],
                         s=130, facecolors='none', edgecolors='firebrick',
                         linewidths=1.6, zorder=5, label='Clipped response (*)')
        axis_fit.annotate(
            '{:.0f} nm'.format(hi), xy=(hi, 1.0), xycoords=('data', 'axes fraction'),
            xytext=(-3, -2), textcoords='offset points', ha='right', va='top',
            fontsize=8, color='firebrick', rotation=90)

    axis_fit.set_xlabel('MSI band [nm]', fontsize=14, fontweight='bold')
    axis_fit.set_ylabel(r'$\mathbf{R_{rs}}$ [sr$^{-1}$]', fontsize=14, fontweight='bold')
    axis_fit.set_ylim(bottom=0.0)
    axis_fit.set_xlim(min(band_wl) - 15, max(band_wl) * 1.03)
    axis_fit.grid(True, alpha=0.3)
    axis_fit.legend(framealpha=0.9, fontsize=8, loc='lower left')
    axis_fit.text(0.99, 0.97, '(C)', transform=axis_fit.transAxes, fontsize=14,
                  fontweight='bold', va='top', ha='right')

    axis_loss = figure.add_subplot(grid[1, 4])
    axis_loss.plot(np.arange(len(losses)), losses, linewidth=2, color='green')
    axis_loss.set_yscale('log')
    axis_loss.set_xlabel('Optimisation step', fontsize=14, fontweight='bold')
    axis_loss.set_ylabel('MSE', fontsize=14, fontweight='bold')
    axis_loss.grid(True, alpha=0.3)
    axis_loss.text(0.99, 0.97, '(D)', transform=axis_loss.transAxes, fontsize=14,
                   fontweight='bold', va='top', ha='right')

    left, right = axis_a.get_position(), axis_b.get_position()
    figure.text((left.x0 + right.x1) / 2, 0.02, 'HYDROPT $\\lambda$ [nm]',
                ha='center', fontsize=14, fontweight='bold')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=300, bbox_inches='tight')

    return figure


# --------------------------------------------------------------------------

def report_domain(problems, n_wavelengths):
    '''Print the outcome of a check_domain() call in full.'''
    if not problems:
        print('  inside the fitted domain at all {} wavelengths'.format(n_wavelengths))

        return
    for problem in problems:
        print('  note: {}'.format(problem))


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--msi-csv', default=str(here / 'example_rrs_dataset_msi.csv'))
    parser.add_argument('--out', default=str(here / 'figures' / 'msi_gradient_inversion.png'))
    parser.add_argument('--subset', type=int, default=2000,
                        help='maximum number of pixels to read')
    parser.add_argument('--sample', type=int, default=512,
                        help='pixels to optimise when selecting one to display')
    parser.add_argument('--jacobian-samples', type=int, default=64)
    parser.add_argument('--gd-steps', type=int, default=400)
    parser.add_argument('--gd-lr', type=float, default=0.05)
    parser.add_argument('--select', default='best',
                        choices=['best', 'worst', 'median', 'improve', 'index'])
    parser.add_argument('--index', type=int, default=None,
                        help='pixel to display; required with --select index')
    parser.add_argument('--basis-centres', type=int, default=6)
    parser.add_argument('--basis-sigma-nm', type=float, default=40.0)
    parser.add_argument('--prior-amplitude', type=float, default=0.35)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    if args.select == 'index' and args.index is None:
        parser.error('--select index requires --index')

    return args


def main(argv=None):
    args = parse_args(argv)

    fine_wl, coeffs = load_pace_poly()
    powers = monomial_powers_2d(4)
    a_water, bb_water = load_water_iops(fine_wl)

    observations, band_wl = load_rrs_csv(args.msi_csv)
    observations = observations[:args.subset]
    sigma = msi_sigma_nm(band_wl)

    bands = classify_bands(band_wl, sigma, fine_wl)
    print('Sentinel-2 MSI bands in {}'.format(Path(args.msi_csv).name))
    print('  coverage = response inside the coefficient grid {:.0f}-{:.0f} nm'
          .format(float(fine_wl.min()), float(fine_wl.max())))
    print('  reliable = response inside {:.0f}-{:.0f} nm, where the coefficients '
          'hold for any state\n'.format(*RELIABLE_WAVELENGTH_RANGE))
    for _, row in bands.iterrows():
        print('  {:>6.0f} nm   coverage {:8.4%}   reliable {:8.4%}   {}'.format(
            row['centre_nm'], row['coverage'], row['reliable'], row['class']))

    keep = (bands['class'] != 'unusable').to_numpy()
    dropped = bands['centre_nm'].to_numpy()[~keep]
    clipped = bands['centre_nm'].to_numpy()[(bands['class'] == 'clipped').to_numpy()]
    if dropped.size:
        print('\n  dropped: {} -- essentially none of their response falls inside the '
              'coefficient\n  table, so band-integrating over it would just re-report '
              'Rrs at its red end.'
              .format(', '.join('{:.0f} nm'.format(w) for w in dropped)))
    if clipped.size:
        print('  fitted with a truncation bias: {} -- response is cut by the end of the '
              'grid,\n  so the modelled value averages only the covered part.'
              .format(', '.join('{:.0f} nm'.format(w) for w in clipped)))

    bands = bands[keep].reset_index(drop=True)
    srf = build_gaussian_srf(fine_wl, bands['centre_nm'].to_numpy(), sigma[keep])
    targets = jnp.asarray(observations[:, keep])

    forward, jacobian_fn = build_forward(coeffs, powers, srf, a_water, bb_water)

    # Start from a clear-water state: 0.05 m^-1 of excess absorption and
    # 0.005 m^-1 of excess backscattering on top of pure water.
    log_da0 = jnp.log(jnp.full(fine_wl.shape, 0.05))
    log_dbb0 = jnp.log(jnp.full(fine_wl.shape, 0.005))

    a_start, bb_start = iops_from_excess(log_da0, log_dbb0, a_water, bb_water)
    print('\nStarting state (clear water over the pure-water floor):')
    report_domain(check_domain(a_start, bb_start, fine_wl, coeffs, powers, error=False),
                  fine_wl.size)

    rrs_start = np.asarray(rrs_poly(a_start, bb_start, coeffs, powers))
    print('  Rrs spans {:.3e} to {:.3e} sr^-1, with {} exact zeros'
          .format(rrs_start.min(), rrs_start.max(), int((rrs_start == 0).sum())))

    key = jax.random.PRNGKey(args.seed)
    states = sample_states(key, int(targets.shape[0]), fine_wl, log_da0, log_dbb0,
                           n_centres=args.basis_centres, sigma_nm=args.basis_sigma_nm,
                           amplitude=args.prior_amplitude)

    median_jacobian, cov, support = jacobian_statistics(
        jacobian_fn, states, srf, fine_wl, n_samples=args.jacobian_samples)

    on_support = np.asarray(support)
    print('\nSensitivity across the ensemble, on each band\'s spectral support:')
    print('  median |dRrs/dln a|   {:.3e}'.format(
        float(np.median(np.abs(np.asarray(median_jacobian[0]))[on_support]))))
    print('  median |dRrs/dln bb|  {:.3e}'.format(
        float(np.median(np.abs(np.asarray(median_jacobian[1]))[on_support]))))
    print('  median CoV (a)        {:.3f}'.format(
        float(np.median(np.asarray(cov[0])[on_support]))))
    print('  median CoV (bb)       {:.3f}'.format(
        float(np.median(np.asarray(cov[1])[on_support]))))

    if args.select == 'index':
        index = args.index
    else:
        index, _, _ = select_pixel(
            forward, states, targets, mode=args.select,
            steps=args.gd_steps, lr=args.gd_lr, sample=args.sample, seed=args.seed + 1)
    print('\nSelected pixel {}'.format(index))

    trajectory, losses = descend(
        forward, states[index], targets[index], steps=args.gd_steps, lr=args.gd_lr)
    print('  MSE {:.3e} -> {:.3e}  ({:.1f}% reduction over {} steps)'.format(
        losses[0], losses[-1], 100.0 * (losses[0] - losses[-1]) / max(losses[0], 1e-12),
        args.gd_steps))

    a_final, bb_final = iops_from_excess(
        trajectory[-1][0], trajectory[-1][1], a_water, bb_water)
    print('\nRetrieved state:')
    report_domain(check_domain(a_final, bb_final, fine_wl, coeffs, powers, error=False),
                  fine_wl.size)

    predicted_final = np.asarray(forward(trajectory[-1]))
    print('  predicted Rrs spans {:.3e} to {:.3e} sr^-1, with {} exact zeros'
          .format(predicted_final.min(), predicted_final.max(),
                  int((predicted_final == 0).sum())))

    out_path = Path(args.out)
    plot_results(median_jacobian, support, fine_wl, bands,
                 np.asarray(targets[index]), np.asarray(forward(trajectory[0])),
                 predicted_final, losses, out_path)
    print('\nSaved figure to {}'.format(out_path))


if __name__ == '__main__':
    main()
