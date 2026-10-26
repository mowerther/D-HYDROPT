"""
Inversion of Sentinel-2 MSI reflectance by gradient descent with D-HYDROPT.

Purpose
-------
Automatic differentiation through the HYDROPT polynomial radiative transfer
model serves two purposes here: the Jacobian, which gives the sensitivity of
each satellite band to absorption and backscattering at each wavelength, and
the inversion of MSI Rrs for the optical state by gradient descent.

How the state is parameterised
------------------------------
The free parameters are the logarithms of the *excess* IOPs above pure water,
passed through hydropt_jax.domain.iops_from_excess(). Pure water sets the
lower bound of the IOPs of natural waters, and at 710 nm pure water absorbs
0.827 m^-1. The polynomial of HYDROPT was fitted with water included, so a
state below that bound is extrapolation. There the polynomial returns exp()
of a large negative number, which underflows to zero in value and gradient,
and the optimiser receives a zero gradient and no error.

exp() is non-negative, so each gradient step, however large, keeps the state
at or above the pure water floor.

Which bands are used
--------------------
The example uses the bands whose spectral response lies inside the range
where the coefficients are reliable. Two limits apply:

  * The coefficient table ends at 710 nm. MSI bands 740 and 783 have 1e-6 and
    4e-18 of their Gaussian response inside it. Integration over the table
    puts nearly all weight of both bands on the last wavelengths of the grid,
    and both values approximate Rrs at 710 nm.
  * The coefficients grow before the end of the table. The largest
    coefficient at each wavelength stays near 3.0 from 400 to 685 nm, then
    grows to 707 at 710 nm. Above 680 nm the model holds only for low excess
    absorption: at 685 nm it tolerates 5.9 m^-1 of excess absorption, at
    700 nm 0.61, at 710 nm 0.50.

classify_bands() therefore labels each band with one of three classes. 'Full'
bands have at least 95% of their response inside the grid. 'Clipped' bands
are cut by the end of the grid (MSI 704 keeps 83% of its Gaussian response).
Their modelled value is the average over the covered part and carries a
truncation bias, and the figure marks them. The example fits 'full' and
'clipped' bands and discards 'unusable' bands.

Scope of the example
--------------------
The inversion is underdetermined: the state has two free parameters at each
of 63 wavelengths, constrained by five MSI bands. Many states fit these bands
equally well. The example illustrates exact gradients and stable convergence
within the physical domain. A well-posed retrieval reduces the state to a few
bio-optical parameters, as hydropt.bio_optics does.

Other assumptions
-----------------
- Nadir view, 30 degree solar zenith angle (the geometry of the polynomial).
- Gaussian approximation of the spectral response functions, from the nominal
  MSI FWHM (Drusch et al., 2012). --srf S2A, S2B or S2C uses the tabulated
  ESA response functions of that satellite instead.
- Vertically homogeneous water column, Rrs already atmospherically corrected.
- Spectral smoothness imposed through a radial basis function expansion.

Usage
-----
    python examples/msi_gradient_inversion.py
    python examples/msi_gradient_inversion.py --gd-steps 800 --select worst
    python examples/msi_gradient_inversion.py --srf S2B
"""
import argparse
import re
import sys
from pathlib import Path

# Run from a clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as mpe  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from hydropt_jax import (  # noqa: E402
    RELIABLE_WAVELENGTH_RANGE,
    build_gaussian_srf,
    build_tabulated_srf,
    check_domain,
    classify_bands,
    classify_tabulated_bands,
    iops_from_excess,
    load_msi_srf,
    load_pace_poly,
    load_water_iops,
    monomial_powers_2d,
    msi_sigma_nm,
    rrs_poly,
)
from hydropt_jax.bands import CLIPPED_SRF_FRACTION  # noqa: E402


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_rrs_csv(path):
    '''
    Load Rrs from a CSV whose column headers are wavelengths.

    Returns (X, wl) with X of shape (n_pixels, n_bands). Rows containing NaN
    or infinite values are dropped.
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
        raise ValueError('no column headers containing wavelengths in {}'.format(path))

    columns.sort()
    wl = np.array([w for w, _ in columns], float)
    X = df[[c for _, c in columns]].to_numpy(float)

    return X[np.isfinite(X).all(axis=1)], wl


# --------------------------------------------------------------------------
# Forward model and priors
# --------------------------------------------------------------------------

def build_forward(coeffs, powers, srf, a_water, bb_water):
    '''
    Forward model integrated over each band, and its Jacobian, over log excess
    IOPs.

    Returns (forward, jacobian). forward maps a state of shape
    (2, n_wavelengths), holding [log excess a, log excess bb], to band Rrs of
    shape (n_bands,). The pure water floor is part of the parameterisation, so
    each state satisfies it.
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

    Returns a matrix of shape (n_wavelengths, n_centres), with each column
    normalised to sum to one.
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
    n_wavelengths) for the support mask. Entries outside the spectral support
    of a band are set to zero, because the band has negligible response there.
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

    Adam scales the step of each parameter separately. The problem needs this
    because it is badly scaled across wavelength: Rrs at 443 nm is roughly
    twenty times Rrs at 665 nm for these spectra, and the Jacobian scales with
    Rrs. A single step size normalised by the global gradient RMS follows the
    blue bands, and the red bands then receive steps too small to move them.
    With Adam, 665 nm converges with the other bands.
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
    '''Vectorised Adam over many pixels. Returns (states, initial, final).'''
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
    Optimise a subset of pixels and select one to display in detail.

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
                 colour_limit=1.5e-3, linear_threshold=1e-7):
    '''
    Summary figure with four panels: two Jacobian heatmaps, the pixel fit, and
    convergence.

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

    figure = plt.figure(figsize=(15, 5))
    grid = figure.add_gridspec(
        1, 5, width_ratios=[1.1, 1.1, 0.05, 0.75, 1.6],
        wspace=0.15, top=0.90, bottom=0.13)
    # A dedicated legend row keeps all explanatory text clear of the spectra.
    right_grid = grid[0, 4].subgridspec(
        3, 1, height_ratios=[0.40, 1.0, 1.0], hspace=0.55)
    legend_axis = figure.add_subplot(right_grid[0])
    legend_axis.axis('off')

    colours = plt.get_cmap('seismic').copy()
    colours.set_bad(color='none', alpha=0.0)
    # Sensitivity scales with Rrs itself, which falls by two orders of magnitude
    # from the blue bands to the red ones. On a linear scale the red rows are
    # uniformly white, so the colour axis is symmetric logarithmic: linear
    # around zero, logarithmic outside that range, with the sign kept.
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
            # Each row has its own vertical scale, so the response shape of a
            # red band is as legible as that of a blue band. The colour
            # underneath carries the absolute magnitude.
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
                         r'$\partial R_{rs}^{\mathrm{band}} / \partial \ln \Delta a(\lambda)$')
    axis_a.set_ylabel('MSI band [nm]', fontsize=14, fontweight='bold')
    axis_a.text(0.98, 0.98, '(A)', transform=axis_a.transAxes, fontsize=14,
                fontweight='bold', va='top', ha='right')

    axis_b = figure.add_subplot(grid[:, 1])
    draw_heatmap(axis_b, median_jacobian[1],
                 r'$\partial R_{rs}^{\mathrm{band}} / \partial \ln \Delta b_b(\lambda)$')
    axis_b.set_yticklabels([])
    axis_b.text(0.98, 0.98, '(B)', transform=axis_b.transAxes, fontsize=14,
                fontweight='bold', va='top', ha='right')

    colour_axis = figure.add_subplot(grid[:, 2])
    bar = figure.colorbar(image, cax=colour_axis, extend='both')
    bar.set_label('Sensitivity [sr$^{-1}$]', fontsize=13, fontweight='bold')
    # Label powers of ten from the colour limit down to the linear region
    # around zero, so each decade of the symmetric logarithmic scale has a
    # tick.
    exponents = np.arange(int(np.ceil(np.log10(linear_threshold))),
                          int(np.floor(np.log10(colour_limit))) + 1)
    positive_ticks = 10.0 ** exponents
    ticks = np.concatenate((-positive_ticks[::-1], [0.0], positive_ticks))
    labels = ([rf'$-10^{{{power}}}$' for power in exponents[::-1]] + ['$0$']
              + [rf'$10^{{{power}}}$' for power in exponents])
    bar.set_ticks(ticks, labels=labels)
    bar.minorticks_off()
    bar.ax.tick_params(labelsize=10)

    figure.add_subplot(grid[:, 3]).axis('off')

    axis_fit = figure.add_subplot(right_grid[1])
    axis_fit.plot(band_wl, np.asarray(observed), 'o-', color='black', markersize=9,
                  linewidth=2.5, label=r'MSI R$_{\mathrm{rs}}$', zorder=3)
    axis_fit.plot(band_wl, np.asarray(predicted_initial), 'o--', color='tab:blue',
                  markersize=5, label='Initial state', zorder=4)
    axis_fit.plot(band_wl, np.asarray(predicted_final), 'o--', color='limegreen',
                  markersize=5, linewidth=1.8, label='After gradient descent', zorder=6)

    if is_clipped.any():
        # Shade the range where the coefficients only hold for low absorption,
        # and ring the bands whose response the grid cuts short.
        axis_fit.axvspan(hi, max(band_wl) * 1.03, alpha=0.12, color='grey', zorder=0)
        axis_fit.axvline(hi, color='firebrick', linestyle='--', linewidth=1.2,
                         alpha=0.8, zorder=1,
                         label='Above {:.0f} nm: valid for low absorption'.format(hi))
        axis_fit.scatter(band_wl[is_clipped], np.asarray(predicted_final)[is_clipped],
                         s=130, facecolors='none', edgecolors='firebrick',
                         linewidths=1.6, zorder=5,
                         label='Response cut off at {:.0f} nm (*)'.format(
                             float(np.max(fine_wl))))

    axis_fit.set_xlabel('MSI band [nm]', fontsize=14, fontweight='bold')
    axis_fit.set_ylabel(r'$\mathbf{R_{rs}}$ [sr$^{-1}$]', fontsize=14, fontweight='bold')
    # Round the upper limit up to a labelled tick above the highest spectrum.
    peak_rrs = max(float(np.max(values))
                   for values in (observed, predicted_initial, predicted_final))
    rrs_ticks = mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(
        0.0, 1.05 * peak_rrs)
    axis_fit.set_yticks(rrs_ticks)
    axis_fit.set_ylim(0.0, rrs_ticks[-1])
    axis_fit.set_xlim(min(band_wl) - 15, max(band_wl) * 1.03)
    axis_fit.grid(True, alpha=0.3)
    handles, labels = axis_fit.get_legend_handles_labels()
    legend_axis.legend(handles, labels, frameon=False, fontsize=8.5,
                       loc='center', ncol=2, columnspacing=1.5)
    axis_fit.text(0.0, 1.04, '(C)', transform=axis_fit.transAxes, fontsize=14,
                  fontweight='bold', va='bottom', ha='left')

    axis_loss = figure.add_subplot(right_grid[2])
    axis_loss.plot(np.arange(len(losses)), losses, linewidth=2, color='green')
    axis_loss.set_yscale('log')
    axis_loss.set_xlabel('Optimisation step', fontsize=14, fontweight='bold')
    axis_loss.set_ylabel('MSE', fontsize=14, fontweight='bold')
    axis_loss.grid(True, alpha=0.3)
    axis_loss.text(0.0, 1.04, '(D)', transform=axis_loss.transAxes, fontsize=14,
                   fontweight='bold', va='bottom', ha='left')

    left, right = axis_a.get_position(), axis_b.get_position()
    figure.text((left.x0 + right.x1) / 2, 0.02, 'Wavelength $\\lambda$ [nm]',
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
                        help='pixel to display, required with --select index')
    parser.add_argument('--basis-centres', type=int, default=6)
    parser.add_argument('--basis-sigma-nm', type=float, default=40.0)
    parser.add_argument('--prior-amplitude', type=float, default=0.35)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--srf', default='gaussian', choices=['gaussian', 'S2A', 'S2B', 'S2C'],
                        help='spectral response functions: Gaussian from the nominal '
                             'MSI widths, or the tabulated ESA functions of a satellite')
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

    if args.srf == 'gaussian':
        sigma = msi_sigma_nm(band_wl)
        bands = classify_bands(band_wl, sigma, fine_wl)
        srf_name = 'Gaussian from the nominal MSI widths'
    else:
        srf_wl, response = load_msi_srf(band_wl, args.srf)
        bands = classify_tabulated_bands(band_wl, srf_wl, response, fine_wl)
        srf_name = 'tabulated ESA functions of {}'.format(args.srf)
    print('Sentinel-2 MSI bands in {}, {}'.format(Path(args.msi_csv).name, srf_name))
    print('  coverage = response inside the coefficient grid {:.0f}-{:.0f} nm'
          .format(float(fine_wl.min()), float(fine_wl.max())))
    print('  reliable = response inside {:.0f}-{:.0f} nm, where log(Rrs) stays above -30 '
          'for plausible states\n'.format(*RELIABLE_WAVELENGTH_RANGE))
    for _, row in bands.iterrows():
        print('  {:>6.0f} nm   coverage {:8.4%}   reliable {:8.4%}   {}'.format(
            row['centre_nm'], row['coverage'], row['reliable'], row['class']))

    keep = (bands['class'] != 'unusable').to_numpy()
    dropped = bands['centre_nm'].to_numpy()[~keep]
    clipped = bands['centre_nm'].to_numpy()[(bands['class'] == 'clipped').to_numpy()]
    if dropped.size:
        print('\n  discarded: {}. Less than {:.0%} of their response lies inside the '
              'coefficient table.'
              .format(', '.join('{:.0f} nm'.format(w) for w in dropped), CLIPPED_SRF_FRACTION))
    if clipped.size:
        print('  fitted with a truncation bias: {}. The end of the grid cuts their '
              'response,\n  and the modelled value averages the covered part.'
              .format(', '.join('{:.0f} nm'.format(w) for w in clipped)))

    bands = bands[keep].reset_index(drop=True)
    if args.srf == 'gaussian':
        srf = build_gaussian_srf(fine_wl, bands['centre_nm'].to_numpy(), sigma[keep])
    else:
        srf = build_tabulated_srf(fine_wl, srf_wl, response[keep])
    targets = jnp.asarray(observations[:, keep])

    forward, jacobian_fn = build_forward(coeffs, powers, srf, a_water, bb_water)

    # Start from a clear water state: 0.05 m^-1 of excess absorption and
    # 0.005 m^-1 of excess backscattering on top of pure water.
    log_da0 = jnp.log(jnp.full(fine_wl.shape, 0.05))
    log_dbb0 = jnp.log(jnp.full(fine_wl.shape, 0.005))

    a_start, bb_start = iops_from_excess(log_da0, log_dbb0, a_water, bb_water)
    print('\nStarting state (clear water over the pure water floor):')
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
    print('\nSensitivity across the ensemble, on the spectral support of each band:')
    print('  median |dRrs/dln excess a|   {:.3e}'.format(
        float(np.median(np.abs(np.asarray(median_jacobian[0]))[on_support]))))
    print('  median |dRrs/dln excess bb|  {:.3e}'.format(
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
