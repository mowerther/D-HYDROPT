"""
The differentiable HYDROPT forward model.

These are bare kernels: no validation, no data-dependent branching, nothing
that would stop them being traced. That is deliberate, so they compose freely
with jit, grad and vmap. It also means rrs_poly() will evaluate the polynomial
anywhere it is asked to, including well outside the region it was fitted on,
where it returns exactly zero without complaint. hydropt_jax.domain holds the
guards; use them around these functions rather than inside them.
"""
import jax
import jax.numpy as jnp


def monomial_powers_2d(max_degree=4):
    '''
    Exponent pairs (i, j) for every monomial a^i * bb^j up to max_degree.

    Ordered by total degree, then by descending power of the first variable:
    (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ... This matches both the column
    order of the shipped coefficient tables and sklearn's PolynomialFeatures,
    which the reference NumPy implementation uses. tests/test_jax_equivalence.py
    pins that agreement.
    '''
    powers = []
    for degree in range(max_degree + 1):
        for i in range(degree, -1, -1):
            powers.append((i, degree - i))

    return jnp.asarray(powers, dtype=jnp.int32)


def polynomial_features_2d(x, powers):
    '''
    Evaluate every monomial at x.

    x      - stacked variables, shape (2, n_bands)
    powers - exponent pairs, shape (n_terms, 2)

    Returns shape (n_bands, n_terms).
    '''
    x0 = x[0][jnp.newaxis, :]
    x1 = x[1][jnp.newaxis, :]
    i = powers[:, 0][:, jnp.newaxis]
    j = powers[:, 1][:, jnp.newaxis]

    return ((x0 ** i) * (x1 ** j)).T


def rrs_poly(a, bb, coeffs, powers):
    '''
    Remote sensing reflectance from total absorption and backscattering.

    a, bb  - total IOPs in m^-1, shape (n_bands,). Total means including pure
             water; see hydropt_jax.domain.iops_from_excess.
    coeffs - coefficient table, shape (n_bands, n_terms)
    powers - exponent pairs from monomial_powers_2d, shape (n_terms, 2)

    Returns Rrs in sr^-1, shape (n_bands,).

    The fit is in log space on both sides, so this is
    exp(sum_k c_k * log(a)^i_k * log(bb)^j_k). Outside the fitted domain the
    exponent can reach several thousand in magnitude and the result underflows
    to exactly zero, gradient included -- silently. See hydropt_jax.domain.
    '''
    x = jnp.stack([jnp.log(a), jnp.log(bb)], axis=0)
    log_rrs = jnp.sum(coeffs * polynomial_features_2d(x, powers), axis=1)

    return jnp.exp(log_rrs)


def _rrs_poly_single(state, coeffs, powers):
    return rrs_poly(state[0], state[1], coeffs, powers)


#: rrs_poly over a leading batch axis: states of shape (n, 2, n_bands).
rrs_poly_batched = jax.vmap(_rrs_poly_single, in_axes=(0, None, None))
