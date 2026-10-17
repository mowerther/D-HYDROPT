"""
The differentiable HYDROPT forward model.

These kernels contain only array operations, so JAX traces them and they
compose with jit, grad and vmap. rrs_poly() evaluates the polynomial at the
IOPs it receives, including IOPs far outside the region the polynomial was
fitted to, where Rrs can underflow to zero. The guards in hydropt_jax.domain
validate a state before and after these functions, outside traced code.
"""
import jax
import jax.numpy as jnp


def monomial_powers_2d(max_degree=4):
    '''
    Exponent pairs (i, j) for each monomial a^i * bb^j up to max_degree.

    Ordered by total degree, then by descending power of the first variable:
    (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ... The order matches both the
    column order of the shipped coefficient tables and PolynomialFeatures of
    scikit-learn, which the reference NumPy implementation uses.
    tests/test_jax_equivalence.py verifies that agreement.
    '''
    powers = []
    for degree in range(max_degree + 1):
        for i in range(degree, -1, -1):
            powers.append((i, degree - i))

    return jnp.asarray(powers, dtype=jnp.int32)


def polynomial_features_2d(x, powers):
    '''
    Evaluate each monomial at x.

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
             water. See hydropt_jax.domain.iops_from_excess.
    coeffs - coefficient table, shape (n_bands, n_terms)
    powers - exponent pairs from monomial_powers_2d, shape (n_terms, 2)

    Returns Rrs in sr^-1, shape (n_bands,).

    The fit is in log space on both sides, so rrs_poly computes
    exp(sum_k c_k * log(a)^i_k * log(bb)^j_k). Outside the fitted domain the
    exponent can reach several thousand in magnitude. The result and its
    gradient then underflow to zero, and rrs_poly returns these zeros.
    hydropt_jax.domain detects such states.
    '''
    x = jnp.stack([jnp.log(a), jnp.log(bb)], axis=0)
    log_rrs = jnp.sum(coeffs * polynomial_features_2d(x, powers), axis=1)

    return jnp.exp(log_rrs)


def _rrs_poly_single(state, coeffs, powers):
    return rrs_poly(state[0], state[1], coeffs, powers)


#: rrs_poly over a leading batch axis: states of shape (n, 2, n_bands).
rrs_poly_batched = jax.vmap(_rrs_poly_single, in_axes=(0, None, None))
