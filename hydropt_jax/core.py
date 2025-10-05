
import jax
import jax.numpy as jnp

def monomial_powers_2d(max_degree=4):
    ps = []
    for d in range(max_degree+1):
        for i in range(d, -1, -1):
            ps.append((i, d-i))
    return jnp.asarray(ps, dtype=jnp.int32)

def polynomial_features_2d(x, powers):
    x0 = x[0][jnp.newaxis, :]
    x1 = x[1][jnp.newaxis, :]
    i = powers[:, 0][:, jnp.newaxis]
    j = powers[:, 1][:, jnp.newaxis]
    feats = (x0 ** i) * (x1 ** j)
    return feats.T

def rrs_poly(a, bb, coeffs, powers):
    x = jnp.stack([jnp.log(a), jnp.log(bb)], axis=0)
    f = polynomial_features_2d(x, powers)
    log_rrs = jnp.sum(coeffs * f, axis=1)
    return jnp.exp(log_rrs)

def _rrs_poly_single(params, coeffs, powers):
    a, bb = params
    return rrs_poly(a, bb, coeffs, powers)

rrs_poly_batched = jax.vmap(_rrs_poly_single, in_axes=(0, None, None))
