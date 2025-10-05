
import jax, jax.numpy as jnp, numpy as np
from hydropt_jax.core import monomial_powers_2d, rrs_poly

def test_grad_shapes():
    W = 5
    a = jnp.linspace(0.01,0.1,W)
    bb = jnp.linspace(0.001,0.01,W)
    powers = monomial_powers_2d(4)
    coeffs = jnp.ones((W, powers.shape[0]))
    def f(a_, bb_): return rrs_poly(a_, bb_, coeffs, powers)
    ga, gb = jax.jacrev(f, argnums=(0,1))(a, bb)
    assert ga.shape == (W, W)
    assert gb.shape == (W, W)
