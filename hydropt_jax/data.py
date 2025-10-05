
import numpy as np
import jax.numpy as jnp
import pandas as pd
import pkg_resources

ORDER = ['1','a','bb','a^2','a bb','bb^2','a^3','a^2 bb','a bb^2','bb^3','a^4','a^3 bb','a^2 bb^2','a bb^3','bb^4']

def load_pace_poly():
    import pandas as pd, pkg_resources, jax.numpy as jnp
    stream = pkg_resources.resource_filename('hydropt', 'data/PACE_polynom_04_h2o.csv')
    df = pd.read_csv(stream)
    if 'wavelength' not in df.columns:
        df = pd.read_csv(stream, index_col=0).reset_index().rename(columns={'index':'wavelength'})
    lam = df['wavelength'].to_numpy()
    coeffs = df[ORDER].to_numpy()
    return jnp.asarray(lam, jnp.float64), jnp.asarray(coeffs, jnp.float64)

def resample_coeffs(lam_src, coeffs_src, lam_tgt):
    lam_src = np.asarray(lam_src, float); lam_tgt = np.asarray(lam_tgt, float)
    out = np.empty((lam_tgt.size, coeffs_src.shape[1]), float)
    for k in range(coeffs_src.shape[1]):
        out[:, k] = np.interp(lam_tgt, lam_src, np.asarray(coeffs_src[:, k]))
    return jnp.asarray(out, dtype=jnp.float64)
