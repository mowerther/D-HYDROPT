"""Access to the packaged data files (polynomial coefficients and IOP spectra)."""
from importlib.resources import files

__all__ = ['data_path']


def data_path(name):
    '''
    Absolute path to a data file shipped inside hydropt/data.

    name - file name, e.g. 'PACE_polynom_04_h2o.csv'
    '''
    return str(files(__package__).joinpath(name))
