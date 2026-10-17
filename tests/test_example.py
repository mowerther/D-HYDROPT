"""The worked example runs end to end with Gaussian and tabulated response functions."""
import contextlib
import importlib.util
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402

EXAMPLE = Path(__file__).resolve().parent.parent / 'examples' / 'msi_gradient_inversion.py'


def load_example():
    '''Import the example script as a module.'''
    spec = importlib.util.spec_from_file_location('msi_gradient_inversion', EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


class TestExample(unittest.TestCase):

    def test_runs_with_each_kind_of_response_function(self):
        example = load_example()
        for srf in ('gaussian', 'S2B'):
            with self.subTest(srf=srf), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'figure.png'
                stdout = io.StringIO()
                # check_domain() warns about the wavelengths above 680 nm.
                with contextlib.redirect_stdout(stdout), warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    example.main(['--subset', '40', '--sample', '8', '--gd-steps', '5',
                                  '--jacobian-samples', '4', '--srf', srf, '--out', str(out)])
                plt.close('all')
                text = stdout.getvalue()
                self.assertTrue(out.is_file())
                self.assertIn('704 nm   coverage', text)
                self.assertIn('discarded: 740 nm, 780 nm', text)
                self.assertIn('fitted with a truncation bias: 704 nm', text)
                self.assertIn('with 0 exact zeros', text)


if __name__ == '__main__':
    unittest.main()
