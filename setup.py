from setuptools import setup

import versioneer


def get_long_description():
    '''README.md, for PyPI.'''
    try:
        with open('README.md', encoding='utf-8') as readme:
            return readme.read()
    except IOError:
        return None


setup(
    name='d-hydropt',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url='https://github.com/mowerther/D-HYDROPT',
    author='Mortimer Werther',
    description=(
        'D-HYDROPT: differentiable HYDROPT for gradient-based inversion of '
        'ocean colour observations'),
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    license='AGPL-3.0',
    python_requires='>=3.10',
    # hydropt.data is a real package, not just a data directory: it carries the
    # data_path() helper that both hydropt and hydropt_jax import. Omitting it
    # here still ships the CSVs, but leaves hydropt.data as an implicit
    # namespace package with no data_path, so the install imports and then
    # fails on first use.
    packages=['hydropt', 'hydropt.data', 'hydropt_jax'],
    package_data={'hydropt.data': ['*.csv']},
    install_requires=[
        'numpy>=2.0',
        'pandas>=2.0',
        'matplotlib>=3.7',
        'scikit-learn>=1.3',
        'xarray>=2023.1',
        'jax>=0.4.31'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Atmospheric Science'],
    zip_safe=False)
