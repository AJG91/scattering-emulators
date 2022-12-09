from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='emulate',
    version='0.1',
    packages=find_packages(where='emulate'),
    package_dir={'': 'emulate'},
    py_modules=[splitext(basename(path))[0] for path in glob('emulate/*.py')],
    description='Uses KVP and NVP emulators to make predictions for SMS chiral interaction',
    author='Alberto J. Garcia',
    author_email='garcia.823@osu.edu',
    zip_safe=False
)
