"""Install command: pip3 install -e ."""
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['torchdiffeq', 'wfdb', 'pandas', 'numpy', 'matplotlib', 'biosppy', 'torch']

setup(name='ecg_ode_lib',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      description='ECG ODE paper',
      url='http://github.com/tomergolany/ecg_ode',
      author='Tomer Golany',
      author_email='tomer.golany@gmail.com',
      license='Technion',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)