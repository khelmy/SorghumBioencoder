# http://liufuyang.github.io/2017/04/02/just-another-tensorflow-beginner-guide-4.html
from setuptools import setup, find_packages

setup(name='src',
  version='0.1',
  packages=find_packages(),
  description='helps run keras on gcloud ml-engine',
  author='karim helmy',
  author_email='khelmy@andrew.cmu.edu',
  license='MIT',
  install_requires=[
      'keras',
      'h5py'
  ],
  zip_safe=False)
