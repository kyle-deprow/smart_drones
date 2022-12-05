"""This module integrates with setuptools so that we can pip install our package easily
"""

from setuptools import setup, find_packages
setup(
  name='speech_bootstrap',
  version='0.0.0',
  description='speech_bootstrap: synthetic speech bootstrapping library',
  packages=find_packages('src'),
  install_requires=['num2words',\
                    'nltk',\
                    'tables',\
                    'torchaudio==0.9.0',\
                    'webrtcvad',\
                    'pillow==4.1.1',\
                    'llvmlite',\
                    'Cython',\
                    'pydub',\
                    'ffmpeg-normalize'],
  package_dir={'': 'src'},
  test_suite='test'
)
