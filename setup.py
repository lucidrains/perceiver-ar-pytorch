from setuptools import setup, find_packages

setup(
  name = 'perceiver-ar-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.6',
  license='MIT',
  description = 'Perceiver AR',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/perceiver-ar-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'long context',
    'attention'
  ],
  install_requires=[
    'einops>=0.4',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
