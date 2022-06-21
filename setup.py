from setuptools import setup, find_packages

setup(
  name = 'metaformer-gpt',
  packages = find_packages(exclude=[]),
  version = '0.0.5',
  license='MIT',
  description = 'Metaformer - GPT',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/metaformer-gpt',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention-less'
  ],
  install_requires=[
    'einops>=0.4',
    'scipy',
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
