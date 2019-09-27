from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='ml',
    version='0.0.1',
    description='',
    long_description=readme,
    author='Tomoya Koike',
    author_email='makeffort134@gmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
