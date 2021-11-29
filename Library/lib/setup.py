from setuptools import setup, find_packages

def get_requirements(requirements_path='requirements.txt'):
    with open(requirements_path) as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]

from setuptools import find_packages, setup
setup(
    name='libjs',
    packages=find_packages(include=['libjs']),
    version='0.1.0',
    description='Project 2  library classification_covertype',
    author='searjo',
    license='MIT',
    install_requires=get_requirements(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
    
