import os

from setuptools import setup


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


packages = [
    'attack_splitnn',
    'attack_splitnn.attack',
    'attack_splitnn.defense',
    'attack_splitnn.measure',
    'attack_splitnn.splitnn',
    'attack_splitnn.utils'
]

console_scripts = [
]

setup(
    name='attack_splitnn',
    version='0.0.0',
    description='Attacking SplitNN',
    author='Hideaki Takahashi',
    author_email='koukyosyumei@hotmail.com',
    license="MIT",
    install_requires=read_requirements(),
    url="https://github.com/Koukyosyumei/Attack_SplitNN",
    package_dir={"": "src"},
    packages=packages,
    entry_points={'console_scripts': console_scripts},
)
