from setuptools import setup, find_packages

setup(
    name="deepbde",
    version="0.1.0",
    description="DeepBDE: a graph neural network for fast and accurate bond dissociation enthalpies",
    author="MSRG",
    packages=find_packages(),
    install_requires=[],
    include_package_data=True,
    python_requires=">=3.7",
)
