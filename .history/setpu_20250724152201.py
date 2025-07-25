from setuptools import setup, find_packages

setup(
    name="geister",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pennylane",
        "qiskit",
        "matplotlib",
        "tqdm",
        "graphviz",
        "seaborn",
        "pandas",
        "scipy"
    ]
)
