"""
Setup Script for Modular DRL Framework.

This script configures the installation of the Modular Deep Reinforcement Learning (DRL) Framework, as described in the DRL survey and project engineering blueprint.

Detailed Description:
The setup script uses setuptools to package the framework, specifying dependencies, package structure, and metadata. It ensures that all required libraries for DRL research and experimentation are installed, and that the modular source code is discoverable by Python. The script is designed for reproducibility and ease of installation in research and production environments.

Key Concepts/Algorithms:
- Python packaging with setuptools
- Dependency management for DRL research
- Modular source code structure (src/)

Important Parameters/Configurations:
- Package name, version, author, and description
- List of required dependencies (see requirements.txt)
- Python version requirement (>=3.7)

Expected Inputs/Outputs:
- Inputs: None (run as a script)
- Outputs: Installed Python package and dependencies

Dependencies:
- setuptools, torch, numpy, gym, PyYAML, matplotlib, tensorboard, tqdm, seaborn

Author: REL Project Team
Date: 2025-07-13
"""
from setuptools import setup, find_packages

setup(
    name="modular_drl_framework",
    version="0.1.0",
    description="A modular Deep Reinforcement Learning framework based on Wang et al. survey.",
    author="REL Project Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "gym",
        "PyYAML",
        "matplotlib",
        "tensorboard",
        "tqdm",
        "seaborn"
    ],
    include_package_data=True,
    python_requires=">=3.7",
)
