from setuptools import setup, find_packages

# Reading requirements from 'requirements.txt'
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="wmbench",
    version="0.1",
    packages=find_packages(),
    py_modules=["cli", "utils", "tree_ring", "dev_scripts/reverse"],
    install_requires=requirements,
    entry_points={
        "console_scripts": ["wmbench=cli:cli"]  # Pointing to the cli function in cli.py
    },
    # Other metadata
)
