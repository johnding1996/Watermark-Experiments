from setuptools import setup, find_packages

# Reading requirements from 'requirements.txt'
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="wmbench",
    version="0.1",
    packages=find_packages(),
    py_modules=["cli"],
    install_requires=requirements,
    dependency_links=[
        "https://download.pytorch.org/whl/cu118",
    ],
    entry_points={
        "console_scripts": ["wmbench=cli:cli"]  # Pointing to the cli function in cli.py
    },
    # Other metadata
)
