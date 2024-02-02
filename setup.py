from setuptools import setup, find_packages

# Reading requirements from 'requirements_cli.txt'
with open("requirements_cli.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="wmbench",
    version="0.2.1",
    packages=find_packages(),
    py_modules=["cli"],
    install_requires=requirements,
    entry_points={
        "console_scripts": ["wmbench=cli:cli"]  # Pointing to the cli function in cli.py
    },
    # Other metadata
)
