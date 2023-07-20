from setuptools import setup, find_packages

setup(name="eugene-tools", packages=find_packages())

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="eugene-tools",
    version="0.1.0",
    author="Adam Klie",
    author_email="aklie@ucsd.edu",
    description="Elucidating the Utility of Genomic Elements with Neural Nets",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/ML4GLand/EUGENe",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
