from setuptools import setup, find_packages

setup(name = 'eugene', packages = find_packages())

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="eugene-tools",
    version="0.0.4",
    author="Adam Klie",
    author_email="aklie@eng.ucsd.edu",
    description="Elucidating the Utility of Genomic Elements with Neural Nets",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/EUGENe",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
