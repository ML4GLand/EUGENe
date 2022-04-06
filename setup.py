from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="eugene",
    version="0.0.0",
    author="Adam Klie",
    author_email="aklie@eng.ucsd.edu",
    description="Elucidating and Understanding Grammar of Enhancers with Neural Nets",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/EUGENE",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
