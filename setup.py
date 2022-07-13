from itertools import chain

from setuptools import find_packages, setup

from KED import __author__, __author_email__, __description__, __name__, __version__

# Projects with optional features for building the documentation and running
# tests. From setuptools:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
extra_feature_requirements = {
    "doc": [
        "furo",
        "ipykernel",  # https://github.com/spatialaudio/nbsphinx/issues/121
        "nbsphinx >= 0.7",
        "sphinx >= 3.0.2",
        "sphinx-gallery >= 0.6",
        "sphinxcontrib-bibtex >= 1.0",
        "scikit-learn",
    ],
    "tests": ["pytest >= 5.4", "pytest-cov >= 2.8.1", "coverage >= 5.0"],
}
extra_feature_requirements["dev"] = [
    "black >= 22.3",
    "isort",
    "manifix",
    "pre-commit >= 1.16",
] + list(chain(*list(extra_feature_requirements.values())))

setup(
    name=__name__,
    version=str(__version__),
    license="All rights reserved",
    author=__author__,
    author_email=__author_email__,
    description=__description__,
    long_description=open("README.md").read(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(exclude=["KED/tests"]),
    extras_require=extra_feature_requirements,
    install_requires=[
        "ase",
        "diffpy.structure >= 3",
        "IPython",
        "ipywidgets",
        "matplotlib >= 3.4",
        "numba",
        "numpy",
        "orix >= 0.9",
        "pandas",
        "scikit-image",
        "scipy",
        "tqdm",
    ],
    package_data={"": ["LICENSE", "README.md", "readthedocs.yml"], "KED": ["*.py"]},
)
