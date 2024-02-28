# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from setuptools import find_packages, setup

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import allennlp whilst setting up.
VERSION = {}  # type: ignore
with open("llmlingua/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

INSTALL_REQUIRES = [
    "transformers>=4.26.0",
    "accelerate",
    "torch",
    "tiktoken",
    "nltk",
    "numpy",
]
QUANLITY_REQUIRES = [
    "black==21.4b0",
    "flake8>=3.8.3",
    "isort>=5.5.4",
    "pre-commit",
    "pytest",
    "pytest-xdist",
]
DEV_REQUIRES = INSTALL_REQUIRES + QUANLITY_REQUIRES

setup(
    name="llmlingua",
    version=VERSION["VERSION"],
    author="The LLMLingua team",
    author_email="hjiang@microsoft.com",
    description="To speed up LLMs' inference and enhance LLM's perceive of key information, compress the prompt and KV-Cache, which achieves up to 20x compression with minimal performance loss.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    keywords="Prompt Compression, LLMs, Inference Acceleration, Black-box LLMs, Efficient LLMs",
    license="MIT License",
    url="https://github.com/microsoft/LLMLingua",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "."},
    packages=find_packages("."),
    extras_require={
        "dev": DEV_REQUIRES,
        "quality": QUANLITY_REQUIRES,
    },
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    python_requires=">=3.8.0",
    zip_safe=False,
)
