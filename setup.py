from setuptools import setup
from setuptools import find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyaudioaliasingmetrics",
    version="0.1.3",
    description="Computes SNR and SINAD from spectral data",
    author="Maxime COUTANT",
    author_email="maxime.coutant@protonmail.com",
    url="https://github.com/maxmarsc/pyaudioaliasingmetrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    install_requires=["numpy", "numba"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    packages=find_packages(),
)
