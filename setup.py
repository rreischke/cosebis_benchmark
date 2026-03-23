from setuptools import setup, find_packages

setup(
    name="cosebis_benchmark",
    version="0.1.0",
    description="High-precision COSEBIs benchmark filter functions",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "mpmath",
        "scipy",
        "joblib",
        "tqdm",
        # levin is part of OneCovariance — install from https://github.com/rreischke/OneCovariance
    ],
    extras_require={
        "plots": ["matplotlib"],
    },
)
