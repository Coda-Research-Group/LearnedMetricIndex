from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name="li",
    version="0.1",
    packages=find_packages(where="search"),
    package_dir={"": "search"},
    py_modules=[splitext(basename(path))[0] for path in glob("search/*.py")],
)
