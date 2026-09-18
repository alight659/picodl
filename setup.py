from setuptools import setup, find_packages

setup(
    name="picodl-nn",
    version="0.1.1",
    author="Aanis Ali Shah",
    description="A tiny deep learning library built from scratch requiring just numpy.",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
    ]
)