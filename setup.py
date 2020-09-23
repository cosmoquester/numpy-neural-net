from setuptools import find_packages, setup

setup(
    name="numpy-neural-net",
    version="0.0.1",
    description="This is neural network samples implemented with numpy without machine learning libraries.",
    python_requires='>=3.7',
    install_requires=[],
    url="https://github.com/psj8252/numpy-neural-net.git",
    author="Park Sangjun",
    packages=find_packages(exclude=["tests"]),
)
