from setuptools import setup, find_packages

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="candatascience",
    version="0.1.0",
    author="Can Şentürk",
    author_email="cansenturks@hotmail.com",
    description="Personal library for data science and machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cansenturk1/candatascience",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "tensorflow>=2.5.0",
        "keras>=2.4.0",
    ],
)