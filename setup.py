from setuptools import setup, find_packages

setup(
    name="ml_determination",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,  # to include non-code files
    install_requires=[
        # Add your dependencies here
        "spacy",
        "numpy",
        "torch",
        "transformers"
    ],
    author="Olga Iakovenko",
    description="A package for determining the matrix language in bilingual sentences.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)