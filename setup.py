import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rankerNN2pmml",
    version="0.1.2",
    author="Yinxiao Li",
    author_email="liyinxiao1227@gmail.com",
    description="Exporter of pairwise ranker with Neural Nets as underlying model into PMML.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liyinxiao/rankerNN2pmml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
