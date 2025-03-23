from setuptools import setup, find_packages

setup(
    name="promptbuilder",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "aisuite",
    ],
    author="Kapulkin Stanislav",
    author_email="kapulkin@gmail.com",
    description="Library for building prompts for LLMs",
    long_description=open("Readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kapulkin/promptbuilder",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
) 