from setuptools import setup, find_packages

setup(
    name='InFlags',
    version='0.1',
    packages=find_packages(),
    install_requires=[],
    url='https://github.com/Kiryukhasemenov/InFlags',
    license='CC BY 4.0',
    author='Kirill Semenov (Kiryukhasemenov)',
    author_email='kirill.semenov@uzh.ch',
    description='Python package for dictionary-based inline tokenization preprocessing',
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
)
