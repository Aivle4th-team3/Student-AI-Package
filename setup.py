from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='student-ai',
    version='0.2.7',
    description='Package that bundles student AI functions',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='aivle-4th-team3',
    packages=find_packages(),
    install_requires=[
    ],
)
