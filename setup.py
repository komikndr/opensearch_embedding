from setuptools import setup, find_packages

setup(
    name='opensearch-embedding',
    version='0.1.0',
    description='A library for generating embeddings with OpenSearch',
    author='Micko Lesmana',
    author_email='mickolesmana@gmail.com',
    license='???',
    packages=find_packages(),
    install_requires=[
        'opensearch-py',
        'langchain-core',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
