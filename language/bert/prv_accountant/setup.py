# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import setuptools

version = '0.1.0a2'

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.readlines()

with open('requirements_extra.txt') as f:
    requirements_extra = f.readlines()

setuptools.setup(
    name='prv_accountant',
    version=version,
    description='A fast algorithm to optimally compose privacy guarantees of differentially private (DP) mechanisms to arbitrary accuracy.',  # noqa: E501
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/microsoft/prv_accountant',
    author='Microsoft Corporation',
    packages=["prv_accountant"],
    python_requires=">=3.6.0",
    include_package_data=True,
    extras_require={
        "extra": requirements_extra
    },
    install_requires=requirements,
    scripts=[
        'bin/compute-dp-epsilon',
    ],
    test_suite="tests",
    zip_safe=False
)
