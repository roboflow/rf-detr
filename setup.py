from setuptools import setup, find_packages
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
req_file = os.path.join(base_dir, "requirements.txt")

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setup(
    name="rfdetr",
    version="1.0.8",
    description="RF-DETR",
    author="Roboflow, Inc",
    author_email="peter@roboflow.com",
    license="Apache License 2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts': [
            'rfdetr = rfdetr.cli.main:trainer',
        ],
    },
    url="https://github.com/roboflow/rf-detr",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        'Typing :: Typed',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    keywords="machine-learning, deep-learning, vision, ML, DL, AI, DETR, RF-DETR, Roboflow",
)
